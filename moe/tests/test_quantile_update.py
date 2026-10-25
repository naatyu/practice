"""AI-authored local QB lifecycle tests: exact snapshots or pooled histograms."""

import pytest
import torch

from moe.router import TopKRouter


@pytest.mark.parametrize("num_bins", [None, 32])
@pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
def test_qb_update_replaces_from_full_window_and_resets_for_next_step(
    num_bins, score_func
):
    torch.manual_seed(150)
    router = TopKRouter(
        4, 4, 2, score_func=score_func, use_quantile_balancing=True, num_bins=num_bins
    )
    with torch.no_grad():
        router.expert_bias.copy_(torch.tensor([0.15, -0.2, 0.05, 0.0]))
    bias_object = router.expert_bias
    for _step in range(2):
        old_bias = router.expert_bias.clone()
        scores = []
        for sequence_length in (3, 1):
            x = torch.randn(2, sequence_length, 4, requires_grad=True)
            with torch.no_grad():
                logits = router.router(x)
                raw = (
                    logits.sigmoid() if score_func == "sigmoid" else logits.softmax(-1)
                )
                scores.append(raw.reshape(-1, 4))
            _ids, weights, _aux, _z = router(x)
            (weights * torch.tensor([1.0, -2.0])).sum().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all()
            torch.testing.assert_close(router.expert_bias, old_bias)
        if num_bins is None:
            assert router.bias_histogram is None
            assert len(router._quantile_score_chunks) == 2
            for observed, expected_scores in zip(
                router._quantile_score_chunks, scores, strict=True
            ):
                assert not observed.requires_grad and observed.grad_fn is None
                torch.testing.assert_close(observed, expected_scores)
            expected = router.estimate_quantile_bias(torch.cat(scores), old_bias, 2)
        else:
            assert router.bias_histogram is not None
            torch.testing.assert_close(
                router.bias_histogram.sum(-1), torch.full((4,), 8)
            )
            lower, upper = old_bias.min() - 1, old_bias.max() + 1
            expected = router.estimate_bias_from_histogram(
                router.bias_histogram, lower, (upper - lower) / num_bins, 2
            )
        # Diagnostic reset must not destroy the QB window or its token count.
        router.reset_metrics()
        assert not any(
            "chunks" in name or "histogram" in name for name in router.state_dict()
        )
        router.update_quantile_bias()
        assert router.expert_bias is bias_object  # Preserve registered-buffer identity.
        torch.testing.assert_close(router.expert_bias, expected)
        assert torch.isfinite(router.expert_bias).all()
        assert not router.expert_bias.requires_grad
        assert router._quantile_score_chunks == []
        if router.bias_histogram is not None:
            assert router.bias_histogram.count_nonzero() == 0
        # No new forwards: repeated end-of-step calls are harmless.
        router.update_quantile_bias()
        torch.testing.assert_close(router.expert_bias, expected)


@pytest.mark.parametrize("num_bins", [None, 32])
def test_empty_qb_window_leaves_nonzero_bias_unchanged(num_bins):
    router = TopKRouter(4, 4, 2, use_quantile_balancing=True, num_bins=num_bins)
    with torch.no_grad():
        router.expert_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.2]))
    original = router.expert_bias.clone()
    router.update_quantile_bias()
    torch.testing.assert_close(router.expert_bias, original)


@pytest.mark.parametrize("num_bins", [None, 32])
def test_eval_neither_collects_nor_applies_pending_window(num_bins):
    router = TopKRouter(4, 4, 2, use_quantile_balancing=True, num_bins=num_bins)
    router(torch.randn(1, 3, 4))
    before_bias = router.expert_bias.clone()
    histogram = None if router.bias_histogram is None else router.bias_histogram.clone()
    chunks = [chunk.clone() for chunk in router._quantile_score_chunks]
    router.eval()
    router(torch.randn(1, 2, 4))
    router.update_quantile_bias()
    torch.testing.assert_close(router.expert_bias, before_bias)
    if histogram is not None:
        torch.testing.assert_close(router.bias_histogram, histogram)
    assert len(router._quantile_score_chunks) == len(chunks)
    for observed, expected in zip(router._quantile_score_chunks, chunks, strict=True):
        torch.testing.assert_close(observed, expected)


def test_disabled_qb_allocates_no_histogram_and_does_not_update():
    router = TopKRouter(4, 4, 4)
    assert router.num_bins is None and router.bias_histogram is None
    assert "bias_histogram" not in dict(router.named_buffers())
    original = router.expert_bias.clone()
    router(torch.randn(1, 3, 4))
    router.update_quantile_bias()
    torch.testing.assert_close(router.expert_bias, original)
    assert router._quantile_score_chunks == []
