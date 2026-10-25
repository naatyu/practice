"""AI-authored checks for the learner's exact, frozen-cutoff QB estimator."""

import pytest
import torch

from moe.router import TopKRouter


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_quantile_bias_matches_hand_computed_example(dtype):
    router = TopKRouter(4, 4, 1, score_func="sigmoid")
    scores = torch.tensor(
        [
            [0.9, 0.7, 0.4, 0.1],
            [0.2, 0.8, 0.6, 0.3],
            [0.5, 0.1, 0.9, 0.4],
            [0.3, 0.6, 0.2, 0.8],
        ],
        dtype=dtype,
    )
    bias = torch.tensor([0.1, -0.1, 0.05, 0.0], dtype=dtype)
    actual = router.estimate_quantile_bias(scores, bias, top_k=1)
    # Biased second-largest cutoffs: [.6, .65, .6, .5].
    # Column-wise required-bias 25% quantiles, with linear interpolation:
    # [0, -.1125, -.0375, .075], whose mean is -.01875.
    expected = torch.tensor([0.01875, -0.09375, -0.01875, 0.09375], dtype=dtype)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (4,) and actual.dtype == dtype
    torch.testing.assert_close(actual.mean(), actual.new_zeros(()), atol=1e-7, rtol=0)


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_single_token_uses_k_plus_one_cutoff_and_raw_scores(top_k):
    router = TopKRouter(4, 4, top_k, score_func="sigmoid")
    scores = torch.tensor([[0.9, 0.7, 0.4, 0.1]], dtype=torch.float64)
    bias = torch.tensor([0.0, -0.6, 0.8, 0.0], dtype=torch.float64)
    actual = router.estimate_quantile_bias(scores, bias, top_k)
    # With one token, its shared cutoff disappears during mean-centering.
    # Subtracting biased scores instead would incorrectly retain the old biases.
    expected = scores.mean() - scores.squeeze(0)
    torch.testing.assert_close(actual, expected)


def test_estimator_preserves_inputs_and_router_state_without_autograd():
    router = TopKRouter(4, 4, 2, score_func="sigmoid")
    scores = torch.rand(7, 4, requires_grad=True)
    bias = torch.tensor([0.1, -0.1, 0.05, 0.0], requires_grad=True)
    before_scores, before_bias = scores.detach().clone(), bias.detach().clone()
    before_buffers = {name: value.clone() for name, value in router.named_buffers()}
    result = router.estimate_quantile_bias(scores, bias, 2)
    assert not result.requires_grad and result.grad_fn is None
    torch.testing.assert_close(scores, before_scores)
    torch.testing.assert_close(bias, before_bias)
    for name, value in router.named_buffers():
        torch.testing.assert_close(value, before_buffers[name])


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_estimator_respects_expert_token_order_and_common_bias_shift(top_k):
    generator = torch.Generator().manual_seed(120)
    router = TopKRouter(5, 5, top_k, score_func="sigmoid")
    # Exercise a noncontiguous [T,E] score matrix as well.
    scores = torch.rand(5, 13, generator=generator, dtype=torch.float64).T
    bias = torch.randn(5, generator=generator, dtype=torch.float64) * 0.1
    assert not scores.is_contiguous()
    expected = router.estimate_quantile_bias(scores, bias, top_k)
    expert_order = torch.tensor([2, 4, 0, 3, 1])
    token_order = torch.randperm(13, generator=generator)
    torch.testing.assert_close(
        router.estimate_quantile_bias(
            scores[:, expert_order], bias[expert_order], top_k
        ),
        expected[expert_order],
    )
    torch.testing.assert_close(
        router.estimate_quantile_bias(scores[token_order], bias, top_k),
        expected,
    )
    torch.testing.assert_close(
        router.estimate_quantile_bias(scores, bias + 10, top_k),
        expected,
    )


def test_equal_scores_and_biases_produce_zero_centered_targets():
    router = TopKRouter(4, 4, 2, score_func="sigmoid")
    actual = router.estimate_quantile_bias(torch.full((5, 4), 0.5), torch.ones(4), 2)
    torch.testing.assert_close(actual, torch.zeros(4))
    # This checks finite estimates under ties, NOT guaranteed balanced top-k ties.
