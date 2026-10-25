"""AI-authored tests for the requested histogram-accumulation teaching helper."""

import math

import pytest
import torch

from moe.router import TopKRouter


def _reference_counts(scores, bias, top_k, num_bins):
    """Scalar loops/sorting, independent of tensor top-k and scatter_add_."""
    scores, bias = scores.tolist(), bias.tolist()
    lower, upper = min(bias) - 1, max(bias) + 1
    width = (upper - lower) / num_bins
    counts = torch.zeros(len(bias), num_bins, dtype=torch.int64)
    for row in scores:
        cutoff = sorted(
            (score + b for score, b in zip(row, bias, strict=True)), reverse=True
        )[top_k]
        for expert, score in enumerate(row):
            index = math.floor((cutoff - score - lower) / width)
            counts[expert, min(max(index, 0), num_bins - 1)] += 1
    return counts, lower, width


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("top_k", [1, 2, 3])
@pytest.mark.parametrize("counter_dtype", [torch.int32, torch.int64])
def test_histogram_matches_scalar_reference_and_counts_every_token(
    dtype, top_k, counter_dtype
):
    router = TopKRouter(4, 4, top_k, score_func="sigmoid")
    scores = torch.tensor(
        [
            [0.9, 0.7, 0.4, 0.1],
            [0.2, 0.8, 0.6, 0.3],
            [0.5, 0.1, 0.9, 0.4],
            [0.3, 0.6, 0.2, 0.8],
        ],
        dtype=dtype,
    ).T
    bias = torch.tensor([0.1, -0.1, 0.05, 0.0], dtype=dtype)
    assert not scores.is_contiguous()
    histogram = torch.zeros(4, 7, dtype=counter_dtype)
    lower, width = router.accumulate_bias_histogram(scores, bias, top_k, histogram)
    expected, expected_lower, expected_width = _reference_counts(scores, bias, top_k, 7)
    torch.testing.assert_close(histogram, expected.to(counter_dtype))
    assert histogram.dtype == counter_dtype
    torch.testing.assert_close(
        histogram.sum(-1), torch.full((4,), 4, dtype=torch.int64)
    )
    assert lower.item() == pytest.approx(expected_lower)
    assert width.item() == pytest.approx(expected_width)
    assert lower.ndim == width.ndim == 0
    assert lower.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)


@pytest.mark.parametrize("num_bins", [1, 8])
@pytest.mark.parametrize("counter_dtype", [torch.int32, torch.int64])
def test_microbatch_addition_matches_pooled_histogram_without_mutating_inputs(
    num_bins, counter_dtype
):
    router = TopKRouter(4, 4, 2, score_func="sigmoid")
    torch.manual_seed(125)
    scores = torch.rand(11, 4, requires_grad=True)
    bias = torch.tensor([0.1, -0.1, 0.05, 0.0], requires_grad=True)
    before_scores, before_bias = scores.detach().clone(), bias.detach().clone()
    before_buffers = {name: value.clone() for name, value in router.named_buffers()}
    pooled = torch.zeros(4, num_bins, dtype=counter_dtype)
    chunks = torch.zeros_like(pooled)
    lower, width = router.accumulate_bias_histogram(scores, bias, 2, pooled)
    for microbatch in scores.split([2, 0, 6, 3]):
        chunk_lower, chunk_width = router.accumulate_bias_histogram(
            microbatch, bias, 2, chunks
        )
        torch.testing.assert_close(chunk_lower, lower)
        torch.testing.assert_close(chunk_width, width)
    torch.testing.assert_close(chunks, pooled)
    # Separate rank-local histograms can be summed in the same way.
    rank_histograms = []
    for part in scores.split([4, 7]):
        local = torch.zeros_like(pooled)
        router.accumulate_bias_histogram(part, bias, 2, local)
        rank_histograms.append(local)
    torch.testing.assert_close(sum(rank_histograms), pooled)
    assert not lower.requires_grad and not width.requires_grad
    torch.testing.assert_close(scores, before_scores)
    torch.testing.assert_close(bias, before_bias)
    for name, value in router.named_buffers():
        torch.testing.assert_close(value, before_buffers[name])


def test_required_bias_endpoints_and_internal_bin_edges():
    router = TopKRouter(3, 3, 1, score_func="sigmoid")
    # With zero biases the range is [-1,1], width .5.
    # Cutoffs: 1, 0, .5. Required biases: [0,0,1], [-1,0,0], [0,0,0].
    scores = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    histogram = torch.zeros(3, 4, dtype=torch.int64)
    router.accumulate_bias_histogram(scores, torch.zeros(3), 1, histogram)
    torch.testing.assert_close(
        histogram, torch.tensor([[1, 0, 2, 0], [0, 0, 3, 0], [0, 0, 2, 1]])
    )


def test_empty_microbatch_leaves_existing_counts_unchanged():
    router = TopKRouter(4, 4, 2, score_func="sigmoid")
    histogram = torch.ones(4, 8, dtype=torch.int64)
    router.accumulate_bias_histogram(torch.empty(0, 4), torch.zeros(4), 2, histogram)
    torch.testing.assert_close(histogram, torch.ones_like(histogram))
