"""AI-authored checks for the learner's pooled histogram quantile extraction."""

import pytest
import torch

from moe.router import TopKRouter


def _scalar_reference(histogram, lower, width, top_k):
    """Independent row loops: no searchsorted, gather, or tensor cumulative sum."""
    target = sum(histogram[0].tolist()) * top_k / len(histogram)
    estimates = []
    for row in histogram.tolist():
        before = 0
        for index, count in enumerate(row):
            if before + count >= target:
                estimates.append(lower + (index + (target - before) / count) * width)
                break
            before += count
    mean = sum(estimates) / len(estimates)
    return torch.tensor([value - mean for value in estimates], dtype=torch.float64)


def test_int32_counts_extract_same_biases_as_int64():
    router = TopKRouter(4, 4, 1, score_func="sigmoid")
    histogram = torch.tensor([[2, 5, 1, 2], [0, 1, 4, 5], [5, 0, 0, 5], [0, 0, 0, 10]])
    lower, width = torch.tensor(-1.0), torch.tensor(0.5)
    expected = router.estimate_bias_from_histogram(histogram, lower, width, 1)
    actual = router.estimate_bias_from_histogram(histogram.int(), lower, width, 1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogram_bias_matches_hand_calculated_interpolation(dtype):
    router = TopKRouter(4, 4, 1, score_func="sigmoid")
    histogram = torch.tensor([[2, 5, 1, 2], [0, 1, 4, 5], [5, 0, 0, 5], [0, 0, 0, 10]])
    result = router.estimate_bias_from_histogram(
        histogram,
        torch.tensor(-1.0, dtype=dtype),
        torch.tensor(0.5, dtype=dtype),
        1,
    )
    # Target count 2.5 gives uncentered estimates [-.45, .1875, -.75, .625].
    expected = torch.tensor([-0.353125, 0.284375, -0.653125, 0.721875], dtype=dtype)
    assert result.shape == (4,)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result.mean(), result.new_zeros(()), atol=1e-7, rtol=0)


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_bin_zero_trailing_empty_bins_and_exact_cumulative_boundary(top_k):
    router = TopKRouter(4, 4, top_k)
    # At K=1 target=2. Earliest cumulative >=2 must choose the occupied bin,
    # not an empty bin on the plateau; otherwise interpolation divides by zero.
    histogram = torch.tensor([[2, 0, 0, 6], [0, 2, 6, 0], [8, 0, 0, 0], [0, 0, 0, 8]])
    actual = router.estimate_bias_from_histogram(
        histogram, torch.tensor(-1.0), torch.tensor(0.5), top_k
    )
    expected = _scalar_reference(histogram, -1.0, 0.5, top_k).to(actual.dtype)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("num_bins", [1, 8, 64])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_accumulate_then_extract_matches_scalar_histogram_reference(num_bins, top_k):
    torch.manual_seed(130)
    router = TopKRouter(5, 5, top_k, score_func="sigmoid")
    scores = torch.rand(19, 5)
    bias = torch.tensor([0.1, -0.1, 0.05, 0.0, 0.2])
    histogram = torch.zeros(5, num_bins, dtype=torch.int64)
    for chunk in scores.split([3, 7, 9]):
        lower, width = router.accumulate_bias_histogram(chunk, bias, top_k, histogram)
    actual = router.estimate_bias_from_histogram(histogram, lower, width, top_k)
    expected = _scalar_reference(histogram, lower.item(), width.item(), top_k).to(
        actual.dtype
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_extraction_preserves_inputs_and_state_and_is_no_grad():
    router = TopKRouter(4, 4, 2, score_func="sigmoid")
    histogram = torch.tensor([[2, 5, 1, 2], [0, 1, 4, 5], [5, 0, 0, 5], [0, 0, 0, 10]])
    lower = torch.tensor(-1.0, requires_grad=True)
    width = torch.tensor(0.5, requires_grad=True)
    originals = histogram.clone(), lower.detach().clone(), width.detach().clone()
    buffers = {name: value.clone() for name, value in router.named_buffers()}
    result = router.estimate_bias_from_histogram(histogram, lower, width, 2)
    assert not result.requires_grad and result.grad_fn is None
    for value, original in zip((histogram, lower, width), originals, strict=True):
        torch.testing.assert_close(value, original)
    for name, value in router.named_buffers():
        torch.testing.assert_close(value, buffers[name])


def test_expert_permutation_and_shared_lower_bound_shift_do_not_change_bias_differences():
    router = TopKRouter(4, 4, 2)
    # Noncontiguous histogram, to exercise the batched lookup on strided data.
    histogram = torch.tensor(
        [[2, 0, 5, 0], [5, 1, 0, 0], [1, 4, 0, 0], [2, 5, 5, 10]]
    ).T
    assert not histogram.is_contiguous()
    lower, width = torch.tensor(-1.0), torch.tensor(0.5)
    original = router.estimate_bias_from_histogram(histogram, lower, width, 2)
    order = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(
        router.estimate_bias_from_histogram(histogram[order], lower, width, 2),
        original[order],
    )
    torch.testing.assert_close(
        router.estimate_bias_from_histogram(histogram, lower + 10, width, 2),
        original,
    )
