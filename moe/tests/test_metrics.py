"""AI-authored tests for MoE logging metrics."""

import pytest
import torch

from moe.metrics import (
    assignment_entropy,
    assignment_fraction,
    maxvio,
    router_entropy,
    unused_expert_count,
)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([10, 20, 30, 20], 0.5),
        ([20, 20, 20, 20], 0.0),
        ([0, 0, 0, 8], 3.0),
        ([0, 0, 0, 0], 0.0),
        ([7], 0.0),
        ([20, 40, 60, 40], 0.5),  # Relative imbalance is scale-invariant.
    ],
)
def test_maxvio_controlled_values(counts, expected):
    counts = torch.tensor(counts, dtype=torch.int64)
    original = counts.clone()
    result = maxvio(counts)
    assert result.shape == ()
    assert result.device == counts.device
    assert result.is_floating_point()
    assert torch.isfinite(result)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=result.dtype))
    torch.testing.assert_close(counts, original)


def test_maxvio_does_not_track_gradients():
    counts = torch.tensor([10.0, 20.0, 30.0, 20.0], requires_grad=True)
    result = maxvio(counts)
    assert not result.requires_grad
    assert result.grad_fn is None


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([10, 20, 30, 20], [0.125, 0.25, 0.375, 0.25]),
        ([20, 20, 20, 20], [0.25, 0.25, 0.25, 0.25]),
        ([0, 0, 0, 8], [0.0, 0.0, 0.0, 1.0]),
        ([0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_assignment_fraction_values_shape_and_unchanged_counts(counts, expected):
    counts = torch.tensor(counts, dtype=torch.int64)
    original = counts.clone()
    result = assignment_fraction(counts)
    assert result.shape == counts.shape
    assert result.device == counts.device
    assert result.is_floating_point()
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor(expected, dtype=result.dtype))
    torch.testing.assert_close(counts, original)
    expected_sum = 1.0 if counts.sum() > 0 else 0.0
    torch.testing.assert_close(result.sum(), result.new_tensor(expected_sum))


def test_assignment_fraction_is_scale_invariant():
    counts = torch.tensor([1, 2, 0, 5])
    torch.testing.assert_close(
        assignment_fraction(counts), assignment_fraction(counts * 10)
    )


def test_assignment_fraction_does_not_track_gradients():
    counts = torch.tensor([10.0, 20.0, 30.0, 20.0], requires_grad=True)
    result = assignment_fraction(counts)
    assert not result.requires_grad
    assert result.grad_fn is None


@pytest.mark.parametrize(
    ("counts", "expected"),
    [([0, 10, 0, 30], 2), ([1, 2, 3, 4], 0), ([0, 0, 0, 0], 4), ([0], 1)],
)
def test_unused_expert_count(counts, expected):
    counts = torch.tensor(counts, dtype=torch.int64)
    original = counts.clone()
    result = unused_expert_count(counts)
    assert result.shape == ()
    assert result.dtype == torch.int64
    assert result.device == counts.device
    assert result.item() == expected
    assert not result.requires_grad
    torch.testing.assert_close(counts, original)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([1, 1, 1, 1], 2.0),
        ([1, 1, 0, 0], 1.0),
        ([4, 0, 0, 0], 0.0),
        ([0, 0, 0, 0], 0.0),
    ],
)
def test_assignment_entropy_in_bits(counts, expected):
    counts = torch.tensor(counts, dtype=torch.int64)
    original = counts.clone()
    result = assignment_entropy(counts)
    assert result.shape == ()
    assert result.device == counts.device
    assert result.is_floating_point()
    assert torch.isfinite(result)
    assert result >= 0
    torch.testing.assert_close(result, result.new_tensor(expected))
    torch.testing.assert_close(counts, original)


def test_assignment_entropy_is_scale_and_expert_order_invariant():
    counts = torch.tensor([1, 2, 0, 5])
    expected = assignment_entropy(counts)
    torch.testing.assert_close(assignment_entropy(counts * 10), expected)
    torch.testing.assert_close(assignment_entropy(counts.flip(0)), expected)


def test_assignment_entropy_does_not_track_gradients():
    counts = torch.tensor([10.0, 20.0, 30.0, 20.0], requires_grad=True)
    result = assignment_entropy(counts)
    assert not result.requires_grad
    assert result.grad_fn is None


def test_router_entropy_returns_one_value_per_token_in_bits():
    probabilities = torch.tensor(
        [[[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    )
    original = probabilities.clone()
    result = router_entropy(probabilities)
    assert result.shape == (1, 3)
    assert result.device == probabilities.device
    torch.testing.assert_close(result, torch.tensor([[2.0, 1.0, 0.0]]))
    torch.testing.assert_close(probabilities, original)
    assert torch.isfinite(result).all()


def test_confident_router_can_have_balanced_assignments():
    probabilities = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    torch.testing.assert_close(router_entropy(probabilities).mean(), torch.tensor(0.0))
    counts = torch.bincount(probabilities.argmax(-1).flatten(), minlength=2)
    torch.testing.assert_close(assignment_entropy(counts), torch.tensor(1.0))


def test_router_entropy_does_not_track_gradients():
    probabilities = torch.tensor([[[0.75, 0.25]]], requires_grad=True)
    result = router_entropy(probabilities)
    assert not result.requires_grad
    assert result.grad_fn is None
