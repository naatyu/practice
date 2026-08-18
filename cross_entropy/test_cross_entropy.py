import pytest
import torch
import torch.nn.functional as F

from cross_entropy import CrossEntropy


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_matches_pytorch(reduction: str) -> None:
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 3, 5
    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float64)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    actual = CrossEntropy(reduction)(logits, targets)  # type: ignore[arg-type]
    expected = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction=reduction,
    )
    if reduction == "none":
        expected = expected.reshape(batch_size, seq_len)

    torch.testing.assert_close(actual, expected)


def test_extreme_logits_remain_finite() -> None:
    logits = torch.tensor([[[1_000.0, 999.0, -1_000.0]]])
    targets = torch.tensor([[0]])

    actual = CrossEntropy()(logits, targets)
    expected = F.cross_entropy(logits.reshape(1, 3), targets.reshape(1))

    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)


def test_gradients_match_pytorch() -> None:
    torch.manual_seed(0)
    vocab_size = 5
    logits = torch.randn(2, 3, vocab_size, dtype=torch.float64, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    targets = torch.randint(0, vocab_size, (2, 3))

    actual = CrossEntropy()(logits, targets)
    expected = F.cross_entropy(
        reference_logits.reshape(-1, vocab_size), targets.reshape(-1)
    )
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_invalid_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        CrossEntropy("median")  # type: ignore[arg-type]


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_ignore_index_matches_pytorch_without_mutating_targets(
    reduction: str,
) -> None:
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 3, 5
    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float64)
    targets = torch.tensor([[2, -100, 0], [4, 1, -100]])
    original_targets = targets.clone()

    actual = CrossEntropy(reduction, ignore_index=-100)(  # type: ignore[arg-type]
        logits, targets
    )
    expected = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        original_targets.reshape(-1),
        reduction=reduction,
        ignore_index=-100,
    )
    if reduction == "none":
        expected = expected.reshape(batch_size, seq_len)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(targets, original_targets)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_all_targets_ignored_matches_pytorch(reduction: str) -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.full((2, 3), -100)
    original_targets = targets.clone()

    actual = CrossEntropy(reduction, ignore_index=-100)(  # type: ignore[arg-type]
        logits, targets
    )
    expected = F.cross_entropy(
        logits.reshape(-1, 5),
        original_targets.reshape(-1),
        reduction=reduction,
        ignore_index=-100,
    )
    if reduction == "none":
        expected = expected.reshape_as(original_targets)

    if reduction == "mean":
        assert torch.isnan(actual)
        assert torch.isnan(expected)
    else:
        torch.testing.assert_close(actual, expected)
