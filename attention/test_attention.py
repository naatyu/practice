import pytest
import torch
import torch.nn.functional as F

from attention import attention


def test_zero_scores_produce_mean_of_values() -> None:
    """With equal scores, every value receives the same weight."""
    q = torch.zeros(1, 1, 3, 2)
    k = torch.zeros(1, 1, 3, 2)
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]]]])

    actual = attention(q, k, v)
    expected_token = v.mean(dim=-2, keepdim=True)
    expected = expected_token.expand_as(v)

    torch.testing.assert_close(actual, expected)


def test_matches_pytorch_reference() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 7, dtype=torch.float64)
    k = torch.randn(2, 3, 5, 7, dtype=torch.float64)
    v = torch.randn(2, 3, 5, 7, dtype=torch.float64)

    actual = attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)

    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected)


def test_causal_mask() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 7, dtype=torch.float64)
    k = torch.randn(2, 3, 5, 7, dtype=torch.float64)
    v = torch.randn(2, 3, 5, 7, dtype=torch.float64)

    actual = attention(q, k, v, causal=True)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected)


def test_cached_causal_attention_matches_full_attention_suffix() -> None:
    """Suffix queries over the full K/V prefix must match a full causal pass."""
    torch.manual_seed(0)
    q = torch.randn(2, 3, 6, 7, dtype=torch.float64)
    k = torch.randn(2, 3, 6, 7, dtype=torch.float64)
    v = torch.randn(2, 3, 6, 7, dtype=torch.float64)

    full_output = attention(q, k, v, causal=True)

    query_len = 2
    cached_output = attention(q[:, :, -query_len:, :], k, v, causal=True)

    assert cached_output.shape == (2, 3, query_len, 7)
    torch.testing.assert_close(cached_output, full_output[:, :, -query_len:, :])


def test_causal_attention_rejects_query_longer_than_keys() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)

    with pytest.raises(ValueError):
        attention(q, k, v, causal=True)
