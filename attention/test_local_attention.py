import pytest
import torch

from attention import attention
from attention.local_attention import efficient_local_attention, local_attention


def test_controlled_local_window_output() -> None:
    """Equal scores make each query average exactly its visible values."""
    q = torch.zeros(1, 1, 5, 1)
    k = torch.zeros(1, 1, 5, 1)
    v = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)

    actual = local_attention(q, k, v, window_size=3)

    expected = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0]).reshape(1, 1, 5, 1)
    torch.testing.assert_close(actual, expected)


def test_window_size_one_only_attends_to_current_token() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 7)
    k = torch.randn(2, 3, 5, 7)
    v = torch.randn(2, 3, 5, 7)

    actual = local_attention(q, k, v, window_size=1)

    torch.testing.assert_close(actual, v)


def test_large_window_matches_dense_causal_attention() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 6, 7, dtype=torch.float64)
    k = torch.randn(2, 3, 6, 7, dtype=torch.float64)
    v = torch.randn(2, 3, 6, 5, dtype=torch.float64)

    actual = local_attention(q, k, v, window_size=6)
    expected = attention(q, k, v, causal=True)

    assert actual.shape == (2, 3, 6, 5)
    torch.testing.assert_close(actual, expected)


def test_cached_suffix_matches_full_local_attention() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    k = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    v = torch.randn(2, 3, 7, 4, dtype=torch.float64)

    full_output = local_attention(q, k, v, window_size=3)

    query_len = 2
    cached_output = local_attention(
        q[..., -query_len:, :],
        k,
        v,
        window_size=3,
    )

    assert cached_output.shape == (2, 3, query_len, 4)
    torch.testing.assert_close(cached_output, full_output[..., -query_len:, :])


@pytest.mark.parametrize("window_size", [1, 2, 4, 10])
def test_efficient_implementation_matches_dense_reference(
    window_size: int,
) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    k = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    v = torch.randn(2, 3, 7, 4, dtype=torch.float64)

    expected = local_attention(q, k, v, window_size)
    actual = efficient_local_attention(q, k, v, window_size)

    assert actual.shape == (2, 3, 7, 4)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("query_len", [1, 2, 5])
def test_efficient_cached_suffix_matches_dense_reference(query_len: int) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, query_len, 5, dtype=torch.float64)
    k = torch.randn(2, 3, 8, 5, dtype=torch.float64)
    v = torch.randn(2, 3, 8, 4, dtype=torch.float64)

    expected = local_attention(q, k, v, window_size=3)
    actual = efficient_local_attention(q, k, v, window_size=3)

    assert actual.shape == (2, 3, query_len, 4)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("window_size", [0, -1])
def test_rejects_non_positive_window_size(window_size: int) -> None:
    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)

    with pytest.raises(ValueError, match="(?i)window size"):
        local_attention(q, k, v, window_size=window_size)


def test_rejects_more_queries_than_keys() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)

    with pytest.raises(ValueError, match="(?i)key.*query"):
        local_attention(q, k, v, window_size=2)
