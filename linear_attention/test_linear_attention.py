import pytest
import torch

from linear_attention import naive_recurrent_linear_attention


def test_controlled_key_value_associations() -> None:
    q = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    v = torch.tensor([[[[2.0], [3.0]]]])

    output, final_state = naive_recurrent_linear_attention(
        q,
        k,
        v,
        output_final_state=True,
    )

    expected_output = torch.tensor([[[[2.0], [5.0]]]])
    expected_state = torch.tensor([[[[2.0], [3.0]]]])
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_state, expected_state)


def test_matches_parallel_causal_reference() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    k = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    v = torch.randn(2, 3, 5, 6, dtype=torch.float64)

    output, final_state = naive_recurrent_linear_attention(
        q,
        k,
        v,
        output_final_state=True,
    )

    causal_scores = (q @ k.transpose(-1, -2)).tril()
    expected_output = causal_scores @ v
    expected_state = k.transpose(-1, -2) @ v
    assert output.shape == (2, 3, 5, 6)
    assert final_state is not None
    assert final_state.shape == (2, 3, 4, 6)
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_state, expected_state)


def test_split_sequence_matches_single_recurrence() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 7, 4, dtype=torch.float64)
    k = torch.randn(2, 3, 7, 4, dtype=torch.float64)
    v = torch.randn(2, 3, 7, 5, dtype=torch.float64)

    full_output, full_state = naive_recurrent_linear_attention(
        q,
        k,
        v,
        output_final_state=True,
    )
    first_output, first_state = naive_recurrent_linear_attention(
        q[..., :3, :],
        k[..., :3, :],
        v[..., :3, :],
        output_final_state=True,
    )
    second_output, split_state = naive_recurrent_linear_attention(
        q[..., 3:, :],
        k[..., 3:, :],
        v[..., 3:, :],
        initial_state=first_state,
        output_final_state=True,
    )

    split_output = torch.cat((first_output, second_output), dim=-2)
    torch.testing.assert_close(split_output, full_output)
    torch.testing.assert_close(split_state, full_state)


def test_final_state_is_optional() -> None:
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 5)

    output, final_state = naive_recurrent_linear_attention(q, k, v)

    assert output.shape == (1, 2, 3, 5)
    assert final_state is None


def test_rejects_incorrect_initial_state_shape() -> None:
    q = torch.randn(2, 3, 4, 5)
    k = torch.randn(2, 3, 4, 5)
    v = torch.randn(2, 3, 4, 6)
    wrong_state = torch.zeros(1, 3, 5, 6)

    with pytest.raises(ValueError, match="(?i)state"):
        naive_recurrent_linear_attention(q, k, v, initial_state=wrong_state)


def test_gradients_flow_through_recurrence() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 4, 5, dtype=torch.float64, requires_grad=True)

    output, _ = naive_recurrent_linear_attention(q, k, v)
    output.square().sum().backward()

    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
