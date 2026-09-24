import pytest
import torch

from linear_attention.delta_net import naive_recurrent_delta_net


def test_beta_controls_correction_without_erasing_other_key_directions() -> None:
    q = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]])
    k = q.clone()
    v = torch.tensor([[[[100.0], [7.0], [1.0]]]])
    beta = torch.tensor([[[0.0, 0.5, 1.0]]])
    initial_state = torch.tensor([[[[2.0], [4.0]]]])

    output, final_state = naive_recurrent_delta_net(
        q, k, v, beta, initial_state, output_final_state=True
    )

    torch.testing.assert_close(output, torch.tensor([[[[2.0], [4.5], [1.0]]]]))
    torch.testing.assert_close(final_state, torch.tensor([[[[1.0], [4.0]]]]))


def test_beta_varies_by_batch_head_and_position() -> None:
    q = torch.ones(2, 2, 2, 1)
    k = q.clone()
    v = torch.tensor([[[[2.0], [6.0]], [[2.0], [6.0]]]]).expand(2, -1, -1, -1)
    beta = torch.tensor([[[0.0, 1.0], [1.0, 0.0]], [[0.5, 0.5], [0.0, 0.5]]])

    output, _ = naive_recurrent_delta_net(q, k, v, beta)

    expected = torch.tensor(
        [[[[0.0], [6.0]], [[2.0], [2.0]]], [[[1.0], [3.5]], [[0.0], [3.0]]]]
    )
    torch.testing.assert_close(output, expected)


def test_split_sequence_matches_full_recurrence() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    k = torch.nn.functional.normalize(
        torch.randn(2, 3, 5, 4, dtype=torch.float64), dim=-1
    )
    v = torch.randn(2, 3, 5, 6, dtype=torch.float64)
    beta = torch.rand(2, 3, 5, dtype=torch.float64)

    full_output, full_state = naive_recurrent_delta_net(
        q, k, v, beta, output_final_state=True
    )
    first_output, first_state = naive_recurrent_delta_net(
        q[..., :2, :],
        k[..., :2, :],
        v[..., :2, :],
        beta[..., :2],
        output_final_state=True,
    )
    second_output, split_state = naive_recurrent_delta_net(
        q[..., 2:, :],
        k[..., 2:, :],
        v[..., 2:, :],
        beta[..., 2:],
        initial_state=first_state,
        output_final_state=True,
    )

    assert full_output.shape == (2, 3, 5, 6)
    assert full_state is not None and full_state.shape == (2, 3, 4, 6)
    torch.testing.assert_close(
        torch.cat((first_output, second_output), dim=-2), full_output
    )
    torch.testing.assert_close(split_state, full_state)


def test_final_state_is_optional_and_wrong_state_shape_is_rejected() -> None:
    q = torch.ones(1, 2, 1, 3)
    k = torch.nn.functional.normalize(q, dim=-1)
    v = torch.ones(1, 2, 1, 4)
    beta = torch.ones(1, 2, 1)

    _, final_state = naive_recurrent_delta_net(q, k, v, beta)
    assert final_state is None

    with pytest.raises(ValueError, match="(?i)state"):
        naive_recurrent_delta_net(q, k, v, beta, initial_state=torch.zeros(1, 2, 4, 4))


def test_gradients_flow_through_inputs_gate_and_initial_state() -> None:
    torch.manual_seed(1)
    q = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 3, 5, dtype=torch.float64, requires_grad=True)
    beta = torch.rand(1, 2, 3, dtype=torch.float64, requires_grad=True)
    initial_state = torch.randn(1, 2, 4, 5, dtype=torch.float64, requires_grad=True)

    output, final_state = naive_recurrent_delta_net(
        q, k, v, beta, initial_state, output_final_state=True
    )
    assert final_state is not None
    (output.square().sum() + final_state.square().sum()).backward()

    for tensor in (q, k, v, beta, initial_state):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
