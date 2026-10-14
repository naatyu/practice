"""AI-authored tests for the educational chunkwise KDA worked example."""

import pytest
import torch

from linear_attention import KimiDeltaAttention, chunked_kda, recurrent_kda


def make_inputs(dtype=torch.float64):
    torch.manual_seed(34)
    q = torch.nn.functional.normalize(torch.randn(2, 2, 7, 3, dtype=dtype), dim=-1)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(2, 2, 7, 4, dtype=dtype)  # Dv need not equal Dk.
    beta = torch.rand(2, 2, 7, dtype=dtype)
    log_retention = -torch.rand_like(k)
    state = torch.randn(2, 2, 3, 4, dtype=dtype)
    return q, k, v, beta, log_retention, state


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 7, 16])
def test_outputs_states_and_gradients_match_recurrence(chunk_size):
    inputs = [x.requires_grad_() for x in make_inputs()]
    q, k, v, beta, log_retention, state = inputs

    actual, actual_state = chunked_kda(
        *inputs, chunk_size=chunk_size, output_final_state=True
    )
    expected, expected_state = recurrent_kda(
        q, k, v, beta, log_retention.exp(), state, output_final_state=True
    )
    assert actual_state is not None and expected_state is not None
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, expected_state)

    # Compare every differentiable input, not just the output's requires_grad.
    actual_gradients = torch.autograd.grad(
        actual.square().sum() + actual_state.square().sum(), inputs, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum() + expected_state.square().sum(), inputs
    )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_controlled_video_example():
    k = torch.tensor([[[[1.0, 0.0], [0.6, 0.8], [0.8, 0.6]]]], dtype=torch.float64)
    v = torch.tensor([[[[2.0], [2.0], [3.0]]]], dtype=torch.float64)
    beta = torch.tensor([[[0.5, 0.75, 1.0]]], dtype=torch.float64)
    alpha = torch.tensor([[[[0.8, 0.9], [0.5, 0.8], [0.9, 0.5]]]], dtype=torch.float64)
    state = torch.ones(1, 1, 2, 1, dtype=torch.float64)

    output, final_state = chunked_kda(
        k, k, v, beta, alpha.log(), state, chunk_size=3, output_final_state=True
    )

    torch.testing.assert_close(
        output.flatten(), torch.tensor([1.4, 1.749, 3.0], dtype=k.dtype)
    )
    assert final_state is not None
    torch.testing.assert_close(
        final_state.flatten(), torch.tensor([2.4558072, 1.7255904], dtype=k.dtype)
    )


def test_split_calls_with_different_chunk_boundaries_match_full_sequence():
    q, k, v, beta, log_retention, state = make_inputs()
    original_state = state.clone()
    full, full_state = chunked_kda(
        q, k, v, beta, log_retention, state, chunk_size=4, output_final_state=True
    )
    prefix, prefix_state = chunked_kda(
        q[:, :, :3],
        k[:, :, :3],
        v[:, :, :3],
        beta[:, :, :3],
        log_retention[:, :, :3],
        state,
        chunk_size=2,
        output_final_state=True,
    )
    suffix, split_state = chunked_kda(
        q[:, :, 3:],
        k[:, :, 3:],
        v[:, :, 3:],
        beta[:, :, 3:],
        log_retention[:, :, 3:],
        prefix_state,
        chunk_size=3,
        output_final_state=True,
    )
    torch.testing.assert_close(torch.cat((prefix, suffix), dim=-2), full)
    assert split_state is not None and full_state is not None
    torch.testing.assert_close(split_state, full_state)
    torch.testing.assert_close(
        state, original_state
    )  # Caller-owned cache is not mutated.


@pytest.mark.parametrize("prefix_length", [2, 4])
def test_future_changes_do_not_affect_prefix(prefix_length):
    inputs = make_inputs()
    original, _ = chunked_kda(*inputs, chunk_size=4)
    changed = [x.clone() for x in inputs]
    for x in changed[:5]:
        x[:, :, prefix_length:] *= 2
    modified, _ = chunked_kda(*changed, chunk_size=4)
    torch.testing.assert_close(
        modified[:, :, :prefix_length], original[:, :, :prefix_length]
    )


@pytest.mark.parametrize("log_gate", [0.0, -1000.0])
@pytest.mark.parametrize("write_strength", [0.0, 1.0])
def test_gate_extremes_are_finite_and_match_recurrence(log_gate, write_strength):
    q, k, v, beta, log_retention, state = make_inputs()
    log_retention = torch.full_like(log_retention, log_gate).requires_grad_()
    beta = torch.full_like(beta, write_strength)
    actual, actual_state = chunked_kda(
        q, k, v, beta, log_retention, state, chunk_size=4, output_final_state=True
    )
    expected, expected_state = recurrent_kda(
        q, k, v, beta, log_retention.exp(), state, output_final_state=True
    )
    torch.testing.assert_close(actual, expected)
    assert actual_state is not None and expected_state is not None
    torch.testing.assert_close(actual_state, expected_state)
    (gradient,) = torch.autograd.grad(actual.sum() + actual_state.sum(), log_retention)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_uses_float32_state_and_preserves_output_dtype(dtype):
    inputs = make_inputs(dtype)
    output, state = chunked_kda(*inputs, chunk_size=4, output_final_state=True)
    q, k, v, beta, log_retention, initial_state = (x.float() for x in inputs)
    expected, expected_state = recurrent_kda(
        q, k, v, beta, log_retention.exp(), initial_state, output_final_state=True
    )
    assert output.dtype == dtype
    assert state is not None and state.dtype == torch.float32
    torch.testing.assert_close(output, expected.to(dtype))
    torch.testing.assert_close(state, expected_state)


def test_default_zero_state_and_empty_sequence():
    inputs = make_inputs()
    q, k, v, beta, log_retention, state = inputs
    actual, returned_state = chunked_kda(*inputs[:5], chunk_size=4)
    expected, _ = recurrent_kda(q, k, v, beta, log_retention.exp())
    torch.testing.assert_close(actual, expected)
    assert returned_state is None

    empty_inputs = [x[:, :, :0] for x in inputs[:5]]
    output, final_state = chunked_kda(
        *empty_inputs, initial_state=state, output_final_state=True
    )
    assert output.shape == (2, 2, 0, 4)
    assert final_state is not None
    torch.testing.assert_close(final_state, state)


def test_autocast_does_not_downcast_internal_state_math():
    inputs = make_inputs(torch.float32)
    expected, expected_state = chunked_kda(*inputs, output_final_state=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, actual_state = chunked_kda(*inputs, output_final_state=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_state is not None and expected_state is not None
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 2.5])
def test_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="positive integer"):
        chunked_kda(*make_inputs(), chunk_size=chunk_size)
    with pytest.raises(ValueError, match="positive integer"):
        KimiDeltaAttention(12, 3, 3, chunk_size=chunk_size)


@pytest.mark.parametrize("index", range(6))
def test_invalid_shapes(index):
    inputs = list(make_inputs())
    inputs[index] = inputs[index][..., :-1]
    with pytest.raises(ValueError):
        chunked_kda(*inputs)


@pytest.mark.parametrize("value", [0.1, float("inf"), float("-inf"), float("nan")])
def test_invalid_log_gates(value):
    inputs = list(make_inputs())
    inputs[4].fill_(value)
    with pytest.raises(ValueError, match="finite and nonpositive"):
        chunked_kda(*inputs)


def test_integer_inputs_are_rejected():
    inputs = list(make_inputs())
    inputs[2] = inputs[2].long()
    with pytest.raises(ValueError, match="floating-point"):
        chunked_kda(*inputs)


def test_full_layer_matches_recurrence_including_parameter_gradients():
    torch.manual_seed(35)
    recurrent = KimiDeltaAttention(12, 3, 3).double()
    chunked = KimiDeltaAttention(12, 3, 3, chunk_size=4).double()
    chunked.load_state_dict(recurrent.state_dict())  # No new learned parameters.
    x = torch.randn(2, 7, 12, dtype=torch.float64, requires_grad=True)
    x_chunked = x.detach().clone().requires_grad_()

    expected, expected_cache = recurrent(x, use_cache=True)
    actual, actual_cache = chunked(x_chunked, use_cache=True)
    torch.testing.assert_close(actual, expected)
    assert expected_cache is not None and actual_cache is not None
    for name in ("state", "q_tail", "k_tail", "v_tail"):
        torch.testing.assert_close(
            getattr(actual_cache, name), getattr(expected_cache, name)
        )

    (actual.square().sum() + actual_cache.state.square().sum()).backward()
    (expected.square().sum() + expected_cache.state.square().sum()).backward()
    torch.testing.assert_close(x_chunked.grad, x.grad)
    for actual_param, expected_param in zip(
        chunked.parameters(), recurrent.parameters()
    ):
        assert actual_param.grad is not None and expected_param.grad is not None
        torch.testing.assert_close(actual_param.grad, expected_param.grad)


def test_full_layer_token_by_token_cache_matches_full_sequence():
    torch.manual_seed(36)
    layer = KimiDeltaAttention(12, 3, 3, chunk_size=4)
    x = torch.randn(2, 7, 12)
    full, full_cache = layer(x, use_cache=True)
    outputs = []
    cache = None
    for t in range(x.shape[1]):
        output, cache = layer(x[:, t : t + 1], cache, use_cache=True)
        outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, dim=1), full)
    assert cache is not None and full_cache is not None
    for name in ("state", "q_tail", "k_tail", "v_tail"):
        torch.testing.assert_close(getattr(cache, name), getattr(full_cache, name))
