import pytest
import torch

from linear_attention.delta_net import naive_recurrent_gated_delta_net
from linear_attention.kimi_delta_attention import (
    CausalDepthwiseConv1d,
    KimiDeltaAttention,
    recurrent_kda,
)


def test_causal_depthwise_conv_preserves_sequence_and_channel_shapes() -> None:
    conv = CausalDepthwiseConv1d(3, kernel_size=4)
    x = torch.randn(2, 7, 3)

    assert conv(x).shape == x.shape


def test_causal_depthwise_conv_cannot_see_future_tokens() -> None:
    torch.manual_seed(8)
    conv = CausalDepthwiseConv1d(3, kernel_size=4)
    x = torch.randn(2, 7, 3)
    changed_future = x.clone()
    changed_future[:, 4:, :] += 100

    torch.testing.assert_close(conv(x)[:, :4], conv(changed_future)[:, :4])


def test_causal_depthwise_conv_keeps_channels_independent() -> None:
    torch.manual_seed(9)
    channels = 3
    conv = CausalDepthwiseConv1d(channels, kernel_size=3)
    x = torch.randn(2, 6, channels)
    changed_one_channel = x.clone()
    changed_one_channel[:, :, 1] += 100

    assert conv.conv.groups == channels
    unchanged_channels = [0, 2]
    torch.testing.assert_close(
        conv(x)[:, :, unchanged_channels],
        conv(changed_one_channel)[:, :, unchanged_channels],
    )


def test_kda_layer_passes_head_shaped_inputs_to_recurrence(monkeypatch) -> None:
    import linear_attention.kimi_delta_attention as kda_module

    captured = {}

    def capture_recurrence(
        q, k, v, beta, alpha, initial_state=None, *, output_final_state=False
    ):
        captured.update(
            q=q,
            k=k,
            v=v,
            beta=beta,
            alpha=alpha,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return torch.zeros_like(v), None

    monkeypatch.setattr(kda_module, "recurrent_kda", capture_recurrence)
    layer = KimiDeltaAttention(d_model=12, num_heads=3, kernel_size=3)
    layer(torch.randn(2, 5, 12))

    assert captured["q"].shape == (2, 3, 5, 4)
    assert captured["k"].shape == (2, 3, 5, 4)
    assert captured["v"].shape == (2, 3, 5, 4)
    assert captured["beta"].shape == (2, 3, 5)
    assert captured["alpha"].shape == (2, 3, 5, 4)
    for gate in (captured["beta"], captured["alpha"]):
        assert ((0 < gate) & (gate < 1)).all()
    for name in ("q", "k"):
        torch.testing.assert_close(
            torch.linalg.vector_norm(captured[name], dim=-1),
            torch.ones(2, 3, 5),
        )


def test_kda_layer_normalizes_each_head_independently(monkeypatch) -> None:
    import linear_attention.kimi_delta_attention as kda_module

    layer = KimiDeltaAttention(d_model=4, num_heads=2, kernel_size=2)
    recurrent_output = torch.tensor([[[[3.0, 4.0]], [[1.0, 2.0]]]])

    def fake_recurrence(q, k, v, beta, alpha, initial_state=None, *, output_final_state=False):
        return recurrent_output, None

    monkeypatch.setattr(kda_module, "recurrent_kda", fake_recurrence)
    x = torch.randn(1, 1, 4)
    normalized = []
    hook = layer.norm.register_forward_hook(
        lambda _module, _inputs, output: normalized.append(output.detach().clone())
    )
    first, _ = layer(x)

    recurrent_output = torch.tensor([[[[3.0, 4.0]], [[100.0, 200.0]]]])
    second, _ = layer(x)
    hook.remove()

    assert first.shape == (1, 1, 4)
    assert second.shape == first.shape
    torch.testing.assert_close(normalized[0][:, 0], normalized[1][:, 0])
    torch.testing.assert_close(
        normalized[0][:, 0].square().mean(), torch.tensor(1.0)
    )


def test_kda_layer_output_gate_has_a_learned_projection() -> None:
    layer = KimiDeltaAttention(d_model=12, num_heads=3, kernel_size=3)
    linear_layers = [module for module in layer.modules() if isinstance(module, torch.nn.Linear)]

    # QKV, alpha, beta, output projection, and at least one gate projection.
    assert len(linear_layers) >= 5


def test_channelwise_decay_and_correction_affect_distinct_state_rows() -> None:
    q = torch.tensor([[[[0.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0]]]])  # Normalized key.
    v = torch.tensor([[[[10.0]]]])
    beta = torch.ones(1, 1, 1)
    alpha = torch.tensor([[[[0.5, 0.25]]]])
    initial_state = torch.tensor([[[[4.0], [7.0]]]])

    output, final_state = recurrent_kda(
        q, k, v, beta, alpha, initial_state, output_final_state=True
    )

    # The first row is corrected to 10; the second only decays to 7 / 4.
    torch.testing.assert_close(final_state, torch.tensor([[[[10.0], [1.75]]]]))
    torch.testing.assert_close(output, torch.tensor([[[[1.75]]]]))


def test_uniform_channel_decay_matches_scalar_gated_delta_net() -> None:
    torch.manual_seed(5)
    q = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    k = torch.nn.functional.normalize(
        torch.randn(2, 3, 4, 5, dtype=torch.float64), dim=-1
    )
    v = torch.randn(2, 3, 4, 6, dtype=torch.float64)
    beta = torch.rand(2, 3, 4, dtype=torch.float64)
    scalar_alpha = torch.rand(2, 3, 4, dtype=torch.float64)
    channel_alpha = scalar_alpha.unsqueeze(-1).expand_as(k)

    expected, expected_state = naive_recurrent_gated_delta_net(
        q, k, v, beta, scalar_alpha, output_final_state=True
    )
    actual, actual_state = recurrent_kda(
        q, k, v, beta, channel_alpha, output_final_state=True
    )

    assert actual.shape == (2, 3, 4, 6)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, expected_state)


def test_split_sequence_matches_full_kda_recurrence() -> None:
    torch.manual_seed(6)
    q = torch.randn(2, 2, 5, 3, dtype=torch.float64)
    k = torch.randn(2, 2, 5, 3, dtype=torch.float64)
    v = torch.randn(2, 2, 5, 4, dtype=torch.float64)
    beta = torch.rand(2, 2, 5, dtype=torch.float64)
    alpha = torch.rand(2, 2, 5, 3, dtype=torch.float64)

    full_output, full_state = recurrent_kda(
        q, k, v, beta, alpha, output_final_state=True
    )
    prefix_output, prefix_state = recurrent_kda(
        q[..., :2, :],
        k[..., :2, :],
        v[..., :2, :],
        beta[..., :2],
        alpha[..., :2, :],
        output_final_state=True,
    )
    suffix_output, suffix_state = recurrent_kda(
        q[..., 2:, :],
        k[..., 2:, :],
        v[..., 2:, :],
        beta[..., 2:],
        alpha[..., 2:, :],
        initial_state=prefix_state,
        output_final_state=True,
    )

    torch.testing.assert_close(torch.cat((prefix_output, suffix_output), -2), full_output)
    torch.testing.assert_close(suffix_state, full_state)


def test_kda_gradients_optional_state_and_state_shape_validation() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 3, 5, dtype=torch.float64, requires_grad=True)
    beta = torch.rand(1, 2, 3, dtype=torch.float64, requires_grad=True)
    alpha = torch.rand(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    initial_state = torch.randn(1, 2, 4, 5, dtype=torch.float64, requires_grad=True)

    output, final_state = recurrent_kda(
        q, k, v, beta, alpha, initial_state, output_final_state=True
    )
    assert final_state is not None
    (output.square().sum() + final_state.square().sum()).backward()

    for tensor in (q, k, v, beta, alpha, initial_state):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()

    _, omitted_state = recurrent_kda(q, k, v, beta, alpha)
    assert omitted_state is None

    with pytest.raises(ValueError, match="(?i)state"):
        recurrent_kda(q, k, v, beta, alpha, torch.zeros(1, 2, 5, 4))
