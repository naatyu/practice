"""AI-authored tests for packed and grouped expert computation."""

import math

import pytest
import torch

from mlp import SwiGLU
from moe.experts import GroupedExperts, PackedExperts, _expert_activation


@pytest.mark.parametrize("counts", [[0, 2, 0, 3], [0, 0, 0, 0]])
def test_packed_experts_accept_empty_groups_and_preserve_grouped_order(counts):
    torch.manual_seed(32)
    experts = PackedExperts(3, 5, 4).double()
    offsets = torch.tensor(counts).cumsum(0)
    tokens = torch.randn(sum(counts), 3, dtype=torch.float64)
    actual = experts(tokens, offsets)
    outputs = []
    start = 0
    for e, count in enumerate(counts):
        reference = SwiGLU(3, 5).double()
        with torch.no_grad():
            reference.up_proj.weight.copy_(experts.up_weight[e].T)
            reference.down_proj.weight.copy_(experts.down_weight[e].T)
        outputs.append(reference(tokens[start : start + count]))
        start += count
    torch.testing.assert_close(actual, torch.cat(outputs))


def test_packed_parameters_have_fan_in_scaled_normal_initialization():
    torch.manual_seed(33)
    experts = PackedExperts(128, 256, 4)
    assert set(dict(experts.named_parameters())) == {"up_weight", "down_weight"}
    assert experts.up_weight.shape == (4, 128, 512)
    assert experts.down_weight.shape == (4, 256, 128)
    for weight, fan_in in [(experts.up_weight, 128), (experts.down_weight, 256)]:
        std = fan_in**-0.5
        assert abs(weight.mean().item()) < std * 0.02
        assert abs(weight.std().item() / std - 1) < 0.02


def test_grouped_mm_matches_packed_with_empty_experts():
    torch.manual_seed(35)
    packed = PackedExperts(16, 32, 4)
    grouped = GroupedExperts(16, 32, 4)
    grouped.load_state_dict(packed.state_dict())
    offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    tokens = torch.randn(5, 16, requires_grad=True)
    grouped_tokens = tokens.detach().clone().requires_grad_()
    probe = torch.randn(5, 16)
    expected = packed(tokens, offsets)
    actual = grouped(grouped_tokens, offsets)
    torch.testing.assert_close(actual, expected)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(grouped_tokens.grad, tokens.grad)
    torch.testing.assert_close(grouped.up_weight.grad, packed.up_weight.grad)
    torch.testing.assert_close(grouped.down_weight.grad, packed.down_weight.grad)


@pytest.mark.parametrize("expert_cls", [PackedExperts, GroupedExperts])
@pytest.mark.parametrize(
    "bad_beta", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
@pytest.mark.parametrize("position", [0, 1])
def test_softcap_betas_must_both_be_finite_and_positive(expert_cls, bad_beta, position):
    betas = (bad_beta, 3.0) if position == 0 else (0.5, bad_beta)
    with pytest.raises(ValueError, match="finite, strictly positive"):
        expert_cls(16, 32, 4, softcap_betas=betas)


def test_situ_matches_scalar_math_and_bounds_each_hidden_feature():
    # Unequal caps catch swapped tuple order and multiplication vs division.
    values = [-1000.0, -3.0, 0.0, 2.0, 1000.0]
    gates = [1000.0, -2.0, 1.0, 3.0, 1000.0]
    gate_beta, value_beta = 0.5, 3.0
    expected = [
        value_beta
        * math.tanh(value / value_beta)
        * gate_beta
        * math.tanh(gate / gate_beta)
        / (1 + math.exp(-gate))
        for value, gate in zip(values, gates, strict=True)
    ]
    hidden = _expert_activation(
        torch.tensor(values, dtype=torch.float64),
        torch.tensor(gates, dtype=torch.float64),
        (gate_beta, value_beta),
    )
    torch.testing.assert_close(hidden, torch.tensor(expected, dtype=torch.float64))
    assert hidden.abs().max() <= gate_beta * value_beta


@pytest.mark.parametrize("betas", [None, (0.5, 3.0)])
def test_expert_activation_passes_numerical_gradient_check(betas):
    torch.manual_seed(102)
    value = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    gate = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value, gate: _expert_activation(value, gate, betas), (value, gate)
    )


@pytest.mark.parametrize("betas", [None, (0.5, 3.0), (4.0, 25.0)])
def test_situ_backends_match_outputs_and_gradients_with_empty_experts(betas):
    torch.manual_seed(103)
    packed = PackedExperts(16, 32, 4, softcap_betas=betas)
    grouped = GroupedExperts(16, 32, 4, softcap_betas=betas)
    grouped.load_state_dict(packed.state_dict(), strict=True)
    assert set(packed.state_dict()) == {"up_weight", "down_weight"}
    offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    x = torch.randn(5, 16, requires_grad=True)
    x_grouped = x.detach().clone().requires_grad_()
    expected = packed(x, offsets)
    actual = grouped(x_grouped, offsets)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(x_grouped.grad, x.grad)
    torch.testing.assert_close(grouped.up_weight.grad, packed.up_weight.grad)
    torch.testing.assert_close(grouped.down_weight.grad, packed.down_weight.grad)
