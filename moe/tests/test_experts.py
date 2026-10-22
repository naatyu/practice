"""AI-authored tests for packed and grouped expert computation."""

import pytest
import torch

from mlp import SwiGLU
from moe.experts import GroupedExperts, PackedExperts


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
