"""AI-authored tests for the learner's packed experts and grouped dispatch."""

import pytest
import torch

from mlp import SwiGLU
from moe.expert_batched_moe import ExpertBatchedMoe
from moe.packed_experts import GroupedMoE, PackedExperts
from moe.router import TopKRouter


def copy_reference_weights(reference, grouped):
    grouped.router.load_state_dict(reference.router.state_dict())
    with torch.no_grad():
        for e, expert in enumerate(reference.experts):
            assert isinstance(expert, SwiGLU)
            # nn.Linear stores [out, in]; our packed weights store [in, out].
            grouped.experts.up_weight[e].copy_(expert.up_proj.weight.T)
            grouped.experts.down_weight[e].copy_(expert.down_proj.weight.T)


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grouped_outputs_and_gradients_match_reference(top_k, dtype):
    torch.manual_seed(31)
    reference = ExpertBatchedMoe(3, 5, 4, top_k).to(dtype)
    grouped = GroupedMoE(3, 5, 4, top_k).to(dtype)
    copy_reference_weights(reference, grouped)
    x_reference = torch.randn(2, 4, 3, dtype=dtype, requires_grad=True)
    x_grouped = x_reference.detach().clone().requires_grad_()

    expected = reference(x_reference)
    actual = grouped(x_grouped)
    assert actual.shape == x_grouped.shape
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x_grouped.grad, x_reference.grad)
    torch.testing.assert_close(
        grouped.router.router.weight.grad, reference.router.router.weight.grad
    )
    assert grouped.experts.up_weight.grad is not None
    assert grouped.experts.down_weight.grad is not None
    for e, expert in enumerate(reference.experts):
        assert isinstance(expert, SwiGLU)
        assert expert.up_proj.weight.grad is not None
        assert expert.down_proj.weight.grad is not None
        torch.testing.assert_close(
            grouped.experts.up_weight.grad[e], expert.up_proj.weight.grad.T
        )
        torch.testing.assert_close(
            grouped.experts.down_weight.grad[e], expert.down_proj.weight.grad.T
        )


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


def test_grouped_accepts_noncontiguous_input():
    reference = ExpertBatchedMoe(3, 5, 4, 2)
    grouped = GroupedMoE(3, 5, 4, 2)
    copy_reference_weights(reference, grouped)
    x = torch.randn(2, 4, 3).transpose(0, 1)
    assert not x.is_contiguous()
    torch.testing.assert_close(grouped(x), reference(x))


class FixedRouter(TopKRouter):
    def __init__(self):
        super().__init__(2, 4, 2)

    def forward(self, x):
        ids = torch.tensor([[[1, 3], [3, 1]]], device=x.device)
        weights = x.new_tensor([[[0.25, 0.75], [0.6, 0.4]]])
        return ids, weights


def test_controlled_weight_permutation_and_repeated_token_accumulation():
    grouped = GroupedMoE(2, 1, 4, 2).double()
    grouped.router = FixedRouter()
    # All experts use the same gated hidden scalar, scaled by their down weights.
    with torch.no_grad():
        grouped.experts.up_weight.fill_(1)
        for e, scale in enumerate([1.0, 2.0, 3.0, 4.0]):
            grouped.experts.down_weight[e].fill_(scale)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64)
    hidden = x.sum(-1, keepdim=True)
    hidden = hidden * torch.nn.functional.silu(hidden)
    expected_scales = x.new_tensor([[[3.5], [3.2]]])
    expected = (hidden * expected_scales).expand_as(x)
    torch.testing.assert_close(grouped(x), expected)


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
