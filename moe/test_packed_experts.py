"""AI-authored tests for the learner's packed experts and grouped dispatch."""

import pytest
import torch

from mlp import SwiGLU
from moe.expert_batched_moe import ExpertBatchedMoe
from moe.packed_experts import GroupedExperts, GroupedMoE, PackedExperts
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
    grouped = GroupedMoE(3, 5, 4, top_k, expert_backend="loop").to(dtype)
    copy_reference_weights(reference, grouped)
    x_reference = torch.randn(2, 4, 3, dtype=dtype, requires_grad=True)
    x_grouped = x_reference.detach().clone().requires_grad_()

    expected, expected_aux = reference(x_reference)
    actual, actual_aux = grouped(x_grouped)
    torch.testing.assert_close(actual_aux, expected_aux)
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
    grouped = GroupedMoE(3, 5, 4, 2, expert_backend="loop")
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
        return ids, weights, None


def test_controlled_weight_permutation_and_repeated_token_accumulation():
    grouped = GroupedMoE(2, 1, 4, 2, expert_backend="loop").double()
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
    torch.testing.assert_close(grouped(x)[0], expected)


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


@pytest.mark.parametrize(
    ("backend", "expert_cls"), [("loop", PackedExperts), ("grouped_mm", GroupedExperts)]
)
def test_grouped_moe_selects_expert_backend(backend, expert_cls):
    model = GroupedMoE(16, 32, 4, 2, expert_backend=backend)
    assert isinstance(model.experts, expert_cls)


def test_grouped_moe_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown expert backend"):
        GroupedMoE(16, 32, 4, 2, expert_backend="unknown")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_grouped_mm_outputs_and_gradients_match_expert_batched_reference(top_k):
    torch.manual_seed(34)
    reference = ExpertBatchedMoe(16, 32, 4, top_k)
    packed = GroupedMoE(16, 32, 4, top_k, expert_backend="loop")
    grouped = GroupedMoE(16, 32, 4, top_k, expert_backend="grouped_mm")
    copy_reference_weights(reference, packed)
    grouped.load_state_dict(packed.state_dict())
    tokens = torch.randn(2, 3, 16)
    probe = torch.randn_like(tokens)
    reference_input = tokens.clone().requires_grad_()
    expected, _ = reference(reference_input)
    (expected * probe).sum().backward()

    for model in (packed, grouped):
        x = tokens.clone().requires_grad_()
        actual, _ = model(x)
        torch.testing.assert_close(actual, expected)
        (actual * probe).sum().backward()
        torch.testing.assert_close(x.grad, reference_input.grad)
        torch.testing.assert_close(
            model.router.router.weight.grad, reference.router.router.weight.grad
        )
        assert model.experts.up_weight.grad is not None
        assert model.experts.down_weight.grad is not None
        for expert_id, expert in enumerate(reference.experts):
            assert isinstance(expert, SwiGLU)
            assert expert.up_proj.weight.grad is not None
            assert expert.down_proj.weight.grad is not None
            torch.testing.assert_close(
                model.experts.up_weight.grad[expert_id], expert.up_proj.weight.grad.T
            )
            torch.testing.assert_close(
                model.experts.down_weight.grad[expert_id],
                expert.down_proj.weight.grad.T,
            )


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
