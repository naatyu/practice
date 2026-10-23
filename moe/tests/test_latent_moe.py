"""AI-authored checks for the learner's latent routed branch."""

import copy
import math

import pytest
import torch
from torch.nn import functional as F

from moe import GroupedMoE


def _reference(model, x):
    """Independent per-assignment math; normalize the combined latent vector."""
    ids, weights, aux, z = model.router(x)
    tokens = x.reshape(-1, model.d_model)
    ids, weights = ids.reshape(-1, model.top_k), weights.reshape(-1, model.top_k)
    latent = F.linear(tokens, model.latent_down.weight)
    capacity = (
        None
        if model.capacity_factor is None
        else math.ceil(model.capacity_factor * ids.numel() / model.num_experts)
    )
    # Build admission lists with Python loops, independently of packed sorting.
    admitted = set()
    for expert in range(model.num_experts):
        candidates = [
            (token, slot)
            for token in range(len(tokens))
            for slot in range(model.top_k)
            if ids[token, slot].item() == expert
        ]
        if capacity is not None and model.drop_policy == "probs":
            candidates.sort(key=lambda pair: -weights[pair].detach().item())
        admitted.update(candidates if capacity is None else candidates[:capacity])

    rows = []
    for token in range(len(tokens)):
        # Zero keeps dropped-token input/projection paths in the autograd graph.
        row = latent[token] * 0
        for slot in range(model.top_k):
            if (token, slot) not in admitted:
                continue
            expert = int(ids[token, slot].item())
            value, gate = (latent[token] @ model.experts.up_weight[expert]).chunk(2)
            betas = model.experts.softcap_betas
            if betas is None:
                hidden = value * F.silu(gate)
            else:
                gate_beta, value_beta = betas
                hidden = (
                    gate_beta
                    * (gate / gate_beta).tanh()
                    * gate.sigmoid()
                    * value_beta
                    * (value / value_beta).tanh()
                )
            y = hidden @ model.experts.down_weight[expert]
            row = row + weights[token, slot] * y
        # Native PyTorch is the norm reference, not our RMSNorm implementation.
        # The norm must follow the sum: it cannot move inside each contribution.
        row = F.rms_norm(
            row, (model.d_latent,), model.latent_norm.weight, model.latent_norm.eps
        )
        rows.append(F.linear(row, model.latent_up.weight))
    output = torch.stack(rows).reshape_as(x)
    if model.shared_expert is not None:
        shared = model.shared_expert(x)
        if model.shared_expert_gate is not None:
            shared = shared * model.shared_expert_gate(x).sigmoid()
        output = output + shared
    return output, aux, z


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
@pytest.mark.parametrize("policy", [None, "position", "probs"])
@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("betas", [None, (0.5, 3.0)])
def test_latent_outputs_and_all_gradients_match_reference(
    backend, policy, score_func, betas
):
    torch.manual_seed(72)
    model = GroupedMoE(
        d_model=32,
        hidden_dim=32,
        num_experts=4,
        top_k=2,
        d_latent=16,
        softcap_betas=betas,
        shared_hidden_dim=24,
        shared_expert_gate=True,
        expert_backend=backend,
        capacity_factor=None if policy is None else 0.3,
        drop_policy=policy or "probs",
        score_func=score_func,
    )
    with torch.no_grad():
        model.latent_norm.weight.copy_(torch.linspace(0.5, 1.5, 16))
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, 32).transpose(0, 1).requires_grad_()
    assert not x.is_contiguous()
    x_ref = x.detach().clone().requires_grad_()
    output, aux, z = model(x)
    expected, expected_aux, expected_z = _reference(reference, x_ref)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(aux, expected_aux)
    torch.testing.assert_close(z, expected_z)

    upstream = torch.randn_like(output)
    assert aux is not None and expected_aux is not None
    loss = (output * upstream).sum() + 0.01 * aux
    ref_loss = (expected * upstream).sum() + 0.01 * expected_aux
    if z is not None:
        assert expected_z is not None
        loss, ref_loss = loss + 0.001 * z, ref_loss + 0.001 * expected_z
    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
    ref_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        expected_grad = ref_params[name].grad
        if expected_grad is None:
            expected_grad = torch.zeros_like(param)
        torch.testing.assert_close(param.grad, expected_grad, rtol=1e-4, atol=1e-5)


def test_latent_branch_boundaries_and_projection_call_counts():
    model = GroupedMoE(
        d_model=32,
        hidden_dim=32,
        num_experts=4,
        top_k=2,
        d_latent=16,
        shared_hidden_dim=24,
        shared_expert_gate=True,
    ).eval()
    x = torch.randn(2, 3, 32)
    seen = {}

    def record(name):
        def hook(module, inputs):
            seen.setdefault(name, []).append(inputs[0].detach().clone())

        return hook

    assert model.shared_expert is not None and model.shared_expert_gate is not None
    handles = [
        module.register_forward_pre_hook(record(name))
        for name, module in (
            ("router", model.router),
            ("shared", model.shared_expert),
            ("gate", model.shared_expert_gate),
            ("down", model.latent_down),
            ("up", model.latent_up),
            ("norm", model.latent_norm),
            ("experts", model.experts),
        )
    ]
    try:
        output, aux, z = model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert output.shape == x.shape and aux is None and z is None
    assert all(len(calls) == 1 for calls in seen.values())
    for name in ("router", "shared", "gate"):
        torch.testing.assert_close(seen[name][0], x)
    torch.testing.assert_close(seen["down"][0], x.reshape(6, 32))
    assert seen["experts"][0].shape == (12, 16)
    assert seen["up"][0].shape == (6, 16)  # Once per token, not per assignment.
    assert seen["norm"][0].shape == (6, 16)
    assert model.latent_norm.weight.shape == (16,)
    assert "latent_norm.weight" in model.state_dict()
    torch.testing.assert_close(
        seen["up"][0],
        F.rms_norm(
            seen["norm"][0], (16,), model.latent_norm.weight, model.latent_norm.eps
        ),
    )
    assert model.latent_down.bias is None and model.latent_up.bias is None
    assert model.experts.up_weight.shape == (4, 16, 64)
    assert model.experts.down_weight.shape == (4, 32, 16)


@pytest.mark.parametrize("d_latent", [0, -1])
def test_nonpositive_latent_width_rejected(d_latent):
    with pytest.raises(ValueError, match="d_latent.*strictly positive"):
        GroupedMoE(32, 32, 4, 2, d_latent=d_latent)


def test_none_keeps_original_parameter_layout():
    torch.manual_seed(90)
    original = GroupedMoE(32, 32, 4, 2)
    torch.manual_seed(90)
    explicit_none = GroupedMoE(32, 32, 4, 2, d_latent=None)
    assert not any("latent" in key for key in explicit_none.state_dict())
    explicit_none.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2, 3, 32)
    torch.testing.assert_close(original(x), explicit_none(x))


@pytest.mark.parametrize("shared", [False, True])
def test_no_admitted_routes_stays_zero_before_optional_shared_branch(shared):
    # Route every token to the same two experts; capacity keeps only token zero.
    model = GroupedMoE(
        32,
        32,
        4,
        2,
        d_latent=16,
        capacity_factor=0.1,
        drop_policy="position",
        shared_hidden_dim=24 if shared else None,
    ).eval()
    with torch.no_grad():
        model.router.router.weight.zero_()
        model.router.expert_bias.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    x = torch.randn(1, 3, 32)
    output, _, _ = model(x)
    expected = torch.zeros_like(x)
    if shared:
        assert model.shared_expert is not None
        expected = model.shared_expert(x)
    torch.testing.assert_close(output[:, 1:], expected[:, 1:])
    output.square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
@pytest.mark.parametrize("d_latent", [None, 16])
def test_caps_reach_only_routed_experts(backend, d_latent):
    torch.manual_seed(110)
    baseline = GroupedMoE(
        32,
        32,
        4,
        2,
        d_latent=d_latent,
        shared_hidden_dim=24,
        shared_expert_gate=True,
        expert_backend=backend,
    ).eval()
    torch.manual_seed(110)
    capped = GroupedMoE(
        32,
        32,
        4,
        2,
        d_latent=d_latent,
        shared_hidden_dim=24,
        shared_expert_gate=True,
        expert_backend=backend,
        softcap_betas=(0.5, 3.0),
    ).eval()
    assert capped.experts.softcap_betas == (0.5, 3.0)
    assert baseline.experts.softcap_betas is None
    # Configuration changes no parameters and does not alter the shared branch.
    capped.load_state_dict(baseline.state_dict(), strict=True)
    assert capped.shared_expert is not None and baseline.shared_expert is not None
    x = torch.randn(2, 3, 32, requires_grad=True)
    torch.testing.assert_close(capped.shared_expert(x), baseline.shared_expert(x))
    actual, _, _ = capped(x)
    assert not torch.allclose(actual, baseline(x)[0])
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
@pytest.mark.parametrize("betas", [(float("nan"), 3.0), (0.5, 0.0)])
def test_grouped_moe_forwards_cap_validation(backend, betas):
    with pytest.raises(ValueError, match="finite, strictly positive"):
        GroupedMoE(32, 32, 4, 2, expert_backend=backend, softcap_betas=betas)
