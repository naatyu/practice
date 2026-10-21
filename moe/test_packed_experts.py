"""AI-authored tests for the learner's packed experts and grouped dispatch."""

import copy
import math

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


class CapacityRouter(TopKRouter):
    """Uneven routing across batches, including an empty expert and mixed slots."""

    def __init__(self):
        super().__init__(16, 4, 2)
        self.register_buffer(
            "ids", torch.tensor([[[1, 3], [1, 2], [3, 1]], [[1, 3], [2, 3], [1, 2]]])
        )
        self.weights = torch.nn.Parameter(
            torch.tensor(
                [
                    [[0.25, 0.75], [0.6, 0.4], [0.7, 0.3]],
                    [[0.2, 0.8], [0.9, 0.1], [0.35, 0.65]],
                ]
            )
        )

    def forward(self, x):
        return self.ids, self.weights, None


def capacity_reference(model, x, capacity, policy):
    """Python selection and token loops, independent of tensor sorts/offsets."""
    ids, weights, _ = model.router(x)
    keep = torch.zeros_like(ids, dtype=torch.bool)
    for expert in range(model.num_experts):
        assignments = [
            (batch, position, slot)
            for batch in range(x.shape[0])
            for position in range(x.shape[1])
            for slot in range(model.top_k)
            if ids[batch, position, slot].item() == expert
        ]
        if policy == "probs":
            assignments.sort(key=lambda index: -weights[index].detach().item())
        for index in assignments[:capacity]:
            keep[index] = True
    outputs = []
    for batch in range(x.shape[0]):
        for position in range(x.shape[1]):
            token = x[batch, position]
            output = token * 0
            for slot in range(model.top_k):
                expert = ids[batch, position, slot].item()
                if keep[batch, position, slot]:
                    value, gate = (token @ model.experts.up_weight[expert]).chunk(2)
                    result = (
                        value * torch.nn.functional.silu(gate)
                    ) @ model.experts.down_weight[expert]
                    output = output + weights[batch, position, slot] * result
            outputs.append(output)
    return torch.stack(outputs).reshape_as(x)


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
@pytest.mark.parametrize("factor", [0.5, 1.0])
@pytest.mark.parametrize("policy", ["position", "probs"])
def test_capacity_matches_reference_outputs_gradients_and_retained_offsets(
    backend, factor, policy
):
    torch.manual_seed(40)
    model = GroupedMoE(
        16, 32, 4, 2, capacity_factor=factor, drop_policy=policy, expert_backend=backend
    )
    model.router = CapacityRouter()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    capacity = math.ceil(factor * 6 * 2 / 4)
    observed = []

    def capture(_module, args):
        observed.append((args[0].detach().clone(), args[1].clone()))

    handle = model.experts.register_forward_pre_hook(capture)
    actual, _ = model(x)
    handle.remove()
    expected = capacity_reference(reference, reference_x, capacity, policy)
    torch.testing.assert_close(actual, expected)
    counts = torch.tensor([0, 5, 3, 4]).clamp(max=capacity)
    grouped_tokens, offsets = observed[0]
    torch.testing.assert_close(offsets, counts.cumsum(0, dtype=torch.int32))
    assert grouped_tokens.shape == (counts.sum().item(), 16)
    if factor == 0.5:
        # ceil(1.5) = 2; partial and fully dropped tokens preserve rectangular output.
        if policy == "position":
            torch.testing.assert_close(actual[1, 0], torch.zeros(16))
            torch.testing.assert_close(actual[1, 2], torch.zeros(16))
            retained_tokens = [0, 1, 1, 4, 0, 2]
        else:
            torch.testing.assert_close(actual[0, 2], torch.zeros(16))
            retained_tokens = [1, 5, 4, 5, 3, 0]
        torch.testing.assert_close(grouped_tokens, x.reshape(-1, 16)[retained_tokens])
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    for name in ("router.weights", "experts.up_weight", "experts.down_weight"):
        actual_grad = dict(model.named_parameters())[name].grad
        expected_grad = dict(reference.named_parameters())[name].grad
        assert actual_grad is not None and expected_grad is not None
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
def test_large_capacity_matches_dropless_and_preserves_auxiliary_loss(backend):
    torch.manual_seed(41)
    dropless = GroupedMoE(16, 32, 4, 2, expert_backend=backend)
    limited = GroupedMoE(16, 32, 4, 2, capacity_factor=2.0, expert_backend=backend)
    limited.load_state_dict(dropless.state_dict())
    x = torch.randn(2, 3, 16)
    torch.testing.assert_close(limited(x), dropless(x))


def test_dropping_keeps_auxiliary_loss_and_logging_counts_before_capacity():
    torch.manual_seed(42)
    dropless = GroupedMoE(16, 32, 4, 2, expert_backend="loop")
    limited = GroupedMoE(16, 32, 4, 2, capacity_factor=0.1, expert_backend="loop")
    limited.load_state_dict(dropless.state_dict())
    x = torch.randn(2, 3, 16)
    _, expected_aux = dropless(x)
    _, actual_aux = limited(x)
    torch.testing.assert_close(actual_aux, expected_aux)
    torch.testing.assert_close(
        limited.router.tokens_per_expert, dropless.router.tokens_per_expert
    )
    assert limited.router.tokens_per_expert.sum() == 12  # Requested, not retained.


@pytest.mark.parametrize(
    "factor", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_capacity_factor_must_be_finite_and_positive(factor):
    with pytest.raises(ValueError, match="capacity_factor"):
        GroupedMoE(16, 32, 4, 2, capacity_factor=factor)


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
def test_score_priority_ties_preserve_original_assignment_order(backend):
    model = GroupedMoE(16, 32, 4, 2, capacity_factor=0.5, expert_backend=backend)
    model.router = CapacityRouter()
    with torch.no_grad():
        model.router.weights.fill_(0.5)
    reference = copy.deepcopy(model)
    reference.drop_policy = "position"
    x = torch.randn(2, 3, 16)
    torch.testing.assert_close(model(x), reference(x))


def test_dropless_path_does_not_sort_by_weight(monkeypatch):
    original = torch.argsort
    dtypes = []

    def capture(input, *args, **kwargs):
        dtypes.append(input.dtype)
        return original(input, *args, **kwargs)

    monkeypatch.setattr(torch, "argsort", capture)
    GroupedMoE(16, 32, 4, 2)(torch.randn(2, 3, 16))
    assert dtypes == [torch.int64]  # Only expert IDs, not floating-point weights.


def test_unknown_drop_policy_is_rejected():
    with pytest.raises(ValueError, match="drop policy"):
        GroupedMoE(16, 32, 4, 2, drop_policy="invalid")  # ty: ignore[invalid-argument-type]


def make_overflow_model(backend="loop", policy="probs", factor=0.5):
    model = GroupedMoE(
        16, 32, 4, 2, capacity_factor=factor, drop_policy=policy, expert_backend=backend
    )
    with torch.no_grad():
        model.router.router.weight.zero_()
        model.router.router.weight[:, :4].copy_(torch.eye(4))
    tokens = torch.zeros(1, 3, 16)
    tokens[..., :4] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    return model, tokens  # Every token requests experts 0 and 1.


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
@pytest.mark.parametrize("policy", ["position", "probs"])
def test_drop_counts_accumulate_actual_losses_across_unequal_forwards(backend, policy):
    model, tokens = make_overflow_model(backend, policy)
    model(tokens)
    # T=3, K=2, E=4, factor=0.5 -> capacity=1; two drops per requested expert.
    torch.testing.assert_close(
        model.dropped_assignments_per_expert, torch.tensor([2, 2, 0, 0])
    )
    model(tokens[:, :1])  # Capacity=1 again; no overflow this time.
    torch.testing.assert_close(
        model.router.tokens_per_expert, torch.tensor([4, 4, 0, 0])
    )
    torch.testing.assert_close(
        model.dropped_assignments_per_expert, torch.tensor([2, 2, 0, 0])
    )
    assert model.dropped_assignments_per_expert.grad_fn is None
    assert not model.dropped_assignments_per_expert.requires_grad


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
def test_evaluation_preserves_drop_statistics_and_reset_clears_all_in_place(backend):
    model, tokens = make_overflow_model(backend)
    model(tokens)
    buffers = dict(model.named_buffers())
    before = {name: value.clone() for name, value in buffers.items()}
    model.eval()
    model(tokens)  # Autograd is enabled; eval alone must prevent metric updates.
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name])
    model.reset_metrics()
    for name, value in model.named_buffers():
        assert value is buffers[name]
        assert torch.count_nonzero(value) == 0
    model.train()
    model(tokens[:, :1])
    torch.testing.assert_close(
        model.router.tokens_per_expert, torch.tensor([1, 1, 0, 0])
    )
    assert model.dropped_assignments_per_expert.sum() == 0


@pytest.mark.parametrize("factor", [None, 10.0])
def test_dropless_and_sufficient_capacity_record_requests_but_no_drops(factor):
    model, tokens = make_overflow_model(factor=factor)
    model(tokens)
    torch.testing.assert_close(
        model.router.tokens_per_expert, torch.tensor([3, 3, 0, 0])
    )
    assert model.dropped_assignments_per_expert.sum() == 0


def test_drop_counts_are_nonpersistent_int64_buffer():
    model, _ = make_overflow_model()
    buffer = model.dropped_assignments_per_expert
    assert dict(model.named_buffers())["dropped_assignments_per_expert"] is buffer
    assert "dropped_assignments_per_expert" not in model.state_dict()
    assert buffer.shape == (4,) and buffer.dtype == torch.int64
    model.double()
    assert model.dropped_assignments_per_expert.dtype == torch.int64


@pytest.mark.parametrize(
    "requested,dropped,expected_global,expected_per_expert",
    [
        ([90, 10, 0, 0], [9, 5, 0, 0], 0.14, [0.1, 0.5, 0.0, 0.0]),
        ([0, 0, 0, 0], [0, 0, 0, 0], 0.0, [0.0, 0.0, 0.0, 0.0]),
        ([10, 20, 0, 30], [0, 0, 0, 0], 0.0, [0.0, 0.0, 0.0, 0.0]),
        ([4, 0, 6, 0], [4, 0, 6, 0], 1.0, [1.0, 0.0, 1.0, 0.0]),
    ],
)
def test_drop_metrics_controlled_values_shapes_and_zero_requests(
    requested, dropped, expected_global, expected_per_expert
):
    model = GroupedMoE(16, 32, 4, 1)
    model.router.tokens_per_expert.copy_(torch.tensor(requested))
    model.dropped_assignments_per_expert.copy_(torch.tensor(dropped))
    metrics = model.compute_metrics()
    global_rate = metrics["dropped_assignment_rate"]
    per_expert = metrics["dropped_assignment_rate_per_expert"]
    assert global_rate.shape == ()
    assert per_expert.shape == (4,)
    torch.testing.assert_close(global_rate, torch.tensor(expected_global))
    torch.testing.assert_close(per_expert, torch.tensor(expected_per_expert))
    for rate in (global_rate, per_expert):
        assert rate.is_floating_point() and torch.isfinite(rate).all()
        assert not rate.requires_grad and rate.grad_fn is None


def test_drop_metric_reporting_preserves_router_metrics_and_does_not_reset():
    model, tokens = make_overflow_model()
    model(tokens)
    model(tokens[:, :1])  # Combined rate is 4/8, not the mean of 4/6 and 0/2.
    before = {name: value.clone() for name, value in model.named_buffers()}
    router_metrics = model.router.compute_metrics()
    first = model.compute_metrics()
    second = model.compute_metrics()
    assert first.keys() == router_metrics.keys() | {
        "dropped_assignment_rate",
        "dropped_assignment_rate_per_expert",
    }
    for name, value in router_metrics.items():
        torch.testing.assert_close(first[name], value)
    for name in first:
        torch.testing.assert_close(first[name], second[name])
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name])
    torch.testing.assert_close(first["dropped_assignment_rate"], torch.tensor(0.5))
