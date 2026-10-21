"""AI-authored tests for the learner's top-k router."""

import pytest
import torch

from moe.losses import load_balancing_loss
from moe.router import TopKRouter


def test_auxiliary_loss_uses_current_forward_not_accumulated_metrics():
    router = TopKRouter(4, 4, 2).float()
    with torch.no_grad():
        router.router.weight.copy_(torch.eye(4, dtype=torch.float32))
    router(torch.tensor([[[4.0, 3.0, 0.0, -1.0]]], dtype=torch.float32))
    x = torch.tensor([[[0.0, -1.0, 4.0, 3.0]]], dtype=torch.float32)
    ids, _, loss, _ = router(x)
    counts = ids.flatten().bincount(minlength=4)
    assert loss is not None
    torch.testing.assert_close(loss, load_balancing_loss(x.softmax(-1), counts))
    assert router.tokens_per_expert.sum() == 4
    loss.backward()
    assert router.router.weight.grad is not None
    assert router.router.weight.grad.abs().sum() > 0


def test_evaluation_returns_no_auxiliary_loss():
    router = TopKRouter(4, 4, 2)
    x = torch.randn(2, 3, 4)
    _, _, training_loss, _ = router(x)
    assert training_loss is not None and training_loss.requires_grad
    router.eval()
    assert router(x)[2] is None
    router.train()
    assert router(x)[2] is not training_loss


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_shapes_ids_and_normalized_weights(top_k):
    torch.manual_seed(0)
    router = TopKRouter(d_model=6, num_experts=4, top_k=top_k)

    expert_ids, weights, aux_loss, _ = router(torch.randn(2, 5, 6))
    assert aux_loss is not None and aux_loss.shape == () and aux_loss.requires_grad

    assert expert_ids.shape == weights.shape == (2, 5, top_k)
    assert expert_ids.dtype == torch.int64
    assert ((expert_ids >= 0) & (expert_ids < 4)).all()
    assert (weights > 0).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5))
    # topk cannot select the same expert twice for a token.
    sorted_ids = expert_ids.sort(dim=-1).values
    assert (sorted_ids[..., 1:] != sorted_ids[..., :-1]).all()


def test_controlled_selection_and_weight_normalization():
    router = TopKRouter(d_model=4, num_experts=4, top_k=2).float()
    with torch.no_grad():
        router.router.weight.copy_(torch.eye(4, dtype=torch.float32))

    # The identity projection makes logits log(p), so softmax recovers p.
    probabilities = torch.tensor(
        [[[0.1, 0.4, 0.2, 0.3], [0.5, 0.1, 0.25, 0.15]]], dtype=torch.float32
    )
    expert_ids, weights, _, _ = router(probabilities.log())

    torch.testing.assert_close(expert_ids, torch.tensor([[[1, 3], [0, 2]]]))
    torch.testing.assert_close(
        weights,
        torch.tensor([[[4 / 7, 3 / 7], [2 / 3, 1 / 3]]], dtype=torch.float32),
    )


def test_assignment_counts_computed_once_per_forward(monkeypatch):
    calls = []
    original = torch.bincount

    def capture(*args, **kwargs):
        counts = original(*args, **kwargs)
        calls.append(counts)
        return counts

    monkeypatch.setattr(torch, "bincount", capture)
    router = TopKRouter(d_model=6, num_experts=4, top_k=2)
    x = torch.randn(2, 5, 6)
    router(x)
    router(x)
    assert len(calls) == 2
    torch.testing.assert_close(router.tokens_per_expert, calls[0] + calls[1])


@pytest.mark.parametrize("top_k", [0, -1, 5])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="Top K must be between"):
        TopKRouter(d_model=6, num_experts=4, top_k=top_k)


def test_top_two_weights_provide_router_gradients():
    torch.manual_seed(1)
    router = TopKRouter(d_model=6, num_experts=4, top_k=2).float()
    x = torch.randn(2, 5, 6, dtype=torch.float32, requires_grad=True)

    _, weights, _, _ = router(x)
    # Unequal expert outputs give a meaningful task signal. Summing weights
    # alone would be constant and could not test router learning.
    expert_outputs = torch.tensor([1.0, -2.0], dtype=torch.float32)
    loss = (weights * expert_outputs).sum()
    loss.backward()

    assert router.router.weight.grad is not None
    assert torch.isfinite(router.router.weight.grad).all()
    assert router.router.weight.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_normalized_top_one_has_no_router_gradient_from_mixing_weight():
    torch.manual_seed(2)
    router = TopKRouter(d_model=6, num_experts=4, top_k=1).float()
    x = torch.randn(2, 5, 6, dtype=torch.float32, requires_grad=True)

    _, weights, _, _ = router(x)
    torch.testing.assert_close(weights, torch.ones_like(weights))
    weights.sum().backward()

    assert router.router.weight.grad is not None
    assert x.grad is not None
    torch.testing.assert_close(
        router.router.weight.grad, torch.zeros_like(router.router.weight.grad)
    )
    torch.testing.assert_close(x.grad, torch.zeros_like(x.grad))


def test_assignment_counts_accumulate_across_microbatches():
    torch.manual_seed(3)
    router = TopKRouter(4, 5, 2).train()
    first_ids, _, _, _ = router(torch.randn(2, 3, 4, requires_grad=True))
    second_ids, _, _, _ = router(torch.randn(1, 2, 4, requires_grad=True))
    expected = torch.bincount(first_ids.flatten(), minlength=5)
    expected += torch.bincount(second_ids.flatten(), minlength=5)
    torch.testing.assert_close(router.tokens_per_expert, expected)
    assert router.tokens_per_expert.sum().item() == (2 * 3 + 1 * 2) * 2
    assert not router.tokens_per_expert.requires_grad
    assert router.tokens_per_expert.grad_fn is None


def test_eval_does_not_change_training_counts():
    router = TopKRouter(4, 5, 2).train()
    router(torch.randn(2, 3, 4))
    statistics = {name: buffer.clone() for name, buffer in router.named_buffers()}
    router.eval()
    router(torch.randn(2, 3, 4))
    for name, buffer in router.named_buffers():
        torch.testing.assert_close(buffer, statistics[name])


def test_reset_clears_counts_in_place_without_changing_routing():
    router = TopKRouter(4, 5, 2).train()
    x = torch.randn(2, 3, 4)
    expected_ids, expected_weights, _, _ = router(x)
    original_buffers = dict(router.named_buffers())
    router.reset_metrics()
    for name, buffer in router.named_buffers():
        assert buffer is original_buffers[name]
        assert torch.count_nonzero(buffer).item() == 0
    actual_ids, actual_weights, _, _ = router(x)
    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(
        router.tokens_per_expert, torch.bincount(actual_ids.flatten(), minlength=5)
    )


def test_count_buffer_is_nonpersistent_and_stays_integer():
    router = TopKRouter(4, 5, 2)
    assert dict(router.named_buffers())["tokens_per_expert"] is router.tokens_per_expert
    assert "tokens_per_expert" not in router.state_dict()
    assert "tokens_per_expert" not in dict(router.named_parameters())
    router.double()
    assert router.router.weight.dtype == torch.float64
    assert router.tokens_per_expert.dtype == torch.int64


def test_router_projection_stays_float32_under_cpu_autocast():
    router = TopKRouter(4, 3, 2)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16, requires_grad=True)
    projection_dtypes = []

    def capture(_module, _args, output):
        projection_dtypes.append(output.dtype)

    handle = router.router.register_forward_hook(capture)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, weights, aux, z = router(x)
    handle.remove()
    assert projection_dtypes == [torch.float32]
    assert aux is not None and z is not None
    assert weights.dtype == aux.dtype == z.dtype == torch.float32
    (aux + z).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert router.router.weight.grad is not None


def test_z_loss_detects_common_logit_shift_without_changing_routing():
    router = TopKRouter(3, 3, 2)
    with torch.no_grad():
        router.router.weight.copy_(torch.eye(3))
    x = torch.tensor([[[2.0, 1.0, 0.0]]])
    ids, weights, aux, z = router(x)
    shifted_ids, shifted_weights, shifted_aux, shifted_z = router(x + 100)
    torch.testing.assert_close(ids, shifted_ids)
    torch.testing.assert_close(weights, shifted_weights)
    torch.testing.assert_close(aux, shifted_aux)
    assert z is not None and shifted_z is not None
    torch.testing.assert_close(z, x.logsumexp(-1).square().mean())
    assert shifted_z > z
    shifted_z.backward()
    assert router.router.weight.grad is not None
    assert router.router.weight.grad.abs().sum() > 0
    router.eval()
    assert router(x)[2:] == (None, None)


def test_full_probability_sums_weight_unequal_microbatches_by_token_count():
    router = TopKRouter(4, 4, 2).train()
    with torch.no_grad():
        router.router.weight.copy_(torch.eye(4))
    first_probs = torch.tensor([[[0.7, 0.1, 0.1, 0.1]]])
    second_probs = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]).expand(1, 3, 4)
    router(first_probs.log().requires_grad_())
    router(second_probs.log().requires_grad_())

    expected_sum = first_probs.sum((0, 1)) + second_probs.sum((0, 1))
    torch.testing.assert_close(router.router_prob_sum, expected_sum)
    assert router.num_tokens.item() == 4  # Tokens, not the 8 expert assignments.
    torch.testing.assert_close(router.router_prob_sum.sum(), torch.tensor(4.0))
    expected_mean = torch.cat((first_probs, second_probs), dim=1).mean((0, 1))
    torch.testing.assert_close(
        router.router_prob_sum / router.num_tokens, expected_mean
    )
    unweighted_mean = (first_probs.mean((0, 1)) + second_probs.mean((0, 1))) / 2
    assert not torch.allclose(expected_mean, unweighted_mean)
    for buffer in (router.router_prob_sum, router.num_tokens):
        assert not buffer.requires_grad
        assert buffer.grad_fn is None


def test_probability_and_token_buffers_are_registered_nonpersistent_statistics():
    router = TopKRouter(4, 5, 2)
    buffers = dict(router.named_buffers())
    assert buffers["router_prob_sum"] is router.router_prob_sum
    assert buffers["num_tokens"] is router.num_tokens
    assert router.router_prob_sum.shape == (5,)
    assert router.router_prob_sum.dtype == torch.float32
    assert router.num_tokens.shape == ()
    assert router.num_tokens.dtype == torch.int64
    assert buffers["router_entropy_sum"] is router.router_entropy_sum
    assert router.router_entropy_sum.shape == ()
    assert router.router_entropy_sum.dtype == torch.float32
    assert buffers["selected_prob_mass_sum"] is router.selected_prob_mass_sum
    assert router.selected_prob_mass_sum.shape == ()
    assert router.selected_prob_mass_sum.dtype == torch.float32
    for name in (
        "router_prob_sum",
        "num_tokens",
        "router_entropy_sum",
        "selected_prob_mass_sum",
    ):
        assert name not in router.state_dict()
        assert name not in dict(router.named_parameters())


def test_compute_metrics_reports_controlled_values():
    router = TopKRouter(4, 4, 2)
    router.tokens_per_expert.copy_(torch.tensor([10, 20, 30, 20]))
    router.router_prob_sum.copy_(torch.tensor([4.0, 8.0, 12.0, 16.0]))
    router.num_tokens.fill_(40)
    router.router_entropy_sum.fill_(28)
    router.selected_prob_mass_sum.fill_(32)
    metrics = router.compute_metrics()
    expected = {
        "maxvio": torch.tensor(0.5),
        "assignment_fraction": torch.tensor([0.125, 0.25, 0.375, 0.25]),
        "unused_expert_count": torch.tensor(0),
        "assignment_entropy": torch.tensor(1.9056390622),
        "mean_router_probability": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "mean_router_entropy": torch.tensor(0.7),
        "mean_selected_probability_mass": torch.tensor(0.8),
    }
    assert metrics.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(metrics[name], value)


@pytest.mark.parametrize("after_reset", [False, True])
def test_compute_metrics_without_data_is_finite(after_reset):
    router = TopKRouter(4, 5, 2)
    if after_reset:
        router(torch.randn(2, 3, 4))
        router.reset_metrics()
    metrics = router.compute_metrics()
    for value in metrics.values():
        assert torch.isfinite(value).all()
        assert not value.requires_grad
    assert metrics["maxvio"].item() == 0
    assert metrics["assignment_entropy"].item() == 0
    assert metrics["mean_router_entropy"].item() == 0
    assert metrics["mean_selected_probability_mass"].item() == 0
    assert metrics["unused_expert_count"].item() == 5
    torch.testing.assert_close(metrics["assignment_fraction"], torch.zeros(5))
    torch.testing.assert_close(metrics["mean_router_probability"], torch.zeros(5))


def test_compute_metrics_is_read_only_and_reports_survive_reset():
    router = TopKRouter(4, 5, 2)
    router(torch.randn(2, 3, 4, requires_grad=True))
    original = {name: buffer.clone() for name, buffer in router.named_buffers()}
    first = router.compute_metrics()
    saved = {name: value.clone() for name, value in first.items()}
    second = router.compute_metrics()
    for name in first:
        torch.testing.assert_close(first[name], second[name])
        assert not first[name].requires_grad
        assert first[name].grad_fn is None
    for name, buffer in router.named_buffers():
        torch.testing.assert_close(buffer, original[name])
    router.reset_metrics()
    for name in first:
        torch.testing.assert_close(first[name], saved[name])


def test_router_entropy_accumulates_token_weighted_microbatch_statistics():
    router = TopKRouter(2, 2, 1).train()
    with torch.no_grad():
        router.router.weight.copy_(torch.eye(2))
    first_probs = torch.tensor([[[0.5, 0.5]]])
    second_probs = torch.tensor([[[0.9, 0.1]]]).expand(1, 3, 2)
    router(first_probs.log())
    router(second_probs.log())

    # One 1-bit token and three tokens with binary entropy H(0.9).
    expected_sum = torch.tensor(1.0 + 3 * 0.4689955936)
    torch.testing.assert_close(router.router_entropy_sum, expected_sum)
    metrics = router.compute_metrics()
    torch.testing.assert_close(metrics["mean_router_entropy"], expected_sum / 4)
    assert not router.router_entropy_sum.requires_grad
    assert router.router_entropy_sum.grad_fn is None
    assert not torch.allclose(
        metrics["mean_router_entropy"], torch.tensor((1.0 + 0.4689955936) / 2)
    )  # Averaging the two microbatch means would be wrong.


def test_router_reports_confidence_separately_from_assignment_balance():
    router = TopKRouter(2, 2, 1).train()
    with torch.no_grad():
        router.router.weight.copy_(torch.eye(2))
    router(torch.tensor([[[40.0, -40.0], [-40.0, 40.0]]]))
    metrics = router.compute_metrics()
    torch.testing.assert_close(metrics["assignment_entropy"], torch.tensor(1.0))
    assert metrics["mean_router_entropy"].item() < 1e-5


@pytest.mark.parametrize(("top_k", "expected_mass"), [(1, 0.4), (2, 0.7), (4, 1.0)])
def test_selected_mass_uses_probabilities_before_renormalization(top_k, expected_mass):
    router = TopKRouter(4, 4, top_k).train()
    with torch.no_grad():
        router.router.weight.copy_(torch.eye(4))
    probabilities = torch.tensor([[[0.4, 0.3, 0.2, 0.1]]])
    _, weights, _, _ = router(probabilities.log())
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 1))
    torch.testing.assert_close(
        router.selected_prob_mass_sum, torch.tensor(expected_mass)
    )
    torch.testing.assert_close(
        router.compute_metrics()["mean_selected_probability_mass"],
        torch.tensor(expected_mass),
    )
    assert not router.selected_prob_mass_sum.requires_grad
    assert router.selected_prob_mass_sum.grad_fn is None


def test_selected_mass_is_token_weighted_across_unequal_microbatches():
    router = TopKRouter(4, 4, 2).train()
    with torch.no_grad():
        router.router.weight.copy_(torch.eye(4))
    first = torch.tensor([[[0.4, 0.3, 0.2, 0.1]]])  # Mass 0.7 for one token.
    second = torch.tensor([[[0.7, 0.2, 0.05, 0.05]]]).expand(1, 3, 4)  # Mass 0.9 each.
    router(first.log())
    router(second.log())
    torch.testing.assert_close(router.selected_prob_mass_sum, torch.tensor(3.4))
    torch.testing.assert_close(
        router.compute_metrics()["mean_selected_probability_mass"], torch.tensor(0.85)
    )


def test_selected_mass_reduction_uses_float32_under_autocast():
    router = TopKRouter(4, 4, 2).train()
    # 70001 * 0.5 is not exactly representable in bfloat16. The reduction must
    # happen in float32, not merely be cast when added to the float32 buffer.
    with torch.no_grad():
        router.router.weight.zero_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        router(torch.zeros(1, 70001, 4))
    torch.testing.assert_close(
        router.selected_prob_mass_sum, torch.tensor(70001 * 0.5), rtol=0, atol=0
    )
