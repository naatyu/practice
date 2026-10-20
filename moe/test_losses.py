"""AI-authored tests for the learner's auxiliary balancing loss."""

import pytest
import torch

from moe.losses import load_balancing_loss


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_balanced_assignments_give_one(top_k):
    base = torch.tensor([0.4, 0.3, 0.2, 0.1])
    probabilities = torch.stack([base.roll(i) for i in range(4)]).reshape(2, 2, 4)
    expert_ids = probabilities.topk(top_k, dim=-1).indices
    counts = expert_ids.flatten().bincount(minlength=4)
    loss = load_balancing_loss(probabilities, counts)
    assert loss.shape == ()
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_controlled_top_two_loss_and_inputs_unchanged():
    probabilities = torch.tensor([[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]])
    counts = torch.tensor([1, 2, 1])
    original_probabilities = probabilities.clone()
    original_counts = counts.clone()
    # f = [1/4, 2/4, 1/4], p = [0.4, 0.4, 0.2].
    loss = load_balancing_loss(probabilities, counts)
    torch.testing.assert_close(loss, torch.tensor(1.05))
    torch.testing.assert_close(probabilities, original_probabilities)
    torch.testing.assert_close(counts, original_counts)


def test_collapsed_routing_includes_unselected_experts():
    probabilities = torch.tensor([[[0.7, 0.1, 0.1, 0.1]]])
    counts = torch.tensor([1, 0, 0, 0])
    # Only expert 0 has assignments; f must still have four entries.
    torch.testing.assert_close(
        load_balancing_loss(probabilities, counts), torch.tensor(2.8)
    )


def test_repeating_batch_does_not_scale_loss():
    probabilities = torch.tensor([[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]])
    counts = torch.tensor([1, 2, 1])
    torch.testing.assert_close(
        load_balancing_loss(probabilities.repeat(3, 1, 1), counts * 3),
        load_balancing_loss(probabilities, counts),
    )


def test_unused_expert_receives_gradient_through_softmax():
    logits = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    probabilities = logits.softmax(dim=-1)
    expert_ids = probabilities.topk(1, dim=-1).indices
    counts = expert_ids.flatten().bincount(minlength=2)
    loss = load_balancing_loss(probabilities, counts)
    loss.backward()
    assert logits.grad is not None
    # Gradient descent lowers the busy expert's logit and raises the unused one's.
    assert logits.grad[0, 0, 0] > 0
    assert logits.grad[0, 0, 1] < 0
    updated_logits = logits.detach() - 0.1 * logits.grad
    assert load_balancing_loss(updated_logits.softmax(-1), counts) < loss.detach()


def test_logit_gradients_match_finite_differences():
    logits = torch.tensor(
        [[[1.0, 0.4, -0.5], [0.8, 0.6, -0.2]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    expert_ids = logits.topk(2, dim=-1).indices
    counts = expert_ids.flatten().bincount(minlength=3)
    assert torch.autograd.gradcheck(
        lambda x: load_balancing_loss(x.softmax(-1), counts), (logits,)
    )
