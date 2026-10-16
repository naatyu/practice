"""AI-authored tests for the learner's top-k router."""

import pytest
import torch

from moe.router import TopKRouter


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_shapes_ids_and_normalized_weights(top_k):
    torch.manual_seed(0)
    router = TopKRouter(d_model=6, num_experts=4, top_k=top_k)

    expert_ids, weights = router(torch.randn(2, 5, 6))

    assert expert_ids.shape == weights.shape == (2, 5, top_k)
    assert expert_ids.dtype == torch.int64
    assert ((expert_ids >= 0) & (expert_ids < 4)).all()
    assert (weights > 0).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5))
    # topk cannot select the same expert twice for a token.
    sorted_ids = expert_ids.sort(dim=-1).values
    assert (sorted_ids[..., 1:] != sorted_ids[..., :-1]).all()


def test_controlled_selection_and_weight_normalization():
    router = TopKRouter(d_model=4, num_experts=4, top_k=2).double()
    with torch.no_grad():
        router.router.weight.copy_(torch.eye(4, dtype=torch.float64))

    # The identity projection makes logits log(p), so softmax recovers p.
    probabilities = torch.tensor(
        [[[0.1, 0.4, 0.2, 0.3], [0.5, 0.1, 0.25, 0.15]]], dtype=torch.float64
    )
    expert_ids, weights = router(probabilities.log())

    torch.testing.assert_close(expert_ids, torch.tensor([[[1, 3], [0, 2]]]))
    torch.testing.assert_close(
        weights,
        torch.tensor([[[4 / 7, 3 / 7], [2 / 3, 1 / 3]]], dtype=torch.float64),
    )


@pytest.mark.parametrize("top_k", [0, -1, 5])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="Top K must be between"):
        TopKRouter(d_model=6, num_experts=4, top_k=top_k)


def test_top_two_weights_provide_router_gradients():
    torch.manual_seed(1)
    router = TopKRouter(d_model=6, num_experts=4, top_k=2).double()
    x = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)

    _, weights = router(x)
    # Unequal expert outputs give a meaningful task signal. Summing weights
    # alone would be constant and could not test router learning.
    expert_outputs = torch.tensor([1.0, -2.0], dtype=torch.float64)
    loss = (weights * expert_outputs).sum()
    loss.backward()

    assert router.router.weight.grad is not None
    assert torch.isfinite(router.router.weight.grad).all()
    assert router.router.weight.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_normalized_top_one_has_no_router_gradient_from_mixing_weight():
    torch.manual_seed(2)
    router = TopKRouter(d_model=6, num_experts=4, top_k=1).double()
    x = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)

    _, weights = router(x)
    torch.testing.assert_close(weights, torch.ones_like(weights))
    weights.sum().backward()

    assert router.router.weight.grad is not None
    assert x.grad is not None
    torch.testing.assert_close(
        router.router.weight.grad, torch.zeros_like(router.router.weight.grad)
    )
    torch.testing.assert_close(x.grad, torch.zeros_like(x.grad))
