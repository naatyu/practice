"""AI-authored equivalence and dispatch checks for expert-batched MoE."""

import pytest
import torch
from torch import nn

from mlp import SwiGLU
from moe.expert_batched_moe import ExpertBatchedMoe
from moe.naive_moe import NaiveMoe
from moe.router import TopKRouter


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_outputs_and_gradients_match_naive(top_k, dtype):
    torch.manual_seed(20)
    naive = NaiveMoe(3, 5, 4, top_k).to(dtype)
    batched = ExpertBatchedMoe(3, 5, 4, top_k).to(dtype)
    batched.load_state_dict(naive.state_dict())
    x_naive = torch.randn(2, 4, 3, dtype=dtype, requires_grad=True)
    x_batched = x_naive.detach().clone().requires_grad_()

    expected = naive(x_naive)
    actual = batched(x_batched)
    assert actual.shape == x_batched.shape
    torch.testing.assert_close(actual, expected)

    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x_batched.grad, x_naive.grad)
    expected_parameters = dict(naive.named_parameters())
    for name, parameter in batched.named_parameters():
        expected_gradient = expected_parameters[name].grad
        # The batched implementation can call an unused expert on [0, D],
        # giving zero gradients where the naive version has no graph (None).
        if expected_gradient is None:
            expected_gradient = torch.zeros_like(parameter)
        actual_gradient = parameter.grad
        if actual_gradient is None:
            actual_gradient = torch.zeros_like(parameter)
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient)


class FixedRouter(TopKRouter):
    def __init__(self):
        super().__init__(d_model=2, num_experts=4, top_k=2)

    def forward(self, x):
        # Expert 2 appears in different slots and in both batch rows.
        ids = torch.tensor([[[0, 2], [2, 1]], [[2, 0], [1, 0]]], device=x.device)
        weights = x.new_tensor([[[0.7, 0.3], [0.6, 0.4]], [[0.2, 0.8], [0.9, 0.1]]])
        return ids, weights


class RecordingExpert(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return self.scale * x


def test_controlled_dispatch_across_batches_slots_and_unused_expert():
    model = ExpertBatchedMoe(2, 4, 4, 2)
    model.router = FixedRouter()
    experts = [RecordingExpert(scale) for scale in [2, -1, 4, 10]]
    model.experts = nn.ModuleList(experts)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    output = model(x)

    # Each token receives BOTH contributions, with the weight of its own slot.
    expected_scales = x.new_tensor([[2.6, 2.0], [2.4, -0.7]])
    torch.testing.assert_close(output, x * expected_scales.unsqueeze(-1))
    expected_inputs = [
        torch.stack((x[0, 0], x[1, 0], x[1, 1])),
        torch.stack((x[0, 1], x[1, 1])),
        torch.stack((x[0, 0], x[0, 1], x[1, 0])),
        x.new_empty(0, 2),
    ]
    for expert, expected_input in zip(experts, expected_inputs):
        if expected_input.shape[0] == 0 and not expert.inputs:
            continue  # Skipping an unused expert is also a valid optimization.
        assert len(expert.inputs) == 1
        torch.testing.assert_close(expert.inputs[0], expected_input)


def test_single_token_matches_naive():
    naive = NaiveMoe(3, 5, 4, 2)
    batched = ExpertBatchedMoe(3, 5, 4, 2)
    batched.load_state_dict(naive.state_dict())
    x = torch.randn(1, 1, 3)
    torch.testing.assert_close(batched(x), naive(x))


@pytest.mark.parametrize("model_type", [NaiveMoe, ExpertBatchedMoe])
def test_experts_use_shared_bias_free_swiglu(model_type):
    model = model_type(3, 5, 4, 2)
    for expert in model.experts:
        assert isinstance(expert, SwiGLU)
        assert expert.up_proj.bias is None and expert.down_proj.bias is None
    # E experts each hold 3*D*F weights; the bias-free router holds D*E.
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == 4 * 3 * 3 * 5 + 3 * 4
    )
