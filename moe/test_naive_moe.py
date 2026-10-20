"""AI-authored checks for the learner's naive MoE implementation."""

import pytest
import torch
from torch import nn

from mlp import SwiGLU
from moe.naive_moe import NaiveMoe
from moe.router import TopKRouter


class FixedRouter(TopKRouter):
    def __init__(self):
        super().__init__(d_model=2, num_experts=4, top_k=2)

    def forward(self, x):
        ids = torch.tensor([[[0, 2], [1, 0]]], device=x.device)
        weights = x.new_tensor([[[0.25, 0.75], [0.6, 0.4]]])
        return ids, weights, None


class RecordingExpert(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return self.scale * x


def test_controlled_routing_weights_and_original_token_inputs():
    model = NaiveMoe(2, 4, 4, 2)
    model.router = FixedRouter()
    recording_experts = [RecordingExpert(scale) for scale in [2, -1, 4, 10]]
    model.experts = nn.ModuleList(recording_experts)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    output, _ = model(x)

    # Token 0: (.25*2 + .75*4)*x = 3.5*x.
    # Token 1: (.6*(-1) + .4*2)*x = .2*x.
    torch.testing.assert_close(output, torch.tensor([[[3.5, 7.0], [0.6, 0.8]]]))
    for expert_id, token_indices in enumerate([[0, 1], [1], [0], []]):
        calls = recording_experts[expert_id].inputs
        assert len(calls) == len(token_indices)
        for actual_input, token_index in zip(calls, token_indices):
            torch.testing.assert_close(actual_input, x[0, token_index])


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_matches_dense_reference_outputs_and_gradients(top_k):
    torch.manual_seed(10)
    model = NaiveMoe(3, 5, 4, top_k).double()
    x = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    inputs = (x, *model.parameters())

    actual, _ = model(x)
    ids, weights, _ = model.router(x)
    # Evaluating every expert is acceptable only in this small test reference.
    all_outputs = torch.stack([expert(x) for expert in model.experts], dim=-2)
    selected = all_outputs.gather(
        dim=-2, index=ids.unsqueeze(-1).expand(*ids.shape, x.shape[-1])
    )
    expected = (selected * weights.unsqueeze(-1)).sum(dim=-2)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)

    actual_gradients = torch.autograd.grad(
        actual.square().sum(), inputs, retain_graph=True, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(expected.square().sum(), inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        # An expert never selected has no graph in the sparse implementation.
        if actual_gradient is None:
            actual_gradient = torch.zeros_like(expected_gradient)
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient)
    if top_k > 1:
        router_gradient = actual_gradients[1]
        assert router_gradient is not None and router_gradient.abs().sum() > 0


def test_expert_parameters_are_registered_and_independent():
    model = NaiveMoe(3, 5, 4, 2)
    parameters = dict(model.named_parameters())
    weight_pointers = set()
    for i, expert in enumerate(model.experts):
        assert isinstance(expert, SwiGLU)
        assert parameters[f"experts.{i}.up_proj.weight"] is expert.up_proj.weight
        weight_pointers.add(expert.up_proj.weight.data_ptr())
    assert len(weight_pointers) == 4
    assert "router.router.weight" in model.state_dict()
    model.double()
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
