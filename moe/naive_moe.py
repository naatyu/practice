import torch
from torch import nn

from mlp import SwiGLU
from .router import TopKRouter


class NaiveMoe(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        """
        x: [B, S, D]
        """
        batch, sequence, d_model = x.shape
        experts, weights = self.router(x)  # [B, S, K], [B, S, K]
        output = torch.empty_like(x)  # [B, S, D]

        for b in range(batch):
            for s in range(sequence):
                weighted_output = torch.zeros(
                    (d_model), device=x.device, dtype=x.dtype
                )  # [D]

                for k in range(experts.shape[-1]):
                    expert_id = experts[b, s, k]
                    expert_weight = weights[b, s, k]
                    weighted_output += (
                        self.experts[expert_id](x[b, s]) * expert_weight
                    )  # [D]

                output[b, s] = weighted_output

        return output
