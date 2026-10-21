import torch
from torch import nn

from mlp import SwiGLU

from .router import TopKRouter


class ExpertBatchedMoe(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, hidden_dim) for _ in range(num_experts)]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        x: [B, S, D]
        """
        experts, weights, aux_loss, z_loss = self.router(
            x
        )  # [B, S, K], [B, S, K], scalar
        output = torch.zeros_like(x)  # [B, S, D]

        for expert_id, expert in enumerate(self.experts):
            batch_indices, sequence_indices, slots = torch.where(
                experts == expert_id
            )  # [N], [N], [N]
            expert_input = x[batch_indices, sequence_indices]  # [N, D]
            experts_weights = weights[batch_indices, sequence_indices, slots]  # [N]
            expert_output = expert(expert_input)  # [N, D]
            weighted_output = expert_output * experts_weights.unsqueeze(-1)  # [N, D]

            output[batch_indices, sequence_indices] += weighted_output

        return output, aux_loss, z_loss
