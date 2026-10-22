"""Packed expert weights with looped or grouped-matmul execution."""

import math

import torch
from torch import nn


class PackedExperts(nn.Module):
    """Packed bias-free SwiGLU experts: E experts, model width D, hidden width F."""

    def __init__(self, d_model: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.up_weight = nn.Parameter(
            torch.empty(num_experts, d_model, 2 * hidden_dim)
        )  # [E, D, 2F]
        self.down_weight = nn.Parameter(
            torch.empty(num_experts, hidden_dim, d_model)
        )  # [E, F, D]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use zero-mean normal weights with std = 1/sqrt(projection fan-in)."""
        nn.init.normal_(self.up_weight, mean=0.0, std=1 / math.sqrt(self.d_model))
        nn.init.normal_(self.down_weight, mean=0.0, std=1 / math.sqrt(self.hidden_dim))

    def forward(
        self, grouped_tokens: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Process token assignments already grouped by expert.

        grouped_tokens: [N, D], N = retained assignments after any dropping.
        offsets: [E], cumulative exclusive ends; counts [2, 0, 3] -> [2, 2, 5].
        N_e: assignments for expert e; repeated offsets mean an empty group.

        Returns [N, D] in grouped order, before router weighting and combining.
        Assumes valid nondecreasing offsets ending at N.
        """
        output = torch.empty_like(grouped_tokens)  # [N, D]

        for expert_id in range(self.num_experts):
            start = offsets[expert_id - 1] if expert_id > 0 else 0
            end = offsets[expert_id]
            expert_tokens = grouped_tokens[start:end]  # [N_e, D]
            up = expert_tokens @ self.up_weight[expert_id]  # [N_e, 2F]
            value, gate = torch.chunk(up, 2, dim=-1)  # [N_e, F], [N_e, F]
            hidden = value * nn.functional.silu(gate)  # [N_e, F]
            output[start:end] = hidden @ self.down_weight[expert_id]  # [N_e, D]

        return output


class GroupedExperts(nn.Module):
    """Packed bias-free SwiGLU experts: E experts, model width D, hidden width F."""

    def __init__(self, d_model: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.up_weight = nn.Parameter(
            torch.empty(num_experts, d_model, 2 * hidden_dim)
        )  # [E, D, 2F]
        self.down_weight = nn.Parameter(
            torch.empty(num_experts, hidden_dim, d_model)
        )  # [E, F, D]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use zero-mean normal weights with std = 1/sqrt(projection fan-in)."""
        nn.init.normal_(self.up_weight, mean=0.0, std=1 / math.sqrt(self.d_model))
        nn.init.normal_(self.down_weight, mean=0.0, std=1 / math.sqrt(self.hidden_dim))

    def forward(
        self, grouped_tokens: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Process token assignments already grouped by expert.

        grouped_tokens: [N, D], N = retained assignments after any dropping.
        offsets: [E], cumulative exclusive ends; counts [2, 0, 3] -> [2, 2, 5].

        Returns [N, D] in grouped order, before router weighting and combining.
        Assumes valid nondecreasing offsets ending at N.
        """
        up = nn.functional.grouped_mm(
            grouped_tokens, self.up_weight, offs=offsets
        )  # [N, 2F]
        value, gate = torch.chunk(up, 2, dim=-1)  # [N, F], [N, F]
        hidden = value * nn.functional.silu(gate)  # [N, F]
        output = nn.functional.grouped_mm(
            hidden, self.down_weight, offs=offsets
        )  # [N, D]

        return output
