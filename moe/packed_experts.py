import math
from typing import Literal

import torch
from torch import nn

from .router import TopKRouter


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

    def forward(  # ty: ignore[invalid-mutable-override]
        self, grouped_tokens: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Process token assignments already grouped by expert.

        grouped_tokens: [N, D], N = B*S*K assignments before any dropping.
        offsets: [E], cumulative exclusive ends; counts [2, 0, 3] -> [2, 2, 5].
        N_e: assignments for expert e; repeated offsets mean an empty group.
        K: number of selected experts.

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

    def forward(  # ty: ignore[invalid-mutable-override]
        self, grouped_tokens: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Process token assignments already grouped by expert.

        grouped_tokens: [N, D], N = B*S*K assignments before any dropping.
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


class GroupedMoE(nn.Module):
    """Route, group, process, and combine assignments using packed expert weights."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        *,
        expert_backend: Literal["loop", "grouped_mm"] = "grouped_mm",
    ):
        super().__init__()
        if expert_backend not in ("loop", "grouped_mm"):
            raise ValueError(f"Unknown expert backend: {expert_backend!r}")
        self.router = TopKRouter(d_model, num_experts, top_k)
        expert_cls = PackedExperts if expert_backend == "loop" else GroupedExperts
        self.experts = expert_cls(d_model, hidden_dim, num_experts)

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        """
        x: [B, S, D]
        T = B * S
        K = top k experts
        E = total num of experts
        """

        # Token routing
        expert_ids, expert_weights = self.router(x)  # [B, S, K], [B, S, K]

        tokens = x.reshape(-1, self.d_model)  # [T, D]
        flat_expert_ids = expert_ids.flatten()  # [T * K]

        # Group tokens by expert
        permutation = torch.argsort(flat_expert_ids, stable=True)  # [T * K]
        token_indices = permutation // self.top_k  # [T * K]
        grouped_tokens = tokens[token_indices]  # [T * K, D]

        # Compute token offsets of experts
        expert_counts = torch.bincount(
            flat_expert_ids, minlength=self.num_experts
        )  # [E]
        offsets = expert_counts.cumsum(dim=0, dtype=torch.int32)  # [E]

        # Process expert groups
        grouped_output = self.experts(grouped_tokens, offsets)  # [T * K, D]

        # Apply matching router weights
        grouped_weights = expert_weights.flatten()[permutation]  # [T * K]
        weighted_output = grouped_output * grouped_weights.unsqueeze(-1)  # [T * K, D]

        # Recombine tokens
        output = torch.zeros_like(tokens)  # [T, D]
        output.index_add_(dim=0, index=token_indices, source=weighted_output)  # [T, D]

        return output.reshape_as(x)  # [B, S, D]


if __name__ == "__main__":
    tokens = torch.randn((3, 4, 2))
    moe = GroupedMoE(2, hidden_dim=4, num_experts=4, top_k=2)
    moe(tokens)
