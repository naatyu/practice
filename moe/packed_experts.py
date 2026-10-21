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


class GroupedMoE(nn.Module):
    """Packed dispatch with optional position- or weight-priority capacity limits.

    capacity_factor=None is dropless; otherwise capacity is ceil(factor * T*K/E).
    Dropped assignments are not computed; surviving weights are not renormalized.
    """

    dropped_assignments_per_expert: torch.Tensor

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        *,
        capacity_factor: float | None = None,
        drop_policy: Literal["position", "probs"] = "probs",
        expert_backend: Literal["loop", "grouped_mm"] = "grouped_mm",
    ):
        super().__init__()
        if expert_backend not in ("loop", "grouped_mm"):
            raise ValueError(f"Unknown expert backend: {expert_backend!r}")
        if drop_policy not in ("position", "probs"):
            raise ValueError(f"Unknown drop policy: {drop_policy!r}")
        if capacity_factor is not None and (
            not math.isfinite(capacity_factor) or capacity_factor <= 0
        ):
            raise ValueError(
                f"Expected capacity_factor to be finite and strictly positive, got: {capacity_factor}"
            )

        self.router = TopKRouter(d_model, num_experts, top_k)
        expert_cls = PackedExperts if expert_backend == "loop" else GroupedExperts
        self.experts = expert_cls(d_model, hidden_dim, num_experts)

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.drop_policy = drop_policy

        self.register_buffer(
            "dropped_assignments_per_expert",
            torch.zeros((num_experts), dtype=torch.int64),
            persistent=False,
        )

    @torch.no_grad()
    def reset_metrics(self):
        self.router.reset_metrics()
        self.dropped_assignments_per_expert.zero_()

    @torch.no_grad()
    def compute_metrics(self) -> dict[str, torch.Tensor]:
        metrics = self.router.compute_metrics()
        metrics["dropped_assignment_rate_per_expert"] = (
            self.dropped_assignments_per_expert
            / self.router.tokens_per_expert.clamp_min(1)
        )
        metrics["dropped_assignment_rate"] = (
            self.dropped_assignments_per_expert.sum()
            / self.router.tokens_per_expert.sum().clamp_min(1)
        )

        return metrics

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        x: [B, S, D]
        T = B * S
        K = top k experts
        E = total num of experts
        N = retained assignments, at most T * K
        """

        # Token routing
        expert_ids, expert_weights, aux_loss, z_loss = self.router(
            x
        )  # [B, S, K], [B, S, K], scalar

        tokens = x.reshape(-1, self.d_model)  # [T, D]
        flat_expert_ids = expert_ids.flatten()  # [T * K]

        # Group by expert; capacity determines whether token or weight order wins.
        if self.capacity_factor is not None and self.drop_policy == "probs":
            # Original assignment indices, ranked by descending weight.
            weights_permutation = torch.argsort(
                expert_weights.flatten(), stable=True, descending=True
            )
            # Positions in that weight-sorted array, grouped stably by expert.
            expert_order = torch.argsort(
                flat_expert_ids[weights_permutation], stable=True
            )
            # Map back to original indices: each expert's strongest come first.
            permutation = weights_permutation[expert_order]  # [T * K]
        else:
            permutation = torch.argsort(flat_expert_ids, stable=True)  # [T * K]

        expert_counts = torch.bincount(
            flat_expert_ids, minlength=self.num_experts
        )  # [E]

        # Keep the first C assignments in each expert's sorted group.
        if self.capacity_factor is not None:
            capacity = math.ceil(
                self.capacity_factor * flat_expert_ids.numel() / self.num_experts
            )
            group_starts = expert_counts.cumsum(dim=0) - expert_counts  # [E]
            positions_in_group = torch.arange(
                flat_expert_ids.numel(), device=x.device
            ) - torch.repeat_interleave(group_starts, expert_counts)  # [T * K]
            permutation = permutation[positions_in_group < capacity]  # [N]
            retained_counts = expert_counts.clamp(max=capacity)  # [E]

            # Accumulate drop metric
            if self.training:
                with torch.no_grad():
                    self.dropped_assignments_per_expert += (
                        expert_counts - retained_counts
                    )
        else:
            retained_counts = expert_counts

        # Offsets must describe the retained groups, not the original requests.
        offsets = retained_counts.cumsum(dim=0, dtype=torch.int32)  # [E]
        token_indices = permutation // self.top_k  # [N]
        grouped_tokens = tokens[token_indices]  # [N, D]

        # Process only retained assignments.
        grouped_output = self.experts(grouped_tokens, offsets)  # [N, D]

        # Preserve original mixing weights; dropping does not renormalize them.
        grouped_weights = expert_weights.flatten()[permutation]  # [N]
        weighted_output = grouped_output * grouped_weights.unsqueeze(-1)  # [N, D]

        # Tokens with no retained assignments get zero MoE output, not removed rows.
        output = torch.zeros_like(tokens)  # [T, D]
        # FP32 routing weights can promote low-precision expert outputs.
        output.index_add_(
            dim=0, index=token_indices, source=weighted_output.to(output.dtype)
        )  # [T, D]

        return output.reshape_as(x), aux_loss, z_loss  # [B, S, D], optional scalars


if __name__ == "__main__":
    tokens = torch.randn((3, 4, 2))
    moe = GroupedMoE(2, hidden_dim=4, num_experts=4, top_k=2)
    moe(tokens)
