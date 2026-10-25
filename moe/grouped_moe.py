"""Expert-sorted MoE dispatch, capacity handling, and weighted combination."""

import math
from typing import Literal

import torch
from torch import nn

from mlp import SwiGLU
from normalization import RMSNorm

from .experts import GroupedExperts, PackedExperts
from .router import TopKRouter


class GroupedMoE(nn.Module):
    """Packed dispatch with optional position- or weight-priority capacity limits.

    capacity_factor=None is dropless; otherwise capacity is ceil(factor * T*K/E).
    Dropped assignments are not computed; surviving weights are not renormalized.
    softcap_betas=None keeps SwiGLU; (beta_gate, beta_value) enables SiTU-GLU
    for routed experts only. Both caps must be finite and strictly positive.
    QB uses exact quantiles when num_bins=None, otherwise histogram estimation.
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
        shared_hidden_dim: int | None = None,
        shared_expert_gate: bool = False,
        d_latent: int | None = None,
        softcap_betas: tuple[float, float] | None = None,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        use_quantile_balancing: bool = False,
        num_bins: int | None = None,
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
        if shared_hidden_dim is None and shared_expert_gate:
            raise ValueError(
                "Can't set shared expert gating when no shared expert is set."
            )
        if d_latent is not None and d_latent <= 0:
            raise ValueError(
                f"Expected d_latent to be strictly positive, got: {d_latent}"
            )

        self.router = TopKRouter(
            d_model,
            num_experts,
            top_k,
            score_func=score_func,
            use_quantile_balancing=use_quantile_balancing,
            num_bins=num_bins,
        )
        expert_cls = PackedExperts if expert_backend == "loop" else GroupedExperts
        self.experts = expert_cls(
            d_model if d_latent is None else d_latent,
            hidden_dim,
            num_experts,
            softcap_betas=softcap_betas,
        )
        self.shared_expert = (
            SwiGLU(d_model, shared_hidden_dim)
            if shared_hidden_dim is not None
            else None
        )
        self.shared_expert_gate = (
            nn.Linear(d_model, 1, bias=False) if shared_expert_gate else None
        )
        if d_latent is not None:
            self.latent_down = nn.Linear(d_model, d_latent, bias=False)
            self.latent_up = nn.Linear(d_latent, d_model, bias=False)
            self.latent_norm = RMSNorm(d_latent)

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.drop_policy = drop_policy
        self.d_latent = d_latent

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
        x/output: [B, S, d_model]. Routed width is d_latent when enabled,
        otherwise d_model; the router/shared expert always use d_model.
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

        # Latent down projection
        if self.d_latent is not None:
            tokens = self.latent_down(tokens)  # [T, D_latent]

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
        grouped_tokens = tokens[token_indices]  # [N, D | D_latent]

        # Process only retained assignments.
        grouped_output = self.experts(grouped_tokens, offsets)  # [N, D | D_latent]

        # Preserve original mixing weights; dropping does not renormalize them.
        grouped_weights = expert_weights.flatten()[permutation]  # [N]
        weighted_output = grouped_output * grouped_weights.unsqueeze(
            -1
        )  # [N, D | D_latent]

        # Tokens with no retained assignments get zero MoE output, not removed rows.
        output = torch.zeros_like(tokens)  # [T, D | D_latent]
        # FP32 routing weights can promote low-precision expert outputs.
        output.index_add_(
            dim=0, index=token_indices, source=weighted_output.to(output.dtype)
        )  # [T, D | D_latent]

        # Latent up projection
        if self.d_latent is not None:
            output = self.latent_norm(output)  # [T, D_latent]
            output = self.latent_up(output)  # [T, D]

        output = output.reshape_as(x)  # [B, S, D]
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                gate = torch.sigmoid(self.shared_expert_gate(x))  # [B, S, 1]
                shared_output = gate * shared_output

            output += shared_output

        return output, aux_loss, z_loss


if __name__ == "__main__":
    tokens = torch.randn((3, 4, 2))
    # Tiny dimensions are useful for debugging but violate grouped-MM alignment.
    moe = GroupedMoE(2, hidden_dim=4, num_experts=4, top_k=2, expert_backend="loop")
    moe(tokens)
