from typing import Literal

import torch
from torch import nn

from moe.losses import load_balancing_loss, router_z_loss

from .metrics import (
    assignment_entropy,
    assignment_fraction,
    maxvio,
    router_entropy,
    unused_expert_count,
)


class TopKRouter(nn.Module):
    """Top-k routing with detached training statistics."""

    tokens_per_expert: torch.Tensor
    router_prob_sum: torch.Tensor
    num_tokens: torch.Tensor
    router_entropy_sum: torch.Tensor
    selected_prob_mass_sum: torch.Tensor
    expert_bias: torch.Tensor

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.score_func = score_func

        if not (1 <= top_k <= num_experts):
            raise ValueError(f"Top K must be between 1 and {num_experts}, got: {top_k}")
        if score_func not in ["softmax", "sigmoid"]:
            raise ValueError(
                f'score_func must be either "softmax" or "sigmoid", got: {score_func}'
            )

        self.router = nn.Linear(d_model, num_experts, bias=False)

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "router_prob_sum",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "num_tokens", torch.zeros((), dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "router_entropy_sum", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "selected_prob_mass_sum",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts, dtype=torch.float32),
        )

    @torch.no_grad()
    def reset_metrics(self) -> None:
        self.tokens_per_expert.zero_()
        self.router_prob_sum.zero_()
        self.num_tokens.zero_()
        self.router_entropy_sum.zero_()
        self.selected_prob_mass_sum.zero_()

    @torch.no_grad()
    def compute_metrics(self) -> dict[str, torch.Tensor]:
        """Read statistics from the full normalized affinity view; means are [E]."""
        metrics = {}

        metrics["maxvio"] = maxvio(self.tokens_per_expert)
        metrics["assignment_fraction"] = assignment_fraction(self.tokens_per_expert)
        metrics["unused_expert_count"] = unused_expert_count(self.tokens_per_expert)
        metrics["assignment_entropy"] = assignment_entropy(self.tokens_per_expert)
        metrics["mean_router_probability"] = (
            self.router_prob_sum / self.num_tokens.clamp_min(1)
        )
        metrics["mean_router_entropy"] = (
            self.router_entropy_sum / self.num_tokens.clamp_min(1)
        )
        metrics["mean_selected_probability_mass"] = (
            self.selected_prob_mass_sum / self.num_tokens.clamp_min(1)
        )

        return metrics

    @torch.no_grad()
    def update_expert_bias(self, rate: float) -> None:
        target_count = self.tokens_per_expert.sum() / self.num_experts
        self.expert_bias += rate * torch.sign(
            target_count - self.tokens_per_expert
        )  # [E]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        E = number of experts
        K = number of selected experts
        x: [B, S, D]
        Returns IDs, weights [B, S, K], balancing loss, and z-loss (None in eval).
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = self.router(x.float())  # [B, S, E]

            if self.score_func == "softmax":
                router_scores = nn.functional.softmax(
                    router_logits, dim=-1
                )  # [B, S, E]
                normalized_scores = router_scores
            elif self.score_func == "sigmoid":
                router_scores = nn.functional.sigmoid(router_logits)  # [B, S, E]
                normalized_scores = router_scores / router_scores.sum(
                    -1, keepdim=True
                )  # [B, S, E]

        _, experts = torch.topk(
            router_scores + self.expert_bias, k=self.top_k, dim=-1
        )  # [B, S, K]
        scores = router_scores.gather(dim=-1, index=experts)  # [B, S, K]

        counts = torch.bincount(experts.flatten(), minlength=self.num_experts)  # [E]
        weights = scores / torch.sum(scores, dim=-1, keepdim=True)  # [B, S, K]

        aux_loss = (
            load_balancing_loss(normalized_scores, counts) if self.training else None
        )
        z_loss = (
            router_z_loss(router_logits)
            if self.training and self.score_func != "sigmoid"
            else None
        )

        if self.training:
            with torch.no_grad():
                self.tokens_per_expert += counts
                self.router_prob_sum += normalized_scores.float().sum(dim=(0, 1))  # [E]
                self.num_tokens += x.shape[0] * x.shape[1]  # B*S
                self.router_entropy_sum += router_entropy(
                    normalized_scores.float()
                ).sum()
                self.selected_prob_mass_sum += (
                    normalized_scores.gather(-1, experts).float().sum()
                )

        return experts, weights, aux_loss, z_loss


if __name__ == "__main__":
    x = torch.randn((4, 6, 5))
    router = TopKRouter(d_model=5, num_experts=4, top_k=2)
    router(x)
