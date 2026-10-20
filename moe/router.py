import torch
from torch import nn

from moe.losses import load_balancing_loss

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

    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

        if not (1 <= top_k <= num_experts):
            raise ValueError(f"Top K must be between 1 and {num_experts}, got: {top_k}")

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

    @torch.no_grad()
    def reset_metrics(self) -> None:
        self.tokens_per_expert.zero_()
        self.router_prob_sum.zero_()
        self.num_tokens.zero_()
        self.router_entropy_sum.zero_()
        self.selected_prob_mass_sum.zero_()

    @torch.no_grad()
    def compute_metrics(self) -> dict[str, torch.Tensor]:
        """Read accumulated statistics without resetting; mean probabilities are [E]."""
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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        E = number of experts
        K = number of selected experts
        x: [B, S, D]
        Returns IDs, weights [B, S, K], and a live scalar loss (None in eval).
        """
        router_scores = self.router(x)  # [B, S, E]
        router_scores = nn.functional.softmax(router_scores, dim=-1)  # [B, S, E]

        scores, experts = torch.topk(
            router_scores, k=self.top_k, dim=-1
        )  # [B, S, K], [B, S, K]

        counts = torch.bincount(experts.flatten(), minlength=self.num_experts)  # [E]
        aux_loss = load_balancing_loss(router_scores, counts) if self.training else None

        weights = scores / torch.sum(scores, dim=-1, keepdim=True)  # [B, S, K]

        if self.training:
            with torch.no_grad():
                self.tokens_per_expert += counts
                self.router_prob_sum += router_scores.float().sum(dim=(0, 1))  # [E]
                self.num_tokens += x.shape[0] * x.shape[1]  # B*S
                self.router_entropy_sum += router_entropy(router_scores.float()).sum()
                self.selected_prob_mass_sum += scores.float().sum()

        return experts, weights, aux_loss


if __name__ == "__main__":
    x = torch.randn((4, 6, 5))
    router = TopKRouter(d_model=5, num_experts=4, top_k=2)
    router(x)
