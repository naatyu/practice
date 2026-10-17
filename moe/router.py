import torch
from torch import nn


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k

        if not (1 <= top_k <= num_experts):
            raise ValueError(f"Top K must be between 1 and {num_experts}, got: {top_k}")

        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-mutable-override]
        """
        E = number of experts
        K = number of selected experts
        x: [B, S, D]
        """
        router_scores = self.router(x)  # [B, S, E]
        router_scores = nn.functional.softmax(router_scores, dim=-1)  # [B, S, E]
        scores, experts = torch.topk(
            router_scores, k=self.top_k, dim=-1
        )  # [B, S, K], [B, S, K]
        weights = scores / torch.sum(scores, dim=-1, keepdim=True)  # [B, S, K]

        return experts, weights
