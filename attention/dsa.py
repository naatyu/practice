"""
T = tokens
D = dimension
Di = index dimension
Dv = value dimension
k = top k
"""

import math

import torch
from torch import nn


class SimpleIndexer(nn.Module):
    def __init__(self, d_model: int, d_indexer: int, top_k: int):
        super().__init__()

        if top_k <= 0:
            raise ValueError(f"Top-k must be greater than 0, got: {top_k}")

        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_indexer, bias=False)
        self.k_proj = nn.Linear(d_model, d_indexer, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x [T, D]
        """
        q = self.q_proj(x)  # [T, Di]
        k = self.k_proj(x)  # [T, Di]

        selected_tokens = self.select_tokens_causal(q, k)  # [T, k]

        return selected_tokens

    def select_tokens_causal(
        self, index_queries: torch.Tensor, index_keys: torch.Tensor
    ):
        """
        index_queries  [T, Di]
        index_keys     [T, Di]
        """

        index_scores = index_queries @ index_keys.T  # [T, T]

        # Use the causal mask for top-k selection
        positions = torch.arange(
            index_queries.shape[0], device=index_queries.device
        )  # [T]
        valid_mask = positions[None, :] <= positions[:, None]  # [T, T]
        index_scores.masked_fill_(~valid_mask, float("-inf"))  # [T, T]

        # Top-K selection
        top_scores, indices = torch.topk(
            index_scores,
            min(self.top_k, index_queries.shape[0]),
            dim=-1,
        )  # [T, k]

        return indices.masked_fill_(top_scores == float("-inf"), -1)  # [T, k]


def sparse_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
):
    """
    queries [T, D]
    keys: [T, D]
    values: [T, Dv]
    indices: [T, k]
    Output: [T, Dv]
    """
    # Create safe indicies for gathering
    exclude_mask = indices == -1  # [T, k]
    safe_indices = indices.masked_fill(indices == -1, 0)  # [T, k]

    # This reference materializes K/V per query, duplicating rows across queries.
    # Fused kernels load selected rows directly as they compute attention.
    gathered_keys = keys[safe_indices]  # [T, k, D]
    gathered_values = values[safe_indices]  # [T, k, Dv]

    # Compute attention scores
    attn_scores = queries.unsqueeze(1) @ gathered_keys.transpose(2, 1)  # [T, 1, k]
    attn_scores = attn_scores.squeeze(1)  # [T, k]
    attn_scores = attn_scores / math.sqrt(queries.shape[-1])  # [T, k]

    # Exclude non valid indices
    attn_scores.masked_fill_(exclude_mask, float("-inf"))  # [T, k]

    # Sofmtax scoring and values
    attn_scores = torch.softmax(attn_scores, dim=-1)  # [T, k]
    attn_values = attn_scores.unsqueeze(1) @ gathered_values  # [T, 1, Dv]

    return attn_values.squeeze(1)  # [T, Dv]
