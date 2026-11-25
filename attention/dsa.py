"""
B = batch size
T = tokens
D = dimension
H = number of heads
Di = index dimension
Dv = value dimension
k = top k
"""

import math

import torch
from torch import nn

from positional_encoding import RotaryPositionalEncoding


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


class LightningIndexer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_indexer: int,
        n_heads: int,
        top_k: int,
        rope: RotaryPositionalEncoding | None = None,
    ):
        super().__init__()

        if top_k <= 0:
            raise ValueError(f"Top-k must be greater than 0, got: {top_k}")
        if rope is not None and rope.d_head > d_indexer:
            raise ValueError("RoPE width must not exceed the index head width")

        self.n_heads = n_heads
        self.top_k = top_k
        self.rope = rope
        self.q_proj = nn.Linear(d_model, n_heads * d_indexer, bias=False)
        self.k_proj = nn.Linear(d_model, d_indexer, bias=False)
        self.head_weights = nn.Linear(d_model, n_heads, bias=False)
        self.k_norm = nn.LayerNorm(d_indexer, eps=1e-6)

    def forward(self, x: torch.Tensor):
        """Select token positions [B, T, k] from hidden states x [B, T, D]."""
        q = self.q_proj(x)  # [B, T, H * Di]
        k = self.k_norm(self.k_proj(x))  # [B, T, Di]
        head_weights = self.head_weights(x)  # [B, T, H]

        selected_tokens = self.select_tokens_causal(q, k, head_weights)  # [B, T, k]

        return selected_tokens

    def index_scores(
        self,
        index_queries: torch.Tensor,
        index_keys: torch.Tensor,
        head_weights: torch.Tensor,
    ):
        """Return one unmasked ranking score per query–key pair, shaped [B, T, T].

        Split the projected queries into heads [B, T, H, Di] and optionally rotate
        a prefix of Q and K. Each head scores its sequence's keys [B, T, Di] with scaled
        dot products. ReLU zeros negative matches before the weighted head sum.
        These scores rank keys for top-k; they are not attention weights.
        """
        # Compute per head score
        batch_size, seq_len, _ = index_queries.shape
        index_queries = index_queries.reshape(
            batch_size, seq_len, self.n_heads, index_keys.size(-1)
        )  # [B, T, H, Di]
        if self.rope is not None:
            rope_dim = self.rope.d_head
            # K has one shared head.
            q_rot = index_queries[..., :rope_dim].transpose(1, 2)  # [B, H, T, rope_dim]
            k_rot = index_keys[..., :rope_dim].unsqueeze(1)  # [B, 1, T, rope_dim]
            q_rot, k_rot = self.rope(q_rot, k_rot)
            index_queries = torch.cat(
                (q_rot.transpose(1, 2), index_queries[..., rope_dim:]),
                dim=-1,
            )  # [B, T, H, Di]
            index_keys = torch.cat(
                (k_rot.squeeze(1), index_keys[..., rope_dim:]), dim=-1
            )  # [B, T, Di]

        # Broadcast keys across queries within each sequence.
        keys_t = index_keys.transpose(-2, -1).unsqueeze(1)  # [B, 1, Di, T]
        scores = index_queries @ keys_t  # [B, T, H, T]
        scores /= math.sqrt(index_keys.shape[-1])  # [B, T, H, T]
        scores = nn.functional.relu(scores)  # [B, T, H, T]

        # Aggregate with head weights
        scaled_head_weights = head_weights / math.sqrt(self.n_heads)  # [B, T, H]
        scores = scores * scaled_head_weights.unsqueeze(-1)  # [B, T, H, T]
        scores = scores.sum(dim=-2)  # [B, T, T]

        return scores

    def select_tokens_causal(
        self,
        index_queries: torch.Tensor,
        index_keys: torch.Tensor,
        head_weights: torch.Tensor,
    ):
        """Select up to k keys per query and mark unused slots with -1.

        Queries are [B, T, H * Di], keys [B, T, Di], and weights [B, T, H].
        """

        index_scores = self.index_scores(
            index_queries, index_keys, head_weights
        )  # [B, T, T]

        # Share the causal mask across the batch.
        key_count = index_keys.size(-2)
        positions = torch.arange(key_count, device=index_keys.device)  # [T]
        valid_mask = positions[None, :] <= positions[:, None]  # [T, T]
        index_scores.masked_fill_(~valid_mask, float("-inf"))  # [B, T, T]

        # Top-K selection
        top_scores, indices = torch.topk(
            index_scores,
            min(self.top_k, key_count),
            dim=-1,
        )  # [B, T, k]

        return indices.masked_fill_(top_scores == float("-inf"), -1)  # [B, T, k]


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
