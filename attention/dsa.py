"""
B = batch size
T = tokens
Q = query tokens
D = dimension
H = number of heads
Di = index dimension
Dv = value dimension
k = top k
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from positional_encoding import RotaryPositionalEncoding


def _select_causal_topk(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """Select positions [B, Q, k] from scores [B, Q, T] for the last Q tokens."""
    query_count, key_count = scores.shape[-2:]
    if query_count > key_count:
        raise ValueError("Suffix queries cannot outnumber the keys")

    key_positions = torch.arange(key_count, device=scores.device)  # [T]
    # Local query row 0 represents absolute position T - Q.
    query_positions = torch.arange(query_count, device=scores.device) + (
        key_count - query_count
    )  # [Q]
    valid = key_positions[None, :] <= query_positions[:, None]  # [Q, T]
    masked_scores = scores.masked_fill(~valid, float("-inf"))  # [B, Q, T]

    top_scores, indices = masked_scores.topk(min(top_k, key_count), dim=-1)  # [B, Q, k]
    return indices.masked_fill_(top_scores == float("-inf"), -1)  # [B, Q, k]


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


@dataclass
class IndexerCache:
    """Store projected, normalized, RoPE-ready keys [B, T, Di] for one layer."""

    keys: torch.Tensor | None = None


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

    def forward(
        self, x: torch.Tensor, *, cache: IndexerCache | None = None
    ) -> torch.Tensor:
        """Select positions [B, Q, k] for new states x [B, Q, D].

        Pass an empty cache for prefill, then reuse it for the same batch.
        """
        q = self.q_proj(x)  # [B, Q, H * Di]
        k = self.k_norm(self.k_proj(x))  # [B, Q, Di]
        head_weights = self.head_weights(x)  # [B, Q, H]

        return self.select_tokens_causal(q, k, head_weights, cache=cache)  # [B, Q, k]

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
        queries, keys = self._prepare_qk(index_queries, index_keys)
        return self._score_heads(queries, keys, head_weights)  # [B, T, T]

    def _prepare_qk(
        self, index_queries: torch.Tensor, index_keys: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_count, _ = index_queries.shape
        index_queries = index_queries.reshape(
            batch_size, query_count, self.n_heads, index_keys.size(-1)
        )  # [B, Q, H, Di]
        if self.rope is not None:
            rope_dim = self.rope.d_head
            # K has one shared head.
            q_rot = index_queries[..., :rope_dim].transpose(1, 2)  # [B, H, Q, rope_dim]
            k_rot = index_keys[..., :rope_dim].unsqueeze(1)  # [B, 1, Q, rope_dim]
            q_rot, k_rot = self.rope(q_rot, k_rot, offset=offset)
            index_queries = torch.cat(
                (q_rot.transpose(1, 2), index_queries[..., rope_dim:]),
                dim=-1,
            )  # [B, Q, H, Di]
            index_keys = torch.cat(
                (k_rot.squeeze(1), index_keys[..., rope_dim:]), dim=-1
            )  # [B, Q, Di]

        return index_queries, index_keys

    def _score_heads(
        self, queries: torch.Tensor, keys: torch.Tensor, head_weights: torch.Tensor
    ) -> torch.Tensor:
        # Broadcast keys across queries within each sequence.
        keys_t = keys.transpose(-2, -1).unsqueeze(1)  # [B, 1, Di, T]
        scores = queries @ keys_t  # [B, Q, H, T]
        scores /= math.sqrt(keys.size(-1))  # [B, Q, H, T]
        scores = nn.functional.relu(scores)  # [B, Q, H, T]

        # Aggregate with head weights
        scaled_head_weights = head_weights / math.sqrt(self.n_heads)  # [B, Q, H]
        scores = scores * scaled_head_weights.unsqueeze(-1)  # [B, Q, H, T]
        scores = scores.sum(dim=-2)  # [B, Q, T]

        return scores

    def select_tokens_causal(
        self,
        index_queries: torch.Tensor,
        index_keys: torch.Tensor,
        head_weights: torch.Tensor,
        *,
        cache: IndexerCache | None = None,
    ) -> torch.Tensor:
        """Select up to k keys per query and mark unused slots with -1.

        Inputs cover only the Q new tokens. Cached keys are already prepared.
        """
        past_keys = cache.keys if cache is not None else None  # [B, T - Q, Di]
        offset = 0 if past_keys is None else past_keys.size(1)
        queries, keys = self._prepare_qk(index_queries, index_keys, offset)
        if past_keys is not None:
            keys = torch.cat((past_keys, keys), dim=1)  # [B, T, Di]

        scores = self._score_heads(queries, keys, head_weights)  # [B, Q, T]
        indices = _select_causal_topk(scores, self.top_k)  # [B, Q, k]
        if cache is not None:
            cache.keys = keys.detach()
        return indices


def sparse_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Attend to selected K/V positions and return [B, Q, Dv].

    Queries are [B, Q, D], keys [B, T, D], values [B, T, Dv], and indices
    [B, Q, k]. Each query needs at least one valid index; -1 marks unused
    slots. All inputs can also omit the batch dimension.
    """
    if queries.ndim == 2:
        return sparse_attention(
            queries[None], keys[None], values[None], indices[None]
        ).squeeze(0)  # [Q, Dv]

    exclude_mask = indices == -1  # [B, Q, k]
    safe_indices = indices.masked_fill(exclude_mask, 0)  # [B, Q, k]
    # [B, 1, 1] broadcasts across query positions and selected slots.
    batch_indices = torch.arange(keys.size(0), device=keys.device)[:, None, None]

    # This reference materializes K/V per query, duplicating rows across queries.
    # Fused kernels load selected rows directly as they compute attention.
    gathered_keys = keys[batch_indices, safe_indices]  # [B, Q, k, D]
    gathered_values = values[batch_indices, safe_indices]  # [B, Q, k, Dv]

    # Compute attention scores
    attn_scores = queries.unsqueeze(-2) @ gathered_keys.transpose(
        -2, -1
    )  # [B, Q, 1, k]
    attn_scores = attn_scores.squeeze(-2) / math.sqrt(queries.size(-1))  # [B, Q, k]

    attn_scores.masked_fill_(exclude_mask, float("-inf"))  # [B, Q, k]
    attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, Q, k]
    attn_values = attn_weights.unsqueeze(-2) @ gathered_values  # [B, Q, 1, Dv]

    return attn_values.squeeze(-2)  # [B, Q, Dv]
