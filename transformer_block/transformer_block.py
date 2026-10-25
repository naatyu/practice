from typing import Literal

import torch
from torch import nn

from mlp import SwiGLU
from moe import GroupedMoE
from multi_head_attention import MultiHeadAttention
from normalization import RMSNorm
from positional_encoding import RotaryPositionalEncoding


class TransformerBlock(nn.Module):
    """Pre-norm block; optional QB mode/bin count are forwarded to its MoE router."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        num_kv_heads: int | None = None,
        dropout_p: float = 0.0,
        rope: RotaryPositionalEncoding | None = None,
        window_size: int | None = None,
        eps: float = 1e-6,
        num_experts: int | None = None,
        top_k: int | None = None,
        *,
        capacity_factor: float | None = None,
        drop_policy: Literal["position", "probs"] = "probs",
        shared_hidden_dim: int | None = None,
        shared_expert_gate: bool = False,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        use_quantile_balancing: bool = False,
        num_bins: int | None = None,
    ):
        super().__init__()
        if num_experts is None and (use_quantile_balancing or num_bins is not None):
            raise ValueError("Quantile balancing options require MoE experts")
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_p=dropout_p,
            rope=rope,
            window_size=window_size,
        )

        self.norm1 = RMSNorm(d_model, eps=eps)
        self.norm2 = RMSNorm(d_model, eps=eps)

        # Select between ffn or moe
        if num_experts is not None:
            if top_k is None:
                raise ValueError("Top K can't be None when using moe.")
            self.ffn = GroupedMoE(
                d_model,
                hidden_dim,
                num_experts,
                top_k,
                capacity_factor=capacity_factor,
                drop_policy=drop_policy,
                shared_hidden_dim=shared_hidden_dim,
                shared_expert_gate=shared_expert_gate,
                score_func=score_func,
                use_quantile_balancing=use_quantile_balancing,
                num_bins=num_bins,
            )
        else:
            self.ffn = SwiGLU(d_model, hidden_dim, dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool = False,
        position_offset: int | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Transform x [B, Q, d_model].

        Optional per-layer cache: (K, V), each
        [B, num_kv_heads, K, d_head]. Output has x's shape and, when
        requested, an updated cache with the same leading dimensions.
        Returns hidden states, optional cache, balancing loss, and z-loss.
        """
        updated_cache = None
        if use_cache:
            attn_output, updated_cache = self.attn(
                self.norm1(x),
                causal=True,
                use_cache=True,
                kv_cache=kv_cache,
                position_offset=position_offset,
            )
            x_attn = x + attn_output
        else:
            x_attn = x + self.attn(
                self.norm1(x), causal=True, position_offset=position_offset
            )
        if isinstance(self.ffn, GroupedMoE):
            ffn_output, aux_loss, z_loss = self.ffn(self.norm2(x_attn))
        else:
            ffn_output = self.ffn(self.norm2(x_attn))
            aux_loss = z_loss = None
        return x_attn + ffn_output, updated_cache, aux_loss, z_loss
