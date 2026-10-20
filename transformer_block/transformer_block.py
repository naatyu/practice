import torch
from torch import nn

from mlp import SwiGLU
from moe import GroupedMoE
from multi_head_attention import MultiHeadAttention
from normalization import RMSNorm
from positional_encoding import RotaryPositionalEncoding


class TransformerBlock(nn.Module):
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
    ):
        super().__init__()
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
            self.ffn = GroupedMoE(d_model, hidden_dim, num_experts, top_k)
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
        torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None
    ]:
        """Transform x [B, Q, d_model].

        Optional per-layer cache: (K, V), each
        [B, num_kv_heads, K, d_head]. Output has x's shape and, when
        requested, an updated cache with the same leading dimensions.
        Always returns (hidden states, optional cache, optional auxiliary loss).
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
            ffn_output, aux_loss = self.ffn(self.norm2(x_attn))
        else:
            ffn_output = self.ffn(self.norm2(x_attn))
            aux_loss = None
        return x_attn + ffn_output, updated_cache, aux_loss
