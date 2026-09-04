import torch
from torch import nn

from mlp import SwiGLU
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
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_p=dropout_p,
            rope=rope,
        )
        self.ffn = SwiGLU(d_model, hidden_dim, dropout_p)
        self.norm1 = RMSNorm(d_model, eps=eps)
        self.norm2 = RMSNorm(d_model, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if use_cache:
            attn_output, updated_cache = self.attn(
                self.norm1(x), causal=True, use_cache=True, kv_cache=kv_cache
            )
            x_attn = x + attn_output
        else:
            x_attn = x + self.attn(self.norm1(x), causal=True)
        x_ffn = x_attn + self.ffn(self.norm2(x_attn))

        if use_cache:
            return x_ffn, updated_cache
        return x_ffn
