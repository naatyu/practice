import torch
from torch import nn

from mlp import SwiGLU
from multi_head_attention import MultiHeadAttention
from positional_encoding import RotaryPositionalEncoding
from rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        dropout_p: float = 0.0,
        rope: RotaryPositionalEncoding | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout_p, rope)
        self.ffn = SwiGLU(d_model, hidden_dim, dropout_p)
        self.norm1 = RMSNorm(d_model, eps=eps)
        self.norm2 = RMSNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor, position_offset: int = 0):
        x_attn = x + self.attn(
            self.norm1(x), causal=True, position_offset=position_offset
        )
        x_ffn = x_attn + self.ffn(self.norm2(x_attn))

        return x_ffn
