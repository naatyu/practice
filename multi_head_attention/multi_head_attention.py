import torch
from torch import nn

from attention import attention


class MultiHeadAttention(nn.Module):
    """Implement multi-head self-attention.

    Do not use torch.nn.MultiheadAttention or PyTorch's built-in
    scaled-dot-product attention.

    Input and output shape:
        (batch_size, sequence_length, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        # check num heads
        if num_heads <= 0:
            raise ValueError(f"Expected num_heads to be > 0 got {num_heads} instead.")
        # Check for divisibility
        if d_model % num_heads != 0:
            raise ValueError(
                f"Expected d_model to be divisible by num_heads but got {d_model} and {num_heads} instead"
            )
        # Check dropout value
        if not (0 <= dropout_p <= 1):
            raise ValueError(
                f"Expected dropout in [0, 1] interval, got {dropout_p} instead."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout_p = dropout_p
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, S, d_model = x.shape

        qkv = self.qkv(x)  # [B,S,3d_model]
        q, k, v = qkv.split(self.d_model, dim=-1)  # [B,S,d_model]
        # Prepare for attention [B,S,d_model] -> [B, S, H, d_head] -> [B, H, S, d_head]
        q = q.reshape(B, S, self.num_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, S, self.num_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, S, self.num_heads, self.d_head).transpose(1, 2)

        dropout_p = self.dropout_p if self.training else 0.0

        attn_out = attention(q, k, v, dropout_p, causal=causal)  # [B,H,S,d_head]

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, d_model)  # [B,S,d_model]

        return self.out(attn_out)
