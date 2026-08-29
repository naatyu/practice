import torch
from torch import nn

from attention import attention
from positional_encoding import RotaryPositionalEncoding


class MultiHeadAttention(nn.Module):
    """Implement multi-head self-attention.

    Do not use torch.nn.MultiheadAttention or PyTorch's built-in
    scaled-dot-product attention.

    Input and output shape:
        (batch_size, sequence_length, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout_p: float = 0.0,
        rope: RotaryPositionalEncoding | None = None,
    ) -> None:
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # check num heads
        if num_heads <= 0:
            raise ValueError(f"Expected num_heads to be > 0 got {num_heads} instead.")
        if num_kv_heads <= 0:
            raise ValueError(
                f"Expected num_kv_heads to be > 0 got {num_kv_heads} instead."
            )
        # Check for divisibility
        if d_model % num_heads != 0:
            raise ValueError(
                f"Expected d_model to be divisible by num_heads but got {d_model} and {num_heads} instead"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"Expected num_heads to be divisible by num_kv_heads but got {num_heads} and {num_kv_heads} instead"
            )
        # Check dropout value
        if not (0 <= dropout_p <= 1):
            raise ValueError(
                f"Expected dropout in [0, 1] interval, got {dropout_p} instead."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_head = d_model // num_heads
        self.dropout_p = dropout_p

        # Check rope dimension value
        if rope is not None and rope.d_head != self.d_head:
            raise ValueError(
                f"Expected rope dimension and head dimension to match, got {rope.d_head} and {self.d_head} instead."
            )

        self.qkv = nn.Linear(
            d_model, num_heads * self.d_head + 2 * num_kv_heads * self.d_head
        )
        self.out = nn.Linear(d_model, d_model)

        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        causal: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, query_len, d_model = x.shape

        # [B, query_len, d_model + 2 * num_kv_heads * d_head]
        qkv = self.qkv(x)
        q, k, v = qkv.split(
            [
                self.d_model,
                self.num_kv_heads * self.d_head,
                self.num_kv_heads * self.d_head,
            ],
            dim=-1,
        )  # Q: [B, query_len, d_model], K/V: [B, query_len, num_kv_heads * d_head]

        # Split projected features into heads and move heads before query_len.
        q = q.view(B, query_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, query_len, self.num_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, query_len, self.num_kv_heads, self.d_head).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(
                q, k, offset=kv_cache[0].shape[-2] if kv_cache is not None else 0
            )

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        dropout_p = self.dropout_p if self.training else 0.0

        # Expose query groups and broadcast each compact K/V head across its
        # corresponding query heads without materializing repeated K/V tensors.
        # q: [B, num_heads, query_len, d_head]
        # k, v: [B, num_kv_heads, key_len, d_head]
        grouped_q = q.view(
            B, self.num_kv_heads, self.group_size, query_len, self.d_head
        )  # [B, num_kv_heads, group_size, query_len, d_head],
        grouped_k = k.unsqueeze(2)  # [B, num_kv_heads, 1, key_len, d_head]
        grouped_v = v.unsqueeze(2)  # [B, num_kv_heads, 1, key_len, d_head]

        attn_out = attention(
            grouped_q, grouped_k, grouped_v, dropout_p, causal=causal
        )  # [B, num_kv_heads, group_size, query_len, d_head]
        attn_out = attn_out.view(
            B, self.num_heads, query_len, self.d_head
        )  # [B, num_heads, query_len, d_head]

        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(B, query_len, d_model)
        )  # [B, query_len, d_model]

        if use_cache:
            return self.out(attn_out), (k, v)
        return self.out(attn_out)
