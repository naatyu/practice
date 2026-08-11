import math

import torch
from torch import nn


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if d_head % 2 != 0 or d_head <= 0:
            raise ValueError(f"Expected d_head to be positive and even, got: {d_head}")

        self.max_seq_len = max_seq_len
        # Compute frequencies = 1 / base**(2i / d_head) = exp(2i / d_head * -log(base))
        frequencies = torch.exp(
            torch.arange(0, d_head, 2) / d_head * -math.log(base)
        )  # [d_head/2]
        angles = (
            torch.arange(0, max_seq_len).unsqueeze(1) * frequencies
        )  # [max_seq_len, d_head/2]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        self.register_buffer("cos_angles", cos_angles, persistent=False)
        self.register_buffer("sin_angles", sin_angles, persistent=False)

    def _apply_rope(self, cos, sin, x):
        batch, heads, seq_len, head_dim = x.shape
        pairs = x.reshape(batch, heads, seq_len, head_dim // 2, 2)

        x0 = pairs[..., 0]
        x1 = pairs[..., 1]

        y0 = x0 * cos - sin * x1
        y1 = x0 * sin + cos * x1

        return torch.stack((y0, y1), dim=-1).reshape(batch, heads, seq_len, head_dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Expected sequence size to be at most {self.max_seq_len}, got: {seq_len}"
            )

        if offset < 0 or seq_len + offset > self.max_seq_len:
            raise ValueError(
                f"Expected offset to be non-negative and at most {self.max_seq_len - seq_len}, got: {offset}"
            )

        cos = self.cos_angles[offset : offset + seq_len].to(q.dtype)  # ty: ignore[not-subscriptable]
        sin = self.sin_angles[offset : offset + seq_len].to(q.dtype)  # ty: ignore[not-subscriptable]

        q = self._apply_rope(cos, sin, q)
        k = self._apply_rope(cos, sin, k)

        return q, k
