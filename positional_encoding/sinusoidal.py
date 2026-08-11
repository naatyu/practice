import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0 or d_model <= 0:
            raise ValueError(
                f"Expected d_model to be positive and even, got: {d_model}"
            )

        self.max_seq_len = max_seq_len
        frequencies = torch.exp(
            (torch.arange(0, d_model, 2) / d_model) * -math.log(base)
        )
        angles = torch.arange(0, max_seq_len).unsqueeze(1) * frequencies
        table = torch.empty((max_seq_len, d_model))
        table[:, 0::2] = torch.sin(angles)
        table[:, 1::2] = torch.cos(angles)
        self.register_buffer("table", table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Expected sequence size to be at most {self.max_seq_len}, got: {seq_len}"
            )

        return x + self.table[:seq_len].to(x.dtype)  # ty: ignore[not-subscriptable]
