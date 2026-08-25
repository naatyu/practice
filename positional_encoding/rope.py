import torch
from torch import nn


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if d_head % 2 != 0 or d_head <= 0:
            raise ValueError(f"Expected d_head to be positive and even, got: {d_head}")

        self.max_seq_len = max_seq_len
        self.d_head = d_head
        # Compute frequencies = 1 / base**(2i / d_head).
        even_indices = torch.arange(0, d_head, 2, dtype=torch.float32)
        frequencies = base ** (-even_indices / d_head)  # [d_head/2]
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


class RotaryPositionalEncodingComplex(nn.Module):
    """
    Complex-valued RoPE implementation based on Umar Jamil's video.

    Using complex numbers simplifies the codebase by expressing each pairwise
    rotation as a multiplication by cis. It also packs sine and cosine into a
    single tensor instead of two separate buffers. A complex value still contains
    both components, so any performance difference depends on the available
    complex kernels and should be measured.
    """

    def __init__(self, d_head: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if d_head % 2 != 0 or d_head <= 0:
            raise ValueError(f"Expected d_head to be positive and even, got: {d_head}")

        self.max_seq_len = max_seq_len
        self.d_head = d_head
        # Compute frequencies = 1 / base**(2i / d_head).
        even_indices = torch.arange(0, d_head, 2, dtype=torch.float32)
        frequencies = base ** (-even_indices / d_head)  # [d_head/2]
        frequencies = (
            torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1) * frequencies
        )  # [max_seq_len, d_head/2]
        # Store the real view so module.to(dtype=...) cannot discard the
        # imaginary component of a complex buffer.
        freqs_cis = torch.view_as_real(
            torch.polar(torch.ones_like(frequencies), frequencies)
        )  # [max_seq_len, d_head/2, 2]

        self.freqs_cis: torch.Tensor
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @staticmethod
    def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype

        batch, heads, seq_len, head_dim = x.shape
        pairs = x.reshape(
            batch, heads, seq_len, head_dim // 2, 2
        )  # [B, H, S, D] -> [B, H, S, D/2, 2]

        # view_as_complex does not support BF16. Using FP32 for both half types
        # also avoids relying on limited complex-half operator support.
        if pairs.dtype in (torch.float16, torch.bfloat16):
            pairs = pairs.float()

        x_complex = torch.view_as_complex(pairs)  # [B, H, S, D/2]
        # The buffer may have followed the module to BF16/FP16, while
        # view_as_complex requires a supported real dtype.
        complex_frequencies = torch.view_as_complex(freqs_cis.float())
        rotated = x_complex * complex_frequencies[None, None, :, :]  # [B, H, S, D/2]
        rotated = torch.view_as_real(rotated)  # [B, H, S, D/2, 2]
        rotated = rotated.reshape(batch, heads, seq_len, head_dim)  # [B, H, S, D]

        return rotated.to(dtype)

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

        freqs_cis = self.freqs_cis[offset : offset + seq_len]
        q = self._apply_rope(q, freqs_cis)
        k = self._apply_rope(k, freqs_cis)

        return q, k
