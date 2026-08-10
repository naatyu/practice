import torch
from torch import nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_table = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check for un-allowed seq len
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Expected sequence size to be at most {self.max_seq_len}, got: {seq_len}"
            )

        positions = torch.arange(seq_len, device=x.device)

        return x + self.pos_table(positions)
