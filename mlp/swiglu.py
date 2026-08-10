import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, 2 * hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        value, gate = torch.chunk(x, 2, dim=-1)
        gate = torch.nn.functional.silu(gate)

        return self.dropout(self.down_proj(value * gate))
