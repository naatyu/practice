import torch
from torch import nn


class MLP(nn.Module):
    """Implement a position-wise Transformer GELU MLP.

    Input and output shape:
        (batch_size, sequence_length, d_model)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"Expected d_model to be positive, got: {d_model}")
        if hidden_dim <= 0:
            raise ValueError(f"Expected hidden_dim to be positive, got: {hidden_dim}")
        self.up_proj = nn.Linear(d_model, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.gelu(self.up_proj(x))
        x = self.dropout(x)
        return self.dropout(self.down_proj(x))
