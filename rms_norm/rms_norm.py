import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"Expected d_model to be positive, got: {d_model}")
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Implement RMSNorm over the final dimension of x.

        Do not use torch.nn.RMSNorm or torch.nn.functional.rms_norm.

        Shapes:
            x: (..., hidden_dim)
        """
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = (x / rms) * self.weight

        return x
