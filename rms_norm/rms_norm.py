import torch


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Implement RMSNorm over the final dimension of x.

    Do not use torch.nn.RMSNorm or torch.nn.functional.rms_norm.

    Shapes:
        x: (..., hidden_dim)
        weight: (hidden_dim,)
    """
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    x = x / rms * weight

    return x
