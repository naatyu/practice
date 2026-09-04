import torch

from normalization import RMSNorm


def test_versus_pytorch() -> None:
    torch.manual_seed(0)
    x = torch.randn((4, 2, 7))
    weight = torch.randn(7)
    eps = 1e-6
    rms_norm = RMSNorm(d_model=7)
    with torch.no_grad():
        rms_norm.weight.copy_(weight)

    torch.testing.assert_close(
        rms_norm(x),
        torch.nn.functional.rms_norm(
            x, normalized_shape=(x.shape[-1],), weight=weight, eps=eps
        ),
    )
