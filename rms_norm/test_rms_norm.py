import torch

from rms_norm import rms_norm


def test_versus_pytorch() -> None:
    torch.manual_seed(0)
    x = torch.randn((4, 2, 7))
    weight = torch.randn(7)
    eps = 1e-6

    torch.testing.assert_close(
        rms_norm(x, weight, eps),
        torch.nn.functional.rms_norm(
            x, normalized_shape=(x.shape[-1],), weight=weight, eps=eps
        ),
    )
