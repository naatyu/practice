import torch

from mlp import SwiGLU


def test_output_shape():
    x = torch.randn((4, 10, 32))
    ffn = SwiGLU(32, 64)
    assert ffn(x).shape == (4, 10, 32)


def test_train():
    x = torch.randn((4, 10, 32))
    ffn = SwiGLU(32, 64, dropout_p=0.5).train()

    assert not (torch.allclose(ffn(x), ffn(x)))


def test_eval():
    x = torch.randn((4, 10, 32))
    ffn = SwiGLU(32, 64).eval()

    assert torch.allclose(ffn(x), ffn(x))


def test_behavior():
    x = torch.Tensor([[1]])
    ffn = SwiGLU(1, 1, dropout_p=0.0)
    with torch.no_grad():
        ffn.up_proj.weight.copy_(torch.Tensor([[2], [1]]))
        ffn.up_proj.bias.zero_()
        ffn.down_proj.weight.copy_(torch.Tensor([[3]]))
        ffn.down_proj.bias.zero_()
    expected = 3 * 2 * (1 / (1 + torch.exp(-torch.Tensor([1]))))

    torch.testing.assert_close(ffn(x).squeeze(0), expected)
