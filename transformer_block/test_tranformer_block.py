import torch

from transformer_block import TransformerBlock


def test_output_shape():
    x = torch.randn((4, 10, 32))
    block = TransformerBlock(d_model=32, num_heads=4, hidden_dim=32)

    assert block(x).shape == x.shape


def test_eval():
    x = torch.randn((4, 10, 32))
    block = TransformerBlock(d_model=32, num_heads=4, hidden_dim=32, dropout_p=0.5)
    block.eval()

    torch.testing.assert_close(block(x), block(x))


def test_causality():
    x = torch.randn((4, 10, 32))
    block = TransformerBlock(d_model=32, num_heads=4, hidden_dim=32)
    x1 = block(x)

    x[:, 6:, :] = 0
    x2 = block(x)

    assert torch.allclose(x1[:, :6, :], x2[:, :6, :])
    assert not torch.allclose(x1[:, 6:, :], x2[:, 6:, :])
