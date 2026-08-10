import pytest
import torch

from mlp import MLP


def test_output_shape():
    x = torch.randn((4, 10, 32))
    ffn = MLP(32, 64)
    assert ffn(x).shape == (4, 10, 32)


def test_dimensions():
    with pytest.raises(ValueError, match="d_model"):
        MLP(-3, 32)
    with pytest.raises(ValueError, match="hidden_dim"):
        MLP(32, -3)


def test_train():
    x = torch.randn((4, 10, 32))
    ffn = MLP(32, 64, dropout_p=0.5)
    ffn.train()

    assert not (torch.allclose(ffn(x), ffn(x)))


def test_val():
    x = torch.randn((4, 10, 32))
    ffn = MLP(32, 64, dropout_p=0.5)
    ffn.eval()

    assert torch.allclose(ffn(x), ffn(x))
