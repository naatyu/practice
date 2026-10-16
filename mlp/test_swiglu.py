import pytest
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
        ffn.down_proj.weight.copy_(torch.Tensor([[3]]))
    expected = 3 * 2 * (1 / (1 + torch.exp(-torch.Tensor([1]))))

    torch.testing.assert_close(ffn(x).squeeze(0), expected)


@pytest.mark.parametrize("bias", [False, True])
def test_projection_bias_configuration(bias):
    d_model, hidden_dim = 3, 5
    ffn = SwiGLU(d_model, hidden_dim, bias=bias)
    if bias:
        assert ffn.up_proj.bias is not None and ffn.down_proj.bias is not None
        assert ffn.up_proj.bias.shape == (2 * hidden_dim,)
        assert ffn.down_proj.bias.shape == (d_model,)
    else:
        assert ffn.up_proj.bias is None and ffn.down_proj.bias is None
        assert set(ffn.state_dict()) == {"up_proj.weight", "down_proj.weight"}

    expected_count = 3 * d_model * hidden_dim
    if bias:
        expected_count += 2 * hidden_dim + d_model
    assert sum(parameter.numel() for parameter in ffn.parameters()) == expected_count


def test_biases_are_disabled_by_default():
    ffn = SwiGLU(3, 5)
    assert ffn.up_proj.bias is None and ffn.down_proj.bias is None


def test_optional_nonzero_biases_affect_output():
    ffn = SwiGLU(1, 1, bias=True)
    assert ffn.up_proj.bias is not None and ffn.down_proj.bias is not None
    with torch.no_grad():
        ffn.up_proj.weight.copy_(torch.tensor([[2.0], [1.0]]))
        ffn.up_proj.bias.copy_(torch.tensor([1.0, -1.0]))
        ffn.down_proj.weight.fill_(3.0)
        ffn.down_proj.bias.fill_(5.0)
    # x=2 -> value=5, gate=1 -> 3 * 5 * silu(1) + 5.
    expected = 15 * torch.nn.functional.silu(torch.tensor(1.0)) + 5
    torch.testing.assert_close(ffn(torch.tensor([2.0])).squeeze(), expected)
