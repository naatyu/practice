import pytest
import torch

from multi_head_attention import MultiHeadAttention


def test_construction() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        MultiHeadAttention(d_model=768, num_heads=-1)

    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=768, num_heads=11)

    with pytest.raises(ValueError, match="dropout"):
        MultiHeadAttention(d_model=768, num_heads=12, dropout_p=-1)


def test_output_shape() -> None:
    torch.manual_seed(0)
    x = torch.randn((4, 10, 64))
    mha = MultiHeadAttention(d_model=64, num_heads=32, dropout_p=0.0)

    res = mha(x)

    assert res.shape == (4, 10, 64)


def test_dropout() -> None:
    torch.manual_seed(0)
    x = torch.randn((4, 10, 64))
    mha = MultiHeadAttention(d_model=64, num_heads=32, dropout_p=0.9)

    torch.manual_seed(1)
    train_output_1 = mha(x)
    torch.manual_seed(2)
    train_output_2 = mha(x)
    assert not torch.allclose(train_output_1, train_output_2)

    mha.eval()
    torch.manual_seed(1)
    eval_output_1 = mha(x)
    torch.manual_seed(2)
    eval_output_2 = mha(x)
    torch.testing.assert_close(eval_output_1, eval_output_2)

    no_dropout_mha = MultiHeadAttention(d_model=64, num_heads=32, dropout_p=0.0)
    torch.testing.assert_close(no_dropout_mha(x), no_dropout_mha(x))


def test_against_torch() -> None:
    torch.manual_seed(0)
    x = torch.randn((4, 10, 64))
    mha = MultiHeadAttention(d_model=64, num_heads=32, dropout_p=0.0)
    torch_mha = torch.nn.MultiheadAttention(
        embed_dim=64, num_heads=32, dropout=0.0, batch_first=True
    )

    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(mha.qkv.weight)
        torch_mha.in_proj_bias.copy_(mha.qkv.bias)
        torch_mha.out_proj.weight.copy_(mha.out.weight)
        torch_mha.out_proj.bias.copy_(mha.out.bias)

    mha.eval()
    torch_mha.eval()
    torch_output, _ = torch_mha(x, x, x, need_weights=False)

    torch.testing.assert_close(mha(x), torch_output)
