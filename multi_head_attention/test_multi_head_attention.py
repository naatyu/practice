import pytest
import torch

from multi_head_attention import MultiHeadAttention
from positional_encoding import RotaryPositionalEncoding


def test_construction() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        MultiHeadAttention(d_model=768, num_heads=-1)

    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=768, num_heads=11)

    with pytest.raises(ValueError, match="dropout"):
        MultiHeadAttention(d_model=768, num_heads=12, dropout_p=-1)

    with pytest.raises(ValueError, match="num_kv_heads"):
        MultiHeadAttention(d_model=64, num_heads=8, num_kv_heads=0)

    with pytest.raises(ValueError, match="divisible by num_kv_heads"):
        MultiHeadAttention(d_model=64, num_heads=8, num_kv_heads=3)


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


def test_wrong_rope_construction():
    d_model = 64
    num_heads = 32
    seq_len = 10
    dropout_p = 0.0

    rope = RotaryPositionalEncoding(
        d_head=((d_model // num_heads) + 2) & ~(1), max_seq_len=seq_len
    )

    with pytest.raises(ValueError, match="rope dimension"):
        MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout_p=dropout_p, rope=rope
        )


def test_cache_shapes_after_prefill() -> None:
    torch.manual_seed(0)
    x = torch.randn((2, 5, 32))
    rope = RotaryPositionalEncoding(d_head=8, max_seq_len=16)
    mha = MultiHeadAttention(d_model=32, num_heads=4, rope=rope).eval()

    output, (k_cache, v_cache) = mha(x, causal=True, use_cache=True)

    assert output.shape == x.shape
    assert k_cache.shape == (2, 4, 5, 8)
    assert v_cache.shape == (2, 4, 5, 8)


def test_cache_is_appended_along_sequence_dimension() -> None:
    torch.manual_seed(0)
    prompt = torch.randn((2, 5, 32))
    next_token = torch.randn((2, 1, 32))
    rope = RotaryPositionalEncoding(d_head=8, max_seq_len=16)
    mha = MultiHeadAttention(d_model=32, num_heads=4, rope=rope).eval()

    _, cache = mha(prompt, causal=True, use_cache=True)
    output, (k_cache, v_cache) = mha(
        next_token, kv_cache=cache, causal=True, use_cache=True
    )

    assert output.shape == (2, 1, 32)
    assert k_cache.shape == (2, 4, 6, 8)
    assert v_cache.shape == (2, 4, 6, 8)


def test_cached_attention_matches_full_attention_suffix() -> None:
    torch.manual_seed(0)
    x = torch.randn((2, 7, 32))
    prompt_len = 4
    rope = RotaryPositionalEncoding(d_head=8, max_seq_len=16)
    mha = MultiHeadAttention(d_model=32, num_heads=4, rope=rope).eval()

    full_output = mha(x, causal=True)

    _, cache = mha(x[:, :prompt_len], causal=True, use_cache=True)
    cached_suffix_output, updated_cache = mha(
        x[:, prompt_len:], kv_cache=cache, causal=True, use_cache=True
    )

    torch.testing.assert_close(
        cached_suffix_output,
        full_output[:, prompt_len:],
        atol=1e-6,
        rtol=1e-5,
    )
    assert updated_cache[0].shape[-2] == x.shape[1]
    assert updated_cache[1].shape[-2] == x.shape[1]


@pytest.mark.parametrize("num_kv_heads", [2, 1])
def test_gqa_and_mqa_output_and_projection_shapes(num_kv_heads: int) -> None:
    torch.manual_seed(0)
    d_model = 32
    num_heads = 4
    d_head = d_model // num_heads
    x = torch.randn((2, 5, d_model))
    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    ).eval()

    output = mha(x)

    assert output.shape == x.shape
    assert mha.qkv.out_features == (
        num_heads + 2 * num_kv_heads
    ) * d_head


@pytest.mark.parametrize("num_kv_heads", [2, 1])
def test_gqa_and_mqa_cache_stays_compact_and_matches_full_forward(
    num_kv_heads: int,
) -> None:
    torch.manual_seed(0)
    x = torch.randn((2, 7, 32))
    prompt_len = 4
    rope = RotaryPositionalEncoding(d_head=8, max_seq_len=16)
    mha = MultiHeadAttention(
        d_model=32,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        rope=rope,
    ).eval()

    full_output = mha(x, causal=True)

    _, cache = mha(x[:, :prompt_len], causal=True, use_cache=True)
    cached_suffix_output, updated_cache = mha(
        x[:, prompt_len:], kv_cache=cache, causal=True, use_cache=True
    )

    torch.testing.assert_close(
        cached_suffix_output,
        full_output[:, prompt_len:],
        atol=1e-6,
        rtol=1e-5,
    )
    expected_cache_shape = (2, num_kv_heads, x.shape[1], 8)
    assert updated_cache[0].shape == expected_cache_shape
    assert updated_cache[1].shape == expected_cache_shape
