import torch

from positional_encoding import RotaryPositionalEncoding
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


def test_cache_shapes_after_prefill():
    torch.manual_seed(0)
    x = torch.randn((2, 5, 32))
    rope = RotaryPositionalEncoding(d_head=8, max_seq_len=16)
    block = TransformerBlock(
        d_model=32, num_heads=4, hidden_dim=48, rope=rope
    ).eval()

    output, (k_cache, v_cache) = block(x, use_cache=True)

    assert output.shape == x.shape
    assert k_cache.shape == (2, 4, 5, 8)
    assert v_cache.shape == (2, 4, 5, 8)


def test_cached_block_matches_full_block_suffix():
    torch.manual_seed(0)
    x = torch.randn((2, 7, 32))
    prompt_len = 4
    rope = RotaryPositionalEncoding(d_head=8, max_seq_len=16)
    block = TransformerBlock(
        d_model=32, num_heads=4, hidden_dim=48, rope=rope
    ).eval()

    full_output = block(x)

    _, cache = block(x[:, :prompt_len], use_cache=True)
    cached_suffix_output, updated_cache = block(
        x[:, prompt_len:], kv_cache=cache, use_cache=True
    )

    torch.testing.assert_close(
        cached_suffix_output,
        full_output[:, prompt_len:],
        atol=1e-6,
        rtol=1e-5,
    )
    assert updated_cache[0].shape[-2] == x.shape[1]
    assert updated_cache[1].shape[-2] == x.shape[1]
