import pytest
import torch

from mlp import SwiGLU
from moe import GroupedMoE
from positional_encoding import RotaryPositionalEncoding
from transformer_block import TransformerBlock


def test_output_shape():
    x = torch.randn((4, 10, 32))
    block = TransformerBlock(d_model=32, num_heads=4, hidden_dim=32)

    assert block(x)[0].shape == x.shape


def test_eval():
    x = torch.randn((4, 10, 32))
    block = TransformerBlock(d_model=32, num_heads=4, hidden_dim=32, dropout_p=0.5)
    block.eval()

    torch.testing.assert_close(block(x), block(x))


def test_causality():
    x = torch.randn((4, 10, 32))
    block = TransformerBlock(d_model=32, num_heads=4, hidden_dim=32)
    x1, _, _ = block(x)

    x[:, 6:, :] = 0
    x2, _, _ = block(x)

    assert torch.allclose(x1[:, :6, :], x2[:, :6, :])
    assert not torch.allclose(x1[:, 6:, :], x2[:, 6:, :])


def test_cache_shapes_after_prefill():
    torch.manual_seed(0)
    x = torch.randn((2, 5, 32))
    rope = RotaryPositionalEncoding(d_head=8, max_seq_len=16)
    block = TransformerBlock(d_model=32, num_heads=4, hidden_dim=48, rope=rope).eval()

    output, cache, aux_loss = block(x, use_cache=True)
    assert cache is not None and aux_loss is None
    k_cache, v_cache = cache

    assert output.shape == x.shape
    assert k_cache.shape == (2, 4, 5, 8)
    assert v_cache.shape == (2, 4, 5, 8)


def test_cached_block_matches_full_block_suffix():
    torch.manual_seed(0)
    x = torch.randn((2, 7, 32))
    prompt_len = 4
    rope = RotaryPositionalEncoding(d_head=8, max_seq_len=16)
    block = TransformerBlock(d_model=32, num_heads=4, hidden_dim=48, rope=rope).eval()

    full_output, _, _ = block(x)

    _, cache, _ = block(x[:, :prompt_len], use_cache=True)
    cached_suffix_output, updated_cache, _ = block(
        x[:, prompt_len:], kv_cache=cache, use_cache=True
    )

    torch.testing.assert_close(
        cached_suffix_output,
        full_output[:, prompt_len:],
        atol=1e-6,
        rtol=1e-5,
    )
    assert updated_cache is not None
    assert updated_cache[0].shape[-2] == x.shape[1]
    assert updated_cache[1].shape[-2] == x.shape[1]


def test_dense_ffn_remains_the_default():
    block = TransformerBlock(d_model=16, num_heads=2, hidden_dim=32)
    assert isinstance(block.ffn, SwiGLU)


def test_moe_block_selects_requested_expert_configuration():
    block = TransformerBlock(
        d_model=16, num_heads=2, hidden_dim=32, num_experts=4, top_k=2
    )
    assert isinstance(block.ffn, GroupedMoE)
    assert block.ffn.num_experts == 4
    assert block.ffn.top_k == 2
    x = torch.randn(2, 5, 16)
    output, cache, aux_loss = block(x)
    assert output.shape == x.shape
    assert cache is None and aux_loss is not None and aux_loss.requires_grad


def test_moe_block_requires_top_k():
    with pytest.raises(ValueError, match="Top K"):
        TransformerBlock(d_model=16, num_heads=2, hidden_dim=32, num_experts=4)


def test_moe_block_preserves_causality():
    torch.manual_seed(36)
    block = TransformerBlock(
        d_model=16, num_heads=2, hidden_dim=32, num_experts=4, top_k=2
    ).eval()
    x = torch.randn(2, 6, 16)
    changed_future = x.clone()
    changed_future[:, 3:] += 10
    with torch.inference_mode():
        expected, _, _ = block(x)
        actual, _, _ = block(changed_future)
    torch.testing.assert_close(actual[:, :3], expected[:, :3])


def test_legacy_positional_eps_is_preserved():
    block = TransformerBlock(16, 2, 32, None, 0.0, None, None, 1e-5)
    assert isinstance(block.ffn, SwiGLU)
    assert block.norm1.eps == 1e-5
    assert block.norm2.eps == 1e-5
