import math

import pytest
import torch

from decoder_model import DecoderCache, DecoderModel
from moe import GroupedMoE
from transformer_block import TransformerBlock


@pytest.fixture
def model_args():
    d_model = 64
    # Approximately parameter-match a standard GELU MLP with hidden_dim=4*d_model
    # aligned to next multiple of 8 (recommended for BF16)
    align_to = 8
    hidden_dim = math.ceil(((8 * d_model) / 3) / align_to) * align_to

    return {
        "vocab_size": 256,
        "d_model": d_model,
        "num_heads": 4,
        "max_seq_len": 20,
        "n_layers": 4,
        "hidden_dim": hidden_dim,
        "dropout_p": 0.0,
        "rope_base": 200,
    }


def test_output_shape(model_args):
    batch_size = 4
    input_ids = torch.randint(
        0, model_args["vocab_size"], (batch_size, model_args["max_seq_len"])
    )
    model = DecoderModel(
        vocab_size=model_args["vocab_size"],
        d_model=model_args["d_model"],
        num_heads=model_args["num_heads"],
        max_seq_len=model_args["max_seq_len"],
        n_layers=model_args["n_layers"],
        hidden_dim=model_args["hidden_dim"],
        dropout_p=model_args["dropout_p"],
        rope_base=model_args["rope_base"],
    )

    assert model(input_ids).logits.shape == (
        batch_size,
        model_args["max_seq_len"],
        model_args["vocab_size"],
    )


def test_weight_tying(model_args):

    model = DecoderModel(
        vocab_size=model_args["vocab_size"],
        d_model=model_args["d_model"],
        num_heads=model_args["num_heads"],
        max_seq_len=model_args["max_seq_len"],
        n_layers=model_args["n_layers"],
        hidden_dim=model_args["hidden_dim"],
        dropout_p=model_args["dropout_p"],
        rope_base=model_args["rope_base"],
        tie_weights=True,
    )
    assert model.tok_emb.weight is model.lm_head.weight

    model = DecoderModel(
        vocab_size=model_args["vocab_size"],
        d_model=model_args["d_model"],
        num_heads=model_args["num_heads"],
        max_seq_len=model_args["max_seq_len"],
        n_layers=model_args["n_layers"],
        hidden_dim=model_args["hidden_dim"],
        dropout_p=model_args["dropout_p"],
        rope_base=model_args["rope_base"],
        tie_weights=False,
    )
    assert model.tok_emb.weight is not model.lm_head.weight


def test_num_blocks(model_args):
    model = DecoderModel(
        vocab_size=model_args["vocab_size"],
        d_model=model_args["d_model"],
        num_heads=model_args["num_heads"],
        max_seq_len=model_args["max_seq_len"],
        n_layers=model_args["n_layers"],
        hidden_dim=model_args["hidden_dim"],
        dropout_p=model_args["dropout_p"],
        rope_base=model_args["rope_base"],
        tie_weights=True,
    )
    assert len(model.blocks) == model_args["n_layers"]


@pytest.mark.parametrize("num_kv_heads", [None, 2, 1])
def test_cache_shapes_after_prefill(model_args, num_kv_heads):
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 5
    input_ids = torch.randint(0, model_args["vocab_size"], (batch_size, seq_len))
    model = DecoderModel(**model_args, num_kv_heads=num_kv_heads).eval()

    output = model(input_ids, use_cache=True)
    logits, kv_caches = output.logits, output.kv_caches
    assert kv_caches is not None and output.aux_loss is None

    assert logits.shape == (batch_size, seq_len, model_args["vocab_size"])
    assert kv_caches.position == seq_len
    assert len(kv_caches) == model_args["n_layers"]
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        assert block.attn.num_kv_heads == model.num_kv_heads
    for k_cache, v_cache in kv_caches:
        expected_shape = (
            batch_size,
            model.num_kv_heads,
            seq_len,
            model_args["d_model"] // model_args["num_heads"],
        )
        assert k_cache.shape == expected_shape
        assert v_cache.shape == expected_shape


def test_rejects_wrong_number_of_layer_caches(model_args):
    model = DecoderModel(**model_args).eval()
    input_ids = torch.randint(0, model_args["vocab_size"], (2, 1))

    with pytest.raises(ValueError, match="number of layers"):
        model(
            input_ids,
            kv_caches=DecoderCache(layers=[], position=0),
            use_cache=True,
        )


@pytest.mark.parametrize("num_kv_heads", [None, 2, 1])
def test_token_by_token_cached_logits_match_full_forward(model_args, num_kv_heads):
    torch.manual_seed(0)
    input_ids = torch.randint(0, model_args["vocab_size"], (2, 8))
    prompt_len = 4
    model = DecoderModel(**model_args, num_kv_heads=num_kv_heads).eval()

    full_logits = model(input_ids).logits

    output = model(input_ids[:, :prompt_len], use_cache=True)
    prompt_logits, kv_caches = output.logits, output.kv_caches
    assert kv_caches is not None
    cached_logits = [prompt_logits]

    for position in range(prompt_len, input_ids.shape[1]):
        output = model(
            input_ids[:, position : position + 1],
            kv_caches=kv_caches,
            use_cache=True,
        )
        kv_caches = output.kv_caches
        assert kv_caches is not None
        cached_logits.append(output.logits)

    cached_logits = torch.cat(cached_logits, dim=1)

    torch.testing.assert_close(cached_logits, full_logits, atol=1e-5, rtol=1e-5)
    for k_cache, v_cache in kv_caches:
        assert k_cache.shape[1] == model.num_kv_heads
        assert v_cache.shape[1] == model.num_kv_heads
        assert k_cache.shape[-2] == input_ids.shape[1]
        assert v_cache.shape[-2] == input_ids.shape[1]


@pytest.mark.parametrize("num_kv_heads", [None, 2, 1])
def test_local_attention_uses_bounded_cache_and_absolute_positions(
    model_args, num_kv_heads
):
    torch.manual_seed(0)
    input_ids = torch.randint(0, model_args["vocab_size"], (2, 10))
    prompt_len = 5
    window_size = 3
    model = DecoderModel(
        **model_args,
        num_kv_heads=num_kv_heads,
        window_size=window_size,
    ).eval()

    full_logits = model(input_ids).logits
    output = model(input_ids[:, :prompt_len], use_cache=True)
    prompt_logits, cache = output.logits, output.kv_caches
    assert cache is not None
    cached_logits = [prompt_logits]

    assert cache.position == prompt_len
    for key_cache, value_cache in cache:
        assert key_cache.shape[-2] == window_size
        assert value_cache.shape[-2] == window_size

    for position in range(prompt_len, input_ids.shape[1]):
        output = model(
            input_ids[:, position : position + 1],
            kv_caches=cache,
            use_cache=True,
        )
        cache = output.kv_caches
        assert cache is not None
        cached_logits.append(output.logits)

    cached_logits = torch.cat(cached_logits, dim=1)

    torch.testing.assert_close(cached_logits, full_logits, atol=1e-5, rtol=1e-5)
    assert cache.position == input_ids.shape[1]
    for key_cache, value_cache in cache:
        assert key_cache.shape[-2] == window_size
        assert value_cache.shape[-2] == window_size


def test_moe_decoder_registers_experts_in_every_layer_and_backpropagates():
    torch.manual_seed(37)
    model = DecoderModel(
        vocab_size=32,
        d_model=16,
        num_heads=2,
        max_seq_len=12,
        n_layers=2,
        hidden_dim=32,
        num_experts=4,
        top_k=2,
    )
    input_ids = torch.randint(0, 32, (2, 6))
    logits = model(input_ids).logits
    assert logits.shape == (2, 6, 32)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 32), input_ids[:, 1:].reshape(-1)
    )
    loss.backward()

    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        assert isinstance(block.ffn, GroupedMoE)
        assert block.ffn.num_experts == 4
        assert block.ffn.top_k == 2
        router_grad = block.ffn.router.router.weight.grad
        assert router_grad is not None
        assert torch.isfinite(router_grad).all()
        assert router_grad.abs().sum() > 0
        for weight in (block.ffn.experts.up_weight, block.ffn.experts.down_weight):
            assert weight.grad is not None
            assert torch.isfinite(weight.grad).all()
            assert weight.grad.abs().sum() > 0
    assert model.blocks[0].ffn is not model.blocks[1].ffn


@pytest.mark.parametrize("window_size", [None, 3])
def test_moe_decoder_cached_logits_match_full_forward(window_size):
    torch.manual_seed(38)
    model = DecoderModel(
        vocab_size=32,
        d_model=16,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=12,
        n_layers=2,
        hidden_dim=32,
        num_experts=4,
        top_k=2,
        window_size=window_size,
    ).eval()
    input_ids = torch.randint(0, 32, (2, 7))
    with torch.inference_mode():
        expected = model(input_ids).logits
        output = model(input_ids[:, :3], use_cache=True)
        prompt_logits, cache = output.logits, output.kv_caches
        assert cache is not None
        chunks = [prompt_logits]
        for position in range(3, input_ids.shape[1]):
            output = model(
                input_ids[:, position : position + 1], kv_caches=cache, use_cache=True
            )
            cache = output.kv_caches
            assert cache is not None
            chunks.append(output.logits)
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, atol=1e-5, rtol=1e-5)
    assert cache.position == input_ids.shape[1]


def test_decoder_averages_live_losses_per_layer_and_auxiliary_only_gradients():
    torch.manual_seed(39)
    model = DecoderModel(
        vocab_size=16,
        d_model=16,
        num_heads=2,
        max_seq_len=8,
        n_layers=3,
        hidden_dim=32,
        num_experts=4,
        top_k=2,
    )
    layer_losses = []

    def capture(_module, _args, output):
        layer_losses.append(output[2])

    handles = [block.register_forward_hook(capture) for block in model.blocks]
    output = model(torch.randint(0, 16, (2, 6)))
    for handle in handles:
        handle.remove()
    assert output.kv_caches is None
    assert output.aux_loss is not None and output.aux_loss.requires_grad
    assert len(layer_losses) == 3
    torch.testing.assert_close(output.aux_loss, torch.stack(layer_losses).mean())
    output.aux_loss.backward()
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        assert isinstance(block.ffn, GroupedMoE)
        gradient = block.ffn.router.router.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0
    # A layer's loss bypasses its own experts, but later router losses can
    # backpropagate through earlier experts that produced their hidden inputs.
    last_block = model.blocks[-1]
    assert isinstance(last_block, TransformerBlock)
    assert isinstance(last_block.ffn, GroupedMoE)
    assert last_block.ffn.experts.up_weight.grad is None
    assert last_block.ffn.experts.down_weight.grad is None


@pytest.mark.parametrize("num_experts", [None, 4])
def test_decoder_auxiliary_loss_is_absent_for_dense_or_eval(num_experts):
    model = DecoderModel(
        vocab_size=16,
        d_model=16,
        num_heads=2,
        max_seq_len=8,
        n_layers=2,
        hidden_dim=32,
        num_experts=num_experts,
        top_k=2 if num_experts else None,
    )
    tokens = torch.randint(0, 16, (1, 4))
    if num_experts is None:
        assert model(tokens).aux_loss is None
    model.eval()
    output = model(tokens, use_cache=True)
    assert output.aux_loss is None
    assert output.kv_caches is not None and output.kv_caches.position == 4
