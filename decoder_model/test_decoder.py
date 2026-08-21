import math

import pytest
import torch

from decoder_model import DecoderModel


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

    assert model(input_ids).shape == (
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
    input_ids = torch.randint(
        0, model_args["vocab_size"], (batch_size, seq_len)
    )
    model = DecoderModel(**model_args, num_kv_heads=num_kv_heads).eval()

    logits, kv_caches = model(input_ids, use_cache=True)

    assert logits.shape == (batch_size, seq_len, model_args["vocab_size"])
    assert len(kv_caches) == model_args["n_layers"]
    assert all(
        block.attn.num_kv_heads == model.num_kv_heads for block in model.blocks
    )
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
        model(input_ids, kv_caches=[], use_cache=True)


@pytest.mark.parametrize("num_kv_heads", [None, 2, 1])
def test_token_by_token_cached_logits_match_full_forward(
    model_args, num_kv_heads
):
    torch.manual_seed(0)
    input_ids = torch.randint(0, model_args["vocab_size"], (2, 8))
    prompt_len = 4
    model = DecoderModel(**model_args, num_kv_heads=num_kv_heads).eval()

    full_logits = model(input_ids)

    prompt_logits, kv_caches = model(
        input_ids[:, :prompt_len], use_cache=True
    )
    cached_logits = [prompt_logits]

    for position in range(prompt_len, input_ids.shape[1]):
        next_logits, kv_caches = model(
            input_ids[:, position : position + 1],
            kv_caches=kv_caches,
            use_cache=True,
        )
        cached_logits.append(next_logits)

    cached_logits = torch.cat(cached_logits, dim=1)

    torch.testing.assert_close(cached_logits, full_logits, atol=1e-5, rtol=1e-5)
    for k_cache, v_cache in kv_caches:
        assert k_cache.shape[1] == model.num_kv_heads
        assert v_cache.shape[1] == model.num_kv_heads
        assert k_cache.shape[-2] == input_ids.shape[1]
        assert v_cache.shape[-2] == input_ids.shape[1]
