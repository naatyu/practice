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


def test_position_offset(model_args):
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
    input_ids = torch.randint(
        0,
        model_args["vocab_size"],
        (2, 3),
    )

    with pytest.raises(ValueError, match="offset"):
        model(input_ids, position_offset=18)
