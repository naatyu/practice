import math

import pytest
import torch

from positional_encoding import SinusoidalPositionalEncoding


def test_output_shape():
    x = torch.randn((4, 10, 32))
    pos_table = SinusoidalPositionalEncoding(32, 128)

    assert pos_table(x).shape == (4, 10, 32)


def test_dmodel():
    with pytest.raises(ValueError, match="d_model"):
        x = torch.randn((4, 10, 32))
        pos_table = SinusoidalPositionalEncoding(31, 9)
        pos_table(x)


def test_max_sequence_length():
    with pytest.raises(ValueError, match="sequence size"):
        x = torch.randn((4, 10, 32))
        pos_table = SinusoidalPositionalEncoding(32, 9)
        pos_table(x)


def test_controlled_output():

    d_model = 4
    base = 10_000
    batch_size = 2
    x = torch.ones(batch_size, 3, d_model)
    pos_encoding = SinusoidalPositionalEncoding(
        d_model=d_model, base=base, max_seq_len=4
    )

    frequency0 = 1 / base ** (2 * 0 / d_model)
    frequency1 = 1 / base ** (2 * 1 / d_model)
    expected = torch.Tensor(
        [
            [
                1 + math.sin(0 * frequency0),
                1 + math.cos(0 * frequency0),
                1 + math.sin(0 * frequency1),
                1 + math.cos(0 * frequency1),
            ],
            [
                1 + math.sin(1 * frequency0),
                1 + math.cos(1 * frequency0),
                1 + math.sin(1 * frequency1),
                1 + math.cos(1 * frequency1),
            ],
            [
                1 + math.sin(2 * frequency0),
                1 + math.cos(2 * frequency0),
                1 + math.sin(2 * frequency1),
                1 + math.cos(2 * frequency1),
            ],
        ]
    ).repeat(batch_size, 1, 1)

    torch.testing.assert_close(pos_encoding(x), expected)
