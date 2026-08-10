import pytest
import torch

from positional_encoding import LearnedPositionalEncoding


def test_output_shape():
    x = torch.randn((4, 10, 32))
    pos_table = LearnedPositionalEncoding(32, 128)

    assert pos_table(x).shape == (4, 10, 32)


def test_max_sequence_length():
    with pytest.raises(ValueError, match="sequence size"):
        x = torch.randn((4, 10, 32))
        pos_table = LearnedPositionalEncoding(32, 9)
        pos_table(x)


def test_controlled_output():
    x = torch.zeros(2, 3, 2)
    pos_encoding = LearnedPositionalEncoding(d_model=2, max_seq_len=4)

    table = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    with torch.no_grad():
        pos_encoding.pos_table.weight.copy_(table)

    expected_one_sequence = table[:3]
    expected = expected_one_sequence.unsqueeze(0).expand(2, -1, -1)

    torch.testing.assert_close(pos_encoding(x), expected)
