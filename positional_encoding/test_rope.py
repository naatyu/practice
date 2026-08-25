import math

import pytest
import torch

from positional_encoding import (
    RotaryPositionalEncoding,
    RotaryPositionalEncodingComplex,
)


def test_output_shape():
    q = torch.randn((4, 3, 10, 32))
    k = torch.randn((4, 3, 10, 32))
    rope = RotaryPositionalEncoding(32, 128)
    rot_q, rot_k = rope(q, k)

    assert rot_q.shape == (4, 3, 10, 32)
    assert rot_k.shape == (4, 3, 10, 32)


def test_output_dtype():
    q = torch.randn((4, 3, 10, 32), dtype=torch.bfloat16)
    k = torch.randn((4, 3, 10, 32), dtype=torch.bfloat16)
    rope = RotaryPositionalEncoding(32, 128)
    rot_q, rot_k = rope(q, k)

    assert rot_q.dtype == torch.bfloat16
    assert rot_k.dtype == torch.bfloat16


def test_dmodel():
    with pytest.raises(ValueError, match="d_head"):
        q = torch.randn((4, 3, 10, 32))
        k = torch.randn((4, 3, 10, 32))
        rope = RotaryPositionalEncoding(31, 9)
        rope(q, k)


def test_max_sequence_length():
    with pytest.raises(ValueError, match="sequence size"):
        q = torch.randn((4, 3, 10, 32))
        k = torch.randn((4, 3, 10, 32))
        rope = RotaryPositionalEncoding(32, 9)
        rope(q, k)


def test_large_offset():
    with pytest.raises(ValueError, match="offset"):
        q = torch.randn((4, 3, 6, 32))
        k = torch.randn((4, 3, 6, 32))
        rope = RotaryPositionalEncoding(32, 9)
        rope(q, k, offset=4)


def test_negative_offset():
    with pytest.raises(ValueError, match="offset"):
        q = torch.randn((4, 3, 6, 32))
        k = torch.randn((4, 3, 6, 32))
        rope = RotaryPositionalEncoding(32, 9)
        rope(q, k, offset=-4)


def test_offset_output():
    q = torch.randn((4, 3, 6, 32))
    k = torch.randn((4, 3, 6, 32))
    rope = RotaryPositionalEncoding(32, 9)
    rot_q, rot_k = rope(q, k)
    off_q, off_k = rope(q[:, :, 3:, :], k[:, :, 3:, :], offset=3)

    torch.testing.assert_close(rot_q[:, :, 3:, :], off_q)
    torch.testing.assert_close(rot_k[:, :, 3:, :], off_k)


def test_controlled_output():

    d_head = 4
    base = 10_000
    batch_size = 2
    seq_len = 3
    num_heads = 2
    q = torch.ones(batch_size, num_heads, seq_len, d_head)
    k = torch.ones(batch_size, num_heads, seq_len, d_head) * 2
    rope = RotaryPositionalEncoding(d_head=d_head, base=base, max_seq_len=4)

    frequency0 = 1 / base ** (2 * 0 / d_head)
    frequency1 = 1 / base ** (2 * 1 / d_head)
    q_expected = torch.Tensor(
        [
            [
                math.cos(0 * frequency0) - math.sin(0 * frequency0),
                math.cos(0 * frequency0) + math.sin(0 * frequency0),
                math.cos(0 * frequency1) - math.sin(0 * frequency1),
                math.cos(0 * frequency1) + math.sin(0 * frequency1),
            ],
            [
                math.cos(1 * frequency0) - math.sin(1 * frequency0),
                math.cos(1 * frequency0) + math.sin(1 * frequency0),
                math.cos(1 * frequency1) - math.sin(1 * frequency1),
                math.cos(1 * frequency1) + math.sin(1 * frequency1),
            ],
            [
                math.cos(2 * frequency0) - math.sin(2 * frequency0),
                math.cos(2 * frequency0) + math.sin(2 * frequency0),
                math.cos(2 * frequency1) - math.sin(2 * frequency1),
                math.cos(2 * frequency1) + math.sin(2 * frequency1),
            ],
        ]
    ).repeat(batch_size, num_heads, 1, 1)

    rot_q, rot_k = rope(q, k)
    torch.testing.assert_close(rot_q, q_expected)
    torch.testing.assert_close(rot_k, 2 * q_expected)


@pytest.mark.parametrize("offset", [0, 3])
def test_complex_rope_matches_real_implementation(offset: int) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 2, 4, 8)
    real_rope = RotaryPositionalEncoding(d_head=8, max_seq_len=12)
    complex_rope = RotaryPositionalEncodingComplex(d_head=8, max_seq_len=12)

    expected_q, expected_k = real_rope(q, k, offset=offset)
    actual_q, actual_k = complex_rope(q, k, offset=offset)

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_complex_rope_preserves_low_precision_dtype(dtype: torch.dtype) -> None:
    q = torch.randn(2, 3, 4, 8, dtype=dtype)
    k = torch.randn(2, 2, 4, 8, dtype=dtype)
    real_rope = RotaryPositionalEncoding(d_head=8, max_seq_len=8).to(dtype=dtype)
    rope = RotaryPositionalEncodingComplex(d_head=8, max_seq_len=8).to(dtype=dtype)

    rotated_q, rotated_k = rope(q, k, offset=2)
    expected_q, expected_k = real_rope(q, k, offset=2)

    assert rotated_q.dtype == dtype
    assert rotated_k.dtype == dtype
    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape
    tolerance = 4 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        rotated_q, expected_q, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        rotated_k, expected_k, rtol=tolerance, atol=tolerance
    )
