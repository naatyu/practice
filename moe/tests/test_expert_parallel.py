"""AI-authored tests for EP preparation and local computation, not communication."""

import pytest
import torch

from moe.expert_parallel import compute_local_experts, prepare_dispatch
from moe.experts import GroupedExperts


@pytest.mark.parametrize(
    "num_experts,world_size,ids,counts,permutation",
    [
        (4, 2, [[3, 0], [2, 1]], [2, 2], [1, 3, 0, 2]),
        # Experts per rank differs from top-k: these are different divisors.
        (6, 2, [[5, 0], [3, 1]], [2, 2], [1, 3, 0, 2]),
        (8, 4, [[6, 0], [7, 1]], [2, 0, 0, 2], [1, 3, 0, 2]),
        # Stable grouping must not sort expert IDs inside a destination chunk.
        (4, 2, [[3], [2]], [0, 2], [0, 1]),
    ],
)
def test_prepare_dispatch_controlled_order(
    num_experts, world_size, ids, counts, permutation
):
    tokens = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    expert_ids = torch.tensor(ids)
    original_tokens, original_ids = tokens.clone(), expert_ids.clone()

    send_tokens, send_ids, send_counts, actual_permutation = prepare_dispatch(
        tokens, expert_ids, num_experts, world_size
    )

    expected_permutation = torch.tensor(permutation)
    token_indices = expected_permutation // expert_ids.shape[1]
    torch.testing.assert_close(actual_permutation, expected_permutation)
    torch.testing.assert_close(send_counts, torch.tensor(counts))
    torch.testing.assert_close(send_tokens, tokens[token_indices])
    torch.testing.assert_close(send_ids, expert_ids.flatten()[expected_permutation])
    assert send_tokens.shape == (expert_ids.numel(), tokens.shape[1])
    assert send_tokens.dtype == tokens.dtype
    torch.testing.assert_close(tokens, original_tokens)
    torch.testing.assert_close(expert_ids, original_ids)


def test_prepare_dispatch_preserves_noncontiguous_token_rows():
    tokens = torch.arange(12, dtype=torch.float64).reshape(2, 6)[:, ::2]
    assert not tokens.is_contiguous()
    send_tokens, _, _, _ = prepare_dispatch(
        tokens, torch.tensor([[3, 0], [2, 1]]), 4, 2
    )
    torch.testing.assert_close(send_tokens, tokens[[0, 1, 0, 1]])


def test_prepare_dispatch_empty_local_batch():
    send_tokens, send_ids, counts, permutation = prepare_dispatch(
        torch.empty(0, 3), torch.empty(0, 2, dtype=torch.int64), 4, 2
    )
    assert send_tokens.shape == (0, 3)
    assert send_ids.shape == permutation.shape == (0,)
    torch.testing.assert_close(counts, torch.tensor([0, 0]))


@pytest.mark.parametrize(
    "rank,ids,permutation",
    [
        (0, [0, 1, 0, 1, 0], [0, 2, 4, 1, 3]),
        (1, [3, 2, 2], [1, 2, 0]),
        (1, [2, 2, 2], [0, 1, 2]),
        (1, [3, 3], [0, 1]),
    ],
)
def test_local_experts_match_independent_per_row_reference(rank, ids, permutation):
    torch.manual_seed(42)
    experts = GroupedExperts(16, 32, 2)
    tokens = torch.randn(len(ids), 16)
    expert_ids = torch.tensor(ids)
    original_tokens, original_ids = tokens.clone(), expert_ids.clone()

    actual, expert_permutation = compute_local_experts(
        tokens, expert_ids, experts, rank
    )

    # Independent reference: process each received row with its selected weights.
    expected_rows = []
    for token, global_id in zip(tokens, ids, strict=True):
        local_id = global_id - rank * 2
        value, gate = (token @ experts.up_weight[local_id]).chunk(2, dim=-1)
        expected_rows.append(
            (value * torch.nn.functional.silu(gate)) @ experts.down_weight[local_id]
        )
    expected_permutation = torch.tensor(permutation)
    expected = torch.stack(expected_rows)[expected_permutation]
    torch.testing.assert_close(expert_permutation, expected_permutation)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == tokens.shape
    torch.testing.assert_close(tokens, original_tokens)
    torch.testing.assert_close(expert_ids, original_ids)
