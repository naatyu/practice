import math

import torch

from attention.dsa import (
    IndexerCache,
    LightningIndexer,
    SimpleIndexer,
    _select_causal_topk,
    sparse_attention,
)
from positional_encoding import RotaryPositionalEncoding


def test_causal_selection_ranks_keys_for_each_query() -> None:
    queries = torch.tensor([[1.0, -1.0], [1.0, 0.0], [-1.0, 1.0], [1.0, -1.0]])
    keys = torch.tensor([[1.0, 2.0], [3.0, 0.0], [1.0, 0.0], [2.0, -2.0]])

    indexer = SimpleIndexer(d_model=2, d_indexer=2, top_k=1)
    actual = indexer.select_tokens_causal(queries, keys)

    torch.testing.assert_close(actual, torch.tensor([[0], [1], [0], [3]]))


def test_causal_selection_pads_short_prefixes_with_negative_one() -> None:
    queries = torch.ones(3, 1)
    keys = torch.tensor([[1.0], [2.0], [100.0]])

    indexer = SimpleIndexer(d_model=1, d_indexer=1, top_k=2)
    actual = indexer.select_tokens_causal(queries, keys)

    torch.testing.assert_close(actual, torch.tensor([[0, -1], [1, 0], [2, 1]]))


def test_causal_selection_caps_top_k_to_prompt_length() -> None:
    indexer = SimpleIndexer(d_model=1, d_indexer=1, top_k=8)
    queries = torch.ones(3, 1)
    keys = torch.tensor([[1.0], [2.0], [100.0]])

    actual = indexer.select_tokens_causal(queries, keys)

    torch.testing.assert_close(
        actual, torch.tensor([[0, -1, -1], [1, 0, -1], [2, 1, 0]])
    )


def test_simple_indexer_uses_separate_query_and_key_projections() -> None:
    indexer = SimpleIndexer(d_model=2, d_indexer=1, top_k=2)
    with torch.no_grad():
        indexer.q_proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
        indexer.k_proj.weight.copy_(torch.tensor([[0.0, 1.0]]))

    # Read the first feature for queries and the second for keys.
    x = torch.tensor([[1.0, 1.0], [2.0, -1.0], [-1.0, 3.0]])

    torch.testing.assert_close(indexer(x), torch.tensor([[0, -1], [0, 1], [1, 0]]))


def test_lightning_indexer_combines_rectified_head_scores() -> None:
    indexer = LightningIndexer(d_model=4, d_indexer=3, n_heads=2, top_k=1)
    queries = torch.tensor([[2.0, 0, 0, -4, 0, 0], [-2.0, 0, 0, 3, 0, 0]])
    keys = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]])
    weights = torch.tensor([[3.0, -1.0], [2.0, 4.0]])
    original_weights = weights.clone()

    scores = indexer.index_scores(queries[None], keys[None], weights[None])

    expected = torch.tensor([[6.0, -4.0], [12.0, 4.0]]) / math.sqrt(6)
    torch.testing.assert_close(scores, expected[None])
    torch.testing.assert_close(weights, original_weights)


def test_lightning_indexer_rotates_only_the_configured_prefix() -> None:
    rope = RotaryPositionalEncoding(d_head=2, max_seq_len=2)
    indexer = LightningIndexer(d_model=4, d_indexer=4, n_heads=2, top_k=1, rope=rope)
    queries = torch.tensor([[1.0, 0, 2, 0, 0, 1, 0, 2]]).repeat(2, 1)
    keys = torch.tensor([[1.0, 0, 1, 1]]).repeat(2, 1)
    weights = torch.eye(2)

    scores = indexer.index_scores(queries[None], keys[None], weights[None])

    # One head per row: rotary contributions differ, the plain dot stays 2.
    expected = torch.tensor([[3.0, 2 + math.cos(1)], [2 - math.sin(1), 2.0]])
    torch.testing.assert_close(scores, expected[None] / math.sqrt(8))


def test_lightning_indexer_ranks_normalized_keys() -> None:
    indexer = LightningIndexer(d_model=3, d_indexer=3, n_heads=1, top_k=1)
    with torch.no_grad():
        indexer.k_proj.weight.copy_(torch.eye(3))
        indexer.q_proj.weight.copy_(torch.diag(torch.tensor([1.0, 0.0, 0.0])))
        indexer.head_weights.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))

    # The second key has a smaller raw first feature but a larger normalized one.
    x = torch.tensor([[10.0, 9.0, 8.0], [2.0, -1.0, -1.0]])

    torch.testing.assert_close(indexer(x[None]), torch.tensor([[[0], [1]]]))


def test_lightning_indexer_batched_matches_separate_sequences() -> None:
    rng = torch.Generator().manual_seed(0)
    rope = RotaryPositionalEncoding(d_head=2, max_seq_len=5)
    indexer = LightningIndexer(d_model=6, d_indexer=4, n_heads=3, top_k=8, rope=rope)
    with torch.no_grad():
        for projection in (indexer.q_proj, indexer.k_proj, indexer.head_weights):
            projection.weight.copy_(torch.randn(projection.weight.shape, generator=rng))
    x = torch.randn(2, 5, 6, generator=rng)

    actual = indexer(x)
    expected = torch.cat([indexer(sequence[None]) for sequence in x], dim=0)

    assert actual.shape == (2, 5, 5)
    torch.testing.assert_close(actual, expected)


def test_causal_topk_uses_absolute_positions_for_suffix_queries() -> None:
    # Queries are at positions 3 and 4. Key 4 scores highest for both queries.
    scores = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 100.0], [0.0, 1.0, 2.0, 3.0, 100.0]]])

    actual = _select_causal_topk(scores, top_k=2)

    torch.testing.assert_close(actual, torch.tensor([[[3, 2], [4, 3]]]))


@torch.no_grad()
def test_lightning_indexer_cached_chunks_match_full_prefill() -> None:
    rng = torch.Generator().manual_seed(7)
    rope = RotaryPositionalEncoding(d_head=2, max_seq_len=5)
    indexer = LightningIndexer(d_model=6, d_indexer=4, n_heads=3, top_k=2, rope=rope)
    for projection in (indexer.q_proj, indexer.k_proj, indexer.head_weights):
        projection.weight.copy_(torch.randn(projection.weight.shape, generator=rng))
    x = torch.randn(2, 5, 6, generator=rng)
    cache = IndexerCache()

    # Fill the cache, then append a two-token suffix and a single decode token.
    for start, end in ((0, 2), (2, 4), (4, 5)):
        previous_keys = cache.keys
        actual = indexer(x[:, start:end], cache=cache)
        # Use the same key count: top-k can break ties differently on wider rows.
        expected = indexer(x[:, :end])[:, start:end]

        torch.testing.assert_close(actual, expected)
        assert cache.keys is not None
        assert cache.keys.shape == (2, end, 4)
        if previous_keys is not None:
            torch.testing.assert_close(cache.keys[:, :start], previous_keys)


def test_sparse_attention_matches_dense_attention_on_selected_keys() -> None:
    rng = torch.Generator().manual_seed(0)
    queries = torch.randn(4, 3, generator=rng)
    keys = torch.randn(4, 3, generator=rng)
    values = torch.randn(4, 5, generator=rng)
    indices = torch.tensor([[0, -1], [1, 0], [2, 0], [3, 1]])

    # In the dense reference, keep a key if any selected slot points to it.
    positions = torch.arange(keys.size(0))
    keep = (positions == indices.unsqueeze(-1)).any(dim=1)
    scores = (queries @ keys.T) / math.sqrt(queries.size(-1))
    expected = scores.masked_fill(~keep, float("-inf")).softmax(dim=-1) @ values

    torch.testing.assert_close(
        sparse_attention(queries, keys, values, indices), expected
    )


def test_batched_sparse_attention_gathers_within_each_sequence() -> None:
    queries = torch.zeros(2, 1, 1)
    keys = torch.zeros(2, 3, 1)
    values = torch.tensor([[[100.0], [1.0], [3.0]], [[5.0], [50.0], [9.0]]])
    indices = torch.tensor([[[1, 2, -1]], [[0, 2, -1]]])

    actual = sparse_attention(queries, keys, values, indices)

    # Equal scores average the two selected values, ignoring the unused slot.
    torch.testing.assert_close(actual, torch.tensor([[[2.0]], [[7.0]]]))


@torch.no_grad()
def test_sparse_attention_with_cached_indexer_matches_prefill_suffix() -> None:
    rng = torch.Generator().manual_seed(7)
    rope = RotaryPositionalEncoding(d_head=2, max_seq_len=5)
    indexer = LightningIndexer(d_model=6, d_indexer=4, n_heads=3, top_k=2, rope=rope)
    for projection in (indexer.q_proj, indexer.k_proj, indexer.head_weights):
        projection.weight.copy_(torch.randn(projection.weight.shape, generator=rng))
    x = torch.randn(2, 5, 6, generator=rng)
    queries = torch.randn(2, 5, 3, generator=rng)
    keys = torch.randn(2, 5, 3, generator=rng)
    values = torch.randn(2, 5, 4, generator=rng)
    prefix = 3
    expected = sparse_attention(queries, keys, values, indexer(x))[:, prefix:]

    cache = IndexerCache()
    indexer(x[:, :prefix], cache=cache)
    indices = indexer(x[:, prefix:], cache=cache)
    actual = sparse_attention(queries[:, prefix:], keys, values, indices)

    torch.testing.assert_close(actual, expected)
