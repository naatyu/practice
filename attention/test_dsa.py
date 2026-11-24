import math

import torch

from attention.dsa import SimpleIndexer, sparse_attention


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
