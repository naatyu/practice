import json
from pathlib import Path

import pytest

from bpe import ByteLevelBPE


@pytest.fixture
def tokenizer() -> ByteLevelBPE:
    pattern = r"\p{L}+|\p{N}+|\s+|[^\p{L}\p{N}\s]+"
    return ByteLevelBPE(pattern=pattern, vocab_size=300)


def test_constructor_builds_complete_byte_vocabulary() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=256)

    assert len(tokenizer.vocab) == 256
    assert tokenizer.vocab[0] == b"\x00"
    assert tokenizer.vocab[255] == b"\xff"
    assert tokenizer.merges == []
    assert tokenizer.merge_ranks == {}


def test_constructor_rejects_vocab_smaller_than_byte_vocabulary() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        ByteLevelBPE(pattern=r".+", vocab_size=255)


def test_constructor_assigns_special_tokens_after_regular_vocabulary() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=300,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )

    assert tokenizer.special_tokens == {
        "<|endoftext|>": 300,
        "<|pad|>": 301,
    }
    assert tokenizer.special_token_ids == {
        300: "<|endoftext|>",
        301: "<|pad|>",
    }


@pytest.mark.parametrize(
    ("special_tokens", "error"),
    [
        ([""], "empty"),
        (["<|end|>", "<|end|>"], "unique"),
    ],
)
def test_constructor_rejects_invalid_special_tokens(
    special_tokens: list[str], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        ByteLevelBPE(
            pattern=r".+",
            vocab_size=256,
            special_tokens=special_tokens,
        )


def test_split_special_tokens_preserves_text_and_delimiters() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=256,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )

    assert tokenizer._split_special_tokens(
        "hello<|endoftext|>world<|pad|>"
    ) == ["hello", "<|endoftext|>", "world", "<|pad|>"]


def test_split_special_tokens_prefers_longest_overlapping_token() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=256,
        special_tokens=["<end>", "<end>text"],
    )

    assert tokenizer._split_special_tokens("a<end>textb") == [
        "a",
        "<end>text",
        "b",
    ]


def test_split_special_tokens_treats_regex_metacharacters_literally() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=256,
        special_tokens=["[MASK]", "a+b"],
    )

    assert tokenizer._split_special_tokens("x[MASK]ya+bz") == [
        "x",
        "[MASK]",
        "y",
        "a+b",
        "z",
    ]


def test_split_special_tokens_without_configured_tokens() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=256)

    assert tokenizer._split_special_tokens("ordinary text") == ["ordinary text"]
    assert tokenizer._split_special_tokens("") == []


def test_pretokenize_preserves_unicode_whitespace_and_punctuation(
    tokenizer: ByteLevelBPE,
) -> None:
    text = "hé  42!"

    chunks = tokenizer._pretokenize(text)

    assert chunks == [
        list("hé".encode("utf-8")),
        list("  ".encode("utf-8")),
        list("42".encode("utf-8")),
        list("!".encode("utf-8")),
    ]
    assert b"".join(bytes(chunk) for chunk in chunks).decode("utf-8") == text


def test_pretokenize_uses_complete_matches_with_capturing_groups() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r"(\p{L}+)|(\s+)|([^\p{L}\s]+)",
        vocab_size=256,
    )

    assert tokenizer._pretokenize("hi 42") == [
        list(b"hi"),
        list(b" "),
        list(b"42"),
    ]


def test_pretokenize_rejects_pattern_that_skips_text() -> None:
    tokenizer = ByteLevelBPE(pattern=r"\p{L}+", vocab_size=256)

    with pytest.raises(ValueError, match="cover"):
        tokenizer._pretokenize("hello world")


def test_pretokenize_rejects_empty_matches() -> None:
    tokenizer = ByteLevelBPE(pattern=r".*?", vocab_size=256)

    with pytest.raises(ValueError, match="empty"):
        tokenizer._pretokenize("text")


def test_merge_pair_handles_empty_and_single_token_inputs() -> None:
    assert ByteLevelBPE.merge_pair([], pair=(1, 2), new_token_id=10) == []
    assert ByteLevelBPE.merge_pair([1], pair=(1, 2), new_token_id=10) == [1]


def test_merge_pair_replaces_all_non_overlapping_occurrences() -> None:
    token_ids = [1, 2, 1, 2, 3]

    assert ByteLevelBPE.merge_pair(
        token_ids, pair=(1, 2), new_token_id=10
    ) == [10, 10, 3]


def test_merge_pair_does_not_merge_overlapping_occurrences() -> None:
    token_ids = [1, 1, 1]

    assert ByteLevelBPE.merge_pair(
        token_ids, pair=(1, 1), new_token_id=10
    ) == [10, 1]


def test_merge_pair_preserves_input_and_unmatched_tokens() -> None:
    token_ids = [1, 3, 2]
    original = token_ids.copy()

    merged = ByteLevelBPE.merge_pair(
        token_ids, pair=(1, 2), new_token_id=10
    )

    assert merged == token_ids
    assert merged is not token_ids
    assert token_ids == original


def test_train_builds_recursive_tokens_and_zero_based_merge_ranks() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=258)

    tokenizer.train("abab")

    assert tokenizer.merges == [
        (ord("a"), ord("b"), 256),
        (256, 256, 257),
    ]
    assert tokenizer.merge_ranks == {
        (ord("a"), ord("b")): 0,
        (256, 256): 1,
    }
    assert tokenizer.vocab[256] == b"ab"
    assert tokenizer.vocab[257] == b"abab"


def test_train_aggregates_pair_counts_across_documents() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=257)
    documents = (document for document in ["ab", "ab", "ac"])

    tokenizer.train(documents)

    assert tokenizer.merges == [(ord("a"), ord("b"), 256)]
    assert tokenizer.vocab[256] == b"ab"


def test_train_uses_lexicographic_tie_breaking() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=257)

    tokenizer.train(["bc", "ab"])

    assert tokenizer.merges == [(ord("a"), ord("b"), 256)]


def test_train_never_counts_pairs_across_regex_chunks() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r"\p{L}+|\s+",
        vocab_size=257,
    )

    tokenizer.train("a a")

    assert len(tokenizer.vocab) == 256
    assert tokenizer.merges == []


def test_train_never_counts_pairs_across_special_token_boundaries() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=257,
        special_tokens=["<|end|>"],
    )

    tokenizer.train("a<|end|>b")

    assert len(tokenizer.vocab) == 256
    assert tokenizer.merges == []


def test_train_builds_tokens_from_multibyte_utf8_sequences() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=257)

    tokenizer.train("éé")

    first_byte, second_byte = "é".encode("utf-8")
    assert tokenizer.merges == [(first_byte, second_byte, 256)]
    assert tokenizer.vocab[256] == "é".encode("utf-8")


@pytest.mark.parametrize("corpus", ["", "a", ["", "a"]])
def test_train_stops_when_the_corpus_has_no_pairs(corpus) -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=300)

    tokenizer.train(corpus)

    assert len(tokenizer.vocab) == 256
    assert tokenizer.merges == []


def test_train_with_base_vocab_target_learns_no_merges() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=256)

    tokenizer.train("repeated repeated text")

    assert len(tokenizer.vocab) == 256
    assert tokenizer.merges == []


def test_train_rejects_retraining_after_learning_merges() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=260)
    tokenizer.train("ab")
    original_vocab = tokenizer.vocab.copy()
    original_merges = tokenizer.merges.copy()
    original_ranks = tokenizer.merge_ranks.copy()

    with pytest.raises(RuntimeError, match="already been trained"):
        tokenizer.train("abab")

    assert tokenizer.vocab == original_vocab
    assert tokenizer.merges == original_merges
    assert tokenizer.merge_ranks == original_ranks


def test_train_can_retry_when_no_merge_was_learned() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=257)
    tokenizer.train("")

    tokenizer.train("ab")

    assert tokenizer.merges == [(ord("a"), ord("b"), 256)]


def test_encode_without_merges_returns_flat_utf8_byte_ids() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r"\p{L}+|\s+|[^\p{L}\s]+",
        vocab_size=256,
    )
    text = "hé 42!"

    assert tokenizer.encode(text) == list(text.encode("utf-8"))


def test_encode_applies_learned_merges_recursively() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=258)
    tokenizer.train("abab")

    assert tokenizer.encode("ab") == [256]
    assert tokenizer.encode("abab") == [257]


def test_encode_uses_learned_rank_not_input_pair_frequency() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=258)
    a, b, c = map(ord, "abc")
    tokenizer.merges = [
        (a, b, 256),
        (b, c, 257),
    ]
    tokenizer.merge_ranks = {
        (a, b): 0,
        (b, c): 1,
    }

    assert tokenizer.encode("abcbcbc") == [256, c, 257, 257]


def test_encode_never_merges_across_regex_chunk_boundaries() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r"\p{L}+|\s+",
        vocab_size=257,
    )
    a, space = ord("a"), ord(" ")
    tokenizer.merges = [(a, space, 256)]
    tokenizer.merge_ranks = {(a, space): 0}

    assert tokenizer.encode("a a") == [a, space, a]


def test_encode_emits_special_token_as_one_id() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=256,
        special_tokens=["<|end|>"],
    )

    assert tokenizer.encode("a<|end|>b") == [
        ord("a"),
        tokenizer.special_tokens["<|end|>"],
        ord("b"),
    ]


def test_encode_applies_bpe_inside_text_around_special_token() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=258,
        special_tokens=["<|end|>"],
    )
    tokenizer.train("abab<|end|>abab")

    assert tokenizer.encode("abab<|end|>abab") == [
        257,
        tokenizer.special_tokens["<|end|>"],
        257,
    ]


def test_encode_empty_text() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=256)

    assert tokenizer.encode("") == []


def test_decode_joins_fragmented_utf8_bytes_before_decoding() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=256)

    assert tokenizer.decode(list("é🙂".encode("utf-8"))) == "é🙂"


def test_decode_expands_learned_tokens_through_the_vocabulary() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=258)
    tokenizer.train("abab")

    assert tokenizer.decode([257]) == "abab"


def test_decode_reconstructs_special_tokens_between_regular_tokens() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=256,
        special_tokens=["<|end|>"],
    )
    special_id = tokenizer.special_tokens["<|end|>"]

    assert tokenizer.decode([ord("a"), special_id, ord("b")]) == "a<|end|>b"


def test_decode_handles_unicode_special_token() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=256,
        special_tokens=["<🙂>"],
    )

    assert tokenizer.decode([tokenizer.special_tokens["<🙂>"]]) == "<🙂>"


def test_encode_decode_round_trip_after_training(tokenizer: ByteLevelBPE) -> None:
    tokenizer.train(["hello hello", "hé hé", "🙂🙂"])
    text = "hello, hé 42!\n🙂"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_decode_round_trip_with_special_tokens() -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=258,
        special_tokens=["<|end|>", "<🙂>"],
    )
    tokenizer.train("abab<|end|>abab")
    text = "abab<|end|>é<🙂>abab"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_decode_empty_token_sequence() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=256)

    assert tokenizer.decode([]) == ""


def test_decode_rejects_unknown_token_id() -> None:
    tokenizer = ByteLevelBPE(pattern=r".+", vocab_size=256)

    with pytest.raises(ValueError, match="999"):
        tokenizer.decode([999])


def test_save_load_preserves_state_and_tokenization(tmp_path: Path) -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=258,
        special_tokens=["<|end|>", "<🙂>"],
    )
    tokenizer.train("abab<|end|>abab")
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    loaded = ByteLevelBPE.load(path)

    assert loaded.pattern == tokenizer.pattern
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.special_tokens == tokenizer.special_tokens
    assert loaded.special_token_ids == tokenizer.special_token_ids
    assert loaded.merges == tokenizer.merges
    assert loaded.merge_ranks == tokenizer.merge_ranks
    assert loaded.vocab == tokenizer.vocab

    text = "abab<|end|>é<🙂>abab"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_save_uses_minimal_json_state(tmp_path: Path) -> None:
    tokenizer = ByteLevelBPE(
        pattern=r".+",
        vocab_size=257,
        special_tokens=["<🙂>"],
    )
    tokenizer.train("abab")
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)

    with path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    assert set(state) == {"pattern", "vocab_size", "special_tokens", "merges"}
    assert state["special_tokens"] == ["<🙂>"]
    assert state["merges"] == [[ord("a"), ord("b"), 256]]


def test_load_rejects_merge_with_unknown_dependency(tmp_path: Path) -> None:
    path = tmp_path / "invalid-tokenizer.json"
    state = {
        "pattern": ".+",
        "vocab_size": 257,
        "special_tokens": [],
        "merges": [[999, ord("a"), 256]],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(state, file)

    with pytest.raises(ValueError, match="999"):
        ByteLevelBPE.load(path)
