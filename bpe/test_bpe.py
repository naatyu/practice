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


def test_count_pairs_for_empty_and_single_token_inputs() -> None:
    assert ByteLevelBPE.count_pairs([]) == {}
    assert ByteLevelBPE.count_pairs([7]) == {}


def test_count_pairs_includes_overlapping_occurrences() -> None:
    token_ids = [1, 1, 1, 2]

    assert ByteLevelBPE.count_pairs(token_ids) == {
        (1, 1): 2,
        (1, 2): 1,
    }


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
