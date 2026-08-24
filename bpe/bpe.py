import json
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

import regex


class ByteLevelBPE:
    def __init__(
        self, pattern: str, vocab_size: int, special_tokens: list[str] | None = None
    ) -> None:
        if vocab_size < 256:
            raise ValueError(
                f"Expected vocab_size to be at least 256, got: {vocab_size}"
            )

        special_tokens = special_tokens or []
        if any(token == "" for token in special_tokens):
            raise ValueError("Special tokens must not be empty.")
        if len(special_tokens) != len(set(special_tokens)):
            raise ValueError("Special tokens must be unique.")

        self.pattern = pattern
        self._pattern = regex.compile(pattern)
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {v: vocab_size + i for i, v in enumerate(special_tokens)}
        self.special_token_ids = {
            token_id: token for token, token_id in self.special_tokens.items()
        }
        self.merges = []
        self.merge_ranks = {}

        sorted_special_tokens = sorted(
            special_tokens, reverse=True, key=len
        )  # Handle possible special token overlapping by starting with longest match
        if sorted_special_tokens:
            special_pattern = "|".join(
                regex.escape(sp_t) for sp_t in sorted_special_tokens
            )
            self._special_pattern = regex.compile(f"({special_pattern})")
        else:
            self._special_pattern = None

    def _pretokenize(self, text: str) -> list[list[int]]:
        if not text:
            return []

        token_ids = []
        cursor = 0

        for matched in self._pattern.finditer(text):
            if matched.start() != cursor:
                raise ValueError("Regex pattern does not cover the complete input.")
            if matched.end() == matched.start():
                raise ValueError("Regex pattern must not produce empty matches.")

            token_ids.append(list(matched.group(0).encode("utf-8")))
            cursor = matched.end()

        if cursor != len(text):
            raise ValueError("Regex pattern did not cover the complete text.")

        return token_ids

    def _split_special_tokens(self, text: str) -> list[str]:
        if not text:
            return []

        if self._special_pattern is None:
            return [text]

        return [part for part in self._special_pattern.split(text) if part]

    def _text_to_chunks(self, text: str) -> list[list[int]]:
        chunks = []

        for part in self._split_special_tokens(text):
            if part in self.special_tokens:
                chunks.append([self.special_tokens[part]])
            else:
                chunks.extend(self._pretokenize(part))

        return chunks

    @staticmethod
    def merge_pair(
        token_ids: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:
        merged = []
        n = len(token_ids)
        i = 0
        while i < n:
            if i + 1 < n and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
                merged.append(new_token_id)
                i += 2
            else:
                merged.append(token_ids[i])
                i += 1

        return merged

    def train(self, text: str | Iterable[str]) -> None:
        if self.merges:
            raise RuntimeError("Tokenizer has already been trained.")

        if isinstance(text, str):
            text = [text]

        # Pre-tokenize
        chunks = []
        for t in text:
            chunks.extend(self._text_to_chunks(t))

        next_token_id = len(self.vocab)

        while len(self.vocab) < self.vocab_size:
            # Count pair frequency
            freq_count = Counter()
            for chunk in chunks:
                freq_count.update(pairwise(chunk))

            # Stop training if empty, single byte or fully merged
            if not freq_count:
                break

            # Select the highest-frequency pair with lexicographic tie-breaking.
            highest_key = min(
                freq_count,
                key=lambda pair: (-freq_count[pair], pair),
            )

            # Update vocab and merge rank
            rank = len(self.merges)
            self.vocab[next_token_id] = (
                self.vocab[highest_key[0]] + self.vocab[highest_key[1]]
            )
            self.merges.append((highest_key[0], highest_key[1], next_token_id))
            self.merge_ranks[(highest_key[0], highest_key[1])] = rank

            # Merge pair
            chunks = [
                self.merge_pair(chunk, highest_key, next_token_id) for chunk in chunks
            ]
            next_token_id += 1

    def encode(self, text: str) -> list[int]:
        # Pre-tokenization
        chunks = self._text_to_chunks(text)

        while True:
            eligible_pairs = {
                pair for c in chunks for pair in pairwise(c) if pair in self.merge_ranks
            }

            # Stop if nothing to merge
            if not eligible_pairs:
                break

            pair_to_merge = min(eligible_pairs, key=lambda pair: self.merge_ranks[pair])
            rank = self.merge_ranks[pair_to_merge]
            new_token_id = self.merges[rank][2]

            chunks = [
                self.merge_pair(chunk, pair_to_merge, new_token_id) for chunk in chunks
            ]

        return [token_id for chunk in chunks for token_id in chunk]

    def decode(self, token_ids: list[int]) -> str:
        text_bytes = []
        for token_id in token_ids:
            if token_id in self.special_token_ids:
                text_bytes.append(self.special_token_ids[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                text_bytes.append(self.vocab[token_id])
            else:
                raise ValueError(
                    f"Token id {token_id} not found in vocabulary or special tokens."
                )

        return b"".join(text_bytes).decode("utf-8")

    def save(self, path: Path) -> None:
        state = {
            "pattern": self.pattern,
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.special_tokens),
            "merges": self.merges,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                state, f, ensure_ascii=False, indent=2
            )  # Do not ensure ascii to keep unicode special tokens readable

    @classmethod
    def load(cls, path: Path) -> "ByteLevelBPE":
        with path.open("r", encoding="utf-8") as f:
            saved_state = json.load(f)

        tokenizer = cls(
            pattern=saved_state["pattern"],
            vocab_size=saved_state["vocab_size"],
            special_tokens=saved_state["special_tokens"],
        )

        # Reconstruct
        for rank, (left, right, next_id) in enumerate(saved_state["merges"]):
            if left not in tokenizer.vocab:
                raise ValueError(f"Unrecognized token id: {left}")
            elif right not in tokenizer.vocab:
                raise ValueError(f"Unrecognized token id: {right}")

            tokenizer.merges.append((left, right, next_id))
            tokenizer.merge_ranks[(left, right)] = rank
            tokenizer.vocab[next_id] = tokenizer.vocab[left] + tokenizer.vocab[right]

        return tokenizer
