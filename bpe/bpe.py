from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import pairwise

import regex


class ByteLevelBPE:
    def __init__(self, pattern: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError(
                f"Expected vocab_size to be more than 256, got: {vocab_size}"
            )

        self.pattern = pattern
        self._pattern = regex.compile(pattern)
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.merge_ranks = {}

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

    @staticmethod
    def count_pairs(token_ids: list[int]) -> dict[tuple[int, int], int]:
        freq_counter = defaultdict(int)
        for i in range(len(token_ids) - 1):
            a = token_ids[i]
            b = token_ids[i + 1]
            freq_counter[(a, b)] += 1

        return freq_counter

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
        if isinstance(text, str):
            text = [text]

        # Pre-tokenize
        chunks = []
        for t in text:
            chunks.extend(self._pretokenize(t))

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
