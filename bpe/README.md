# Byte-level BPE interview practice

This file contains the exercise prompt and interview questions only. Completed
reasoning is kept separately in `SOLUTIONS.md`; implementation code remains in
`bpe.py`.

## Exercise

Implement a byte-level BPE tokenizer without using an existing tokenizer
implementation.

The tokenizer should eventually support:

- A complete 256-byte base vocabulary.
- User-configurable regex pre-tokenization with Unicode-aware patterns.
- BPE training over one string or multiple documents.
- Deterministic pair selection and ordered merge rules.
- Encoding and decoding arbitrary UTF-8 text.
- Special tokens that cannot be split or merged accidentally.
- Saving and loading a portable tokenizer representation.
- Exact text round trips for ordinary input.

### Discussion questions

1. Why does a 256-byte base vocabulary eliminate unknown Unicode characters?
2. What is the sequence-length tradeoff of starting from bytes?
3. What does one BPE merge add to the vocabulary and change in the corpus?
4. Why are pair occurrences counted with overlap but replaced without overlap?
5. Why must regex chunks be retained as separate token-ID sequences during
   training?
6. Why should a GPT-style pre-tokenization regex match every input character?
7. Why are empty regex matches rejected?
8. Why use the third-party `regex` package rather than standard `re` for
   patterns containing properties such as `\p{L}`?
9. Why should pair-frequency ties be resolved deterministically?
10. Why must merged vocabulary bytes be constructed through vocabulary
    lookups rather than `bytes((left_id, right_id))`?
11. What is the difference between the ordered `merges` list and the derived
    `merge_ranks` lookup?
12. Why does encoding apply the available learned merge with the smallest
    rank instead of relearning frequencies from the new input?
13. How can tokenizer state be serialized without using `pickle`?
14. What are the time and memory costs of the baseline training algorithm?
15. How could repeated chunks, incremental counts, linked token structures, or
    heaps improve training and encoding performance?

## Current checkpoint

Completed:

- Constructor and base vocabulary.
- Regex compilation and lossless pre-tokenization.
- Pair counting.
- Non-overlapping pair replacement.
- Multi-document BPE training.
- Deterministic lexicographic tie-breaking.
- Ordered merges, merge ranks, and merged-byte vocabulary construction.
- Correctness tests for all of the above.

Resume with:

1. Implement `encode(text) -> list[int]` using learned merge ranks.
2. Implement `decode(token_ids) -> str`.
3. Add special-token handling.
4. Add JSON save/load and validation.
5. Profile and improve the baseline algorithm where complexity is justified.

