# Byte-level BPE

This exercise builds a byte-level BPE tokenizer from first principles. The
questions are useful both as an implementation guide and as interview review.
Completed reasoning is kept in `SOLUTIONS.md`; implementation code remains in
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
16. Where does Unicode enter a tokenizer whose BPE algorithm operates on bytes?
17. Why must byte fragments be joined before UTF-8 decoding?
18. Why must special tokens be recognized before ordinary pre-tokenization?
19. How do singleton chunks prevent merges across special-token boundaries?
20. Why should overlapping special tokens use a longest-match rule?
21. Which tokenizer fields are canonical serialized state, and which can be
    reconstructed?
22. Why is an ordered merge list a better JSON representation than a mapping
    with tuple keys?
23. Why is retraining unsafe unless existing merges are first replayed?

## Completed scope

- Constructor and base vocabulary.
- Regex compilation and lossless pre-tokenization.
- Pair-frequency counting.
- Non-overlapping pair replacement.
- Multi-document BPE training.
- Deterministic lexicographic tie-breaking.
- Ordered merges, merge ranks, and merged-byte vocabulary construction.
- Rank-ordered encoding and UTF-8 decoding.
- Indivisible special tokens with longest-match recognition.
- Portable JSON save/load with derived-state reconstruction.
- A one-shot training contract that prevents merge-state corruption.
- Round-trip, boundary, Unicode, serialization, and malformed-state tests.

Performance optimization remains optional and should be guided by profiling.
