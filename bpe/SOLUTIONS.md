# Byte-level BPE: completed reasoning

## Byte vocabulary and UTF-8

Every valid Unicode string can be encoded as UTF-8 bytes whose values lie in
`0 ... 255`. Initializing one token for every byte therefore makes every valid
input representable without an unknown-character token. The cost is that a
single Unicode character may initially require multiple tokens; frequent byte
sequences become shorter after BPE training.

The base vocabulary maps token IDs to byte sequences:

```text
vocab[0]   = b"\x00"
...
vocab[255] = b"\xff"
```

Iterating over a Python `bytes` object yields integer byte values. For example,
the UTF-8 encoding of `é` becomes the token-ID sequence `[195, 169]`. Token
sequences use `list[int]` because learned IDs can be 256 or greater, while a
`bytes` object only accepts individual values through 255.

## Regex pre-tokenization

The tokenizer stores the original regex pattern for serialization and compiles
it once for repeated use. Both training and encoding must call the same
pre-tokenizer. BPE merges are allowed within regex matches but never across
their boundaries.

`finditer` yields matches incrementally and `match.group(0)` always returns the
complete match, even when the pattern contains capturing groups. Match spans
are checked to ensure that every input character is covered exactly once.
Without this check, a pattern such as `\p{L}+` could silently discard spaces
and punctuation, breaking the invariant:

```text
decode(encode(text)) == text
```

Empty matches consume no characters, produce no token IDs, and can create many
meaningless boundaries, so they are rejected. The third-party `regex` package
is used because it supports Unicode properties such as `\p{L}` and `\p{N}`.

The pre-tokenized representation is a list of independent byte-ID sequences:

```text
"hello world!"
-> ["hello", " ", "world", "!"]
-> [[104, ...], [32], [119, ...], [33]]
```

The outer list preserves merge boundaries; each inner list contains the byte
or learned token IDs for one regex match.

## Pair counting and replacement

A sequence of length `n` contains `n - 1` adjacent pair positions. Counting
includes overlapping occurrences: `(a, a)` occurs twice in `[a, a, a]`.
Replacement is non-overlapping, so merging that sequence produces `[aa, a]`.

Pair replacement is implemented as one left-to-right pass into a new list. A
matching pair consumes two input tokens; otherwise the current token is copied.
This avoids index shifting, preserves the input, and costs linear time with
linear output memory.

## Training

Training accepts one string or an iterable of document strings. Strings are
handled specially because they are themselves iterable. Each document is
pre-tokenized independently, after which pair counts are aggregated over
chunks without ever introducing cross-boundary pairs.

For each learned merge:

1. Count all adjacent pairs across all independent chunks.
2. Stop if no pairs remain or the target vocabulary size is reached.
3. Select the highest-frequency pair.
4. Resolve equal frequencies lexicographically by `(left_id, right_id)`.
5. Assign the next token ID and append the ordered merge rule.
6. Record the pair's zero-based rank.
7. Replace the selected pair independently in every chunk.

The deterministic selection key is:

```text
(-frequency, pair)
```

Taking the minimum makes larger frequencies win because their negatives are
smaller. Equal negative frequencies are resolved by normal tuple ordering.
Determinism matters because one selected merge changes all later pair counts.

Merged IDs are vocabulary references, not raw byte values. A learned token's
bytes must therefore be constructed as:

```text
vocab[new_id] = vocab[left_id] + vocab[right_id]
```

Constructing `bytes((left_id, right_id))` only works accidentally for base IDs
and fails once either ID exceeds 255.

## Ordered merges and merge ranks

The ordered merge list is the authoritative, serializable tokenizer state:

```text
[(left_id, right_id, new_id), ...]
```

The derived `merge_ranks` dictionary maps each learned pair to its priority
rank. During encoding it supports constant-time eligibility and priority
lookups. Once a rank is selected, the resulting token ID is available from the
corresponding entry in `merges`.

Keeping the ordered list is convenient for JSON serialization, inspection, and
reconstruction. Keeping the dictionary is analogous to building an index over
that canonical list. On load, validate the ordered merges and rebuild the
dictionary rather than serializing two sources of truth.

## Encoding and decoding

Encoding starts from the same independent regex chunks used during training.
It repeatedly finds adjacent pairs present in `merge_ranks`, selects the pair
with the smallest learned rank, and replaces every non-overlapping occurrence
inside each chunk. It never recounts frequencies on the input: merge priority
is part of the trained tokenizer state. Once no learned pair remains, the
chunks are flattened into one token-ID sequence.

Decoding performs the inverse vocabulary lookup for every ID, joins all byte
fragments, and decodes UTF-8 once at the end. This last detail is necessary
because one Unicode character can be divided across several byte tokens. For
example, the two bytes of `é` are not independently valid UTF-8 even though
their concatenation is.

Unicode therefore appears at the boundaries of this byte-level algorithm:
the regex operates on a Unicode string, ordinary chunks are encoded to UTF-8
before BPE, and the reconstructed bytes are decoded back to Unicode afterward.
No normalization is performed, so canonically equivalent Unicode strings can
have different byte sequences and tokenizations.

## Special tokens

Configured special strings are recognized before ordinary regex
pre-tokenization. Each is assigned one ID at or above `vocab_size`, outside the
regular vocabulary range. Splitting retains both the special matches and the
ordinary text between them. Regex metacharacters in special strings are
escaped, and alternatives are ordered longest-first so overlapping tokens use
a deterministic longest-match rule.

Ordinary pieces are pre-tokenized normally, while a special token becomes its
own singleton chunk:

```text
"hello<|end|>world"
-> [[bytes for hello], [special_id], [bytes for world]]
```

`pairwise` produces no pair for a singleton chunk, so training cannot learn a
merge involving the special ID or either neighbor. Encoding likewise cannot
merge across those boundaries. During decoding, the special string is encoded
to UTF-8 bytes and joined with ordinary vocabulary bytes before the final
decode.

## Serialization and training lifecycle

The portable JSON state contains only the original regex pattern, target
regular vocabulary size, ordered special-token strings, and ordered merges.
The 256-byte base vocabulary is deterministic. The learned vocabulary is
reconstructed by replaying merges, merge ranks are their list indices, special
IDs are regenerated from their order, and compiled regex objects are rebuilt
from their source strings.

An ordered list is a natural JSON representation because JSON object keys
cannot be tuples. It also preserves rank and contains the input IDs and output
ID needed to reconstruct each learned vocabulary entry. Loading validates that
both inputs to a merge already exist, preventing references to unknown or
future tokens.

Training is intentionally one-shot once at least one merge has been learned.
Starting a later corpus from raw bytes without first replaying existing rules
could learn a duplicate pair under a new ID and overwrite its rank. Supporting
true continuation would require applying all existing merges to the new corpus
before counting additional pairs. The simpler implementation rejects that
unsafe operation; a retry remains safe when an earlier corpus learned nothing.

## Current performance characteristics

Pair counting and one pair-replacement pass are linear in the current number
of corpus tokens. Training recounts pairs and rebuilds affected token sequences
for every learned merge, so the baseline cost is approximately:

```text
O(number_of_merges * current_corpus_tokens)
```

The implementation uses one shared `Counter` with lazy `pairwise` iteration,
avoiding slices and temporary per-chunk counters. It still stores all tokenized
chunks and rebuilds them on every merge.

Potential improvements, to evaluate after completing correctness, include:

- Frequency-compressing identical regex chunks and weighting their counts.
- Updating only pair counts adjacent to merged positions.
- Representing token sequences with linked neighbors to avoid repeated shifts.
- Maintaining candidate pairs in a heap with lazy invalidation.
- Using a heap or equivalent ranked structure during encoding.

These optimizations add substantial bookkeeping, so they should be justified
with corpus-scale profiling rather than assumed to be faster for small inputs.

