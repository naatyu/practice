# Autoregressive generation: completed reasoning

## From logits to the next token

For causal decoder logits shaped `[B, S, V]`, position `i` predicts the token
after input position `i`. Only `logits[:, -1, :]` represents the distribution
after the complete current prefix. Greedy decoding takes its vocabulary
`argmax`, producing `[B]`, and retains a singleton sequence axis to obtain
`[B, 1]` before concatenation.

Softmax is unnecessary for greedy selection because exponentiation and division
by the shared positive normalization term preserve logit ordering. The largest
logit and largest probability therefore have the same token index.

## Naive generation

The naive loop forwards the complete growing sequence, selects one token from
the final position, and appends it. It is a useful correctness reference but
recomputes representations, keys, and values for every earlier token at every
step. Python does not mutate the prompt because `torch.cat` returns a new
tensor. `torch.inference_mode` disables unnecessary autograd bookkeeping.

## Cached generation

Generation begins with a prefill call over the complete prompt. The final
prefill logit predicts the first generated token, while the returned cache
contains prompt keys and values for every decoder layer.

Each later decode call receives only the previously selected `[B, 1]` token and
the current caches. Its output logit predicts the following token, and its
updated caches replace the old ones. If `M` new tokens are requested, prefill
selects token one and only `M - 1` decode calls are needed. Running another
decode after the final selection would compute unused logits and extend caches
unnecessarily.

## Batched EOS handling

Different batch rows can emit EOS at different steps, but ordinary tensors and
per-layer caches remain rectangular. A Boolean `finished` tensor shaped `[B]`
stores one state value per sequence. At token selection it is temporarily viewed
as `[B, 1]` so previously finished rows can be forced to emit EOS while active
rows continue normally.

Keeping the stored state as `[B]` avoids accidental broadcasting. Combining a
`[B, 1]` mask with a `[B]` update right-aligns them as `[B, 1]` and `[1, B]`,
creating an unintended `[B, B]` result. Singleton dimensions should be inserted
only at the operation that needs them.

The loop stops when every row is finished. Until then, computation for finished
rows is wasted and their selected outputs are ignored; production systems can
avoid this through dynamic or continuous batching. EOS ID zero is valid, so
optional-EOS logic must test `eos_token_id is not None`, not its Boolean value.

## Temperature and multinomial sampling

Temperature sampling divides logits by a positive temperature before softmax.
One leaves the distribution unchanged, values below one sharpen it, and values
above one flatten it. As temperature approaches zero from above, a distribution
with a unique maximum approaches greedy selection; zero itself would divide by
zero and must be rejected or handled as a separate mode.

Softmax converts `[B, V]` logits into non-negative rows that sum to one.
`torch.multinomial` then draws one vocabulary index per row and naturally
returns `[B, 1]` long token IDs. A caller-provided `torch.Generator` gives an
isolated, reproducible random stream. The same generator object must be reused
across generation steps so its state advances rather than restarting the same
draw on every token.

Token selection is kept separate from model execution. `sample_next_token`
processes last-position logits only, while `generate_sampled` owns prefill,
cache updates, sequence concatenation, and EOS state. This allows filtering to
be tested without running a decoder and lets the same generation mechanics use
different policies.

## Top-k filtering

Top-k retains exactly the `k` largest logits in each vocabulary row. The
selected indices are scattered into a `[B, V]` Boolean mask and all excluded
logits become negative infinity. Softmax therefore assigns excluded tokens zero
probability. `top_k=1` leaves only the maximum and is equivalent to greedy
selection, while `top_k=V` is equivalent to no filtering.

A compact implementation could softmax and sample only the `[B, K]` selected
values, then gather their original vocabulary IDs. The full mask uses more work
after selection but preserves vocabulary coordinates, making it straightforward
to compose top-k with top-p, banned tokens, and later logit processors.

## Top-p or nucleus filtering

Top-p sorts candidates from most to least probable and keeps the smallest
prefix whose cumulative mass reaches the configured threshold. The candidate
that first crosses the threshold remains eligible; otherwise the retained mass
can be substantially below the requested value and a very small threshold can
remove every token.

The raw `cumulative > top_p` mask marks the crossing token, so its decisions are
shifted one position to the right and the first entry is forced to remain. The
shift source is cloned because source and destination slices overlap. Without a
snapshot, writing an early destination value can change a later source value
and propagate incorrect mask entries.

The sorted removal mask is scattered back through the sorted vocabulary
indices. Removed logits become negative infinity. Since `exp(-inf) = 0`, the
final softmax denominator contains only retained exponential weights. It
preserves their relative ratios while rescaling their total probability to one.

When top-k and top-p are combined, top-k is applied first. Top-p's preliminary
softmax then normalizes the remaining top-k candidates, so the nucleus is chosen
from that restricted distribution.

## Cached sampled generation

Sampled generation uses the same prefill, one-token decode, cache replacement,
EOS forcing, early stopping, and output concatenation as cached greedy
generation. Only `argmax` is replaced by `sample_next_token`, and the same
temperature, filters, and generator are passed at every selection. No model or
RNG work is performed when zero new tokens are requested.

## Verification

Focused tests cover final-position selection, whole-prefix growth in the naive
path, inference mode, input preservation, zero-token generation, exact prefill
and decode call counts, cache-length growth, naive/cached equivalence, real
decoder integration, staggered EOS, EOS ID zero, and EOS already present in the
prompt. Sampling tests cover validation, seeded reference draws, top-k and
top-p boundaries, crossing-token retention, processor composition, unchanged
input logits, cached call shapes, reproducibility, RNG preservation, and
`top_k=1` parity with greedy generation.

## Repetition penalty

A repetition penalty modifies only the logits whose vocabulary IDs occur in a
sequence's history. The multiplicative form must be sign-aware. Dividing a
positive logit by a factor greater than one moves it toward zero and makes it
smaller. The same division would move a negative logit toward zero and make it
larger, increasing its probability. Negative logits must instead be multiplied
by the factor so they become more negative. Both branches therefore lower the
selected logit.

An additive subtraction would also lower logits of either sign, but it defines
a different processor. The sign-aware multiplicative rule is the conventional
repetition-penalty behavior used in this exercise.

For next-token logits shaped `[B, V]` and history IDs shaped `[B, S]`, gathering
along the vocabulary dimension selects `[B, S]` values. The sign-aware rule is
applied only to those values, which are then scattered back at the same token
IDs. This avoids constructing a `[B, S, V]` tensor. A Python set is not suitable
because uniqueness must remain independent per batch row, rows can contain
different numbers of unique IDs, and conversion would leave efficient
device-resident tensor execution.

Repeated occurrences gather the same original logit and calculate the same
penalized value. Scattering those identical values to the same vocabulary
position leaves one transformed value, so this processor is presence-based
rather than frequency-based. A frequency penalty would deliberately use the
occurrence count to increase the effect.

The helper preserves its input by writing into a cloned logit tensor. A factor
of one is an identity and can return immediately, avoiding the clone, gather,
conditional transformation, and scatter. Factors below one would reward rather
than discourage repetition and are rejected by this exercise. Validation must
also reject NaN: an ordered comparison such as "less than one" is false for
NaN, so validation instead checks that the value satisfies "at least one" and
rejects it when that statement is false.

The conventional history includes both the prompt and generated continuation.
The final prefill logit is therefore processed with the prompt IDs before the
first token is selected. Each later cached decode logit is processed with the
complete growing sequence. Although the model only receives the newest token
during cached decoding, the generation loop still owns the complete sequence
and can use it as processor state.

A generated-only variant may be preferable when the answer needs to repeat
important terms from the prompt. It requires remembering the prompt length and
using only the generated suffix as history. That is a policy choice distinct
from the conventional all-context behavior implemented here.

Repetition processing occurs before temperature, top-k, and top-p. Logit
processors first alter candidate scores according to sequence constraints;
temperature and filtering then construct the sampling distribution from those
processed scores. Applying a penalty after filtering could not restore a token
that an earlier filter had already removed and would make processor composition
harder to reason about.

## Repetition-penalty verification

Focused tests cover positive and negative logits, batch-specific histories,
unchanged unrelated tokens, duplicate IDs, caller-input preservation, the
identity factor, invalid and NaN factors, prompt processing during prefill, and
growth of the processed history across cached decoding steps.

## Next

Generation can next add other useful logit processors or proceed to final API
cleanup before speculative decoding.
