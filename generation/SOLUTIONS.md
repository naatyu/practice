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

## Verification

Focused tests cover final-position selection, whole-prefix growth in the naive
path, inference mode, input preservation, zero-token generation, exact prefill
and decode call counts, cache-length growth, naive/cached equivalence, real
decoder integration, staggered EOS, EOS ID zero, and EOS already present in the
prompt.

## Next: stochastic sampling

Temperature changes distribution sharpness before token sampling. Values below
one sharpen the distribution, values above one flatten it, and zero must be
rejected or treated as a separate greedy mode. A seeded `torch.Generator` makes
otherwise random draws reproducible without relying on global RNG state.
