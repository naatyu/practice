# KV-cache interview practice: completed reasoning

## What is cached and why

For a new query, all previous keys are still needed as possible matches and all
previous values are still needed as payloads for the weighted sum. Previous
queries produced outputs for earlier positions and are never used again, so
only K and V are cached.

Prefill processes the prompt in parallel and creates the initial caches.
Decode then supplies only the newest sampled token (or a small new chunk). Its
representation becomes contextualized independently at every layer by
attending to that layer's cached K/V tensors.

For past length `P` and new chunk length `Q`:

```text
new Q:       (B, num_heads, Q,     d_head)
new K/V:     (B, num_heads, Q,     d_head)
complete K/V:(B, num_heads, P + Q, d_head)
scores:      (B, num_heads, Q,     P + Q)
output:      (B, num_heads, Q,     d_head)
```

Each layer requires its own `(K, V)` pair because every layer receives a
different hidden representation and owns different projection weights. A
decoder cache is therefore a list of length `n_layers`, with one pair per list
entry. New tensors are appended along the sequence dimension, `dim=-2`.

## RoPE and cached positions

New Q and new K are rotated before cache insertion. Rotated K is cached so past
keys never need to be rotated again; V is not rotated. If the past K cache has
length `P`, the first new token has absolute position `P`, so the RoPE offset
can be inferred from `past_k.shape[-2]`. This makes the cache the source of
truth and avoids a separate caller-provided offset becoming inconsistent.

## Causal masking with unequal lengths

With `P` cached tokens and `Q` new queries, the complete key length is `P + Q`.
New query row `i` has absolute position `P + i`, so key column `j` is valid
when:

```text
j <= P + i
```

A boolean mask of shape `(Q, P + Q)` can be constructed as a lower triangle
with diagonal offset `P`. For single-token decode, the only new query may
attend to every cached token and itself. Constructing an ordinary `(1, P + 1)`
lower triangle with diagonal zero would incorrectly expose only the first key.
Masking occurs before softmax by adding or filling invalid scores with negative
infinity; multiplication by a zero/one mask would turn invalid logits into
zero, which can still receive nonzero softmax probability.

## API ownership and module propagation

The generation loop owns the cache and passes it to the model. This keeps the
model stateless, prevents state leaking between requests, supports concurrent
sequences and beam reordering, and makes cache lifetime explicit.

Multi-head attention returns either its output tensor or the output plus its
updated `(K, V)` pair. A transformer block adds only the attention output to
the residual and forwards the cache separately. The decoder gives layer `i`
only cache `i`, collects the updated pairs, and returns either logits or logits
plus the updated cache list.

## Correctness tests

Shape-only tests are insufficient. The strongest behavioral test compares a
single full causal forward pass with cached execution using the same model in
evaluation mode:

1. Prefill a prompt and retain every layer cache.
2. Feed the remaining tokens one at a time while carrying updated caches.
3. Concatenate the cached logits.
4. Compare every logit with the full-forward result.

This catches incorrect RoPE offsets, caching unrotated keys, concatenating on
the wrong dimension, incorrect unequal-length masks, mixing caches between
layers, broken residual paths, and failure to update the returned caches.

## Memory cost and allocation strategy

For standard multi-head attention, total KV-cache storage in bytes is:

```text
2 * B * n_layers * num_heads * sequence_length * d_head * bytes_per_element
```

The factor two accounts for K and V. Since
`num_heads * d_head = d_model`, this is equivalently:

```text
2 * B * n_layers * sequence_length * d_model * bytes_per_element
```

Repeated `torch.cat` is simple but allocates new storage and copies the entire
past cache at every decoding step, creating cumulative quadratic copy traffic.
A static cache instead preallocates capacity such as
`(B, num_heads, max_seq_len, d_head)` and writes new K/V values into the slice
at the current cache position. It avoids repeated allocation and copying but
reserves maximum capacity upfront. Paged caches allocate smaller blocks to
reduce waste for variable-length requests.

