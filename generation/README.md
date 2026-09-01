# Autoregressive generation

Build generation on top of the decoder and its KV-cache API, starting with
greedy decoding and progressing toward stochastic sampling.

## Exercise 1: Greedy decoding

Implement two functions:

```python
generate_greedy_naive(...)
generate_greedy(...)
```

The naive version forwards the complete growing sequence on every step. The
efficient version performs one prompt prefill and then passes one newly selected
token at a time with the updated per-layer KV caches.

Required behavior:

- Accept batched token IDs shaped `[B, S]`.
- Return the prompt followed by at most `max_new_tokens` generated IDs.
- Select directly from the last-position logits without unnecessary softmax.
- Run with gradient tracking disabled.
- Preserve the caller's input tensor.
- Optionally stop on an EOS token.
- Keep finished batch rows rectangular by repeatedly emitting EOS until every
  row finishes.
- Stop early when all rows finish.
- Produce identical naive and cached outputs in evaluation mode.

### Discussion questions

1. Which `[B, S, V]` logit slice predicts the token after the complete prompt?
2. Why does greedy decoding not require softmax?
3. What shapes do selected token IDs need before concatenation?
4. Why is repeatedly forwarding the complete sequence wasteful?
5. What is the distinction between prefill and decode?
6. Which logits predict the first generated token?
7. Why are there only `max_new_tokens - 1` decode calls after prefill selection?
8. Why must caches be replaced with their updated values on every step?
9. Why can a batched loop not stop when only one row emits EOS?
10. Why is completion state naturally shaped `[B]` rather than `[B, 1]`?
11. Why must checks use `eos_token_id is not None` instead of its truth value?

## Exercise 2: Stochastic sampling

Implement a vocabulary-level selection helper and integrate it with cached
generation:

```python
sample_next_token(...)
generate_sampled(...)
```

Required behavior:

- Accept last-position logits shaped `[B, V]`.
- Apply positive temperature scaling.
- Optionally retain exactly the top `k` vocabulary candidates.
- Optionally retain the smallest high-probability nucleus whose cumulative
  mass reaches `top_p`.
- Keep the token that crosses the top-p threshold.
- Renormalize retained candidates before multinomial sampling.
- Draw one `[B, 1]` token-ID tensor with an optional `torch.Generator`.
- Keep filtering in full vocabulary order so processors compose naturally.
- Reuse one advancing generator throughout cached generation.
- Preserve all greedy cache, EOS, and sequence-shape behavior.

### Discussion questions

1. How does temperature below or above one change distribution sharpness?
2. Why is temperature zero invalid in the sampling formula?
3. Why does `torch.multinomial` require probabilities or non-negative weights
   rather than raw logits?
4. Why does a seeded `torch.Generator` improve reproducibility?
5. Why is top-k based on the largest logits?
6. Why does `top_k=1` reduce sampling to greedy selection?
7. What is the tradeoff between sampling compact `[B, K]` values and masking a
   vocabulary-aligned `[B, V]` tensor?
8. Why does top-p sort tokens before computing cumulative probability?
9. Why must the token that crosses `top_p` remain eligible?
10. Why does shifting an overlapping tensor slice require `clone()`?
11. How does the final softmax renormalize the retained probability mass?
12. When both filters are enabled, why does applying top-k before top-p make
    top-p operate on the top-k distribution?
13. Which parts of cached generation are independent of the token-selection
    policy?

## Current checkpoint

Completed:

- Naive greedy generation.
- KV-cached greedy generation.
- Batched EOS handling and early stopping.
- Controlled fake-model tests and real-decoder equivalence.
- Temperature sampling.
- Top-k filtering.
- Top-p/nucleus filtering.
- Seeded and reproducible multinomial sampling.
- KV-cached sampled generation with EOS handling.
- Greedy equivalence when `top_k=1`.

Next:

1. Repetition penalties.
2. Additional logit processors where useful.
3. Final generation API cleanup and documentation.
