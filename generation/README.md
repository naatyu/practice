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

## Exercise 3: Repetition penalty

Add a repetition-penalty processor to sampled generation. Before temperature
scaling and probability filtering, adjust the logits of vocabulary tokens that
already occur in the current sequence.

Required behavior:

- Accept next-token logits shaped `[B, V]` and token histories shaped `[B, S]`.
- Use a multiplicative penalty greater than or equal to one.
- Make repeated tokens less likely whether their logits are positive or
  negative.
- Apply the penalty once based on token presence, not repeatedly based on the
  number of occurrences.
- Keep unrelated vocabulary logits unchanged and preserve the caller's input
  logits.
- Apply the penalty to prompt tokens for the first post-prefill selection.
- Apply it to the complete prompt-plus-generated history during decoding.
- Treat one as an identity fast path and reject invalid or NaN penalties.
- Compose the processor before temperature, top-k, and top-p sampling.

### Discussion questions

1. Why does dividing every repeated-token logit by the penalty fail for
   negative logits?
2. Why does multiplying every repeated-token logit by the penalty fail for
   positive logits?
3. How does the sign-aware transformation ensure that both cases become less
   likely?
4. How do `gather` and `scatter` operate on `[B, V]` logits using `[B, S]`
   token IDs without constructing a `[B, S, V]` tensor?
5. Why do duplicate history IDs not cause the multiplicative penalty to be
   applied repeatedly?
6. Why is a Python `set` unsuitable for batched, device-resident token IDs?
7. What is the difference between a presence-based repetition penalty and a
   frequency-based penalty?
8. Why does the conventional policy include prompt tokens, and when might a
   generated-only policy be preferable?
9. Which history is available for the first post-prefill selection and for
   later cached decode selections?
10. Why should logit processors run before top-k and top-p filtering?
11. Why does `penalty < 1` fail to reject NaN, while testing whether the
    penalty is not at least one succeeds?
12. What work is avoided by returning immediately when the penalty is one?

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
- Sign-aware repetition penalties over prompt-plus-generated history.
- Batched gather/scatter processing with duplicate-token and input-preservation
  coverage.

Next:

1. Additional logit processors where useful.
2. Final generation API cleanup and documentation.
