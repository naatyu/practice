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

## Current checkpoint

Completed:

- Naive greedy generation.
- KV-cached greedy generation.
- Batched EOS handling and early stopping.
- Controlled fake-model tests and real-decoder equivalence.

Next:

1. Temperature sampling.
2. Top-k filtering.
3. Top-p/nucleus filtering.
4. Sampling integration with cached generation.
5. Repetition penalties.
