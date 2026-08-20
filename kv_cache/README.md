# KV-cache interview practice

This file contains the exercise prompt and interview questions only. Completed
reasoning is kept separately in `SOLUTIONS.md`; the implementation remains in
the attention, multi-head attention, transformer-block, and decoder modules.

## Exercise: Autoregressive KV caching

Extend the decoder-only Transformer so autoregressive inference can reuse keys
and values computed for previous tokens.

The ordinary training and full-sequence APIs must continue to return tensors.
When caching is requested, each relevant module must additionally return its
updated cache.

### Required behavior

- Support causal attention when query and key sequence lengths differ.
- Cache one K tensor and one V tensor per transformer layer.
- Apply RoPE to new queries and keys at their absolute positions.
- Store rotated keys and unrotated values.
- During prefill, process the complete prompt and construct every layer cache.
- During decode, process only the new token or token chunk.
- Preserve equivalence with a full causal forward pass in evaluation mode.
- Reject a decoder cache list whose length differs from the number of layers.

### Discussion questions

1. Why are K and V cached while previous Q tensors are discarded?
2. What is the difference between prefill and decode?
3. For a past length `P` and new chunk length `Q`, what are the shapes of the
   new queries, complete keys and values, scores, and outputs?
4. Why does each transformer layer require its own cache?
5. Along which tensor dimension are new keys and values appended?
6. When should RoPE be applied relative to cache insertion?
7. How can the RoPE offset be inferred from the cache?
8. Why is a standard `Q x Q` lower-triangular mask insufficient when cached
   keys are present?
9. How should the causal mask change for `P` cached positions and `Q` new
   queries?
10. Why should the generation loop own the cache rather than the model storing
    mutable generation state?
11. Why does repeated `torch.cat` remain inefficient?
12. What is the KV-cache memory formula for standard multi-head attention?
13. Which tests distinguish a working cache from an implementation that merely
    returns tensors with plausible shapes?

