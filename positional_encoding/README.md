# Rotary positional encoding interview practice

This file contains only the exercise prompt and interview questions. Completed
answers and explanations are kept separately in `SOLUTIONS.md`.

## Exercise: Rotary positional encoding (RoPE)

Implement rotary positional encoding in PyTorch for query and key tensors.

```text
q, k: (batch_size, num_heads, sequence_length, head_dim)
```

Requirements:

- Require a positive, even `head_dim`.
- Use a configurable maximum sequence length and frequency base.
- Precompute the sine and cosine tables as module buffers.
- Apply rotations to adjacent feature pairs without constructing rotation
  matrices.
- Rotate queries and keys, but not values.
- Preserve tensor shape and dtype.
- Support a sequence offset for autoregressive generation with a KV cache.
- Reject positions beyond the configured maximum sequence length.

Do not use an existing RoPE implementation.

### Discussion questions

1. What is the two-dimensional rotation matrix, and what are the resulting
   coordinate equations?
2. How can all feature pairs be rotated without constructing block-diagonal
   rotation matrices or looping over tokens?
3. What shape should the angle table have so it broadcasts across batches and
   attention heads?
4. Why is the frequency schedule based on `head_dim` rather than `d_model`?
5. Why does standard RoPE rotate queries and keys but not values?
6. How do absolute rotations of queries and keys make their dot product depend
   on relative position?
7. Why must cached sine and cosine values be cast when Q and K use `float16` or
   `bfloat16`?
8. Why is a sequence offset needed during autoregressive generation with a KV
   cache?
9. When should keys be rotated relative to insertion into the KV cache?
10. Which vector norms and dot products are preserved by the rotations, and
    which attention scores change?
11. How would you test the implementation under interview time constraints?

