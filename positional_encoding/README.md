# Positional encoding interview practice

This file contains only the exercise prompt and interview questions. Completed
answers and explanations are kept separately in `SOLUTIONS.md`.

## Exercise 1: Learned absolute positional encoding

Implement learned absolute positional encoding for Transformer activations.

```text
input:  (batch_size, sequence_length, d_model)
output: (batch_size, sequence_length, d_model)
```

Requirements:

- Store a learned table of shape `(max_seq_len, d_model)`.
- Use `nn.Embedding` to look up positions `0 ... sequence_length - 1`.
- Share the selected positional vectors across the batch through broadcasting.
- Reject sequences longer than `max_seq_len`.
- Preserve the input shape.

### Discussion questions

1. What are the shapes of the token-embedding and positional-embedding tables?
2. How do their lookup results broadcast before addition?
3. Why does self-attention need positional information?
4. What happens when inference receives a sequence longer than the learned
   table?
5. What is the relationship between `nn.Embedding` and `nn.Parameter`?
6. How can a controlled test verify position ordering and batch broadcasting?
7. Why should test weights be copied inside `torch.no_grad()`?

## Exercise 2: Sinusoidal positional encoding

Implement deterministic sinusoidal positional encoding and add it to
Transformer activations.

```text
input:  (batch_size, sequence_length, d_model)
table:  (max_seq_len, d_model)
output: (batch_size, sequence_length, d_model)
```

Requirements:

- Require a positive, even `d_model`.
- Use interleaved sine values at even feature indices and cosine values at odd
  feature indices.
- Use a configurable maximum sequence length and frequency base.
- Precompute the table as a registered buffer, not a parameter.
- Preserve the input shape and dtype.
- Reject sequences longer than `max_seq_len`.

### Discussion questions

1. How does sinusoidal encoding differ from learned positional embeddings?
2. Why are multiple frequencies used across the feature pairs?
3. What are the sine and cosine formulas for pair index `i`?
4. How do the angle-addition identities expose relative offsets?
5. Why should the table be a registered buffer?
6. What shapes should the position, frequency, and angle tensors have?
7. What does vectorization mean in this table construction?
8. What should the complete encoding at position zero contain?
9. How is additive sinusoidal encoding different from RoPE?

## Exercise 3: Rotary positional encoding (RoPE)

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
