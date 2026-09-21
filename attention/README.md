# Attention interview practice

This file contains only exercise prompts and interview questions. Completed
answers and explanations are kept separately in `SOLUTIONS.md`.

## Exercise 1: Bidirectional scaled dot-product attention

Implement bidirectional scaled dot-product attention in PyTorch.

```text
q, k, v: (batch_size, num_heads, sequence_length, head_dim)
```

Do not use `torch.nn.functional.scaled_dot_product_attention`.

### Discussion questions

1. What does attention compute?
2. What is the shape of the token-to-token attention matrix?
3. How do `Q` and `K` produce that matrix? Which dimensions are transposed?
4. What is the difference between the attention matrix and the final output?
5. Why are the attention scores scaled?
6. Why does standard attention use softmax?
7. Which dimension does softmax normalize, and why?
8. What are the output shape and its intuitive meaning?
9. What are the time and memory complexities?
10. How would you test the implementation?

## Exercise 2: Causal attention

Extend the attention function to optionally support causal attention while
preserving bidirectional attention as the default.

### Discussion questions

1. What does causal attention change in the attention matrix?
2. Why is causal attention needed in autoregressive language models?
3. At what point in the computation should the causal mask be applied?

## Exercise 3: Attention dropout

Add optional attention dropout while preserving deterministic behavior by
default.

### Discussion questions

1. At what point should attention dropout be applied?
2. How should its behavior differ between training and evaluation?
3. Do the sampled attention weights still sum to one after dropout?

## Exercise 4: Dense causal local attention

Implement causal local attention in `local_attention.py`. Define
`window_size` as the maximum number of keys visible to one query, including
the query's own position.

Begin with a correctness-oriented reference implementation that constructs the
complete `[Q, K]` mask and `[B, H, Q, K]` score tensor.

Required behavior:

- `window_size=1` allows each query to attend only to itself.
- A sufficiently large window is equivalent to ordinary causal attention.
- A query may not attend to future keys or keys older than its window.
- Cached suffix queries with `Q < K` use their positions in the complete key
  sequence.
- Non-positive window sizes and `Q > K` are rejected.

### Discussion questions

1. For query position `i`, which key positions are valid for window size `W`?
2. Why does using `i-W` as the lower boundary introduce an off-by-one error?
3. If there are `Q` queries and `K` keys, what absolute position corresponds
   to query row `r`?
4. Which two Boolean comparisons form a causal local mask?
5. Why does masking a dense score matrix not reduce quadratic computation or
   storage?

## Exercise 5: Compact sliding-window attention

Implement an equivalent version that computes only the `W` candidate scores
for each query. Do not construct a `[Q, K]` mask or `[B, H, Q, K]` score
matrix.

Build overlapping K/V windows using left padding and `Tensor.unfold`. Use
matrix contractions for score computation and value aggregation rather than
materializing elementwise products with an extra feature dimension.

Required behavior:

- K windows have shape `[B, H, Q, W, D]`.
- V windows have shape `[B, H, Q, W, Dv]`.
- Scores and probabilities have shape `[B, H, Q, W]`.
- Padding positions are excluded before softmax.
- Cached calls discard old K/V positions that no supplied query can reach.
- The compact implementation matches the dense reference for full and cached
  inputs.
- Attention dropout remains available to model callers.

### Discussion questions

1. Why must early queries be left-padded before fixed-size windows can be
   constructed?
2. What shape does `unfold` return, and why are its final dimensions swapped?
3. Why do `Q` overlapping windows need only `Q + W - 1` source positions but
   still perform `Q * W` query-key interactions?
4. How can cached attention select only the K/V range reachable by its `Q`
   queries?
5. Why can `matmul` or `einsum` avoid the `[B, H, Q, W, D]` temporary created
   by elementwise multiplication followed by a sum?
6. What matrix multiplication combines `[B, H, Q, W]` probabilities with
   `[B, H, Q, W, Dv]` value windows?
7. Why can this plain-PyTorch implementation be slower than dense attention
   despite its better asymptotic complexity?

## Exercise 6: Bounded local-attention KV cache

Integrate local attention into multi-head attention, Transformer blocks, and
the decoder. Retain only the newest `W` entries in each layer's K/V cache.

Introduce decoder-level cache metadata that tracks the absolute position of
the next token independently of the bounded tensor length. Use that position
as the RoPE offset for every layer.

### Discussion questions

1. After appending new K/V entries, which entries should remain in a cache of
   size `W`?
2. Why can cache tensor length no longer determine the RoPE offset?
3. Why should the absolute position be stored once at decoder level instead
   of duplicated in every layer cache?
4. When should trimming occur during a multi-token prefill?
5. Which state changes when tokens are evicted, and which state must continue
   increasing?
6. Why should a direct bounded-cache MHA call reject a missing absolute
   position instead of silently using cache length?

## Bonus: small performance check

Classic full causal attention, dense masked local attention, and compact local
attention were benchmarked on identical Q/K/V inputs. These are measurements
of the handwritten PyTorch implementations, not fused-kernel benchmarks.

### CPU

CPU timings use `torch.utils.benchmark.Timer.blocked_autorange`.

```text
CPU:     Intel Core i7-14700HX, one benchmark thread
PyTorch: 2.13.0
Inputs:  B=2, H=4, D=32, float32

S       W       classic       dense local    compact local   vs. classic
1024    32      51.46 ms      54.43 ms       27.91 ms        1.84x
1024    64      52.15 ms      53.87 ms       56.25 ms        0.93x
2048    32      212.93 ms     221.04 ms      58.38 ms        3.65x
2048    64      210.32 ms     217.06 ms      111.09 ms       1.89x
```

### GPU

GPU timings are medians of 40 CUDA-event measurements after 15 warm-up calls.
An isolated CUDA 12.8 environment was used because the repository's CUDA 13.0
PyTorch build requires a newer driver than the machine currently provides.

```text
GPU:     NVIDIA RTX 4000 Ada Generation Laptop GPU
PyTorch: 2.8.0+cu128
Inputs:  B=1, H=16, D=64, bfloat16

S       W       classic      dense local   compact local   vs. classic
512     64      0.121 ms     0.139 ms      1.177 ms        0.10x
1024    64      0.653 ms     0.713 ms      2.146 ms        0.30x
2048    64      3.162 ms     3.167 ms      4.183 ms        0.76x
2048    128     2.979 ms     3.046 ms      8.093 ms        0.37x
4096    64      13.772 ms    13.748 ms     8.543 ms        1.61x
```

The compact implementation becomes useful when the sequence is long and the
window is small relative to it. It is not automatically faster. On this GPU,
the dense matrix multiplication wins through sequence length 2,048; compact
attention crosses over at `S=4096, W=64`. Padding, strided windows, and many
small batched matrix multiplications consume enough time to offset the reduced
arithmetic at smaller sizes and wider windows.

The benchmark supports the asymptotic argument without turning it into a
universal runtime claim. Results will change with shapes, thread count,
hardware, and kernels. A fused sliding-window CUDA or Triton implementation
would be the appropriate comparison for production performance.
