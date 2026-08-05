# FlashAttention interview practice

This file contains only exercise prompts and interview questions. Completed
reasoning and corrections are kept separately in `SOLUTIONS.md`.

## Exercise: FlashAttention in PyTorch

Implement the core FlashAttention algorithm in plain PyTorch. Do not use
PyTorch's built-in scaled dot-product attention or a fused FlashAttention
kernel.

The goal is to demonstrate the algorithm. A plain-PyTorch implementation is
not expected to match the performance of a fused CUDA or Triton kernel.

### Discussion questions

1. If FlashAttention still performs quadratic computation, what bottleneck is
   it designed to improve, and how?
2. Why can the score matrix not simply be split into blocks and softmax applied
   independently to each block?
3. How is softmax computed in a numerically stable way for one complete row?
4. When a new score block arrives, how should the running row maximum be
   updated?
5. If the running exponential sum was expressed relative to the old maximum,
   how must it be rescaled when the maximum changes?
6. What happens to the rescaling factor when the new block does or does not
   contain a larger maximum?
7. Besides the running maximum and normalization sum, what quantity must be
   accumulated to produce the final attention output without storing
   probabilities?
8. When the running maximum increases, how must the old weighted-value
   accumulator be adjusted?
9. What are the complete per-block updates for the running maximum,
   normalization sum, and weighted-value accumulator?
