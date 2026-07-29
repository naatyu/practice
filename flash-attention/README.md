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
