# Multi-head attention interview practice

This file contains only the exercise prompt and interview questions. Completed
reasoning and corrections are kept separately in `SOLUTIONS.md`.

## Exercise

Implement multi-head self-attention in PyTorch without using
`torch.nn.MultiheadAttention` or PyTorch's built-in scaled dot-product
attention. Support optional causal masking and attention dropout.

### Discussion questions

1. What operations does multi-head attention add around scaled dot-product
   attention?
2. Why are multiple heads useful?
3. What shapes do the fused QKV projection and output projection parameters
   have?
4. Why must `d_model` be divisible by `num_heads` in this implementation?
5. What tensor-contiguity issue can appear when recombining transposed heads?
6. Which operations separate fused QKV and split the model dimension into
   heads, and why are they different?
7. Which invariants belong in the constructor rather than the forward pass?
8. Where does attention dropout belong, and how does the module's training
   state control it?
9. What equivalent strategies can separate fused QKV features and expose the
   head dimension?
