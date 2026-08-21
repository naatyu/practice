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

## Exercise 2: Grouped-query and multi-query attention

Extend multi-head attention with a configurable number of KV heads while
keeping the original number of query heads. The same implementation should
support MHA, GQA, and MQA, including RoPE and compact KV caching.

### Discussion questions

1. How do MHA, GQA, and MQA relate through `num_heads` and `num_kv_heads`?
2. What divisibility constraint makes equal query groups possible?
3. What are the fused Q, K, and V projection widths?
4. How can unequal fused outputs be separated with `torch.split`?
5. How are query heads mapped to shared KV heads?
6. Why must the cache retain compact KV heads rather than expanded copies?
7. How can broadcasting avoid materializing repeated K/V tensors?
8. Which projection, cache, bandwidth, and attention-matrix costs decrease?
9. Why do the dominant attention matrix multiplications remain approximately
   unchanged?
10. Why can GQA retain MHA-like quality more reliably than MQA?
11. Which tests verify that the complete decoder actually uses compact GQA or
    MQA caches?
