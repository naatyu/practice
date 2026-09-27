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

## Exploratory exercise 3: Multi-Head Latent Attention (work in progress)

Study and implement the main ideas behind Multi-Head Latent Attention (MLA):
low-rank query and KV projections, separate non-positional and rotary key
components, and inference-time weight absorption.

This exercise is exploratory. The current implementation is not yet considered
complete until its normal and absorbed paths have focused equivalence tests and
its cache behavior is integrated with the decoder.

### Discussion questions

1. How is MLA's learned low-rank factorization different from adding LoRA
   adapters to a frozen dense model?
2. Which parameter, activation, and KV-cache costs are affected by the query
   and KV low-rank branches?
3. Why is the KV latent normalized between its down and up projections?
4. Why are the non-positional and RoPE key dimensions separated?
5. Why can the RoPE key branch remain shared across heads while the query RoPE
   branch is produced per query head?
6. What tensor shape should an MLA-specific RoPE helper accept, and how does it
   differ from the repository's standard attention layout?
7. Which dimensions must be contiguous before a `view` can merge them?
8. What is weight absorption, and why can it avoid reconstructing full keys and
   values during inference?
9. What information must an MLA KV cache retain?
10. Which equivalence, shape, dtype, causality, and cached-decoding tests are
    required before the implementation can be considered complete?
