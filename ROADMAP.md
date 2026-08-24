# LLM interview practice roadmap

## Completed foundations

### 1. Scaled dot-product attention

- Bidirectional and causal attention
- Attention dropout
- Score scaling and numerical stability
- Tensor shapes and complexity
- Reference tests against PyTorch

### 2. Multi-head attention

- Fused QKV projection
- Splitting and recombining heads
- Output projection
- Training and evaluation behavior
- Reference tests against PyTorch

### 3. Normalization

- RMSNorm
- Comparison with LayerNorm
- Broadcasting and learned feature scaling

### 4. Transformer MLPs

- GELU MLP
- SwiGLU
- Fused value and gate projections
- Parameter-matched hidden dimensions

### 5. Transformer block

- Pre-norm architecture
- Residual connections
- Causal self-attention
- Position-wise SwiGLU feed-forward network

### 6. Positional encoding

- Learned absolute positional encoding
- Sinusoidal positional encoding
- Rotary positional encoding (RoPE)
- Relative-position derivation
- Position offsets for future cached decoding

### 7. Decoder-only language model

- Token embeddings
- Stacked Transformer blocks
- Final RMSNorm
- Vocabulary logits
- Weight tying
- Shifted next-token loss
- RoPE integration and offset propagation

### 8. Stable cross-entropy

- Softmax, log-softmax, negative log-likelihood, and cross-entropy
- The log-sum-exp trick for numerical stability
- Implementing cross-entropy directly from logits without applying softmax
- Selecting the target-class logit with `gather`
- Mean, sum, and per-token loss reductions
- Padding masks and `ignore_index`
- Comparison with `torch.nn.functional.cross_entropy`
- Tests with extreme logits that would overflow a naive implementation
- Value and gradient comparison with PyTorch
- All-ignored targets and input-immutability tests

### 9. Basic language-model training

- Shifted causal language-model logits and targets
- Autograd and the purpose of `loss.backward()`
- A minimal AdamW training loop
- `zero_grad`, forward pass, loss, backward pass, and optimizer step
- Training versus evaluation mode
- Gradient accumulation and gradient clipping
- Overfitting a tiny batch as an end-to-end correctness test

### 10. KV caching

- Prefill versus decoding
- Per-layer cache shapes
- Appending new K and V tensors
- RoPE offsets inferred from cache length
- Causal masking with unequal query and key lengths
- Full-sequence versus token-by-token cached-logit equivalence
- Cache memory cost
- Dynamic concatenation versus static and paged caches

### 11. Multi-query and grouped-query attention

- Separate query-head and KV-head counts
- MHA, GQA, and MQA as points on one design spectrum
- Unequal fused QKV projection shapes
- Mapping query heads to compact KV heads with grouped broadcasting
- KV-cache and projection savings
- Quality, throughput, compute, and memory tradeoffs
- End-to-end cached decoder equivalence for MHA, GQA, and MQA

## Next core sequence

### 12. Byte-level BPE (completed)

- Bytes versus Unicode characters
- Initial byte vocabulary
- Unicode-aware regex pre-tokenization and merge boundaries
- Pair-frequency counting
- Deterministic merge training
- Encoding and decoding
- Merge ordering
- Special tokens
- Portable JSON save/load
- Round-trip tests
- Efficient implementation strategies

### 13. Generation

- Greedy decoding
- Temperature
- Top-k sampling
- Top-p sampling
- EOS handling
- Repetition penalties
- Batched generation
- Integration with KV caching

### 14. Speculative decoding

- Draft and target model roles
- Proposing multiple tokens with the draft model
- Verifying proposed tokens in one target-model forward pass
- Acceptance probability for each draft token
- Rejection sampling from the corrected residual distribution
- Emitting the extra target token when every proposal is accepted
- Why speculative decoding preserves the target model's distribution
- Greedy speculative decoding as a simpler special case
- KV-cache advancement, rollback, and synchronization
- Expected speedup and when verification overhead removes the benefit
- Tests against ordinary target-model decoding

### 15. FlashAttention

- Memory traffic versus arithmetic complexity
- Query, key, and value tiling
- Numerically stable online softmax
- Running maximum and normalization sum
- Running weighted-value accumulator
- Causal block handling
- Comparison with ordinary attention

### 16. FLOP, parameter, and memory accounting

- Treat performance accounting as a recurring part of every layer exercise
- FLOPs for linear projections, including multiply-add conventions
- QKV and output-projection FLOPs
- Attention score and probability-value matrix-multiplication FLOPs
- GELU MLP and SwiGLU parameter and FLOP counts
- Normalization, activation, and elementwise-operation costs
- Vocabulary-projection and cross-entropy costs
- Forward-pass versus forward-and-backward training FLOPs
- FLOPs per token and per sequence for a complete decoder
- Which terms scale linearly or quadratically with sequence length
- Prefill versus autoregressive decode FLOPs
- How KV caching changes computation without changing model parameters
- How MHA, GQA, and MQA change parameters, cache memory, and computation
- Activation and KV-cache memory estimates by dtype
- Arithmetic intensity, memory bandwidth, and why equal FLOPs can have
  different runtimes
- Comparing hand calculations with profiler-reported operator FLOPs

### 17. Long-context RoPE scaling and YaRN

- Why base RoPE can degrade beyond its training context length
- Position interpolation and RoPE frequency scaling
- NTK-aware and frequency-dependent scaling
- YaRN (Yet another RoPE extensioN)
- Interpolation versus extrapolation across frequency bands
- Attention scaling for extended contexts
- Updating cached sine and cosine tables
- Interaction with position offsets and KV caching
- Preserving short-context behavior while extending context length
- Tests and evaluations across original and extended context lengths

## Systems and scaling

### 18. Mixed precision and numerical stability

- FP32, FP16, and BF16
- Loss scaling
- Accumulation precision
- Stable softmax and normalization
- Hardware-friendly matrix dimensions

### 19. Quantization

- Weight-only versus weight-and-activation quantization
- Per-tensor, per-channel, and group-wise scales
- INT8 and INT4
- GPTQ and AWQ concepts
- KV-cache quantization

### 20. Distributed training

- Data parallelism
- Tensor parallelism
- Pipeline parallelism
- Sequence and context parallelism
- ZeRO and FSDP
- Computation and communication costs

### 21. Efficient inference

- Continuous batching
- Paged KV caches
- Prefix caching
- Decode-time memory-bandwidth bottlenecks

## Advanced architectures and adaptation

### 22. Mixture of Experts

- Top-k routing
- Expert capacity
- Load-balancing losses
- Expert parallelism

### 23. Fine-tuning

- LoRA
- QLoRA
- Supervised fine-tuning
- Preference optimization fundamentals

### 24. Extended training and evaluation

- Perplexity and evaluation aggregation
- Padding and document-boundary masks
- Optimizers and learning-rate schedules
- Weight initialization and residual scaling
- Regularization and label smoothing
- Checkpointing optimizer and random-number-generator state
- Data leakage and evaluation design

## Immediate path

```text
byte-level BPE -> generation -> speculative decoding
-> FlashAttention -> FLOP and memory accounting -> YaRN
```

The order is intentional: the completed training loop proves that the decoder
and stable loss can learn together; the completed KV-cache exercise exposes
the main autoregressive inference memory problem; the completed MQA/GQA work
reduces that cost;
tokenization completes the input side of the model; generation combines the
decoder, tokenizer, and cache; speculative decoding builds on generation and
cache management to accelerate sampling without changing the target
distribution; and FlashAttention deepens the analysis of attention performance
during training and prefill. FLOP and memory accounting then consolidates the
implemented components into quantitative model, training, prefill, and decode
cost estimates.
YaRN then extends the completed RoPE work into long-context scaling and
evaluation.
