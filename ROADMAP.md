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

## Next core sequence

### 8. Stable cross-entropy and basic training

- Softmax, log-softmax, negative log-likelihood, and cross-entropy
- The log-sum-exp trick for numerical stability
- Implementing cross-entropy directly from logits without applying softmax
- Selecting the target-class logit with `gather`
- Mean, sum, and per-token loss reductions
- Shifted causal language-model logits and targets
- Padding masks and `ignore_index`
- Comparison with `torch.nn.functional.cross_entropy`
- Tests with extreme logits that would overflow a naive implementation
- Autograd and the purpose of `loss.backward()`
- A minimal AdamW training loop
- `zero_grad`, forward pass, loss, backward pass, and optimizer step
- Training versus evaluation mode
- Gradient accumulation and gradient clipping
- Overfitting a tiny batch as an end-to-end correctness test

### 9. KV caching

- Prefill versus decoding
- Per-layer cache shapes
- Appending new K and V tensors
- RoPE offsets
- Causal masking with unequal query and key lengths
- Full-sequence versus cached-logit equivalence
- Cache memory cost

### 10. Multi-query and grouped-query attention

- Separate query-head and KV-head counts
- MHA, GQA, and MQA as points on one design spectrum
- Projection parameter shapes
- Mapping query heads to KV heads
- KV-cache memory savings
- Quality, throughput, and memory tradeoffs

### 11. Byte-level BPE

- Bytes versus Unicode characters
- Initial byte vocabulary
- Pair-frequency counting
- Deterministic merge training
- Encoding and decoding
- Merge ordering
- Special tokens
- Round-trip tests
- Efficient implementation strategies

### 12. Generation

- Greedy decoding
- Temperature
- Top-k sampling
- Top-p sampling
- EOS handling
- Repetition penalties
- Batched generation
- Integration with KV caching

### 13. Speculative decoding

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

### 14. FlashAttention

- Memory traffic versus arithmetic complexity
- Query, key, and value tiling
- Numerically stable online softmax
- Running maximum and normalization sum
- Running weighted-value accumulator
- Causal block handling
- Comparison with ordinary attention

## Systems and scaling

### 15. Mixed precision and numerical stability

- FP32, FP16, and BF16
- Loss scaling
- Accumulation precision
- Stable softmax and normalization
- Hardware-friendly matrix dimensions

### 16. Quantization

- Weight-only versus weight-and-activation quantization
- Per-tensor, per-channel, and group-wise scales
- INT8 and INT4
- GPTQ and AWQ concepts
- KV-cache quantization

### 17. Distributed training

- Data parallelism
- Tensor parallelism
- Pipeline parallelism
- Sequence and context parallelism
- ZeRO and FSDP
- Computation and communication costs

### 18. Efficient inference

- Continuous batching
- Paged KV caches
- Prefix caching
- Decode-time memory-bandwidth bottlenecks

## Advanced architectures and adaptation

### 19. Mixture of Experts

- Top-k routing
- Expert capacity
- Load-balancing losses
- Expert parallelism

### 20. Fine-tuning

- LoRA
- QLoRA
- Supervised fine-tuning
- Preference optimization fundamentals

### 21. Extended training and evaluation

- Perplexity and evaluation aggregation
- Padding and document-boundary masks
- Optimizers and learning-rate schedules
- Weight initialization and residual scaling
- Regularization and label smoothing
- Checkpointing optimizer and random-number-generator state
- Data leakage and evaluation design

## Immediate path

```text
stable cross-entropy and training -> KV cache -> MQA/GQA -> byte-level BPE
-> generation -> speculative decoding -> FlashAttention
```

The order is intentional: stable loss and a basic training loop first prove
that the decoder can learn; KV caching then exposes the main autoregressive
inference memory problem; MQA and GQA reduce that cost; tokenization completes
the input side of the model; generation combines the decoder, tokenizer, and
cache; speculative decoding builds on generation and cache management to
accelerate sampling without changing the target distribution; and
FlashAttention deepens the analysis of attention performance during training
and prefill.
