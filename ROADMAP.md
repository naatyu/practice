# LLM interview practice roadmap

The [README learning map](README.md#learning-roadmap) is the visual overview;
this file keeps the detailed exercises and scope. When a topic's status changes,
also update `docs/roadmap/roadmap.json` and run
`uv run python docs/roadmap/build.py` to refresh both README themes.

## Current checkpoint

Completed through cached, batched speculative decoding. FlashAttention theory
notes are complete, but its plain-PyTorch implementation is deferred. Local
and sliding-window attention, including bounded-cache tests, has since been
implemented. The educational linear-attention/KDA track is complete, including
the full layer, fixed-size convolution and recurrent caches, log-space
retention, low-rank projections, and per-head rate initialization. The tested
chunkwise operator is explicitly AI-authored teaching material; recurrence
remains the default. Optimized WY kernels are outside the completed scope.
The learner subsequently completed the educational MoE track. Routing, naive and expert-batched
MoE, packed weights, and grouped-GEMM dispatch/combine are implemented and
tested on CPU, with a reproducible comparison in `moe/SOLUTIONS.md`.
Dense/MoE decoder integration and seven routing diagnostics are implemented and
tested. A tiny training runner now logs/plots per-layer diagnostics and resets
once per optimizer step. Auxiliary load-balancing loss and explicit loss returns
are implemented and integrated into training. Coefficient experiments and their
balance/task-loss tradeoffs have been discussed. Capacity policies, overflow
metrics, and decoder/training configuration are implemented. The capacity run
and overflow/headroom tradeoffs have been discussed. FP32 routing, z-loss,
training integration, and coefficient interpretation are complete. Shared experts,
optional gates, fine-grained compute reasoning, and their toy runs are covered.
Auxiliary-loss-free bias updates, softmax/sigmoid scoring, training
wiring, and toy-run interpretation complete MoE stages 1–9. Batch generation
has an independent RNG; model weights remain unmatched across architectures.
Stage 10's FLOP/memory accounting, expert-parallel communication, and runtime
tradeoffs are complete as reasoning exercises. Stage 11's educational two-rank
CPU/Gloo forward is implemented and matches a single-device reference with
identical expert weights. At the learner's request, a separate AI-authored
commented backward example now includes differentiable exchanges and router
gradient synchronization, with persistent two-rank output/gradient regression
tests covering zero-count and empty-rank cases. Stage 12's learner-written
LatentMoE routed path is implemented in `GroupedMoE` and covered by 38 tests;
router/shared branch stay full-width. Post-combination latent RMSNorm is now
implemented/tested and always enabled with `d_latent`, per learner preference.
Optional routed-expert SiTU-GLU is implemented in both backends and wired through
GroupedMoE, leaving the shared expert as SwiGLU. Stage 13's exact quantile-bias
estimator is implemented and tested. Histogram accumulation is now implemented
as explicitly requested AI-authored teaching code; the learner's histogram
quantile extraction is now implemented/tested. Router-local accumulation and
replacement/reset now support exact (num_bins=None) and histogram QB. Model/
training-runner wiring is now implemented/tested, with exclusive once-per-step
updates and exact/histogram CLI modes. The learner's 800-step toy run has been
interpreted, and optional distributed histogram SUM pooling is implemented/tested,
including empty ranks. Stage 14's final architecture synthesis is also complete:
fine-grained/latent compute and payload ratios, shared/router placement,
balancing timing, and low-precision outlier control. Existing components already
support the miniature design; no duplicate assembly exercise was required.
MoE stages 1–14 are complete for educational scope. Distributed exact-score gathering,
distributed training-runner integration, and decoder/training forwarding of
LatentMoE/SiTU options remain outside the implemented scope.
See `moe/README.md`.
Broader layer-by-layer FLOP/memory accounting and YaRN remain subsequent topics; FlashAttention's
implementation remains deferred.
Multi-Head Latent Attention is exploratory work in progress, not a
completed, tested roadmap stage.

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

## Core implementation sequence

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

### 13. Generation (completed)

- Greedy decoding
- Temperature
- Top-k sampling
- Top-p sampling
- EOS handling
- Repetition penalties
- Batched generation
- Integration with KV caching
- Separation of probability construction from multinomial sampling
- Seeded reproducibility and input-preservation tests
- Controlled fake decoders and real-decoder integration tests

### 14. Speculative decoding (completed)

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
- Uncached single-sequence reference implementation
- Batched acceptance and residual correction
- Independent draft and target KV caches
- Lockstep progress for rectangular batched caches
- Logical cache rollback and final-position cache repair
- Batched EOS handling and exact output-length limits
- Real-decoder parity under deterministic sampling

### 15. FlashAttention (deferred implementation)

- Memory traffic versus arithmetic complexity
- Query, key, and value tiling
- Numerically stable online softmax
- Running maximum and normalization sum
- Running weighted-value accumulator
- Causal block handling
- Comparison with ordinary attention

Current status: interview questions and online-softmax derivation notes are
written; the implementation file remains intentionally empty.

### 16. Local and sliding-window attention

- Dense attention versus sparse attention patterns
- Causal sliding windows: each query attends only to recent tokens
- Window-size semantics and off-by-one boundaries
- Building a correct local causal mask
- Attention-score shapes with full tensors versus compact windowed computation
- Time complexity from `O(S²)` to `O(SW)` for window size `W`
- Attention-matrix memory from `O(S²)` to `O(SW)` when implemented sparsely
- Why applying a dense local mask alone does not realize those savings
- Chunked and banded implementations that avoid the full score matrix
- Receptive-field growth across multiple Transformer layers
- Dilated, block-local, and alternating local/global attention patterns
- Global tokens and attention sinks
- Sliding-window KV-cache eviction during autoregressive decoding
- Absolute RoPE positions after old cache entries are evicted
- Quality and long-range dependency tradeoffs
- Controlled tests for window boundaries, causality, and dense-reference parity
- Cached decoding equivalence with a bounded local KV cache

### 17. FLOP, parameter, and memory accounting

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

### 18. Long-context RoPE scaling and YaRN

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

### 19. Mixed precision and numerical stability

- FP32, FP16, and BF16
- Loss scaling
- Accumulation precision
- Stable softmax and normalization
- Hardware-friendly matrix dimensions

### 20. Quantization

- Weight-only versus weight-and-activation quantization
- Per-tensor, per-channel, and group-wise scales
- INT8 and INT4
- GPTQ and AWQ concepts
- KV-cache quantization

### 21. Distributed training

- Data parallelism
- Tensor parallelism
- Pipeline parallelism
- Sequence and context parallelism
- ZeRO and FSDP
- Computation and communication costs

### 22. Efficient inference

- Continuous batching
- Paged KV caches
- Prefix caching
- Decode-time memory-bandwidth bottlenecks

## Advanced architectures and adaptation

### Exploratory: Multi-Head Latent Attention (work in progress)

- Low-rank query and KV projections
- Decoupled non-positional and RoPE key dimensions
- MLA-specific RoPE tensor layout
- Weight absorption for inference
- Latent KV-cache motivation and parameter-capacity tradeoffs
- Still needs a stable public API, correctness tests, cache tests, and completed
  absorbed-path verification before being marked complete

### Completed: Recurrent linear attention and Kimi Delta Attention

- Plain recurrent linear attention, DeltaNet, scalar-gated DeltaNet, and the
  channel-wise KDA recurrence are implemented and tested.
- The educational KDA layer has causal depthwise Q/K/V convolutions,
  log-space channel-wise retention, a sigmoid beta gate, per-head output
  RMSNorm, a learned output gate, and an output projection. Its cache holds
  the recurrent matrix and Q/K/V convolution tails; full-sequence and
  split-sequence calls match. Log-uniform decay-step initialization and
  meta-device reinitialization are tested. The decay projection now has a
  `d_model -> head_dim -> d_model` bottleneck, as does the output gate; 22
  focused KDA tests pass.
- Per-head positive decay rates now start uniformly in `[1, 16]` before being
  stored in log space.
- At the learner's request, Codex added `linear_attention/chunked_kda.py`:
  an educational triangular-solve operator with 34 focused tests, including
  gradients and cache continuation, and an optional `chunk_size` layer switch.
  This part is AI-authored; production WY kernels remain deferred.
- The educational track is complete. The captioned Manim video, worked example,
  and code walkthrough remain available for review; optimized production
  kernels are optional future work, not a completion requirement.
  See `linear_attention/README.md` for exercise prompts and
  `linear_attention/SOLUTIONS.md` for completed reasoning and authorship notes.

### 23. Mixture of Experts (educational track complete)

1. MoE fundamentals: experts, sparsity, and active compute — done
2. Naive top-k routing and weighted expert combination — done
3. Expert-wise token dispatch and output accumulation — done; packed weights
   and selectable loop/grouped-MM execution tested on CPU and benchmarked
4. Decoder integration, tiny training run, and routing diagnostics — done
5. Auxiliary load-balancing loss — done, including coefficient experiments
6. Expert capacity, overflow, and dropless routing — done; both drop policies,
   overflow diagnostics, training wiring, and capacity experiment interpretation
7. Router numerical stability and z-loss — done; FP32 router, loss, training
   integration, tests, and coefficient experiment interpretation
8. Shared experts and fine-grained expert specialization — done; shared branch,
   optional gate, decoder/training wiring, compute reasoning, and toy-run interpretation
9. Auxiliary-loss-free balancing and training-step updates — done
   - Bias update/scoring implementations, training wiring, tests, and
     softmax/sigmoid toy-run interpretation complete
   - Softmax versus sigmoid affinities: top-k ordering, selected-weight
     normalization, and interactions with expert selection corrections
   - Non-gradient balancing biases and once-per-optimizer-step load updates
   - Normalized-affinity diagnostics and softmax-only z-loss policy;
     softmax baseline retained
10. MoE FLOPs, memory, and expert-parallel systems — done (reasoning)
    - Derive linear/SwiGLU/router and routed/shared expert compute and parameters
    - Separate parameter, gradient, optimizer, activation, and dispatch memory
    - Expert-parallel dispatch/combine, communication, and straggler costs;
      distinguish FLOP estimates from measured runtime
11. Educational expert parallelism in PyTorch — forward and teaching backward verified
    - Done: two CPU/Gloo processes, four experts, top-2, fixed expert IDs and
      mixing weights, contiguous expert ownership, dropless forward
    - Done: destination sorting, count/token/ID exchanges, rank-local
      `GroupedExperts`, both ordering restorations, return exchange, weighted sum
    - Verified: two-rank weighted outputs match an identical-weight single-device
      reference; ten persistent tests cover preparation and local computation
    - Done: separate AI-authored commented autograd exchange/backward demo,
      including replicated-router gradient synchronization
    - Done: persistent two-rank output/gradient reference checks, zero-count
      peers, local-only routes, empty source/receiver, bounded process timeouts
    - No custom kernels, overlap, multi-node setup, or FSDP/TP composition required
12. LatentMoE, normalization, and SiTU-GLU — latent routed path implemented/tested;
    post-combination RMSNorm and optional routed SiTU-GLU implemented/tested
    (no decoder/training or distributed latent wiring yet)
13. Quantile balancing — completed for educational scope;
    exact frozen-cutoff estimator implemented/tested;
    pooled quantile reasoning covered; AI-authored histogram accumulation tested;
    learner-written histogram quantile extraction tested;
    local exact/histogram accumulation and step replacement/reset tested;
    model/runner/CLI forwarding and exclusive once-per-step calls tested;
    toy-run interpretation complete; optional distributed histogram SUM pooling
    tested on two CPU/Gloo ranks, including empty/global-empty windows;
    no distributed exact gathering or distributed training-runner integration
14. Frontier-style architecture synthesis and training tradeoffs — done (reasoning)
    - Matched routed-expert compute across fine-grained and latent designs
    - Distinguished dispatch payload, parameters, FLOPs and measured runtime
    - Full-width router/shared branch, unbiased mixing and QB window timing
    - SiTU-GLU outlier control and block-scale precision tradeoffs
    - Existing GroupedMoE components suffice; no new model reproduction claimed

EP references: [PyTorch distributed collectives](https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.all_to_all_single)
and [torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html).

The learner implements; the assistant interviews, reviews, writes tests, and
records corrected reasoning. Session handoff
and CPU benchmark results are in `moe/README.md` and
`moe/SOLUTIONS.md`.

### 24. Fine-tuning

- LoRA
- QLoRA
- Supervised fine-tuning
- Preference optimization fundamentals

### 25. Reinforcement-learning post-training

- RLHF pipeline: policy, reference policy, reward model, and generated samples
- Token-level log-probabilities for sampled completions
- Sequence rewards, completion masks, and credit assignment
- KL regularization against the reference policy
- Proximal Policy Optimization (PPO)
- Importance-sampling probability ratios
- Advantage estimation and Generalized Advantage Estimation (GAE)
- Clipped policy surrogate objective
- Value-function loss and entropy bonus
- PPO minibatches and multiple optimization epochs per rollout
- Group Relative Policy Optimization (GRPO)
- Sampling multiple completions for each prompt
- Group-relative reward normalization and advantage construction
- Removing the learned value model used by PPO
- GRPO clipped objective and reference-policy KL penalty
- Reward scaling, zero-variance groups, and numerical stability
- Separating rollout generation from policy optimization
- Tests for log-probability ratios, clipping, masked losses, advantages, and KL
- PPO versus GRPO memory, compute, stability, and reward-model tradeoffs

### 26. Extended training and evaluation

- Perplexity and evaluation aggregation
- Padding and document-boundary masks
- Optimizers and learning-rate schedules
- Weight initialization and residual scaling
- Regularization and label smoothing
- Checkpointing optimizer and random-number-generator state
- Data leakage and evaluation design

### 27. Build a small autograd engine

Start with a scalar reverse-mode engine, then optionally extend it to tensors.
Keep this a small learning project, not a production framework.

- Scalar values, computation graphs, and local backward rules
- Addition, multiplication, powers, and simple nonlinearities
- Reverse topological traversal and the chain rule
- Gradient accumulation for reused values and branching graphs
- Explicit gradient reset and a tiny MLP trained with SGD
- Tests against hand calculations, finite differences, and PyTorch autograd
- Optional tensor extension: reductions, broadcasting backward, and matmul
- Optional follow-up: saved intermediates, detach/no-grad, and in-place hazards

References:

- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd): a small
  scalar autograd engine and neural-network library, our starting reference.
- [TinyTorch: Don't Just Import PyTorch. Build It.](https://pytorch.org/blog/tinytorch-dont-just-import-pytorch-build-it/)
  (PyTorch blog, September 21, 2026): a recent build-from-scratch curriculum
  including autograd; likely the post mentioned during planning.
- [TinyTorch Module 06: Autograd](https://mlsysbook.ai/tinytorch/modules/06_autograd.html):
  reference for the optional tensor-level extension rather than taking on the
  entire framework curriculum.

## Immediate path

```text
byte-level BPE [done] -> generation [done] -> speculative decoding [done]
-> local/sliding-window attention [done] -> linear attention and KDA [done]
-> MoE [stages 1–14 educational scope done]
-> FLOP and memory accounting / YaRN / deferred FlashAttention
```

The order is intentional: the completed training loop proves that the decoder
and stable loss can learn together; the completed KV-cache exercise exposes
the main autoregressive inference memory problem; the completed MQA/GQA work
reduces that cost;
tokenization completes the input side of the model; the completed generation
stage combines the decoder and cache with practical sampling; and the completed
speculative-decoding stage adds distribution-preserving draft acceleration,
batched verification, and cache rollback. Local attention changes the
attention pattern itself and has now been implemented with bounded cache
memory. The completed linear-attention/KDA track adds fixed-size recurrent
memory, a full layer resumable across decode calls, and an AI-authored
chunkwise worked example checked against the recurrence.
FLOP and memory accounting, YaRN, and the deferred FlashAttention
implementation remain available subsequent exercises.
