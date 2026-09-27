# LLM implementation practice

This is my working repository for learning how modern language models actually
work by implementing their core components from scratch. The exercises are
often run in an interview-style format, which helps me test both my code and my
understanding.

The point is not to build another production framework. It is to implement the
important pieces myself, understand the math behind them, reason about tensor
shapes and numerical stability, and be able to explain the tradeoffs.

## How I use the repository

Each topic starts as a focused implementation problem, such as causal attention
or RoPE. I work through it while an AI acts as both an interviewer and a
learning partner: it asks follow-up questions, challenges incorrect reasoning,
explains unfamiliar ideas, and helps turn the final answers into useful notes.

Most topic folders follow the same structure:

- `README.md` contains the exercise prompt and discussion questions.
- `SOLUTIONS.md` contains the corrected reasoning and lessons from the
  interview.
- The Python module contains the implementation.
- Tests capture the important invariants, controlled examples, and comparisons
  with trusted PyTorch implementations.

The core exercises are implemented interactively as part of the learning
process. Tests and explicitly requested follow-up implementations may be
written with AI assistance so I can spend more practice time on the algorithms
and systems reasoning. The resulting code and tests are intentionally kept
readable because they are also reference material for later review.

## Topics covered

- Scaled dot-product and causal attention
- Multi-head self-attention
- RMSNorm
- GELU MLPs and SwiGLU
- Pre-norm decoder Transformer blocks
- Learned and sinusoidal positional encodings
- Rotary positional encoding and relative-position reasoning
- RoPE integration and sequence offsets
- A complete decoder-only language model
- Weight tying and shifted next-token loss
- Stable cross-entropy with ignored targets
- Basic training steps, AdamW, clipping, and gradient accumulation
- KV caching across prefill and autoregressive decoding
- Cached versus full-sequence decoder equivalence
- Multi-query attention (MQA) and grouped-query attention (GQA)
- Compact KV caches with separate query-head and KV-head counts
- Byte-level BPE training, encoding, and UTF-8 decoding
- Unicode-aware regex pre-tokenization and protected special tokens
- Portable tokenizer serialization and reconstruction
- Naive and KV-cached greedy generation
- Temperature, top-k, and top-p sampling
- Batched EOS handling and reproducible sampling
- Sign-aware repetition penalties
- Reference speculative sampling with exact rejection correction
- Batched speculative decoding with independent draft and target KV caches
- Cache rollback, synchronized batch progress, and EOS handling during
  speculative decoding
- Vectorized K-means prediction as a separate classical-ML warmup

## Current checkpoint

The main decoder path is complete through cached, batched speculative decoding.
The repository also contains an exploratory Multi-Head Latent Attention (MLA)
implementation, but MLA remains work in progress and does not yet have the same
test coverage as the completed components.

The next implementation milestone is FlashAttention in plain PyTorch. Its
interview questions and online-softmax reasoning notes already exist, while the
implementation itself is intentionally still pending. Local attention, FLOP
and memory accounting, YaRN, and broader systems topics follow. The complete
ordered plan is in [ROADMAP.md](ROADMAP.md).

## Repository map

| Area | Folder | Status |
| --- | --- | --- |
| Attention fundamentals | [`attention`](attention) | Complete |
| MHA, GQA, MQA, exploratory MLA | [`multi_head_attention`](multi_head_attention) | MHA/GQA/MQA complete; MLA WIP |
| Normalization and feed-forward layers | [`normalization`](normalization), [`mlp`](mlp) | Complete |
| Transformer and decoder | [`transformer_block`](transformer_block), [`decoder_model`](decoder_model) | Complete |
| Positional methods | [`positional_encoding`](positional_encoding) | Learned, sinusoidal, and RoPE complete |
| Loss and training | [`cross_entropy`](cross_entropy), [`training`](training) | Complete |
| KV-cache concepts | [`kv_cache`](kv_cache) | Complete |
| Tokenization | [`bpe`](bpe) | Complete |
| Sampling and speculative decoding | [`generation`](generation) | Complete reference and cached batched paths |
| FlashAttention | [`flash-attention`](flash-attention) | Theory notes complete; implementation next |
| Classical ML warmups | [`machine_learning`](machine_learning) | K-means prediction complete |

## Running the exercises

The project uses Python 3.12, PyTorch, `uv`, and pytest.

Run the complete test suite from the repository root:

```bash
uv run pytest
```

Run one topic while working on it:

```bash
uv run pytest -q attention
uv run pytest -q positional_encoding
uv run pytest -q training
uv run pytest -q bpe
uv run pytest -q generation
```

The implementations often avoid PyTorch's high-level equivalent on purpose.
The goal is to understand and reproduce the underlying operation; the built-in
version may still be used in tests as a reference oracle.
