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

All implementation code is handwritten as part of the learning process. Some
tests are written with AI assistance so I can spend more practice
time on the algorithms themselves. Those tests are intentionally kept readable
because they are also examples I can revisit when reviewing how to test tensor
code.

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

The repository is still growing. Generation is next, followed by speculative
decoding, FlashAttention, FLOP and memory accounting, long-context techniques,
and larger training and inference systems topics. The full ordered plan is in
[ROADMAP.md](ROADMAP.md).

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
```

The implementations often avoid PyTorch's high-level equivalent on purpose.
The goal is to understand and reproduce the underlying operation; the built-in
version may still be used in tests as a reference oracle.
