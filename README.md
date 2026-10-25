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
process. Tests, learning visuals, and explicitly requested follow-up
implementations may be written with AI assistance so I can spend more practice
time on the algorithms and systems reasoning. The resulting code and tests
are intentionally kept readable because they are also reference material for
later review.

One explicit exception to my own implementation work is the educational
[chunkwise KDA operator](linear_attention/chunked_kda.py): I asked Codex to
write it as a worked example to study. The recurrent KDA implementation is mine.

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
- Causal sliding-window attention and bounded decoding caches
- Recurrent linear attention, DeltaNet, gated DeltaNet, and Kimi Delta Attention
- Full KDA layer with short convolutions, learned gates, and fixed-size cache
- Educational chunkwise KDA (AI-authored worked example)
- MoE routing, grouped experts, capacity policies, shared experts and balancing
- Educational expert-parallel forward/backward with two-rank reference tests
- LatentMoE, SiTU-GLU, and exact/histogram quantile balancing
- MoE architecture synthesis: compute, communication and numerical tradeoffs
- Vectorized K-means prediction as a separate classical-ML warmup

## Current checkpoint

The main decoder path is complete through cached, batched speculative decoding.
Local/sliding-window attention and the educational linear-attention/KDA track
are also complete, including recurrent caching and a tested chunkwise worked
example. The chunkwise implementation is explicitly credited to Codex.
The repository also contains an exploratory Multi-Head Latent Attention (MLA)
implementation, but MLA remains work in progress and does not yet have the same
test coverage as the completed components.

The [Mixture of Experts](moe/README.md) educational track is complete (stages 1–14): routing, expert batching,
packed weights, and selectable loop/grouped-MM execution are tested on CPU.
Decoder integration, per-layer routing diagnostics, tiny training runs,
auxiliary balancing, and capacity-limited/dropless dispatch are complete.
FP32 routing and z-loss are also complete, including coefficient experiments.
Shared experts, optional gating, and fine-grained routing's compute tradeoffs
are also covered. The MoE notes record the toy runs and their limits.
Auxiliary-loss-free balancing and sigmoid routing are complete too, including
their diagnostic semantics and toy-run comparison. Stage 10's FLOP/memory
accounting and expert-parallel systems reasoning are complete. The educational
two-process CPU expert-parallel forward now works with grouped experts and
matches a single-device reference. A separate AI-authored backward example and
persistent two-rank output/gradient tests cover empty-rank cases. Latent routed
experts, optional SiTU-GLU, exact/histogram balancing, local training integration,
and optional distributed histogram pooling are implemented/tested. The final
design review connects those components; it is not a production-model reproduction.
Full distributed training, global exact-quantile gathering, and decoder/training
forwarding of latent/SiTU options remain optional extensions, not stage blockers.

FLOP and memory accounting and YaRN remain subsequent roadmap topics.
FlashAttention theory and online-softmax notes already exist; its plain-PyTorch
implementation remains deferred. The complete ordered plan is in
[ROADMAP.md](ROADMAP.md).

## Repository map

| Area | Folder | Status |
| --- | --- | --- |
| Attention fundamentals and local/sliding-window attention | [`attention`](attention) | Complete |
| Linear attention and KDA | [`linear_attention`](linear_attention) | Educational track complete; chunkwise example AI-authored |
| Mixture of Experts | [`moe`](moe) | Stages 1–14 complete for educational scope; production extensions deferred |
| MHA, GQA, MQA, exploratory MLA | [`multi_head_attention`](multi_head_attention) | MHA/GQA/MQA complete; MLA WIP |
| Normalization and feed-forward layers | [`normalization`](normalization), [`mlp`](mlp) | Complete |
| Transformer and decoder | [`transformer_block`](transformer_block), [`decoder_model`](decoder_model) | Complete |
| Positional methods | [`positional_encoding`](positional_encoding) | Learned, sinusoidal, and RoPE complete |
| Loss and training | [`cross_entropy`](cross_entropy), [`training`](training) | Complete |
| KV-cache concepts | [`kv_cache`](kv_cache) | Complete |
| Tokenization | [`bpe`](bpe) | Complete |
| Sampling and speculative decoding | [`generation`](generation) | Complete reference and cached batched paths |
| FlashAttention | [`flash_attention`](flash_attention) | Theory notes complete; implementation deferred |
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

### CUDA on the local workstation

On Linux x86-64, `uv` selects the official CUDA 12.9 PyTorch build. This keeps
PyTorch 2.13 compatible with the workstation's NVIDIA 575.57.08 driver, without
changing the system driver or toolkit. Other platforms retain the PyPI source.
CUDA 13 builds need a newer driver; see [NVIDIA's compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

```bash
uv sync --locked --extra plots
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

There is one physical GPU. We can run CUDA exercises on it, but two CPU processes
remain the initial EP exercise; a real multi-GPU comparison needs more hardware.

### Debugging

Open the repo root in VS Code and select the `.venv` Python interpreter.
In Run and Debug, choose **Python: Debug a module**, then press F5 and enter
the dotted module name, e.g. `moe.grouped_moe` (no `.py`). This runs with
the repo root on the import path, supporting both sibling and relative imports.
Set breakpoints as usual; the module's `if __name__ == "__main__":` block runs.

To debug tests instead, open a test file and choose **Python: Debug current
test file**. These launch configurations require the Python Debugger extension.
Terminal equivalent for running a module: `uv run python -m moe.grouped_moe`.

The implementations often avoid PyTorch's high-level equivalent on purpose.
The goal is to understand and reproduce the underlying operation; the built-in
version may still be used in tests as a reference oracle.
