# Mixture of Experts practice

Build a routed MoE from a small readable reference through training and modern
balancing strategies. The learner writes implementations; the assistant reviews
them, writes tests, and records corrected reasoning in `SOLUTIONS.md`.

## Learning order

1. MoE fundamentals: experts, sparsity, and active compute
2. Naive top-k routing and weighted expert combination
3. Expert-wise token dispatch and output accumulation
   - Then study packed expert weights and grouped GEMM, comparing with the
     ModuleList reference before decoder integration.
4. Decoder integration, training, and routing diagnostics
5. Auxiliary load-balancing loss
6. Expert capacity, overflow, and dropless routing
7. Router numerical stability and z-loss
8. Shared experts and fine-grained expert specialization
9. Auxiliary-loss-free balancing and training-step updates
10. MoE FLOPs, memory, and expert-parallel systems
11. LatentMoE, normalization, and SiTU-GLU
12. Quantile balancing: exact implementation to histogram estimation
13. Frontier-model case studies: miniature MoE blocks and training tradeoffs

## Current checkpoint

Fundamentals and the minimal `TopKRouter` are implemented and reviewed. It maps
`[B, S, D]` to expert logits `[B, S, E]`, applies softmax across experts, selects
top-K, and returns IDs and normalized selected weights `[B, S, K]`.

Use K=2 initially. Normalized top-1 weights are constant one and supply no
task-loss gradient to this router through output mixing. Retaining the original
selected probability is a different policy to study later.

The learner's `NaiveMoe` now combines independent SwiGLU experts with the router.
It computes only selected experts and adds their weighted outputs per token.
Tests check controlled routing, original-token inputs, sparse expert calls,
parameter registration, and outputs and gradients against a dense test reference.

The learner's `ExpertBatchedMoe` now gathers each expert's assigned tokens across
the entire batch, processes them together, and adds weighted contributions back
to their original positions. Outputs and gradients match the naive reference;
shape comments correctly distinguish `[B, S, K]`, `[N]`, and `[N, D]`.

The learner's `PackedExperts`, `GroupedExperts`, and `GroupedMoE` implement packed weights,
expert-sorted dispatch, router weighting, and `index_add_` combination. Tests
compare outputs and gradients against the ModuleList reference, including
empty groups and non-contiguous inputs. The assistant made a small naming and
shape-restoration cleanup; the learner wrote the algorithm.

`GroupedMoE` selects `expert_backend="loop"` or `"grouped_mm"` (current default).
The grouped version performs two `nn.functional.grouped_mm` calls with int32
offsets and SwiGLU between them. Identical state dictionaries can be loaded
into either backend. The assistant added this selection and its tests.

The learner has integrated MoE into `TransformerBlock` and `DecoderModel`.
`num_experts=None` keeps dense SwiGLU; a positive expert count requires an
explicit `top_k` and constructs a routed MoE in every block. Pre-norm,
residual paths, and the existing dense default are preserved. The original
positional `eps` argument remains in its original constructor position.

Next: a tiny training run and routing diagnostics. No capacity limit, shared
experts, or auxiliary losses yet. Training diagnostics have not been implemented.

Both versions reuse `mlp.SwiGLU`, whose projections now default to `bias=False`.
The packed representation stores up/gate weights `[E, D, 2F]` and down weights
`[E, F, D]` for direct right multiplication, with no expert bias tensors.

All four test files are assistant-authored. There are 42 passing MoE tests.
`benchmark_moe.py` compares inference and forward/backward runtime for all
four versions. The CPU results and reproduction command are in `SOLUTIONS.md`.

## Resume here (2026-10-17)

Step 3 and the decoder-integration part of step 4 are complete. The learner
explained that MoE replaces the FFN while preserving `[B, S, d_model]`, so
normalization and residual connections stay in place. Integration tests check
dense/MoE selection, missing `top_k`, causality, gradients through routers and
experts, and full-versus-cached logits with both full and sliding-window
attention. GPU grouped-kernel speed remains unmeasured.

Resume with this unanswered interview question:

> During a tiny training run, what would you log to tell whether the router
> uses all experts or concentrates most assignments in a few?

Discuss the diagnostics first, then have the learner implement the tiny MoE
training run. Do not repeat the decoder integration or start auxiliary
balancing losses before inspecting routing behavior.
Keep explanations short and concrete; introduce one shape change at a time.
The learner writes the implementation; the assistant writes tests and reviews.

Last focused verification: 74 tests passed across `moe`, `transformer_block`,
`decoder_model`, and `training`. Run
`uv run pytest moe transformer_block decoder_model training -q` when resuming.
For debugging,
use the reusable VS Code module launch with `moe.packed_experts`.
