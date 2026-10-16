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

The learner's `PackedExperts` and `GroupedMoE` now implement packed weights,
expert-sorted dispatch, router weighting, and `index_add_` combination. Tests
compare outputs and gradients against the ModuleList reference, including
empty groups and non-contiguous inputs. The assistant made a small naming and
shape-restoration cleanup; the learner wrote the algorithm.

Next: replace the packed expert loop with grouped GEMM. No capacity limit,
shared experts, or auxiliary losses yet.

Both versions reuse `mlp.SwiGLU`, whose projections now default to `bias=False`.
The packed representation stores up/gate weights `[E, D, 2F]` and down weights
`[E, F, D]` for direct right multiplication, with no expert bias tensors.

All four test files are assistant-authored. There are 35 passing MoE tests.

## Resume here (2026-10-16)

We are at step 3, just before grouped GEMM. `PackedExperts` still loops over
experts; `GroupedMoE` already handles dispatch and combination correctly.
Continue in `packed_experts.py`, without creating a file for each new step.

Last question, not answered yet:

> Three experts receive [2, 0, 3] assignments. Their input groups have shapes
> [2, D], [0, D], and [3, D]. Why can't we simply stack these groups and use
> ordinary `torch.bmm`?

Discuss that first, then explain grouped GEMM using the existing offsets.
Keep explanations short and concrete; introduce one shape change at a time.
The learner writes the implementation; the assistant writes tests and reviews.

Last verification: 35 MoE tests and 325 repo tests passed, with Ruff and MoE
type checks passing. Run `uv run pytest moe -q` when resuming. For debugging,
use the reusable VS Code module launch with `moe.packed_experts`.
