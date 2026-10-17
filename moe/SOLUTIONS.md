# Mixture of Experts: completed reasoning

## Parameters versus active compute

Replacing one dense MLP with eight independent experts of the same size gives
eight times the expert parameters. Selecting two per token runs twice the
original MLP's compute, ignoring router and dispatch overhead. Attention is
unchanged. Sparsity means activating part of a larger parameter pool, not
automatically being cheaper than the original dense model.

## Router and combination shapes

Each token has E expert logits. Softmax and top-k act along the expert dimension,
not along the sequence dimension. For input `[B, S, D]`, logits are `[B, S, E]`;
selected IDs and weights are both `[B, S, K]`. Each ID has one scalar weight.

The MoE combines expert outputs using a weighted sum, preserving `[D]` per
token. A mean is the special case of equal weights. This initial router divides
selected probabilities by their last-dimension sum, with `keepdim=True`.
The denominator is `[B, S, 1]`; the resulting weights remain `[B, S, K]`.

## Gradient caveat for normalized top-1

For K=1, the selected weight is p/p=1. Top-k IDs are discrete and do not carry
gradients, and the constant mixing weight supplies no task-loss gradient to the
router. The selected expert still receives gradients. With K>1, selected weights
can carry router gradients when expert outputs differ. Summing normalized
weights alone is constant and is not a valid test for router learning.

This is a consequence of our initial normalization policy, not a claim that all
top-1 MoE routers are untrainable. Other policies retain the selected probability,
and auxiliary router objectives provide separate gradients.

## Naive weighted expert combination

The MoE module registers its router and independent SwiGLU instances after
calling `super().__init__()`. `nn.ModuleList` registers the experts for parameter
iteration, device/dtype movement, and checkpointing. It is indexed to select an
expert; the container itself is not a callable expert.

The naive forward loops over batch rows, tokens, and selected slots. Each chosen
expert receives the same original token `[D]`, not the entire batch and not the
previous expert's output. Multiply its output by the matching scalar router
weight and add it to a `[D]` accumulator. Store the result in `output[b, s]`.
The gate and value projection must be split along the last feature dimension.

The learner's revised implementation matches a dense test reference for both
outputs and input/parameter gradients. Its additive in-place accumulator works
for this computation; in-place updates are not inherently incompatible with
autograd. Tensor scalar expert IDs work as ModuleList indices, although reading
them from a GPU entails host synchronization, acceptable for this naive reference.

For modern execution, first batch tokens per expert while retaining ModuleList.
Packed expert weights and grouped GEMM are a later execution change, not a
change to the weighted MoE equation. ModuleList storage itself does not require
processing one token at a time.

## Expert-wise batching and selected slots

`torch.where(expert_ids == expert_id)` searches all `[B, S, K]` assignments.
It returns batch indices, sequence indices, and selected-slot indices, each
`[N]`, where N is this expert's assignment count across the whole batch.
N varies per expert and may be zero. Paired advanced indexing gathers exactly
those N tokens, not a Cartesian product of positions.

Batch and sequence indices identify the original tokens; slots identify the
matching router weights within each token's K selections. One expert can appear
in slot zero for one token and slot one for another.

The computation and shapes are:

- `x[batch_indices, sequence_indices]`: expert inputs `[N, D]`.
- `weights[batch_indices, sequence_indices, slots]`: weights `[N]`.
- `expert(expert_input)`: expert outputs `[N, D]`.
- Multiply outputs by weights expanded to `[N, 1]`, broadcasting over D.
- Add `[N, D]` contributions into `output[batch_indices, sequence_indices]`.

Output starts as zeros `[B, S, D]`. Addition preserves contributions from all
K experts; assignment would overwrite earlier contributions. Within one expert's
update, token indices are unique because top-k selects distinct experts. That
makes this indexed addition valid. A future flattened dispatch combining all
assignments at once has repeated token indices and needs an accumulation
operation such as `index_add`, rather than assuming this uniqueness still holds.

The learner's code and shape comments are correct. Eight assistant-authored tests
check outputs and input/parameter gradients against the naive implementation,
different K and dtypes, one-token input, cross-batch routing, different selected
slots, and an unused expert. Calling SwiGLU on `[0, D]` is supported here;
skipping unused experts is also valid. Neither batching nor weight storage changes
which experts process each token or the mathematical weighted sum.

## Bias-free experts and packed weights

The learner changed the shared `mlp.SwiGLU` to default to `bias=False`, with
`bias=True` still available explicitly. The assistant replaced the naive MoE's
duplicate SwiGLU with an import of the shared implementation so both references
use the same bias-free architecture. Their forward algorithms are unchanged.

For E experts of input width D and hidden width F, packed fused up/gate weights
use `[E, D, 2F]`, and down weights use `[E, F, D]` for direct matmul. No expert
biases need to be packed. Expert parameters total `3EDF`; the existing bias-free
router adds `DE`. Packing preserves independent expert weights; it does not
share parameters across experts or itself change sparse execution.

The nn.Linear convention `[E, 2F, D]` is also valid, but requires transposing
each expert's matrix. The learner chose direct right multiplication instead.
The two packed tensors are registered with `nn.Parameter` and initialized on
construction with zero-mean normal weights, std `1/sqrt(D)` for up/gate and
`1/sqrt(F)` for down. The expert count is not part of fan-in.

## Sorted dispatch and combination

Flatten routing IDs to `[T*K]`, where T = B*S. Stable argsort returns original
assignment positions in expert order. Integer division by K maps those positions
to original token rows. Use that map to gather `[T*K, D]` grouped inputs.
`bincount(..., minlength=E)` gives expert counts including unused experts;
cumsum gives the exclusive ends expected by `PackedExperts`.

Reorder flattened weights with the same permutation, expand them to `[T*K, 1]`,
and multiply grouped outputs. Broadcasting aligns from the right, so a bare
`[T*K]` vector would not apply one weight per row. `index_add_` accumulates
repeated token destinations into `[T, D]`; `reshape_as(x)` restores `[B, S, D]`.
Gather reads indexed values; index-add writes and sums contributions.

The learner's implementation passes output and gradient comparisons with
`ExpertBatchedMoe`, including transposed parameter/gradient mapping. The
learner subsequently added `GroupedExperts`, with grouped up/gate and down
matmuls separated by SwiGLU. `PackedExperts` retains the loop for comparison.
The assistant added the `GroupedMoE(expert_backend="loop" | "grouped_mm")`
selector and backend equivalence tests. The learner subsequently selected
grouped MM as the default; tiny-shape/float64 reference tests explicitly
request the loop. Both backends
have matching state dictionaries, so identical weights can be loaded directly.

## Grouped GEMM reasoning

Ordinary bmm requires equal matrix shapes per batch entry. Expert groups with
counts `[2, 0, 3]` cannot be stacked directly. Grouped GEMM instead accepts
packed input `[N, D]`, expert weights `[E, D, 2F]`, and int32 exclusive-end
offsets `[2, 2, 5]`. The slices are `[0:2]`, `[2:2]`, and `[2:5]`; repeated
offsets retain the empty expert's identity. The down projection uses hidden
inputs `[N, F]` and weights `[E, F, D]`. Both matmuls reuse the offsets because
the activation changes features while preserving assignment rows and order.

Optimized GPU grouped kernels schedule tiles from multiple independent GEMMs
together, reducing per-expert launch overhead and improving utilization when
individual groups are small. The expert arithmetic is unchanged. This machine
has no CUDA build, but PyTorch 2.13's CPU path was checked for output and
gradient equivalence, including empty experts, in supported float32 layouts.
Float64 remains a reference-loop test configuration. CPU timings do not prove
GPU grouped-kernel performance.

## Quick CPU benchmark (2026-10-17)

Reproduce with:

```sh
uv run python -m moe.benchmark_moe --threads 1 --min-run-time 0.3
```

Environment: PyTorch 2.13.0, macOS 27.0.1 arm64, CPU float32, one PyTorch thread,
seed 42, eager execution. Each implementation has identical router and expert
weights and sees the same input; outputs are checked before timing. B=1 and
S=T. Timings are end-to-end medians in milliseconds using
`torch.utils.benchmark.Timer.blocked_autorange`, with warmup and at least 0.3
seconds of measurement per configuration. Inference disables autograd.
Forward + backward includes clearing gradients, the forward pass, a
squared-mean loss, and backward, including input gradients. Routing, sorting
where applicable, dispatch, router weighting, and combination are included.
The largest relative timing IQR was 5.0%; treat this as a quick local sample.

| T | D | F | E | K | Mode | Naive | Expert-batched | Packed loop | Grouped MM |
|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|
| 1 | 64 | 128 | 8 | 2 | inference | 0.039 | 0.131 | 0.061 | 0.037 |
| 1 | 64 | 128 | 8 | 2 | forward + backward | 0.176 | 0.448 | 0.453 | 0.134 |
| 32 | 64 | 128 | 8 | 2 | inference | 0.949 | 0.197 | 0.094 | 0.059 |
| 32 | 64 | 128 | 8 | 2 | forward + backward | 3.815 | 0.582 | 0.614 | 0.202 |
| 256 | 128 | 256 | 8 | 2 | inference | 7.942 | 0.518 | 0.411 | 0.328 |
| 256 | 128 | 256 | 8 | 2 | forward + backward | 44.690 | 1.484 | 1.974 | 0.874 |

At T=256, grouped MM was about 1.25x faster than the packed loop for inference
and 2.26x for forward + backward. At T=1, per-expert overhead makes the
expert-batched loop slower than the naive implementation, which calls only
the two selected experts; the small naive/grouped inference difference is
close to timing noise. Grouping also performs SwiGLU across all packed rows
at once instead of once per expert. These measurements compare runtime,
not peak memory, GPU throughput, or a trained model's quality. Parameter
counts and selected-expert arithmetic are the same for all four methods.

## Handoff

The learner's router, naive MoE, expert-batched MoE, and grouped experts are
implemented. PackedExperts and GroupedExperts remain side by side in
`packed_experts.py`; the assistant added the selectable backend and benchmark.
There are 42 passing assistant-authored MoE tests. The learner has also
completed decoder integration: optional expert-count and top-k arguments
select the FFN in `TransformerBlock` and are forwarded by `DecoderModel`.
Dense SwiGLU remains the default, and each decoder layer owns separate experts.
MoE preserves the FFN's input/output shape, so the existing normalization and
residual paths remain valid. Moving the new arguments after the existing
`eps` argument fixed the positional-constructor regression.

The assistant's integration tests verify module selection, configuration,
causality, nonzero task-loss gradients through routers and expert weights,
and full-versus-cached decoding with full or sliding-window attention. Latest
focused verification: 74 tests passed across MoE, Transformer block, decoder,
and training. A tiny MoE training run and routing diagnostics remain future work.

Resume with the unanswered question: "What would you log to tell whether the
router uses all experts or concentrates most assignments in a few?" Discuss
that first, then let the learner implement diagnostics and a tiny training run.
Keep explanations concrete and introduce one shape transformation at a time.

Earlier verification on 2026-10-17: 332 repository tests passed before decoder
integration. ty 0.0.89 checks passed for the MoE folder after concrete test and
benchmark type fixes and narrow ignores for PyTorch's writable `forward`
annotation. The same compatibility diagnostic currently appears on the
Transformer block and decoder `forward` definitions; it is independent of the
passing runtime integration tests.

Initialization is settled for this exercise: normal with std `1/sqrt(fan_in)`.
Do not reopen the choice of uniform versus normal or switch to 0.02 implicitly.
The learner understands the distinction between fan-in reasoning and model-wide
recipes. Keep docstrings and comments concise; lengthy inline explanations were
explicitly rejected. Notes can carry the longer reasoning.

GPU dtype, alignment, and backward behavior still require validation on the
target GPU. Do not infer a CUDA speedup from this CPU benchmark. The assistant
authored tests and the backend selector/benchmark; the learner authored the
MoE algorithms. Preserve existing changes and do not commit or push unless
asked.
