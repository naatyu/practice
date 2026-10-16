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
`ExpertBatchedMoe`, including transposed parameter/gradient mapping. It still
loops over experts: sorted dispatch and packed storage are ready, but grouped
GEMM has not been implemented. The assistant's cleanup only renamed a misspelled
index variable, simplified flatten/reshape operations and comments, and added a
short class docstring and return annotation on initialization.

## Handoff

The learner's router, naive MoE, and expert-batched MoE are correct for the agreed
initial policy. PackedExperts and GroupedMoE are now verified too, with 35 passing
assistant-authored MoE tests. The learner correctly explained that batching mainly
reduces Python calls and increases matrix sizes, not expert arithmetic. Resume
with replacing the expert loop by grouped GEMM. Keep explanations concrete and
introduce one shape transformation at a time.

Session ended 2026-10-16 before the learner answered why ordinary `bmm` cannot
directly stack input groups of lengths [2, 0, 3]. Resume with that question;
do not treat grouped GEMM as implemented or provide the full implementation.
Use `packed_experts.py` for both PackedExperts and GroupedMoE. Preserve the
existing reference loop for equivalence tests when adding a grouped backend.

Initialization is settled for this exercise: normal with std `1/sqrt(fan_in)`.
Do not reopen the choice of uniform versus normal or switch to 0.02 implicitly.
The learner understands the distinction between fan-in reasoning and model-wide
recipes. Keep docstrings and comments concise; lengthy inline explanations were
explicitly rejected. Notes can carry the longer reasoning.

Before coding a native grouped GEMM call, inspect the installed PyTorch API,
dtype/device and alignment constraints, and backward support. Native grouped
GEMM has not been tested here. Previous runs reported an NVIDIA driver too old
for the installed PyTorch CUDA build, so GPU execution must not be assumed.
If unavailable, explain the limitation and retain a clearly labeled reference
path rather than claiming the expert loop is a real grouped kernel.

Last full verification: 325 repo tests, including 35 MoE tests, passed. The
assistant authored tests and the latest small readability cleanup; the learner
authored the MoE algorithms. Current work is uncommitted; preserve existing
changes and do not commit or push unless asked.
