# Mixture of Experts: completed reasoning

## Router stability and z-loss (stage 7, complete)

Adding the same constant across experts for one token leaves softmax and top-k
unchanged, so balancing loss cannot detect this common shift. Stable softmax
prevents exponential overflow but cannot recover distinctions already lost to
rounding: BF16 cannot distinguish 1000 from 999. FP32 routing and regularization
address different parts of this problem.

The log-normalizer is `log(sum(exp(logits)))`, the log of the softmax denominator.
Unlike softmax, it increases by c when all logits increase by c. Z-loss averages
its square across tokens, pushing positive and negative values toward zero.
Without squaring, minimization would encourage unbounded negative drift.
Normalized log-probabilities have logsumexp zero without requiring uniform
probabilities. This anchors common drift, not every individual logit or balance.

Bounded logits or normalized router inputs and weights with controlled scale
are architectural alternatives, but constrain routing differently. Normalizing
inputs alone cannot prevent weights from growing. Z-loss softly encourages
better-scaled logits without changing the forward routing function.
See [ST-MoE, sections 3.3–3.4](https://arxiv.org/html/2202.08906v2#S3.SS3).

The learner implemented `router_z_loss`: cast `[B, S, E]` logits to FP32,
logsumexp over E, square, then average over B and S. The coefficient stays
outside the helper. Codex added six tests for controlled reductions, the
zero-penalty anchor, large logits across three dtypes, and analytic gradients.
The learner also retained raw router logits and disabled autocast around the
FP32 projection/softmax. Casting only after a BF16 projection cannot recover
lost distinctions. `autocast(dtype=float32)` is not the portable way to disable
AMP: CPU rejects that target; use `enabled=False` and FP32 inputs/weights.
Explicit whole-model dtype casts still affect router weights; this implementation
does not override `.to()` or `.bfloat16()` to keep them FP32 automatically.

Codex propagated separate unweighted losses through all MoE variants, the block,
and `DecoderOutput`; each loss is averaged independently across MoE layers and
is None for dense models/eval. Training applies independent coefficients before
one backward pass, then detaches logging values. Both coefficients default to
zero. Accumulation scales the combined objective per equal-sized microbatch.
The tiny runner logs z-loss, exposes `--z-loss-coefficient`, and tags nonzero
coefficients in plot filenames. A dtype cast before grouped output accumulation
handles FP32 routing weights with low-precision experts. Tests cover projection
dtype under CPU autocast, shift detection, gradients, per-layer averaging,
coefficient independence, eval, logging, and filenames.

### Coefficient experiments and interpretation

A useful starting point is 0.001 for z-loss, versus 0.01 for balancing loss;
these are recommended starting values, not universal optimal coefficients.
A small educational sweep is 0, 0.0001, 0.001, and 0.01. Our decoder averages
losses across layers; summing instead would change their effective strength.
See [Megatron's coefficient guidance](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html).

The learner ran dropless experiments with balancing coefficient fixed at 0.05:

- [Z coefficient 0.001](../training/plots/moe_metrics_aux_0.05_z_0.001_dropless.png):
  unweighted z-loss rises from roughly 2.2 to 3.1, then appears approximately flat
  late in the 800-step run. A rising regularizer is possible because task and
  balancing gradients also update the router. Its weighted contribution at 3.1
  is only 0.0031. A plateau alone does not prove the penalty prevented drift;
  the older no-z-loss baseline did not plot this quantity.
- [Z coefficient 0.01](../training/plots/moe_metrics_aux_0.05_z_0.01_dropless.png):
  z-loss rises initially, then decreases late, ending around 2.5. Compared with
  0.001, router entropy is higher, selected probability mass lower, and several
  layers have worse late-stage MaxVio. Both runs still approach zero task loss.

These are approximate visual readings, not raw-history comparisons. The stronger
penalty controls the log-normalizer more, but has a less favorable late-stage
load balance in this run. Calling it categorically too strong is not justified:
higher token entropy/lower selected mass indicate softer preferences, not
necessarily worse expert outputs. MaxVio does measure relative overload.
No generalization or large-scale numerical-stability benefit was established.

Z-loss can change routing despite softmax shift invariance: its gradients update
the same router parameters and are not restricted to equal shifts. It penalizes
the squared log-normalizer, not entropy, balance, or every individual logit.
An increasing value need not represent common drift alone; relative logit
changes can also increase it. An average near 3 is not a bound on all logits.
Stage 7 is complete; next is shared experts and fine-grained specialization.

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

## Why expert IDs do not carry gradients

A derivative measures how an output changes under a tiny input change.
For logits `[2.0, 1.0]`, the top-1 ID is 0. Moving the second logit to 1.01
still returns 0: the selected ID is locally constant, so its derivative is zero.
Once the second logit passes 2.0, the ID jumps to 1; at the tie boundary the
selection is discontinuous and has no ordinary derivative. Thus hard selection
has zero derivative away from boundaries and no derivative at the boundaries,
not a useful gradient telling the router how to change its choices.

PyTorch returns integer IDs without a gradient path. Removing `no_grad` does
not change this: `topk` IDs -> `bincount` -> assignment fractions/MaxVio cannot
train the router. Selected floating-point scores, however, retain gradients
through the currently selected entries; this does not differentiate the choice
of entries itself. Merely casting IDs to floats does not restore a gradient.

Mean full router probabilities provide a differentiable signal when computed
directly from the current forward pass. Our detached logging buffers do not:
the balancing loss must use live probabilities, not `compute_metrics()`.
Actual assignment fractions can serve as detached load measurements alongside
that differentiable signal.

## Auxiliary load-balancing loss

The learner implemented `moe.losses.load_balancing_loss`. The assistant refactored
its input to current-forward counts `[E]`, computed once in the router from
selected IDs `[B, S, K]` and reused for logging. For full probabilities
`[B, S, E]`, compute assignment fractions
`f = counts / counts.sum()` and mean probabilities
`p = probabilities.mean((0, 1))`, both `[E]`. Return `E * sum(f * p)`.
Balanced assignment fractions give a reference loss of 1, not a guaranteed
global minimum. Flattening IDs in the router is required by `bincount`; normalizing by all
assignments avoids scaling the loss with batch size or K.

Counts do not carry gradients, but full softmax probabilities do. An unused
expert has zero direct coefficient in the sum, yet raising its logit takes
probability away from busy experts through the shared softmax denominator.
Thus its logit can receive a useful gradient even with zero assignments.
Selected, renormalized weights omit this full-expert signal.

Assistant-authored tests cover balanced top-1/top-2/all-expert routing, a hand
calculation, unused experts, batch-repeat invariance, and logit gradients.
Codex implemented the explicit-return plumbing at the learner's request:
router -> (IDs, weights, auxiliary loss), MoE -> (hidden states, auxiliary loss),
block -> (hidden states, optional cache, auxiliary loss), and decoder ->
`DecoderOutput(logits, kv_caches, aux_loss)`. No loss is stored on modules.
The loss field is None in evaluation or for dense models. Each MoE layer is
balanced independently before scalar losses are averaged: pooling counts can
hide opposite imbalances in separate layers.

Training adds `coefficient * output.aux_loss` to task loss before backward and
returns detached `TrainingLoss` details. The coefficient defaults to zero for
the baseline and is exposed in `training.train_moe` as `--aux-loss-coefficient`.
Generation now reads the named logits/cache fields. Tests cover layer averaging,
auxiliary gradients, eval/dense outputs, coefficient scaling, and loss logging.
A layer's balancing objective can also update upstream layers through its
hidden inputs; it does not directly depend on its own expert outputs.
The learner correctly explained the coefficient tradeoff: stronger balancing
pressure can improve load balance at the expense of higher task loss, but neither
perfect balance nor a task-loss penalty is guaranteed. The fixed-batch runs with
coefficients 0, 0.05, and 0.5 were compared visually. The 0.5 run showed tighter
assignment balance while still reaching near-zero task loss; it did not establish
a quality penalty. Its total loss near 0.5 is expected from an unweighted auxiliary
loss near 1, not evidence of failed memorization. These single-batch experiments
do not measure generalization. Auxiliary balancing is complete for this stage.
Capacity and overflow were completed next, as recorded below.

## Capacity-limited dispatch: keep first

The learner implemented optional capacity in `GroupedMoE`, leaving `None`
dropless. Capacity is `ceil(capacity_factor * B*S*K / E)` assignments per expert,
not tokens per sequence. The factor must be finite and strictly positive.
Headroom accommodates ordinary routing variation; experts do not inherently
require a cap. In distributed dispatch, the busiest device can delay the others.

Stable expert sorting preserves flattened batch/sequence order within groups.
Subtracting repeated group starts from global assignment indices gives local
positions. Keep positions below capacity, cap counts, and compute fresh int32
offsets before gathering or processing expert inputs. The same filtered
permutation indexes original tokens and weights; retained assignment count N
may be smaller than B*S*K.

A dropped assignment contributes zero, with no rerouting or weight
renormalization. The token stays in the sequence and task loss; if all its
assignments are dropped, its MoE output is zero and the block residual still
carries its representation. Router logging and balancing retain pre-drop counts
so overload is not concealed by clipping. Keep-first is simple but favors earlier
tokens; score-priority dropping is the next exercise.

Codex cleaned up structure/comments, made the option keyword-only, skipped
local-position work on the dropless path, and completed finite-factor validation.
The learner authored the dropping algorithm. Twelve assistant-authored tests
cover controlled outputs/gradients for both expert backends, empty groups,
partial/complete assignment dropping, rounding, preserved weights, sufficient
capacity, and pre-drop auxiliary/logging statistics. Decoder configuration and
post-drop diagnostics have not yet been extended for capacity.

## Score-priority capacity dispatch

The learner implemented `drop_policy="probs"` (default), with `"position"` keeping
the original policy. Sort flattened router mixing weights descending, then
stably sort expert IDs indexed by that first permutation. The second sort
returns positions in the weight-sorted array, not original assignment IDs:
`permutation = weights_permutation[expert_order]` composes them correctly.
Expert groups are then strongest-first; the same local-position mask and capped
offsets apply. Ranking uses the router's returned, top-k-normalized mixing
weights, not raw logits or full-softmax probabilities. Stable sorts break equal
weights by original assignment order.

Codex added the three short indexing comments, policy validation, and cleanup
to avoid a redundant expert-ID sort. Weight sorting is skipped when dropless.
Tests now compare both policies and gradients against independent Python
selection, check exact retained inputs, deterministic ties, and invalid policies.
Both policies remove the same number of assignments per expert at a given
capacity, but can remove contributions from different tokens.
The learner added `dropped_assignments_per_expert`, a non-persistent int64 [E]
buffer in `GroupedMoE`. During training it accumulates requested minus retained
counts, not retained counts themselves; eval leaves it unchanged and dropless
forwards add zero drops. `GroupedMoE.reset_metrics()` clears both these counts
and the router's requested counts so numerator and denominator share a window.
Requested counts are reused from `router.tokens_per_expert`, not duplicated.
Nine assistant-authored tests verify unequal-forward accumulation, no-overflow
cases, eval isolation, in-place reset, and buffer registration for both backends.
The learner implemented MoE-level `compute_metrics()`, preserving the seven
router metrics and adding global/per-expert dropped-assignment rates. Global
rate is sum(dropped)/sum(requested), not the average expert rate; denominators
are clamped only after the appropriate reduction. Reporting is detached and
does not reset statistics. Five assistant-authored tests cover controlled values,
empty windows, unequal counts, shapes, and repeated non-mutating reads.
At the learner's request, Codex wired `capacity_factor` and `drop_policy` through
ModelConfig/DecoderModel/TransformerBlock, added CLI flags, and switched the tiny
runner to MoE-level reporting/reset. History records nine metrics per layer;
plots show aggregate and per-expert drop rates, label requested shares as
pre-capacity, and tag filenames with capacity/policy as well as auxiliary strength.
Default capacity remains None (dropless), without gradient accumulation.
Tests verify forwarding to all layers, per-step count reset, both policies, and
separate plot filenames. A short CPU CLI run verifies end-to-end plotting.
Capacity is per forward, including eval: score-priority dropping can depend on
future tokens in the current chunk and cached/full outputs need not agree when
capacity is enabled. Keep dropless inference as the reference.

### Capacity experiment and interpretation (2026-10-21)

Capacity mainly bounds per-expert work and dispatch/activation storage, helping
control stragglers. It does not shrink expert weights or optimizer state, and
does not by itself guarantee a speedup. Dropless execution remains a valid path.

The learner's factor-1.1, `probs`, auxiliary-coefficient-0.05 run was compared
visually with the dropless 0.05 run. Both reach near-zero fixed-batch task loss.
Early aggregate drop-rate peaks are roughly 10–14% in the most affected layers;
later most layers are near zero, with one around 2%. An individual expert can
drop roughly a third of its requests early without a third of all assignments
being dropped. These are approximate plot readings, not raw-history statistics.
They establish neither generalization nor runtime/VRAM improvements.

With four experts, equal requested shares are 25%; factor 1.1 allows about 27.5%
per expert, subject to rounding. Zero drops therefore does not mean perfect
balance. Increasing capacity cannot increase drops for fixed routing decisions,
but separate training runs can learn different routes. Factor 1.5 gives more
headroom, not a universal guarantee of zero overflow.

The learner correctly derived a sufficient no-drop factor: each of T tokens
can select a given expert at most once, so its maximum load is T, versus a
balanced load of T*K/E. Factor E/K therefore guarantees enough capacity for
standard distinct top-k routing: E=4 and K=2 gives factor 2.

This completed stages 1–6. Comparing position and score priority at the same
factor remains an optional follow-up. Stage 7 was completed afterward, as above.

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

## Routing diagnostics: practitioner references

The learner proposed assignment counts per expert to reveal overloaded and unused
experts. Keep counts per layer, sum over all microbatches contributing to an
optimizer step, report, then reset. No gradient tracking is needed. A
non-persistent registered buffer moves with the module but is not checkpointed.
Collect training statistics only; evaluation should not contaminate training
counts. Later, checkpoint recomputation must not count forwards twice.

References checked on 2026-10-19:

- [TorchTitan router](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/common/moe/moe.py)
  accumulates non-persistent expert counts, excludes padding, guards against
  recomputation, and pools statistics across ranks for step-level bias updates.
- [DeepSeek Loss-Free Balancing report](https://arxiv.org/html/2408.15664v1)
  defines MaxVio as relative overload of the busiest expert compared with the
  balanced target. Batch and whole-validation-set variants answer different
  questions; the report also evaluates perplexity, not balance alone.
- [Megatron router diagnostics](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router_diagnostics.py)
  distinguishes mean all-expert scores, actual versus auxiliary assignments,
  expert biases, selection overlap, valid-token counts, and top-k boundary margin.
  Its [metric tracker](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_logging.py)
  supports per-layer reporting and detached metric accumulation; its router
  separately logs balancing objectives and z-loss.
- [Kimi K3 report](https://arxiv.org/html/2607.24653)
  distinguishes expert routing imbalance from hardware-rank imbalance: balanced
  rank execution can coexist with skewed per-expert counts. Quantile histograms
  pool information over microbatches and ranks before the step-level update.

Our recommended initial suite is counts/shares, unused experts, MaxVio, mean
all-expert probabilities, per-token router entropy, and selected probability
mass. These are a teaching selection, not a claim that every source logs all
of them. Entropy uses the full expert probability distribution, not the
renormalized top-k weights. Selected mass is measured before renormalization;
afterward it is always one. Uniform aggregate use does not imply uncertain
per-token routing or prove useful specialization. Tiny batches can leave
experts unused without indicating persistent collapse.

Use task/validation loss alongside routing metrics. Add auxiliary objectives,
drop rate, bias diagnostics, and systems metrics when their mechanisms exist.
Accumulate counts and sufficient statistics before computing ratios; do not
average unequal-microbatch metrics blindly. Step-aggregated balance can hide
alternating overloaded microbatches, so per-microbatch measurements become
important when studying expert-parallel throughput. Distributed counts must
include each assignment once, not multiply replicated routing statistics.

## MaxVio implemented

MaxVio means maximal violation: `(counts.max() - counts.mean()) / counts.mean()`.
It is one scalar per layer/window. Zero means balanced nonempty load; 0.5 means
the busiest expert is 50% above target. This differs from max-minus-min and from
load/target, whose balanced value is one. The learner's implementation divides
an integer sum by E to get a floating-point target, and runs under no_grad.

For no assignments, return zero as a no-data convention, not proof of balance.
`counts.new_zeros((), dtype=torch.float32)` creates a floating scalar on the same
device as counts. `zeros_like(counts)` would create an integer vector [E];
`torch.zeros((), device=counts.device, dtype=torch.float32)` is equivalent.
Seven assistant-authored tests cover known values, zero counts, scale invariance,
one expert, scalar shape, unchanged inputs, and absence of autograd tracking.
`assignment_fraction` is also implemented and tested: counts divided by total
assignments, shape [E], or a floating zero vector on the input device if none
exist. It preserves counts and does not track gradients. Marin uses this exact
term in its [router metrics](https://github.com/marin-community/marin/blob/main/experiments/grug/fast_track/router_metrics.py).
This is not fraction of unique tokens; each token contributes K assignments.
Marin's aggregate assignment entropy is distinct from per-token router entropy.
`unused_expert_count` is implemented as `(counts == 0).sum()`, returning an
integer scalar. With all-zero counts it returns E, unlike the zero-valued
no-data conventions for MaxVio and assignment fractions. It describes the
measurement window, not permanent expert death. Four assistant-authored tests
cover partial, absent, and complete use, including a single expert.
`assignment_entropy` now computes `-sum(p * log2(p.clamp_min(eps)))` from the
assignment fractions, outside autograd. It returns a scalar in bits. Four equal
shares give 2 bits, two equal nonzero shares give 1 bit, and a point mass gives
zero. Entropy concerns uncertainty about the expert ID, not the number of
distinct count values. For zero assignments, the zero fractions yield zero by
our no-data convention.

Adding eps to every probability biased the result and made point-mass entropy
slightly negative. Clamping only the log argument avoids changing probabilities
above eps; positive fractions below eps are still approximated. Six new
assistant-authored tests cover the stated examples, nonnegative finite scalar
results, unchanged counts, scale/expert-order invariance, and no gradient graph.
There are now 69 passing MoE tests. Next: mean all-expert router probabilities,
then the remaining diagnostics; training-loop logging/reset is still pending.

Probability-statistics update: the learner added non-persistent `router_prob_sum`
[E] (initially float32) and `num_tokens` (int64 scalar). The training-only no_grad
block sums full probabilities over B and S after casting to float32, and adds
B*S tokens, not B*S*K assignments. `reset_metrics()` clears all three buffers.
The assistant updated reset/eval tests and added unequal-microbatch and buffer
registration checks; 71 MoE tests pass. Next: fill the existing `compute_metrics`
stub with the four count-based metrics and mean all-expert probabilities, using
a zero [E] vector if no tokens exist. Reading metrics must not reset statistics.
`compute_metrics()` is now implemented: keys are `maxvio`, `assignment_fraction`,
`unused_expert_count`, `assignment_entropy`, and `mean_router_probability`.
The mean divides by `num_tokens.clamp_min(1)`, yielding a zero vector when no
data exists. Reporting leaves buffers unchanged and returns tensors that survive
later resets. Four additional tests cover exact mapping/values, no-data before
and after reset, no autograd, repeat reads, and report stability. There are now
75 passing MoE tests at that checkpoint.

The learner has since added `router_entropy(probabilities)`, summing
`-p*log2(p.clamp_min(eps))` over experts only: [B, S, E] -> [B, S]. Three
assistant-authored tests verify per-token values, no autograd, unchanged inputs,
and confident opposite token choices with balanced aggregate assignments.
The learner wired `router_entropy_sum` accumulation inside the existing
training-only no_grad block, using full float32 probabilities, then sums over
tokens. `mean_router_entropy` divides by the clamped token count, giving zero
for no data. Two additional tests verify unequal-microbatch weighting and
balanced assignments with confident routing. Existing reset/eval/report tests
now cover entropy too; there were 80 passing MoE tests at that checkpoint.
Keep explanatory docstrings on metric functions, not TopKRouter
(the learner explicitly removed that class docstring). Selected probability mass,
training-loop logging/reset, and the tiny training run remain pending.

Selected probability mass is now implemented. The learner correctly explained
that top-2 of [0.4, 0.3, 0.2, 0.1] retains 0.7 before renormalization; after
renormalization the selected weights always sum to one. The router accumulates
`scores.float().sum()` into a non-persistent scalar, resets it with the other
statistics, and reports `mean_selected_probability_mass` using clamped token
count. Casting BEFORE reduction prevents low-precision overflow/rounding;
a float32 destination buffer alone does not promote the reduction.
Five additional tests cover known mass for K=1/2/E, unequal-microbatch weighting,
and float32 reduction under CPU autocast. Existing registration, reset, eval,
no-data and read-only-report tests also cover the new buffer/metric.
There are now 85 passing MoE tests. The initial seven routing metrics are done;
resume with a tiny training run and per-layer logging/reset once per optimizer
step, keeping task loss separate from routing diagnostics. Do not restart the
metric implementations or start balancing losses before this experiment.

## Handoff

### Latest checkpoint (2026-10-21)

MoE stages 1–7 are complete, including FP32 routing, z-loss integration, and the
0.001-versus-0.01 coefficient discussion above. Next is stage 8: shared experts
and fine-grained specialization. No shared expert has been added and the learner
has not answered the opening motivation question; see `README.md`.
Last verification: 459 repository tests, including 139 MoE tests, pass.

### Historical checkpoint: tiny training run (2026-10-20)

Pending tasks and test counts below describe earlier sessions, not current work.

The learner's `training/train_moe.py` now reuses `train_step` on one full batch,
without gradient accumulation.
Codex added detached per-layer metric history, once-per-step reporting/reset,
the final plotting helper, and regression tests. Fixes include `[B, S]` token
IDs, aligned labels shifted only by the existing helper, device placement,
and gradient clearing through that helper. Default top-2 routing avoids the
constant normalized top-1 weight and its missing task-loss router gradient.

The full batch contributes to each step's statistics. Scalar summary curves and
separate per-layer expert-share/probability curves are saved after training;
consistent expert colors and dashed equal-share references replace the less
readable heatmaps. Diagnostics remain outside
autograd and no balancing objective has been added. A 100-step CPU run reduced
loss from 12.60 to 3.21, but does not yet demonstrate near-zero overfit.
Resume with plot interpretation, then auxiliary load-balancing loss.
Earlier checkpoint notes below are historical.

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
and training at that checkpoint. Router diagnostics are now implemented and
tested; a tiny training run and training-loop logging/reset remain future work.

The learner implemented a non-persistent int64 count buffer, training-only
accumulation under no_grad, and in-place `reset_metrics()` (renamed from
`reset_tokens_count` when probability buffers were added). A class-level
Tensor annotation resolves ty's dynamic buffer type inference without assigning
the attribute before register_buffer. Four assistant-authored tests verify the
diagnostic lifecycle; there are now 46 MoE tests. Resume by deriving MaxVio,
then the other metrics and the tiny training run. Logging/reset in the training
loop has not been implemented yet.
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
