# Mixture of Experts practice

Build a routed MoE from a small readable reference through training and modern
balancing strategies. The learner writes implementations; the assistant reviews
them, writes tests, and records corrected reasoning in `SOLUTIONS.md`.

## Folder layout

- `naive_moe.py` and `expert_batched_moe.py`: the first two implementation steps.
- `experts.py`: `PackedExperts` (looped matmuls) and `GroupedExperts` (grouped MM),
  both using packed weights.
- `grouped_moe.py`: `GroupedMoE` routing, dispatch, capacity, and combination.
- `router.py`, `losses.py`, and `metrics.py`: routing and training diagnostics.
- `expert_parallel.py`: two-process CPU/Gloo EP forward with grouped local experts.
- `tests/`: assistant-authored tests, separated by component.
- `benchmark_moe.py`: comparison of the single-device implementations.

Public imports such as `from moe import GroupedMoE` stay unchanged. To run just
the MoE tests: `uv run python -m pytest -q moe/tests`.

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
10. MoE FLOPs, memory, and expert-parallel systems — stages 1–10 complete
11. Educational expert parallelism in PyTorch — forward verified; regression coverage next
12. LatentMoE, normalization, and SiTU-GLU
13. Quantile balancing: exact implementation to histogram estimation
14. Frontier-model case studies: miniature MoE blocks and training tradeoffs

## Current checkpoint

Stages 1–10 are complete; stage 10 was a reasoning exercise, not distributed
code. Stage 11's two-process EP forward is implemented and verified against a
single-device reference. Next: persistent distributed regression tests.
Completed reasoning and experiments are in `SOLUTIONS.md`.

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

The tiny training run logs seven router diagnostics plus two drop-rate metrics per
layer and resets them together
once per optimizer step. Auxiliary balancing loss is implemented and wired into
training with a configurable coefficient (zero by default). Capacity settings
are passed through the decoder to each MoE layer, along with optional shared
expert width and gating. Shared computation is outside routed capacity/metrics.
`GroupedMoE` supports
an optional `capacity_factor`: `None` is dropless; a positive factor caps each
expert at `ceil(factor * B*S*K/E)` assignments. `drop_policy="probs"` (default)
keeps the highest returned router weights; `"position"` keeps the first in token order.
Surviving mixing weights are preserved, and dropped assignments are not computed.
The learner implemented this algorithm; Codex cleaned up structure and added tests.

### Run and plot

```bash
uv run --extra plots python -m training.train_moe
```

The summary figure contains loss and per-layer scalar curves. Filenames include
`_no_aux` or `_aux_<coefficient>`, plus `_dropless` or `_capacity_<factor>_<policy>`:
for example, `training/plots/moe_metrics_aux_0.05_capacity_1_probs.png`.
The matching `_experts.png` figure shows requested-share, probability, and drop-rate
curves for each expert, one row per layer, with dashed reference lines. The learner
wrote the initial script; Codex added logging, plotting, tests, and training fixes.
A 100-step CPU run reduced fixed-batch loss from 12.60 to 3.21; this shows learning,
not yet complete memorization at that checkpoint; the learner's later 800-step
run reached near-zero loss. Coefficients 0.05 and 0.5 were subsequently compared;
the stronger setting improved balance without an obvious fixed-batch task-loss penalty.

The learner implemented the balancing formula. Codex added explicit loss returns
through the router/MoE/block/decoder, updated generation callers, and integrated
the weighted objective into training. The decoder always returns `DecoderOutput`;
its `aux_loss` and `z_loss` are averaged independently across MoE layers and are
`None` during evaluation. Task, total, and both unweighted router losses are
logged separately. Codex added z-loss plumbing; the learner wrote the loss and
FP32 router. Router weights must remain FP32; use AMP rather than casting the
whole model to BF16. Example:

```bash
uv run --extra plots python -m training.train_moe --aux-loss-coefficient 0.05 --z-loss-coefficient 0.001 --output training/plots/moe_balanced.png
```

### Routing metrics to learn

Implement progressively, per layer, rather than copying an entire framework:

- Implemented: assignment counts/shares, unused experts, and maximum load violation
  (MaxVio); accumulate counts over an optimizer step, then report and reset.
- Implemented: mean all-expert probabilities, per-token router entropy, selected
  probability mass before top-k renormalization, and global/per-expert drop rates.
  Training also logs task, total, and unweighted balancing loss separately.
- Implemented: FP32 router computation and separately weighted/logged z-loss.
- Implemented: selection-only balancing biases and explicit training-step updates.
- Later: per-device load/throughput. Validation is still needed
  for quality claims beyond the fixed-batch memorization experiment.

Counts are detached diagnostics, not a differentiable balancing objective.
Exclude padding when supported; do not conflate step-level balance with balance
in each microbatch. Research references and scope are in `SOLUTIONS.md`.

Both versions reuse `mlp.SwiGLU`, whose projections now default to `bias=False`.
The packed representation stores up/gate weights `[E, D, 2F]` and down weights
`[E, F, D]` for direct right multiplication, with no expert bias tensors.

The eight test files in `tests/` are assistant-authored. There are 172 passing MoE tests,
including ten EP preparation/local-computation cases, not distributed integration tests.
`benchmark_moe.py` compares inference and forward/backward runtime for all
four versions. The CPU results and reproduction command are in `SOLUTIONS.md`.

## Resume here (2026-10-23)

Stages 1–10 are done. Do not repeat integration, diagnostics, capacity, FP32 router
computation, or z-loss wiring. The learner wrote the losses/router changes;
Codex wrote tests and requested model/training/plot plumbing.

Dropless runs with auxiliary coefficient 0.05 and z-loss coefficients 0.001 and
0.01 were compared. The stronger coefficient lowers late-stage z-loss, produces
softer routing, and shows worse late-stage MaxVio in several layers. Softer
preferences and lower selected mass are not themselves quality failures; both
runs still memorize the batch. Neither large-scale stability nor generalization
has been demonstrated. Details and plot links are in `SOLUTIONS.md`.

Stage 8's shared branch and optional input-conditioned sigmoid gate are implemented
and tested. The learner understands common-versus-specialized knowledge, gate
broadcasting, and fine-grained routing's parameter/compute tradeoffs. Codex added
tests and requested forwarding through block/decoder/ModelConfig, plus CLI and
plot tags. Each layer owns separate shared parameters; defaults remain off.
Do not repeat these tasks or introduce another dispatch implementation.

The auxiliary-0.05, z-0, gated-shared-width-24 run was reviewed: task loss still
approaches zero, while routing changes and some layers have worse load balance.
This is not evidence of improved specialization or generalization. Shared width
24 adds 25% expert matmul work to default routed width 48/K=2. Existing expert
count, width, and top-k settings already support fine-grained routing; no separate
implementation is required. Further toy comparisons are optional, not a blocker.

Batch generation now uses a separate seed-42 generator, independent of model
initialization. This changes the data relative to older plots; weights remain
unmatched across architectures, as requested by the learner.

Stage 9: the learner understands selection-only balancing corrections, unbiased
mixing weights, buffer state, and once-per-optimizer-step updates. The persistent
zero-initialized `expert_bias` and gather-based unbiased weights/mass are now
implemented/tested. The explicit update method and once-per-step placement are
also implemented by the learner. Codex completed CLI/history/plot wiring and
validation. `--bias-update-rate 0.001` enables updates; default 0 disables them.
A positive rate requires auxiliary coefficient zero in this runner; z-loss is
independent. Requested counts drive the update before metric reset. Tests verify
timing, no-ops, invalid rates, exclusivity, persistence, and filenames.
The softmax bias-0.001 run was reviewed: near-uniform assignments and low MaxVio
coexist with increasingly selective token affinities and near-zero task loss.
The learner also implemented sigmoid scoring and the normalized diagnostic view.
Codex forwarded `score_func` through all MoE variants, block, decoder, ModelConfig,
and CLI, with mode-specific plot labels/tags and a sigmoid/nonzero-z-loss guard.
Both scoring paths are tested, and their bias-0.001 runs with gated shared width
24 have been compared. Sigmoid has softer normalized per-token affinities:
higher entropy and lower selected normalized mass, while both runs balance
assignments and memorize the batch. This does not mean fewer experts execute or
prove weaker specialization. Detailed interpretation is in `SOLUTIONS.md`.
Do not repeat the implemented scoring, normalizations, wiring, or experiments.
At the learner's request, stage 9 explicitly includes softmax versus sigmoid
routing: independent affinities, top-k ordering, selected-score normalization,
and interaction with selection corrections. Configurable scoring is implemented,
with softmax retained as the default. Sigmoid affinities are normalized across
all experts for entropy, mean-affinity logging, selected-mass logging, and the
balancing loss. Selection/mixing use original affinities. Sigmoid returns z-loss
None and the runner rejects a nonzero z-loss coefficient; plots omit its curve.
Existing metric keys remain unchanged, but sigmoid labels explicitly describe
normalized affinities rather than original categorical probabilities.
Sigmoid scoring alone is not a balancing method. Use the
[DeepSeek-V3 reference gate](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)
to connect the educational design to a model implementation.
Keep the existing balancing loss available as a baseline.

Stage 10 is complete: compute/parameter formulas, view-aware activation storage,
dispatch/return ownership, weighted combination, all-to-all versus all-reduce,
BF16 forward payload 4TKrD, remote fraction r=1-1/R under uniform placement and
routing, topology, stragglers, and grouped-matmul overhead. T is group-wide in
the communication formula. Notes in `SOLUTIONS.md` include the assumptions;
these are not exact peak-memory or runtime predictions. No accounting helper
was required; the subsequent EP forward implementation is described below.

### Stage 11: educational PyTorch EP forward

The learner implemented preparation, variable-split dispatch, local expert
grouping/computation, both scatter-based order restorations, return exchange,
and weighted combination. Codex supplied requested skeletons/launch plumbing,
fixed example mixing weights, comments, reviews, and tests. `all_to_all_warmup`
keeps the initial one-dimensional exchange as a reference.

The demo uses two real CPU/Gloo processes, four experts, top-2, D=16, F=32,
fixed expert IDs and unequal normalized weights. Each rank owns two
`GroupedExperts` entries. The router is not invoked and no assignments drop.
Global-to-local IDs, empty-count entries, the two permutations, and return
splits are explained in `SOLUTIONS.md`.

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=2 --module moe.expert_parallel
uv run python -m pytest -q moe/tests/test_expert_parallel.py
```

Both ranks produce [2,16] combined outputs. An ad-hoc two-process check using
seed 42+rank reproduced each owner's actual weights in a single-device
per-token/per-slot reference; both weighted outputs matched. That check has
not yet been saved as a regression test. The ten existing tests cover only
preparation and local expert computation, including empty local experts.

The forward-only description above records the original checkpoint. At the
learner's request, Codex subsequently added `expert_parallel_backward.py` as a
separate, commented teaching example; the learner's forward file is unchanged.
Its custom autograd all-to-all reverses send/receive splits in backward. Both
activation exchanges use it; counts and expert IDs remain ordinary metadata.
The demo includes a learned replicated router, local expert/input gradients,
and an explicit router-gradient SUM for the sum of rank-local losses.

```bash
uv run python -m torch.distributed.run --master-addr=127.0.0.1 --nproc-per-node=2 --module moe.expert_parallel_backward
uv run python -m pytest -q moe/tests/test_expert_parallel_backward.py
```

The persistent two-process test compares outputs and input, mixing-weight,
router, and expert gradients against independent single-device math using
identical weights. It covers mixed routes, local-only routes/zero-count peers,
an empty receiving rank, and an empty source rank, with bounded subprocess
timeouts. All ranks must traverse both communication nodes in backward; the
example requires grad-enabled tokens on every rank, including empty ones.
Only first-order gradients are supported. No decoder integration is implied.
The launch uses an explicit loopback master address because standalone hostname
rendezvous hung on this machine. Default port is 29500; use `--master-port` to
choose another free port if needed.

Next learning topic: stage 12, LatentMoE, normalization, and SiTU-GLU.

CPU processes teach correctness, not GPU performance. CUDA/NCCL is a later run
when multiple GPUs are available. The local workstation has one CUDA GPU;
its PyTorch build is configured for CUDA 12.9 (see the root README).
The new example explicitly separates token-gradient communication from
gradient synchronization of replicated router parameters. Ordinary forward
collectives alone do not supply this backward path.
No capacity policies, decoder wiring, custom kernels, or overlap are needed
for the initial exercise. Ask one question at a time; the learner writes the
algorithm, Codex writes tests and requested launch plumbing.

References: [PyTorch all-to-all](https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.all_to_all_single)
and [torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html).

Last verification: 517 repository tests pass. The new backward source and tests
pass Ruff, formatting, and ty checks; the original EP files are unchanged.
A real two-rank CPU forward and the ad-hoc distributed-reference comparison
passed. The subsequent backward demo and persistent two-rank
gradient-equivalence test also pass on CPU.
Multi-GPU EP performance remains unverified.
Keep explanations short, ask one question at a time, and let the learner write
the algorithm. Codex writes tests and requested plumbing.

### Historical checkpoint (2026-10-20)

The following records earlier progress; pending tasks below have since been
completed. The current resume instructions are above.

Step 3 and the decoder-integration part of step 4 are complete. The learner
explained that MoE replaces the FFN while preserving `[B, S, d_model]`, so
normalization and residual connections stay in place. Integration tests check
dense/MoE selection, missing `top_k`, causality, gradients through routers and
experts, and full-versus-cached logits with both full and sliding-window
attention. GPU grouped-kernel speed remains unmeasured.

The learner correctly proposed per-expert assignment counts, accumulation over
microbatches, and collecting diagnostics outside autograd. They identified
`self.register_buffer` for keeping counts on the module's device.

`TopKRouter` now has an int64 non-persistent `tokens_per_expert` buffer,
training-only accumulation outside autograd, and `reset_metrics()`.
Tests cover accumulation, evaluation isolation, reset, and buffer registration.
`moe.metrics.maxvio` now computes relative overload versus the mean count,
returning a scalar zero for no assignments. It is detached and leaves counts
unchanged; controlled-value and autograd tests pass.
`assignment_fraction` now returns per-expert shares of total assignments, with
a zero vector when no assignments exist. Value, shape, scale-invariance, and
autograd tests pass. `unused_expert_count` now returns a scalar count of experts
with zero assignments, including E when no assignments exist. Aggregate
`assignment_entropy` is also implemented in bits with protected logarithms and
tests for known values, invariance, and no autograd. Next: mean all-expert router
probabilities: their full-expert sums and token count are now accumulated across
microbatches and reset by `reset_metrics()`. Tests cover unequal microbatch sizes
and isolation from evaluation/autograd. `compute_metrics()` now reports the four
count-based metrics and mean router probabilities without resetting buffers.
No-data, controlled-value, and repeated-read tests pass. `router_entropy` now
returns per-token entropy in bits from full probabilities, with tests explicitly
distinguishing it from assignment entropy. The router now accumulates token
entropies and reports `mean_router_entropy`, weighted by token count across
microbatches. No-data, reset/eval isolation, and unequal-microbatch tests pass.
`selected_prob_mass_sum` now accumulates original selected probabilities in
float32, before renormalization. `mean_selected_probability_mass` divides by
token count; no data yields zero. Tests cover K=1/2/E, unequal microbatches,
autocast reduction, reset, and evaluation isolation.
The tiny training run now logs these seven metrics separately per layer,
reading and resetting once per optimizer step on one full batch (no gradient accumulation).
The learner has implemented the balancing loss and understands its gradient
path through full softmax probabilities. Explicit per-layer loss returns and
training integration are now assistant-authored. Coefficient experiments and
their limitations have been discussed. Both position- and score-priority capacity
dispatch are implemented and tested in `GroupedMoE`, with a dropless fast path
and unchanged pre-drop router statistics. Per-expert dropped counts now
accumulate during training; MoE-level reset clears both dropped and requested
counts. MoE-level reporting includes global/per-expert drop rates with
zero-request protection and no implicit reset. Codex wired decoder/block capacity
configuration, CLI options, MoE-level reporting/reset, drop-rate plots, and run tags.
Next: compare dropless, position, and probs runs at the same capacity factor.
Do not repeat the completed loss plumbing.
The learner asked to include practitioner metrics from frameworks and recent model reports.
Do not repeat decoder integration or the implemented diagnostics.
Keep explanations short and concrete; introduce one shape change at a time.
The learner writes the implementation; the assistant writes tests and reviews.

Last verification: 444 repo tests pass, including 130 MoE tests.
Targeted model/MoE/generation/training Ruff/ty checks pass. Run
`uv run pytest moe transformer_block decoder_model training -q` when resuming.
For debugging,
use the reusable VS Code module launch with `moe.grouped_moe`.
