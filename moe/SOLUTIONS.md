# Mixture of Experts: completed reasoning

## Final architecture synthesis (stage 14, completed)

The final interview reviewed existing components rather than adding another
assembly wrapper. All 14 MoE stages are complete for educational scope. This is
a miniature frontier-style design exercise, not a faithful named-model replica
or evidence of production training quality. Older checkpoints below record the
learning sequence; their "next" steps are historical, not outstanding tasks.

### One compute budget, different communication costs

Let D=d_model, F=hidden_dim, K=top_k and E=num_experts (E is not model width).
Ignoring biases and elementwise operations, a SwiGLU expert has 3*width*F weights;
routed matmul FLOPs per token are approximately 6*K*width*F (multiply-add = 2).
The exercises compare only this routed term, not whole-block FLOPs or latency.

| Design | E | K | Expert input/output width | F | Routed FLOPs / baseline | Activation payload / baseline |
|---|---|---|---|---|---|---|
| Baseline | 8 | 2 | D | 512 | 1 | 1 |
| Fine-grained | 16 | 4 | D | 256 | 1 | 2 |
| Latent, same F | 16 | 4 | D/4 | 256 | 1/4 | 1/2 |
| Latent, recovered compute | 16 | 4 | D/4 | 1024 | 1 | 1/2 |

Payload means activation bytes sent per assignment at fixed dtype and remote
routing fraction, without deduplication; excludes IDs/counts and communication
protocol overhead. Return vectors have the same width, so the same ratios apply
to the combined dispatch/return activation volume. Assignments grow from 2*T to
4*T. The latent payload ratio is (4*D/4)/(2*D)=1/2, independent of F.

The learner initially reused this 1/2 communication ratio for expert compute.
Compute also depends on F: at latent width D/4, F=512 gives half the baseline
routed compute, while F=1024 restores it. Hidden activations stay on the expert's
rank, so increasing F increases compute, parameters and local activation memory,
not dispatch-vector width. Baseline and fine-grained expert parameter counts are
equal: 8*3*D*512 = 16*3*D*256. Equal active compute does not imply equal total
parameters; the final latent design has twice the baseline routed parameters.

### Placement and whole-block costs

The router reads original [T,D] features, avoiding a forced routing bottleneck.
The shared expert is replicated and full-width, giving every token a path outside
the routed latent compression. Replication avoids its EP activation exchanges,
not training-time synchronization of replicated parameter gradients. A remote
shared branch could instead benefit from compressed messages, with a different
capacity/placement tradeoff. Compression does not automatically improve latency.

Equal routed-expert FLOPs excludes latent down/up projections, latent RMSNorm,
the new full-width shared branch, and the larger router (16 rather than 8 outputs).
More assignments also add dispatch/combination work and metadata. Kernel launch,
memory traffic and communication costs matter. Shared/routed computation does
not automatically overlap; that needs explicit scheduling. Do not call the whole
block compute-matched or faster without including these costs and measuring it.

### Routing, balancing and low precision

Use raw scores plus selection bias for top-k, then normalize the selected RAW
scores for mixing. Scores are learned affinities, not guaranteed expert quality;
balancing encourages, rather than guarantees, equal future load.

With four microbatches per optimizer step, freeze biases/bin boundaries across
all four, sum histogram counts, optionally SUM across the process group, then
estimate replacement biases and reset once at the optimizer boundary (the runner
does this after optimizer.step). These statistics describe the completed window,
not scores recomputed with the newly updated model weights.

Low precision does not inherently enlarge activations. SwiGLU outliers can force
a larger shared block scale, reducing resolution for smaller neighbors and
potentially rounding them to zero. SiTU-GLU softcaps gate/value branches to reduce
outliers and their scale distortion. Its intermediate magnitude is bounded by
beta_gate*beta_value; the subsequent down projection is not bounded by that
constant. This supports numerical stability without guaranteeing stable training.

### Completion boundary and verification

GroupedMoE already exposes the components discussed; at the learner's request,
no redundant configuration module or new implementation was added for stage 14.
Educational EP forward/backward and histogram pooling have their own two-rank
tests, not an integrated distributed latent decoder. Distributed exact gathering,
full distributed training, latent/SiTU decoder/runner forwarding and production
optimizations remain optional extensions. Stage 14 adds reasoning notes, not a
new runtime or numerical benchmark. See ROADMAP.md for subsequent broader topics.

## Quantile balancing (stage 13, educational scope complete)

Completed: exact/histogram estimators, optimizer-window lifecycle, local training
integration, toy-run interpretation and optional distributed histogram pooling.
Latest focused verification: 328 MoE/training tests passed; lint and typing passed
for the pooling implementation and its tests. Full distributed training and exact
global score gathering are not implemented. The final design review is complete
above; no MoE stage remains active.

Toy-run interpretation: the learner's 800-step sigmoid/histogram run with four
experts and top-2 routing maintained assignment shares near 25% per expert (about
50% of tokens per expert), low MaxVio and near-maximal assignment entropy. Task
loss approached zero on the fixed batch: useful balance/memorization evidence,
not a generalization or large-scale performance claim.

### Distributed histogram extension

`update_quantile_bias(process_group=group)` performs an in-place SUM all-reduce of
counts `[num_experts,num_bins]`. Pool counts, not quantile estimates: quantiles of
individual shards cannot generally be averaged into the pooled quantile.
Each histogram row contains one observation per token, so its pooled row sum is
the global token count (no division by top-k and no second collective needed).

The collective precedes the empty-window guard. A locally empty rank must still
participate or peers can hang; a globally empty window safely leaves biases alone.
All ranks use identical old biases/bin boundaries, estimate the same new bias, and
reset their workspace. The group must contain replicas of the same router, with
the intended token population counted once. Caller owns initialization and group
selection. `None` deliberately means local, not the default world group.

Distributed exact mode raises before changing the window: it would need global
raw-score gathering, not an average of local quantiles. The sign-based alternative
would instead SUM assignment counts `[num_experts]`, then apply the sign rule;
this extension does not implement that path or distributed training-runner wiring.
Two-rank CPU/Gloo reference tests exercise unequal and empty shards, one/multiple
bins, successive windows and default-local behavior under an initialized group.

Interview questions and corrected reasoning:

- With T tokens, E experts and top-k K, equal target count is `T*K/E`, or fraction
  `K/E` of tokens for each expert (25% for E=8/K=2).
- A raw expert score .4 and fixed cutoff .6 require bias .2 to reach the boundary.
  Required bias is cutoff minus RAW score, not biased score.
- One expert has a list of required biases over tokens. A chosen bias covers the
  tokens needing less than that bonus, so a 25% target uses the lower 25% quantile,
  not the 75% quantile. The latter confusion came from thinking of high-score tails;
  smaller required bias instead means easier to satisfy.
- The quantile supplies a replacement target, not an increment to the current bias.
  Changing all experts simultaneously changes cutoffs, so a frozen-cutoff update
  does not guarantee perfectly equal next-batch routing. Ties also need care.
- Mean-centering removes a common offset without changing selection: biases
  [10.1,10.3,10.2] and [-.1,.1,0] have identical pairwise differences.
- For top-2 with distinct scores [.9,.7,.4,.2], both `score >= .7` and `score > .4`
  describe the winners. The learner's second-score/>= argument was valid; the
  interviewer initially phrased this ambiguously. The algorithm specifically
  freezes the third score, strongest rejected, with the strict-> convention.
  These two descriptions do NOT give interchangeable bias targets. An unselected
  expert entering exact top-k with other scores fixed must displace the kth score;
  the shared frozen cutoff used by QB is a different construct.
- Broadcasting needs one cutoff per token: `[T,1]` minus `[T,E]` yields `[T,E]`.
  Quantiles reduce the token axis (`dim=0`) to return one target per expert `[E]`.
  Without dim, quantile flattens everything; returning the mean instead of
  subtracting it also mistakenly collapses the expert dimension.

The learner's `TopKRouter.estimate_quantile_bias` implements this exact update
under no-grad. Codex supplied the requested cutoff-operation example, reviewed,
added short comments/docstrings, and wrote 10 tests, not the remaining algorithm.
The method retains explicit score/bias/K inputs and does not mutate state. The
documented initial domain is nonempty sigmoid scores and `1 <= K < E`; full
integration must explicitly address invalid inputs, ties and all-expert routing.
The hand-check test uses PyTorch's default linear quantile interpolation; finite
samples and ties do not imply that a single estimate yields exactly target counts.

Source: [Kimi K3 quantile balancing](https://arxiv.org/html/2607.24653v2#S2.SS3.SSS3).
Pooled-quantile reasoning and histogram accumulation are now covered below.
Exact buffers are O(T*E); histograms summarize distributions with per-expert counts
and aggregate counts, not independently estimated quantiles. The sign-based
update/forward remain unchanged. Histogram extraction is now implemented and
tested below. At this historical estimator checkpoint, QB training-window wiring,
bias application and distributed synchronization were not implemented; the
completed lifecycle and optional histogram synchronization are documented above/below.

### Histogram distribution reasoning and accumulation

- Can we average microbatch quantiles? No: batches [0,0,0,0] and [8,8,8,8]
  have 25% quantiles 0 and 8, whose average 4 differs from pooled quantile 0.
- What can we sum? Histograms with identical boundaries: sum bin counts across
  microbatches/ranks before extraction. Distributed SUM would communicate counts,
  not independently calculated quantiles.
- Example counts [2,5,1,2] in bins [0,1), [1,2), [2,3), [3,4] have cumulative
  counts [2,7,8,10]. The 25% target count is 2.5, located in the second bin.
  There are 2 counts before it and 5 inside it; uniform-within-bin interpolation
  moves (.5/5)=.1 of its width, giving 1+.1=1.1. This does not mean half a token
  is selected; it estimates a location in a summarized distribution.
- Actual example values [.2,.8,1.1,1.3,1.5,1.7,1.9,2.4,3.2,3.8] produce those
  counts. Histogram interpolation gives 1.1 versus torch.quantile's 1.15. There is
  both within-bin information loss and a different finite-sample interpolation
  convention (CDF target q*T versus PyTorch linear sample index q*(T-1)).
- Since sigmoid scores lie in [0,1], biased cutoffs lie in [b_min,1+b_max].
  Subtracting raw scores gives safe required-bias bounds [b_min-1,b_max+1], not
  [0,1]. Freeze these bounds for all histograms intended to be added together.
- Zero-based bins are floor((required_bias-lower_bound)/bin_width), clamped to
  [0,num_bins-1] so the upper endpoint goes into the last bin. For range [-1,1]
  with four bins, required bias .3 gives floor((.3+1)/.5)=2.
- Transpose indices [T,E] to [E,T], then scatter-add integer ones into an [E,B]
  histogram along the bin axis. Every expert receives one observation per token,
  including experts not chosen by top-k. A row's added counts sum to T.

The learner explicitly requested this part be implemented for them after finding
the steps blurry. Codex wrote the numbered-comment `accumulate_bias_histogram`
method and tests. After the learner explicitly requested that validation live
elsewhere, all helper checks and their five test cases were removed. The remaining
tests cover computation, accumulation and nonmutation. Valid shapes, int32/int64
histograms, shared devices, positive bin count and `1 <= K < E` are now caller
preconditions, to be checked at integration boundaries, not inside this helper.
It updates an externally supplied integer count buffer in
place, returns lower bound/bin width, leaves router state and inputs unchanged,
and supports adding empty or multiple microbatches. Fixed-window bias/bin identity
is a caller obligation; it does not silently maintain bounds or reset counters.
Work is vectorized in FP32 (or preserved FP64), with no Python token/expert loops
or value-dependent host synchronization. Persistent count storage is O(E*B), but
the educational helper still materializes O(T*E) temporary margins/bin indices per
microbatch; it is not a fused production histogram kernel.

The learner subsequently completed histogram extraction as follows; full-step
lifecycle, distributed count pooling and bias replacement remain next.

### Histogram quantile extraction (learner-written, tested)

The learner wrote `estimate_bias_from_histogram` in small reviewed steps:

1. Cumsum over bins gives `[E,num_bins]`. The last entry of any row is T, not E.
   The target is `T*K/E`; an initial `K/T` expression was corrected.
2. Find the first cumulative count >= target using batched `torch.searchsorted`.
   The scalar target expands to `[E,1]`; contiguous avoids an internal-copy warning.
   Index outputs are `[E,1]`, including when there is only one chosen bin per row.
3. Gather from histogram and cumsum along dim=1. Their difference is count-before,
   which avoids indexing bin_idx-1 and special-casing bin zero. Gather preserves
   index shape `[E,1]`, not `[E]`.
4. Fraction within the selected bin is `(target-count_before)/bin_count`. Bin left
   edge is `lower_bound + bin_idx*bin_width`; estimate adds fraction*width.
5. Squeeze dim=1 to `[E]`, subtract the estimates' mean, and return new bias targets.

Codex reviewed each step, supplied operation/shape guidance, then added comments
and tests. Tests use independent scalar-loop histogram interpolation, explicit
hand values, exact cumulative boundaries with empty gaps (left search must not
select a zero-count plateau bin), no mutation/autograd, strided counts, expert
permutations, shifted lower bounds and accumulated-microbatch distributions.
No library-style input validation was reintroduced. The helper assumes equal
positive row sums, valid K/bin width, and compatible tensors; empty global windows
must be handled by the caller. Empty local microbatches are already supported by
accumulation, but a globally empty histogram has no quantile to extract.

Next interview topic: window lifecycle and bias-update timing, before wiring
training/CLI or distributed collectives. Both exact and histogram estimators now
exist. Forward accumulation is subsequently enabled explicitly, as recorded
below; bias replacement is still not selected by training. Do not repeat the
completed extraction or silently replace the sign-based update.

Counter dtype follow-up: the learner chose int32 because counters reset every
optimizer step, rather than accumulating over the full run. At their request
Codex changed scatter-add source ones to `histogram.dtype`; indices remain int64.
Int32 therefore halves persistent counter storage while int64 remains supported.
Pooled counts per bin must fit the chosen dtype (int32 max 2,147,483,647); the
caller owns this assumption. Cumsum promotes integer counts to int64 by default.
There are now 30 accumulation cases including both counter dtypes, plus 17
extraction cases including int32/int64 equivalence. The learner correctly chose
a registered non-persistent histogram buffer: it survives microbatches and moves
with the device, but is neither learned nor part of checkpoints. They requested
Codex implement this repeated plumbing rather than repeat a familiar exercise.
The initial buffer implementation used keyword-only `num_bins` with effective default 256 and zero int32 `bias_histogram`
of shape [E,num_bins]. Positive bin-count validation lives at construction, not
in the computation helpers. Six tests check configuration, registration,
checkpoint omission, device/dtype tracking and isolation from forward/metric
reset. The actual training-window lifecycle remains next. The learner clarified
that this delegation was only for this instance to focus on quantile balancing;
do not generalize it into a standing rule to take over repeated wiring.

Forward-accumulation checkpoint (2026-10-25): the learner chose all raw scores,
not only selected mixing weights, because every token contributes a required
bias to every expert. They added the default-off `use_quantile_balancing` flag
and train-only helper call. Review caught a hard-coded False field (ignoring the
argument) and unflattened [B,S,E] input; the learner corrected both. Bias remains
fixed during these forwards. Codex supplied requested boundary checks: enabled
QB rejects K=E, and explicit num_bins is allowed only with QB. An omitted
num_bins=None originally resolved to 256; the mode-selection change below supersedes it.
Six additional tests cover conditional configuration, ordinary all-expert routing,
both score functions, accumulated counts across forwards, matching baseline
outputs/gradients, unchanged biases, and isolation from evaluation/metric reset.
The later local step update below implements replacement/reset. Distributed
pooling and model/training wiring remain next. Do not perform sign updates and
QB replacements together.

### Exact versus histogram mode and local step update (2026-10-25)

The learner requested `num_bins=None` select exact QB, and a positive count select
histogram QB, instead of keeping a seemingly stale exact method only for tests.
The default-off use_quantile_balancing flag still controls whether QB runs at all.
An explicit num_bins while QB is off is rejected; enabled QB requires K<E.
Disabled/exact modes register bias_histogram=None instead of allocating bins.

Exact mode cannot average microbatch quantiles. Codex therefore added a temporary
list of detached raw score chunks [T_i,E] and concatenates it once at the step
boundary, then calls the learner's exact estimator using the unchanged old bias.
This uses O(T*E) on-device storage for the whole window and is intentionally a
small-scale reference, not the scalable global method. A Python list avoids
repeated growing tensor concatenations but does not automatically follow module
device moves or checkpoints: complete/reset a window before those transitions.

The learner's histogram update correctly recovers lower/upper bounds from the old
bias before estimation, copies the result in place, then zeroes counts. Codex
preserved that algorithm and supplied the exact branch and guards against empty,
disabled or evaluation updates. The empty histogram guard reads one row total
and synchronizes once per step; no per-microbatch host check was added. Empty
windows must not replace biases with NaNs. Repeated update calls after reset are
no-ops. New biases take effect on the next forward, not the batch that estimated
them; diagnostic metrics reset is independent of both accumulation strategies.

Nine tests compare complete window estimates in both modes/score functions,
including two successive steps (fresh bounds), detached exact snapshots,
non-checkpointed temporary state, bias buffer identity, task-gradient survival,
empty windows, disabled/eval no-ops and reset behavior. Codex added these tests
and requested mode wiring; the histogram-update algorithm remains learner-authored.
This local-mode change did not add distributed raw-score gathering or histogram
all-reduce. Model/runner forwarding was subsequently completed below.

### Calling QB once per optimizer step (requested wiring)

The learner identified training-loop update insertion as pure wiring and explicitly
asked Codex to implement it. QB/bin options now traverse ModelConfig, DecoderModel,
TransformerBlock and GroupedMoE to each TopKRouter. The toy runner calls update
after train_step returns (backward, clipping and optimizer.step already completed),
then records the completed step's diagnostics and resets metrics. QB reset only
clears its own window, not diagnostic counts. The existing rate-based sign update
is the exclusive fallback when QB is disabled. Neither strategy runs by default.

The runner rejects enabling QB and a positive sign rate together; auxiliary
balancing must also be zero when either strategy is active, matching the existing
runner policy. This exclusivity is a controlled-experiment choice, not a universal
claim that bias balancing and auxiliary losses cannot be combined in research.
Softmax z-loss remains independent; sigmoid still rejects a nonzero z coefficient.
CLI flags are --use-quantile-balancing and optional --num-bins. None selects exact
and positive bins select histogram. History and plot tags distinguish the modes,
preventing accidental plot overwrites between exact/histogram/sign baselines.

Ten assistant-authored tests verify every layer receives configuration, updates
occur once after each optimizer step, sign updates cannot also run, diag counts
remain available for logging, default behavior stays off, conflicts are rejected,
and both exact/histogram plots are saved under distinct filenames. Both sigmoid
and softmax runs execute in timing tests. This is local CPU correctness, not
distributed pooling, a large-scale balance guarantee or a model-quality claim.
The later toy-run interpretation and optional histogram all-reduce extension are
now complete (see the current checkpoint above). Do not repeat local wiring.


## LatentMoE: routed width versus expert intermediate width (stage 12, completed)

The learner implemented optional `d_latent` in `GroupedMoE`; Codex reviewed and
added tests, not the algorithm. Down/up projections surround only the routed
path. The router and optional shared expert/gate stay at `d_model`.

Interview questions and corrected answers:

- How does dispatch/return payload change from width 4096 to 1024 at fixed
  tokens, top-k, and dtype? It becomes one quarter, excluding routing metadata.
- Where does down-projection happen? On the source rank before dispatch;
  projecting after receipt would not reduce sent activation bytes.
- Where does up-projection happen? On the source after return and weighted
  combination, keeping both exchanges narrow and projecting once per token.
- Why can projection follow combination? A shared linear map commutes with
  weighted summation. A nonlinear map generally does not. The implementation
  uses bias-free projections, so even a token with all assignments dropped
  retains a zero routed contribution.
- What enters an expert? `[N_e,d_latent]`, not `[N_e,d_model]`. Its fused value/gate
  projection produces `[N_e,2*hidden_dim]`; SwiGLU produces `[N_e,hidden_dim]`;
  its final projection returns `[N_e,d_latent]`.
- Does each expert grow? No: its bias-free SwiGLU parameters change from
  `3*d_model*hidden_dim` to `3*d_latent*hidden_dim`. Shared down/up projections
  add `2*d_model*d_latent` parameters separately.
- What if top-k grows fourfold while routed width shrinks fourfold? The leading
  activation payload is unchanged, but more experts are active. This connects
  to fine-grained routing, which instead narrows expert intermediate width.
  It does not prove identical runtime: projections, metadata, grouping and
  communication overhead still matter.
- Why not wrap the whole MoE in the projections? That would also compress router
  and shared-expert inputs. In the target design both retain original features.
- Why not project grouped assignments? It can be correct before dispatch, but
  repeats the same projection up to top-k times. Project original tokens once,
  then gather their latent representations.

Source: [NVIDIA LatentMoE](https://research.nvidia.com/labs/nemotron/LatentMoE/).
Our implementation is a single-device educational extension, not a complete
reproduction or distributed performance result. Existing packed/grouped experts,
capacity rules, diagnostics and loss returns are reused. Codex's 18 new tests
check independent forward/gradient equivalence and architectural boundaries.
The original plain latent path above was subsequently extended with normalization
as recorded below. SiTU-GLU backend support and optional GroupedMoE forwarding
are also implemented/tested. Next: quantile balancing. Model/runner forwarding and latent EP
integration are not implemented. Ask one question at a time; learner writes code.

### SiTU-GLU backend activation (implemented and tested)

The learner derived the following reasoning during the interview:

- SiLU approaches the identity for large positive gates and zero for large
  negative gates. The unbounded value branch can multiply a large positive gate
  response, creating large intermediate activations before the final latent norm.
- `softcap(x,beta)=beta*tanh(x/beta)` is bounded in magnitude by positive beta.
- Cap the linear gate factor and the value factor independently, but apply
  sigmoid to the original gate. Hidden activation magnitude is bounded by
  `beta_gate*beta_value`; the following expert projection is not bounded by this.
- Near zero, tanh is approximately identity, making both softcaps approximately
  identity and recovering SwiGLU behavior for small inputs.
- An optional `(beta_gate,beta_value)` tuple avoids duplicating expert classes.
  None must select the uncapped SwiGLU path, not substitute unit caps.

The learner wrote the initial packed implementation. At their request Codex
cleaned it up and applied it to grouped experts using `_expert_activation` and
`_validate_softcap_betas`. This corrected two remaining bugs: division rather
than multiplication inside the gate tanh, and rejection if EITHER cap is
nonpositive OR nonfinite. Store the optional tuple once instead of two optional
attributes. The uncapped branch uses the existing functional SiLU primitive.
Both expert backends retain their original parameter layout.

Codex added 26 tests for both-position invalid betas, scalar formula and bounds
with unequal caps, double-precision numerical gradient checks, and packed versus
grouped outputs/input/parameter gradients with empty expert groups. At the
learner's request, Codex subsequently wired the tuple through GroupedMoE only to
routed experts, with 20 additional integration cases. These cover capped reference
outputs/all gradients across capacity policies and router score modes, shared
branch isolation with/without latent width, both backends and validation forwarding.
Default experts and the shared branch remain SwiGLU. Docstrings describe
`softcap_betas=(beta_gate,beta_value)` as finite positive non-learned cap magnitudes;
None selects classic SwiGLU. End-to-end model/runner configuration and distributed
latent integration remain outside scope. Next learning topic: quantile balancing.

#### Microscaling and why SiTU-GLU helps low precision

Microscaling represents a tensor as small blocks of low-precision numbers plus
one shared scale per block. Reconstruction is approximately
`x = low_precision_value * block_scale`; quantization rounds `x / block_scale`
to the low-precision format. The scale adapts the representable magnitude range;
it does not change the mathematical activation or act like learned RMSNorm.

Conceptual example (illustrative scales, not the scale-selection algorithm):

| Stored low-precision values | Shared scale | Reconstructed values |
|---|---:|---|
| [1, 2, 3] | 1 | [1, 2, 3] |
| [1, 2, 3] | 128 | [128, 256, 384] |

MXFP8 uses one E8M0 power-of-two scale for each block of 32 values. The scale is
derived from the block's maximum absolute value so its largest finite value can
fit into the FP8 payload range. Unlike tensor-wide scaling, different blocks can
adapt to different magnitudes. Scaling extends range but cannot restore the
mantissa precision lost by quantization. An outlier can still force its block's
scale upward and degrade the representation of smaller neighboring values.
Source: [NVIDIA MXFP8 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html).

Low precision does not inherently make the real activations explode. The model's
unbounded projections and SwiGLU multiplication produce outliers; restricted
precision makes those outliers more problematic through quantization error,
underflow/saturation and possible overflow in intermediate arithmetic. A suitable
block scale mitigates payload overflow but does not guarantee stable training.

The Kimi K3 paper explicitly motivates SiTU-GLU with activation-outlier and
low-precision overflow concerns. Smooth-capping both multiplicative factors
limits the intermediate product and helps quantization robustness, but does not
bound the following expert projection. It is a general stability mechanism,
particularly useful in low precision, not a complete FP8/FP4 stability solution.
Its post-training recipe uses MXFP4 routed-expert weights and MXFP8 activations;
non-expert components remain higher precision. This does not mean all activations
are MXFP4, and MXFP4 should not be confused with NVIDIA's distinct NVFP4 recipe.
Sources: [SiTU-GLU motivation, §2.3.2](https://arxiv.org/html/2607.24653v2#S2.SS3.SSS2),
[post-training precision, §4.1.4](https://arxiv.org/html/2607.24653v2#S4.SS1.SSS4).

### Normalized LatentMoE (implemented and tested)

- Does down-projection guarantee controlled activation magnitude? No: its learned
  weights can change scale, and experts and their mixture can change it again.
- Why normalize after aggregation, rather than immediately after down-projection?
  The two placements control different tensors. Post-combination RMSNorm controls
  the actual latent representation entering the shared up-projection, after
  expert transformations and weighted summation.
- Why before up-projection rather than after? It gives that projection a controlled
  input magnitude. Normalizing afterward cannot repair an overflow that already
  occurred inside the projection. The two placements are not equivalent; the
  projection can still learn output scaling, so this is not a bound on all future
  activations or a guarantee against instability.
- Which axis/weight shape? Normalize each `[T,d_latent]` row over `d_latent` with
  a learned per-feature weight `[d_latent]`. RMS normalization is not centering
  or a guarantee of unit variance, and learned weights can change RMS afterward.
- Is another argument needed? The learner explicitly preferred no flag: supplying
  `d_latent` always enables the norm. This chooses the normalized variant for this
  repo; original LatentMoE has no such additional norm. `d_latent=None` remains
  unchanged. The learner wrote the implementation; Codex updated tests/docs.

Reference: [Kimi K3, normalized LatentMoE, §2.3.1](https://arxiv.org/html/2607.24653v2#S2.SS3.SSS1).
The routed path is now down -> experts -> weighted combination -> RMSNorm -> up,
with the full-width shared contribution added afterward. The latent tests use
native `torch.nn.functional.rms_norm` after an independent per-assignment sum,
nonuniform norm weights, all-parameter gradient comparisons, hooks checking the
norm immediately precedes up-projection, and zero-contribution backward checks.
Do not reuse the old plain-LatentMoE reference that moves up inside the sum: the
linear map still commutes with summation, but the inserted norm does not.

## Educational PyTorch EP (stage 11, forward and educational backward verified)

`expert_parallel.py` now runs the learner's dropless forward with two real
CPU/Gloo processes, four experts, top-2, and fixed routing. Each rank owns two
`GroupedExperts` entries; D=16/F=32 avoid the tiny debug example's grouped-MM
alignment issue. Codex supplied requested skeletons/launch boilerplate, fixed
mixing weights, comments, tests, and verification, not the learner's algorithm.

### Counts and dispatch

`all_to_all_single` uses one input/output tensor per rank rather than lists of
tensors. Input rows are already grouped into destination-rank chunks. Default
splits are equal; unequal routing requires explicit splits. It does not inspect
expert IDs to choose destinations or allocate the output buffer for us.

Each rank holds send counts [world_size], one count per destination. Exchanging
this tensor with equal splits produces that rank's receive counts [world_size],
one entry per source, not a matrix of all ranks. Send counts [2,2] on rank 0
and [3,1] on rank 1 yield receive counts [2,3] and [2,1]. A zero entry means an
empty data chunk: ranks still execute all collectives in the same order.

Counts are tensors for their exchange, then `.tolist()` produces Python split
lists accepted by this API. `int(received_counts.sum())` gives the output row
count; `int(tensor.item())` is another explicit scalar conversion. `.item()`'s
broad static return type alone is not enough to guarantee an integer dimension
to ty. Output buffers must have the matching input dtype/device; `new_empty`
preserves those. GPU-to-host count conversion can synchronize, but CPU/Gloo is
our current educational target, not an optimized dispatcher.

Here T is the source rank's local token count, K is top-k, and E/world_size is
experts per rank: K and experts per rank are not interchangeable. Flatten global
expert IDs [T,K] to [TK], derive destination = global_id // experts_per_rank,
and keep the destination argsort permutation. `bincount(destinations,
minlength=world_size)` counts assignments, not distinct tokens. Gather token
rows using permutation // K, not flattened feature values; send tokens are
[TK,D] and aligned expert IDs are [TK]. Both exchanges use the same splits.
Received tokens [N,D] and IDs [N] are in source-rank order, not expert order.

### Local computation and the two order restorations

On rank 1, global experts [2,3] occupy local weight indices [0,1]. Translate via
local_id = global_id - rank*experts_per_rank. Sort local IDs and gather token
rows with an expert permutation. Count local IDs (not permutation indices),
then cumsum along dimension 0 as int32 for the grouped-MM offsets. Include
zero-count experts with minlength so repeated offsets represent empty groups.

Stability is helpful for reproducible inspection, not a correctness requirement
for a row-independent expert MLP. [A,C,E] and [C,E,A] produce the same matching
expert outputs in different row orders; retaining and undoing the actual
permutation is the requirement. Sorting twice with the same permutation does
not generally invert it: [1,2,0] applied twice rotates rather than restores.

The learner chose a scatter to avoid another sort:

```python
received_order_output = torch.empty_like(grouped_output)
received_order_output[expert_permutation] = grouped_output
```

Indexing on the right gathers rows; indexing on the left writes into the named
positions. Grouped row i came from received row expert_permutation[i], so this
puts its result back there. Uniqueness of permutation indices makes it a
one-to-one restoration, not accumulation.

For return all-to-all, input splits become recv_split and output splits become
send_split. Allocate [TK,D] on the original source rank. After return, rows are
still destination-grouped, so scatter with the original dispatch permutation
to restore flattened token/slot order. These are two distinct permutations:
expert_permutation undoes local expert grouping; permutation undoes dispatch
grouping. Reshape the restored outputs to [T,K,D], multiply by original mixing
weights [T,K,1], and sum K to produce [T,D]. This demo does not need index_add_.
Weights remain on the source rank in original token/slot order; each fixed
row sums to one. They are not sent with assignments in this implementation.

### Verification and remaining scope

Ten assistant-authored persistent tests cover preparation and local computation:
controlled ordering, different K versus experts per rank, zero-count
destinations, non-contiguous inputs, an empty local input batch in preparation,
rank-1 global/local translation, and empty local expert groups. These tests do
not launch distributed communication.

Separately, an ad-hoc two-rank Gloo check seeded the actual experts with 42+rank,
reconstructed both owners' weights in a single-device reference, and computed
each selected expert per token with the same unequal mixing weights. Both
ranks' combined [2,16] outputs matched (rtol=1e-4, atol=1e-3). This verifies
the current complete forward example; it is not yet a saved regression test.

The original forward-only checkpoint above is superseded by the separate
AI-authored `expert_parallel_backward.py` example and its persistent distributed
test. The learner requested a commented implementation to study, rather than
an additional learner-written exercise. The forward-only source is unchanged.
CPU correctness is not multi-GPU/NCCL performance validation. LocalTensor was
considered but deliberately skipped in favor of real CPU processes.

### Backward: communication versus parameter synchronization

For `output = sum_k(weight_k * expert_output_k)`, upstream gradient `g` gives
each expert output `weight_k * g`. Each scalar mixing weight receives the
feature-wise dot product `sum(expert_output_k * g)`, computed on the source rank
and propagated through selected-score normalization and router projections.
Discrete top-k expert IDs do not receive gradients.

Backward traverses the forward graph in reverse:

1. The output-return exchange sends weighted output gradients to expert owners.
2. Local expert autograd computes both input and parameter gradients. Expert
   parameter gradients stay on the owner; there is no expert replication here.
3. The input-dispatch exchange sends expert-input gradients back to token owners.
   Autograd through the original repeated-row gather sums contributions from
   all selected experts locally, then continues into the preceding model layers.

`_AllToAll.backward` swaps forward input/output splits. This is an inverse
permutation of rows, not an all-reduce. Only the two activation exchanges need
autograd wrappers; counts and IDs are integer routing metadata.

The replicated router is different: each source computes local parameter
gradients, so the demo explicitly all-reduces their SUM after backward. This
matches a global objective equal to the sum of rank-local losses. An average
objective requires the corresponding scaling; unequal token counts require
appropriate loss weighting. With an additional expert data-parallel group,
replicated copies of an expert would also need synchronization over that group.

The CPU/Gloo regression test checks forward outputs, token and mixing-weight
gradients, both expert projection gradients, and reduced router gradients.
Its independent reference evaluates each assignment with ordinary matrix
multiplication in one global autograd graph. Cases include uneven mixed routes,
local-only routes, zero-count peers, an empty receiving rank, and an empty
source rank. Unused reference experts' absent gradients are compared as zeros.
Workers have process-group and parent-enforced timeouts.

Limitations: first-order autograd only; all ranks must enter backward and traverse
both exchanges in the same order. Grad-enabled tokens on every rank maintain
that graph in this educational example, including ranks with empty source
batches. No frozen-input/mixed-participation training, distributed decoder,
capacity policy, asynchronous overlap, or NCCL performance is claimed.

## MoE FLOPs, memory, and expert-parallel systems (stage 10, complete)

Let T=B*S, D=model width, E=routed expert count, K=selected experts per token,
F=routed SwiGLU hidden width, and F_shared=shared hidden width. Count one
multiply-accumulate as one MAC or approximately two FLOPs. For a bias-free
linear D->F, forward cost is TDF MACs or approximately 2TDF FLOPs. SwiGLU's
D->2F and F->D projections therefore cost 6TDF FLOPs, excluding activations.

For router, K routed experts per token, and one shared expert, forward matmul
FLOPs are 2TDE + 6TKDF + 6TD*F_shared. Learned parameters instead count stored
weights: DE + 3EDF + 3D*F_shared. There is no token-count or two-FLOP factor
in parameter counting. These estimates exclude shared gating, elementwise work,
sorting, dispatch, and communication. Doubling T doubles this MoE compute but
does not change parameter count. All E experts must be stored, not only K;
expert parallelism may partition that storage across devices.

With FP32 parameters, gradients, and two FP32 AdamW moments, parameter-related
training storage is 16P bytes once states/gradients are materialized, excluding
activations, workspaces, temporary allocations, and allocator overhead. Lowering
K reduces routed work/activations but not this architectural storage estimate;
all experts can still receive assignments somewhere in a batch.

Dropless gathering repeats each input vector for its K assignments:
grouped_tokens is [TK,D], containing TKD elements or 2TKD BF16 bytes. The up
projection is [TK,2F], or 4TKF BF16 bytes. Its value/gate chunks [TK,F] are
views, so their element storage is not added again. Out-of-place SiLU(gate)
and value*SiLU(gate) each allocate [TK,F], together another 4TKF BF16 bytes.
Thus those three projection/intermediate allocations total 8TKF bytes if live
together. This is a scoped tensor inventory, not exact peak training memory;
lifetimes, autograd-saved storage, output/dispatch tensors, router/shared paths,
and backward workspaces also matter. Do not count saved references/views as
independent copies of already-counted storage.

The learner derived the corrected costs and view/allocation distinction above.

### Expert ownership, dispatch, and combination

EP places different experts on different GPUs; one GPU may host several experts.
Selected experts are parallel branches: each receives the original token vector
[D], not the output of another selected expert. Token here means a hidden vector,
not a vocabulary ID.

The basic forward has two activation exchanges: all-to-all dispatch to expert
owners, local expert computation, then reverse all-to-all back to token owners.
Local routes need no network transfer. Each expert maps [N_e,D] to [N_e,D].
Across the group, TK expert-output vectors become T final token vectors after
weighted combination. Each token returns to its own source rank, not one global
GPU-0 aggregator. Optimized implementations can fuse return and combination.

Saved permutations/metadata match each returned row to its original token and
selected-expert slot. Router weights come from the source token's router; they
can be applied before return or after restoring rows. For example, C routed to
two experts with weights [0.25,0.75] and outputs [2,4], [6,8] gives [5,7].
The single-device analogue is weighted output followed by `index_add_` on token
indices. Rows are matched by routing metadata, not their vector contents.

All-to-all exchanges destination-specific chunks; it does not sum them.
All-reduce sums corresponding tensor positions, not matching token identities.
With tokens A/B/C owned by ranks 0/1/2 and selections A->{0,1}, B->{1,2},
C->{0,2}, expert-local rows can be [A,C], [A,B], [B,C]. Adding these arrays
row-wise mixes different tokens. A globally aligned [A,B,C] zero-filled layout
would allow reduction, but uses a different layout/communication strategy;
all-reduce also replicates the result. Returning sparse assignments directly
preserves the original token ownership without that global dense layout.
See [PyTorch's collective semantics](https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.all_to_all_single)
and [DeepEP's dispatch/combine example](https://github.com/deepseek-ai/DeepEP#example-use-in-model-training-and-inference).

### Communication volume and topology

For this paragraph, T is the total token count across the EP group, not per rank.
Let r be the fraction of assignments whose expert is remote. One remote BF16
assignment sends an input [D] and returns an output [D]: 4D bytes. The simple
global forward payload is therefore 4TKrD bytes, counting each transmitted
payload once, not again at receipt. This excludes metadata, backward, transport
overhead, compression, and aggregation of multiple same-destination routes.

With experts evenly spread over R ranks and uniform selection independent of
token ownership, expected local assignments are TK/R. Divide by TK to get the
local fraction 1/R, so r=1-1/R, not 1-TK/R. At 800 assignments over eight ranks,
100 are local and 700 remote: r=0.875. More ranks increase the global remote
fraction under these assumptions, but do not imply more bytes per rank or a
proportional runtime increase. Going from R=8 to 16 changes r from 87.5% to
93.75%, not twice as much. Increasing K from 2 to 8 instead quadruples this
payload while leaving the expected remote fraction unchanged.

Equal byte counts need not have equal cost: within-node links are typically
faster than inter-node links; the actual gap depends on the hardware. Keeping
EP within a node is attractive, not an absolute rule. Cross-node EP spreads
expert storage further. TP can split each expert's matrices but adds collectives;
FSDP shards training state/parameters but gathers parameters for computation.
These approaches can be combined, and each moves different data.

### Why FLOPs do not predict runtime alone

An overloaded expert can be a straggler; the practical bottleneck is often the
slowest rank's aggregate expert work, since a rank can host several experts.
Balancing and capacity therefore affect execution time as well as assignment
statistics. Communication bandwidth/latency, topology, and overlap also matter.

Doubling K doubles the estimated expert matmul work but need not double runtime:
larger expert batches can improve GPU utilization, while communication and
activation storage also grow. Our `GroupedExperts` reduces Python-loop and
kernel-launch overhead by grouping expert matmuls; it does not reduce their
mathematical FLOPs. No GPU speedup was established in this discussion.

This completes stage 10's reasoning. Codex recorded the corrected explanations;
no accounting helper or distributed implementation was needed for that reasoning
stage. Stage 11's subsequent two-process forward is recorded above; the plan
and optional backward extension are in `README.md`.

## Auxiliary-loss-free balancing (stage 9, complete)

The learner identified the balancing correction as explicit non-gradient state,
not an optimizer-learned parameter. Adding it to selection affinities encourages
underloaded experts when increased and discourages overloaded ones when decreased.
For counts [10, 30, 20, 20], the target is 20: increase bias 0, decrease bias 1,
and leave 2/3 unchanged. Small fixed adjustments encourage rather than guarantee
balance. Keep the correction out of mixing weights: load pressure is not token
affinity. Selection still affects which experts receive task gradients.

Use one update per optimizer step after accumulating its assignments, not inside
forward. Opposite imbalances in two microbatches can cancel; changing bias between
them would change the policy partway through a single accumulated update.
Auxiliary loss and bias updates are not mathematically incompatible, but our
comparison will keep them separate: auxiliary baseline with no updates, versus
updates with auxiliary coefficient zero. Keeping the loss function available
does not mean both controls are active. Z-loss is a separate stability objective.

The learner implemented a persistent FP32 `expert_bias` buffer [E], initialized
to zero. Top-k uses original softmax affinities plus bias; `gather(-1, ids)`
retrieves unbiased selected scores for normalized mixing weights and mass logging.
`scores[ids]` indexes the batch dimension; `scores[:, :, ids]` applies the entire
index tensor to each token. Gather matches each token's coordinates.
Bias survives metric resets/checkpoint round trips and receives no gradients.
Codex added four regression cases for biased train/eval selection, unbiased
weights/metrics, gradients, checkpoint/reset behavior, and zero-bias equivalence.
The learner implemented `update_expert_bias(rate)` under no_grad:
target = mean requested count, direction = sign(target - counts), bias += rate
times direction. Balanced/no-assignment counts naturally give zero updates.
Counts remain available for logging/reset. Large rates can overshoot, causing
overloaded/underloaded oscillation; smaller rates react more gently but slowly.
Four additional tests cover directions, balanced/empty counts, rate zero,
in-place buffer preservation, and unchanged counts.

The learner placed updates after `train_step` and before metric reset in the
tiny runner. Codex added CLI/history/plot-rate wiring and validation: rate must
be finite/nonnegative, zero disables updates, and positive rates cannot be
combined with a nonzero auxiliary coefficient in this exercise. Z-loss remains
independent; auxiliary loss is still computed/logged but weighted by zero.
Eight training/plot regression cases verify update timing/count windows,
disabled defaults, invalid rates, exclusivity, and tagged filenames. Generic
training helpers remain unchanged; this runner uses no gradient accumulation.
The learner's softmax bias-0.001 run, with gated shared width 24, auxiliary/z-loss
coefficients zero, and dropless dispatch, showed MaxVio near zero after roughly
100 steps and assignment shares near 25% per expert. Assignment entropy was near
2 bits, while per-token router entropy decreased and selected mass increased.
Task loss still approached zero. Small fixed-update fluctuations remained.
This demonstrates load control on the toy batch, not improved generalization;
older auxiliary-only plots also predate the independent batch RNG.

Assignment entropy summarizes the distribution of selected assignments, ignoring
which tokens chose them or their probabilities. MaxVio sees only peak overload:
counts [40,20,20,20] and [40,40,20,0] have identical MaxVio, but assignment
entropies about 1.92 and 1.52 bits. High assignment entropy can coexist with low
per-token entropy when confident token preferences spread across experts.

### Sigmoid scoring and normalized diagnostics

Softmax divides a fixed probability budget across experts; sigmoid scores each
expert independently before top-k. Changing a third logit rescales the softmax
gap between two unchanged experts, changing the influence of a fixed additive
bias. Their sigmoid scores do not change. Final selection and normalized mixing
still couple experts. Adding a constant changes sigmoid affinities/mixing,
unlike softmax. Neither independent scoring nor softer mixing guarantees quality.
See the [Loss-Free Balancing paper's Appendix C](https://arxiv.org/html/2408.15664v1#A3).

The learner implemented both scoring paths, runtime validation, and two views:
raw affinities for bias-aware selection and normalized selected mixing weights;
full-expert-normalized affinities for balancing and diagnostics. For raw scores
[0.8,0.7,0.4,0.1], selecting the first two gives mixing weights [0.533,0.467],
but selected mass (0.8+0.7)/2 = 0.75. Normalizing only selected scores would always
report mass 1. Reductions keep the expert dimension for broadcasting; accumulated
mass sums over tokens before division by total token count in reporting.

Sigmoid returns z-loss None by design: the softmax log-normalizer anchor is not
neutral to sigmoid's common logit shift. This is our policy, not a universal ban
on regularizing sigmoid routers. Codex forwarded scoring through MoE/block/model
and the runner, with CLI/history/filename tags, normalized-affinity plot labels,
no sigmoid z-loss curve, and rejection of nonzero z-loss coefficients with sigmoid.
Softmax defaults and existing metric keys/filenames remain compatible. None in
sigmoid history denotes inapplicable z-loss, not a measured value of zero.
Tests cover both modes, model variants, normalized diagnostics, gradients,
cached evaluation, validation, and plotting.

### Sigmoid-versus-softmax toy-run interpretation

The learner compared the two 800-step runs with bias rate 0.001, gated shared
width 24, auxiliary/z-loss coefficients zero, and dropless routing:

- [Softmax summary](../training/plots/moe_metrics_no_aux_bias_0.001_dropless_shared_24_gated.png)
  and [expert curves](../training/plots/moe_metrics_no_aux_bias_0.001_dropless_shared_24_gated_experts.png).
- [Sigmoid summary](../training/plots/moe_metrics_no_aux_sigmoid_bias_0.001_dropless_shared_24_gated.png)
  and [expert curves](../training/plots/moe_metrics_no_aux_sigmoid_bias_0.001_dropless_shared_24_gated_experts.png).

Sigmoid's final mean per-token normalized-affinity entropy is roughly 1.77–1.93
bits across layers, versus 1.33–1.73 for softmax. Selected normalized mass is
roughly 0.62–0.72 versus 0.73–0.86. The learner correctly interpreted this as
softer relative preferences, supported by both diagnostics. Mean expert
affinities alone cannot establish per-token sharpness: averages can hide opposing
preferences. Both runs have nearly balanced assignments and near-zero task loss.

"Less sparse" is imprecise here: both still execute K=2 routed experts per token.
Sigmoid compresses large positive logits, so scoring alone can make normalized
affinities flatter even for identical logits. Neither these entropy/mass values
nor similar mean affinities establish worse predictions or weaker specialization.
These are approximate plot readings, not generalization or performance results.
This completes stage 9. Next: MoE FLOPs, memory, and expert-parallel systems.

### Hyperparameter starting points and small sweeps

| Setting | Starting point | Educational sweep, including disabled baseline |
| --- | --- | --- |
| Auxiliary balancing coefficient | 0.01 | 0, 0.001, 0.01, 0.1 |
| Router z-loss coefficient | 0.001 | 0, 0.0001, 0.001, 0.01 |
| Expert bias update rate | 0.001 | 0, 0.0001, 0.001, 0.01 |

These are exploration grids, not universal safe/optimal ranges. Earlier learner
auxiliary runs also used 0.05 and 0.5; 0.5 was a deliberately strong comparison,
not a recommended default. Keep auxiliary coefficient zero when testing positive
bias update rates in our runner. Hold z-loss and architecture fixed across runs.

The [Loss-Free Balancing paper, section 4.3](https://arxiv.org/html/2408.15664v1)
compares bias rates 0.0001, 0.001, and 0.01: the smallest adapted slowly, the
largest fluctuated late, and 0.001 worked best in those experiments. Its main
experiments use sigmoid affinities; our router supports sigmoid and softmax.
Score scale, expert count, and update frequency affect transferability.
[DeepSeek-V3](https://arxiv.org/html/2412.19437v2) also used rate 0.001 for most
pretraining, then froze updates with rate zero near the end. The auxiliary/z-loss
starting values follow [Megatron's guidance](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html).

## Shared experts and fine-grained specialization (stage 8, complete)

The learner identified the shared branch as an always-on expert outside the
router. It can learn common patterns, reducing duplication across routed experts
and leaving room for specialization. This does not force general semantics or
guarantee more stable training. Both branches receive the original input;
the ungated output is `routed_output + shared_expert(x)`, each `[B, S, D]`.
Shared computation is excluded from routed counts, capacity, balancing, and z-loss.
The block's existing residual connection stays outside this combination.

The learner implemented the optional bias-free SwiGLU and sigmoid gate in
`GroupedMoE`. The gate is computed from x, not the shared expert's output:
`sigmoid(Linear(D, 1)(x))` has shape `[B, S, 1]` and broadcasts over D.
`[B, S]` alone would align incorrectly from the right. The gate is independent
per token, not a softmax competition, and does not skip shared computation.
Gating requires a shared expert. Codex simplified the final addition and added
eight tests for both expert backends, capacity isolation, gradients, gate input,
registration, defaults, and invalid configuration.

A shared SwiGLU of width F adds 3*D*F parameters and its corresponding active
matrix-multiply work. Adding it alongside K=2 full-width routed experts raises
expert compute by 50%. Selecting one fewer full-width expert, or keeping two
half-width routed experts, preserves the original active budget in this example.

Fine-grained routing already works through existing configuration: replacing
8 experts of width F, selecting 2, with 16 of width F/2, selecting 4, preserves
expert parameters and approximate expert matmul FLOPs. It permits more flexible
combinations of smaller capability bundles; larger experts are not restricted
to one feature and specialization is not guaranteed. Router scoring/dispatch
overhead grows, and smaller matmuls can be less efficient. Balanced assignments
per expert remain 2*T/8 = 4*T/16, so worse imbalance is not inevitable.

At the learner's request, Codex forwarded `shared_hidden_dim` and
`shared_expert_gate` through block, decoder, and training ModelConfig. Defaults
None/False preserve routed-only behavior. Each decoder layer owns its own
shared expert and optional gate. CLI flags and history/plot tags distinguish
shared widths and gated/ungated runs. No output/loss/metric API changed.
Tests check forwarding, task gradients, cached/full eval agreement, training,
and filenames. Last verification: 475 repository tests pass (147 MoE tests),
with targeted Ruff/ty checks passing. The toy-run discussion below completes
this stage; next is auxiliary-loss-free balancing. Do not repeat
the completed shared/gate implementation or its wiring.

The first shared run still memorized the batch but showed different routing;
this alone does not prove specialization or improvement. Model initialization
previously consumed the RNG before batch creation, changing the data across
architectures despite the same seed. At the learner's request, Codex separated
batch RNG with its own seed-42 generator and tested identical batches across
routed-only, shared, and gated-shared runs. No weight matching was added.
Older plots predate this change and therefore use different data.

The reviewed gated-shared-width-24 run used auxiliary coefficient 0.05, z-loss
coefficient 0, and dropless routing. Relative to the earlier auxiliary-only plot,
layer 1's final MaxVio is roughly 0.33 versus 0.17; layer 3 has sharper token
preferences and higher selected mass. Task loss still approaches zero and no
experts are unused. These visual observations do not establish better or worse
task quality, or that common knowledge was learned. Data and later-layer
initialization differed, and the shared branch adds 25% expert matmul work with
default routed width 48 and K=2. The learner chose only independent batch RNG,
not weight matching or a larger experiment framework; further runs are optional.

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

### Latest checkpoint (2026-10-23)

At the learner's request, Codex reorganized the folder without changing the
algorithms: `experts.py` holds both packed-weight execution backends,
`grouped_moe.py` holds dispatch/capacity/combination, and `tests/` holds the
component test files (now eight with EP). The former `packed_experts.py` is removed; public
`from moe import ...` imports remain unchanged. The tiny debug example now
explicitly uses the loop backend to avoid grouped-MM alignment restrictions;
the model default is unchanged. The EP demo instead uses `GroupedExperts` with
aligned D=16/F=32, as requested. Its dispatch, local computation, return,
ordering restorations, and weighted sum are implemented by the learner.

MoE stages 1–10 are complete, including bias-based balancing, both scoring paths,
toy-run interpretation, FLOP/memory accounting, and EP communication/runtime
reasoning. Stage 11's two-rank CPU/Gloo weighted forward matches an identical-weight
single-device reference in an ad-hoc run. Ten persistent EP tests currently cover
only preparation and local computation. Next: save the distributed-reference
test and add zero-count/local-only/empty-receiver communication cases with
timeouts; keep implementation/refactor work small. Backward is an optional
separate extension, not implicitly provided by forward communication.
LatentMoE remains stage 12, quantile balancing 13, and model case studies 14.
See `README.md`; do not restart completed dispatch/return or EP theory questions.
Do not repeat the completed stage-9 implementation, wiring, or experiments.
Last verification: 516 repository tests, including 172 MoE tests, pass. EP source
and tests pass Ruff/format/ty; both ranks pass the ad-hoc forward-reference check.

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
`experts.py`, with dispatch in `grouped_moe.py`; the assistant added the selectable backend and benchmark.
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
