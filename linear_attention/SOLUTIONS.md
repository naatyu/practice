# Linear attention and Kimi Delta Attention: completed reasoning

This file records the reasoning completed during the interview-style exercise.
Implementation code remains in the neighboring Python files.

## Matrix-valued associative memory

For one head, the recurrent state is shaped `[Dk, Dv]`. It behaves as a dynamic
fast-weight regression matrix that maps a key-like vector to a predicted value:

```text
[Dk] @ [Dk, Dv] -> [Dv]
```

This is not a normal static model parameter. It is sequence state constructed
online from token representations. The current key reads what the memory
already predicts for the association being written, while the current query
reads the output requested by the model.

## Plain linear-attention write

A key/value association must have the same shape as the state. It is created
with an outer product:

```python
association = k_t.unsqueeze(-1) @ v_t.unsqueeze(-2)
```

Shapes:

```text
[Dk, 1] @ [1, Dv] -> [Dk, Dv]
```

Plain recurrent linear attention adds that association to the state and then
reads the updated state with the current query. Reading after writing allows a
causal token to attend to itself, matching an inclusive lower-triangular
attention matrix.

Blind accumulation cannot cleanly overwrite an association. If the same
normalized key is written with values `v1` and `v2`, its readout contains both
contributions rather than simply returning `v2`. Similar, non-orthogonal keys
also interfere.

## Why the implementation is recurrent

Every causal query must see a different prefix state:

```text
S0 = k0 outer v0
S1 = S0 + k1 outer v1
S2 = S1 + k2 outer v2

o0 = q0 @ S0
o1 = q1 @ S1
o2 = q2 @ S2
```

The final state can be computed directly as `K.transpose(-1, -2) @ V`, but
using that state for every query leaks future associations. Plain recurrent
linear attention is equivalent to the following causal parallel expression:

```python
scores = (q @ k.transpose(-1, -2)).tril()
output = scores @ v
```

That expression materializes an `[S, S]` matrix. Constructing all tokenwise
outer products and applying a cumulative sum is also parallel, but materializes
one `[Dk, Dv]` prefix state per token. The educational loop retains one state,
makes causality explicit, and naturally supports decoding. Production training
uses scans or specialized chunkwise algorithms to recover parallelism without
these large intermediates.

## Initial and final states

`initial_state` allows a call to continue a sequence processed by an earlier
call. `output_final_state=True` returns the state after the final supplied
token. Splitting a sequence into two calls and passing the first final state to
the second must match one full-sequence call.

During recurrent decoding, this fixed `[B, H, Dk, Dv]` matrix plays the role
served by a growing KV cache in full attention. The flag controls whether the
caller receives the state; the recurrence still computes it internally.

## Delta-rule correction

DeltaNet treats the state as an online regressor. Before writing a key/value
pair, it predicts the value currently associated with the key and computes the
error:

```text
prediction = k_t @ state
error      = v_t - prediction
```

The error is written along the current key direction as a rank-one correction,
scaled by a learned write-strength gate `beta_t`. A zero beta leaves memory
unchanged; larger beta values apply more of the correction.

For an L2-normalized key and `beta_t=1`, reading with that key immediately
afterward returns the desired value. The new prediction equals the old
prediction plus `k_t^T k_t` times the prediction error, and the normalized
inner product is one. This exact correction property motivates key
normalization in the complete layer.

## Forgetting and KDA's channel-wise gate

One rank-one delta correction only changes memory along the current key
direction. Associations in other directions can remain stale. Gated DeltaNet
therefore decays the previous state using one scalar per head before applying
the correction.

A decay of one preserves the old state. A decay near zero erases most old
memory before writing the current association. The error must be computed from
the decayed state because that is the state being corrected and retained.

KDA replaces the scalar decay with a vector containing one value per key
channel. For a state `[Dk, Dv]`, reshape the vector to `[Dk, 1]` and multiply it
with the state. Each key channel then independently controls the lifetime of
its complete row of value information.

The recurrent KDA token order is:

1. Apply channel-wise decay to the previous state.
2. Predict the value associated with the current key from the decayed state.
3. Calculate the value error.
4. Apply the beta-scaled key/error outer-product correction.
5. Read the updated state with the current query.

## Short-convolution layer and output path

The recurrent operator deliberately accepts already prepared Q, K, V, decay,
and beta tensors. The educational layer now implements the preprocessing path:

```text
Q projection -> causal depthwise ShortConv -> SiLU -> L2 normalization
K projection -> causal depthwise ShortConv -> SiLU -> L2 normalization
V projection -> causal depthwise ShortConv -> SiLU
```

The fused QKV projection produces `[B, S, 3*d_model]`, then splits it into
three `[B, S, d_model]` tensors. A depthwise Conv1d has one time filter per
channel: `groups=channels`. For kernel width `K`, left padding by `K-1` and
no right padding preserves length and prevents future-token leakage. Conv1d
expects `[B, C, S]`, so the wrapper transposes to and from `[B, S, C]`.

After convolution and SiLU, split the projected channel dimension into heads:
`[B, S, H, Dk] -> [B, H, S, Dk]`. Transposing the last two axes instead
would swap `H` and `Dk`, not `S` and `H`. L2-normalize Q and K along `Dk`;
V is not L2-normalized. A unit-length key makes a full-strength delta
correction exactly overwrite the prediction for that key and controls the
state-update scale.

The recurrence returns `[B, H, S, Dv]`. The layer applies RMSNorm over each
head's `Dv` features *before* merging heads; normalizing the merged
`[B, S, d_model]` vector would couple their scales. A learned sigmoid gate
projected from the layer input then attenuates features of the merged output,
followed by the final output projection. This output gate is separate from
alpha (memory decay) and beta (correction strength). The educational layer
currently uses `Dk = Dv = d_model/H`. Its output-gate projection has a
`d_model -> head_dim -> d_model` bottleneck: the first linear layer has no
bias, while the final bias lets each feature learn a default sigmoid gate
value when the bottleneck output is zero.

The retention gate is now parameterized in log space. A two-layer projection
maps `d_model -> head_dim -> d_model` to produce a channel-wise decay input.
With no activation between the layers, its combined linear map has rank at
most `head_dim`; it still produces a value for every key channel. A learned per-head
`log_decay_rate` becomes a positive rate after exponentiation; softplus of
the projected input plus a learned per-channel bias is also positive. Their
negative product is `log_retention <= 0`, so exponentiating produces a valid
retention value in `(0, 1]`. The rate and bias control how quickly different
heads and channels forget. The projection has no bias because its bias would
be redundant with the explicit per-channel decay-input bias.

For an initial positive decay step `dt`, inverse-softplus sets the bias to
`log(expm1(dt))`, so softplus recovers `dt` when the projected input is zero.
Steps are sampled log-uniformly from `[0.001, 0.1]` to cover different memory
timescales. Each head's positive rate is sampled uniformly from `[1, 16]`
and its logarithm is stored in `log_decay_rate`. Sampling the logarithm
uniformly would instead make the positive rates log-uniform, favoring smaller
rates. The rate range is a reference initialization choice, not a requirement
of the recurrence. The layer's `reset_parameters()` initializes only its
direct decay parameters. It can be called again after a meta-device module is
materialized with `to_empty`;
child modules must be initialized separately. The low-rank decay and
output-gate projections are also implemented.

The convolution introduces three additional fixed-size decode states:
the recent projected Q/K/V inputs needed by the kernel. The recurrent matrix
state alone is insufficient to resume a full layer exactly. The `KDACache`
holds that matrix plus separate Q/K/V tails, each `[B, K-1, d_model]`.
The tails contain projected inputs *before* convolution and SiLU. Each call
concatenates its tail with new projected inputs, applies a valid depthwise
convolution, and saves the last `K-1` inputs as the next tail. A first call
uses a zero tail. For `K=1`, the tail is empty. The cache is passed explicitly
between calls rather than stored as mutable state in the module.

Plain linear attention does not receive a convolution merely because KDA uses
one. Full wrappers for intermediate architectures should include only the
components belonging to their canonical designs.

## Verification completed

`test_linear_attention.py` covers:

- A hand-computed association and readout example.
- Equivalence with the causal parallel expression.
- Final-state equivalence with `K.transpose(-1, -2) @ V`.
- Full-sequence versus split-sequence recurrence.
- Optional state output.
- Complete initial-state shape validation.
- Finite gradients through Q, K, and V.

`test_delta_net.py` covers controlled beta values, independent gates across
batch items/heads/tokens, split-sequence continuation, optional final state,
state shape validation, and gradients through Q, K, V, beta, and initial state.

`test_kimi_delta_attention.py` has 22 passing tests covering channel-wise
decay, equivalence to scalar-gated DeltaNet when channels share a gate,
split-sequence recurrent continuation, gradients, causal depthwise convolution,
layer input shapes, per-head RMSNorm, the learned output gate, and convolution
and full-layer split-sequence equivalence for kernel widths 1, 3, and 4.
The full-layer tests compare outputs and all four cache fields. Controlled
gate values, decay-parameter gradients, initialization range,
meta-device materialization, both projection bottlenecks, and the learned
output-gate baseline are also tested.

## Chunking intuition studied

An extra chunk loop around the token recurrence does not create parallelism.
Within a chunk, each prediction error depends on earlier writes, which gives
a strictly lower-triangular dependency matrix. The coefficient for a write
at `i` uses its writer strength `beta_i`, key alignment, and channel-wise
retention through gates `i+1 ... t`. The incoming state instead decays through
gates `0 ... t`, including the current token's gate before its prediction.
Prefix sums of log-retention let interval decay use differences of prefix
sums, followed by exponentiation.

Collecting those equations gives `(I + A) E = B`, where `B` is the error
against decayed incoming memory alone. A triangular solve obtains the actual
errors; the dependencies are preserved. The output readout matrix includes
the current write on its diagonal. The
[visual lesson and verified numerical example](visuals/README.md) expand
these points and distinguish the educational solve from optimized WY kernels.

## Chunkwise code walkthrough (AI-authored)

The learner asked Codex to write [chunked_kda.py](chunked_kda.py) as a worked
learning example. Its implementation, tests, and optional layer selection
branch are assistant-authored, not a completed learner coding exercise. The
existing recurrence and layer are the learner's reference implementation.

For a chunk of length `C`, use rows `t` for readers and columns `i` for writers:

1. **Surviving keys.** `cumulative_log[t]` sums gates through `t`.
   `exp(G_t - G_i) * k_i` is write `i`'s key after gates `i+1 ... t`.
   Broadcasting gives `[B, H, C, C, Dk]`. Upper-triangular entries are masked
   before exponentiation to avoid overflowing future-directed differences.
   The incoming memory instead uses `exp(G_t)` through gates `0 ... t`.
2. **Prediction errors.** Build
   `A[t, i] = beta_i * dot(k_t, surviving_key[t, i])` for `i < t`.
   `base[t] = v_t - (k_t * exp(G_t)) @ S_in` ignores within-chunk writes.
   Their effect is restored by solving `(I + A) @ errors = base`, using
   `torch.linalg.solve_triangular(..., upper=False, unitriangular=True)`.
   Both sides have batched shapes: the system is `[B, H, C, C]`, and the
   errors are `[B, H, C, Dv]`. This is a solve, not a matrix inverse, and it
   preserves the dependency on earlier errors without a Python token loop.
3. **Outputs.** Replace the reading keys with queries to form `reads`.
   Keep its diagonal: a query reads the state after its own correction.
   `output = (q * exp(G)) @ S_in + reads @ errors`. These coefficients can
   be negative; they are not softmax probabilities.
4. **Outgoing state.** Take the last reader row of the survival factors.
   `final_keys` contains all keys decayed to the chunk's end and scaled by
   their writer beta. Then
   `S_out = exp(G_last)[..., None] * S_in + final_keys.T @ errors`.
   This `[B, H, Dk, Dv]` matrix is the only state carried between chunks.

The short final chunk needs no padding. Chunk size one reduces to the same
one-token update. Splitting calls at different chunk boundaries preserves
the result up to floating-point rounding, and the input cache is not modified
in place. All operations remain differentiable.

The standalone operator accepts finite nonpositive **log-retention**, not
the recurrent function's `alpha`. The full layer passes its log gate directly.
Using float32 internally (float64 when supplied) supports low-precision inputs
and triangular solving; autocast is disabled inside the chunk math. The
returned output uses `v.dtype`, but the cached state keeps the compute dtype.
The recurrent path's precision behavior is unchanged.

This version spends `O(B H C² Dk)` temporary memory on pairwise decay and
surviving keys; its triangular solve also has dependencies. It is an exact
mathematical regrouping, not unlimited token parallelism or a claim of GPU
speedup. More efficient WY representations and fused kernels are separate
work. Very large chunks or extreme cumulative log magnitudes can still lose
floating-point accuracy, so the recurrence remains a useful reference.

The 34 tests in [test_chunked_kda.py](test_chunked_kda.py) cover controlled
numbers, different chunk sizes, outputs and gradients against the recurrence,
state continuation, causality, zero/unit write gates, strong forgetting,
float16/bfloat16 and AMP, input validation, and full-layer integration.

## Completion and handoff

The recurrent KDA core, educational layer, fixed-size decoding cache,
log-space decay initialization, per-head rate initialization, and low-rank
decay and output-gate projections are implemented and tested. An AI-authored
chunkwise operator is now available as explicitly requested teaching material,
with the recurrence still the default. The learner has marked this educational
track complete. Do not resume KDA as an unfinished exercise; its four numbered
code sections, Manim animation, and companion guide are optional review
material. Production WY kernels are outside the completed scope.

Workflow:

1. Move to the next roadmap topic when the learner is ready; revisit KDA only on request.
2. Keep the assistant-authored worked example clearly attributed.
3. For future exercises, the learner normally implements and the assistant writes tests.
4. Preserve the learner's implementations unless asked to change them.
5. Keep the production WY kernel implementation optional.
