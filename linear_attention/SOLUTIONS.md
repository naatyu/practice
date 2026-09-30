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
currently uses `Dk = Dv = d_model/H` and a full-rank output gate.

The retention gate is now parameterized in log space. A full-rank projection
of the layer input produces a channel-wise decay input. A learned per-head
`log_decay_rate` becomes a positive rate after exponentiation; softplus of
the projected input plus a learned per-channel bias is also positive. Their
negative product is `log_retention <= 0`, so exponentiating produces a valid
retention value in `(0, 1]`. The rate and bias control how quickly different
heads and channels forget. The projection has no bias because its bias would
be redundant with the explicit per-channel decay-input bias.

For an initial positive decay step `dt`, inverse-softplus sets the bias to
`log(expm1(dt))`, so softplus recovers `dt` when the projected input is zero.
Steps are sampled log-uniformly from `[0.001, 0.1]` to cover different memory
timescales. `log_decay_rate` starts at zero (rate one). The layer's
`reset_parameters()` initializes only its direct decay parameters. It can be
called again after a meta-device module is materialized with `to_empty`;
child modules must be initialized separately. The reference uses additional
per-head rate initialization and low-rank projections, which remain optional
follow-up work.

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

`test_kimi_delta_attention.py` has 21 passing tests covering channel-wise
decay, equivalence to scalar-gated DeltaNet when channels share a gate,
split-sequence recurrent continuation, gradients, causal depthwise convolution,
layer input shapes, per-head RMSNorm, the learned output gate, and convolution
and full-layer split-sequence equivalence for kernel widths 1, 3, and 4.
The full-layer tests compare outputs and all four cache fields. Controlled
gate values, decay-parameter gradients, initialization range, and
meta-device materialization are also tested.

## Handoff: resume here

The recurrent KDA core, educational layer, fixed-size decoding cache, and
log-space decay initialization are implemented and tested. Next, compare the
educational full-rank decay and output-gate projections with the reference's
low-rank versions and decide whether to implement them. The reference also
initializes positive per-head decay rates differently; this implementation
currently starts every rate at one.

Workflow:

1. The learner implements the function and answers conceptual questions.
2. The interviewer reviews without replacing the learner's code unnecessarily.
3. The interviewer writes focused tests after the implementation is corrected.
4. Preserve the learner's implementations unless asked to change them.
5. Consider optional low-rank gates and per-head rate initialization; keep
   chunkwise training algorithms optional.
