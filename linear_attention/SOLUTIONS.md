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

## Planned short-convolution layer

The recurrent operator deliberately accepts already prepared Q, K, V, decay,
and beta tensors. The full educational KDA layer will separately implement the
paper's preprocessing path:

```text
Q projection -> causal depthwise ShortConv -> SiLU -> L2 normalization
K projection -> causal depthwise ShortConv -> SiLU -> L2 normalization
V projection -> causal depthwise ShortConv -> SiLU
```

The convolution is part of the actual KDA layer and must not be omitted. It
also introduces three additional fixed-size decode states containing the
recent projected Q/K/V inputs required by the convolution kernel. Keeping this
logic outside the recurrent operator makes both pieces independently testable.

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

## Handoff: resume here

The next implementation is scalar-gated DeltaNet, described in `README.md`,
Exercise 3. Plain recurrent linear attention and DeltaNet are complete and
tested.

Workflow:

1. The learner implements the function and answers conceptual questions.
2. The interviewer reviews without replacing the learner's code unnecessarily.
3. The interviewer writes focused tests after the implementation is corrected.
4. Continue to the channel-wise KDA recurrence after scalar-gated DeltaNet.
5. Only after the recurrence is verified, implement the full layer with causal
   depthwise Q/K/V short convolutions and decoding caches.
