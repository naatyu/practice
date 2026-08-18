# Positional encoding interview practice: completed solutions

## Learned absolute positional encoding

For vocabulary size `V`, maximum sequence length `M`, and model dimension
`d_model`, the two learned tables have shapes:

```text
token embedding table:      (V, d_model)
positional embedding table: (M, d_model)
```

Looking up token IDs shaped `(B, S)` produces token embeddings shaped
`(B, S, d_model)`. Selecting position rows `0 ... S - 1` produces
`(S, d_model)`. The positional tensor broadcasts across the batch when added,
so every example uses the same learned vector for a given position:

```text
token embeddings:      (B, S, d_model)
position embeddings:      (S, d_model)
output:                (B, S, d_model)
```

The table has fixed capacity but learned values. A sequence with `S > M`
cannot be processed because no trained rows exist for those positions.
Enlarging the table creates new, untrained parameters rather than reliable
length generalization.

### Why attention needs positions

Bidirectional self-attention without positional information is permutation
equivariant: reordering the input tokens simply reorders the outputs. It has no
inherent representation of first, previous, or distance. A causal mask exposes
prefix structure indirectly, but it still does not provide an explicit
position or distance representation.

### Embedding versus a raw parameter

`nn.Embedding(M, d_model)` owns an internal `nn.Parameter` named `weight`
with shape `(M, d_model)`. A raw parameter plus slicing is mathematically
possible, but `nn.Embedding` more clearly expresses lookup by integer position
indices.

Contiguous positions may be selected either by embedding lookup with
`torch.arange(S, device=x.device)` or by slicing the first `S` weight rows.
The explicit lookup approach makes the positional-index operation visible.

### Learned-position testing

A table filled entirely with zeros or ones is too weak: an implementation that
returns the input unchanged or always selects the same row may still pass. A
strong controlled test uses:

- A zero input
- Distinct, nonzero values in every positional row
- More than one batch element
- An expected tensor containing the first `S` rows in order for every batch

The embedding weights are parameters that require gradients. Test setup should
modify them with `copy_` inside `torch.no_grad()`. This preserves the
existing parameter object and tells autograd that the initialization is not
part of the differentiable forward computation. Using `.data` bypasses
autograd safety checks and replacing the parameter can invalidate optimizer
references.

## Sinusoidal positional encoding

Sinusoidal encoding has no learned parameters. It deterministically generates
an additive positional vector from the token position. For position `p` and
pair index `i`:

```text
frequency[i] = base^(-2i / d_model)
angle[p, i] = p * frequency[i]

PE[p, 2i]     = sin(angle[p, i])
PE[p, 2i + 1] = cos(angle[p, i])
```

For `d_model = 6`, one row is interleaved as:

```text
[sin(angle_0), cos(angle_0),
 sin(angle_1), cos(angle_1),
 sin(angle_2), cos(angle_2)]
```

Unlike a learned table, the formula can produce values for positions that were
not seen during training. This makes length extrapolation possible, although
it does not guarantee that the model will perform well far beyond its training
range.

### Multiple frequencies and relative offsets

High-frequency pairs change rapidly and distinguish nearby positions.
Low-frequency pairs change slowly and retain information over longer ranges.
If every pair used one frequency, the features would be redundant and many
positions separated by a full period would alias.

The angle-addition identities imply that, for a fixed offset `k`, the
sine/cosine pair at `p + k` is a linear transformation of the pair at `p`:

```text
PE_pair(p + k) = R(k) PE_pair(p)
```

The transformation depends only on the relative offset, which makes relative
relationships accessible even though the vectors are added as absolute
encodings.

### Shapes, vectorization, and buffers

Assuming even `d_model`:

```text
positions:   (max_seq_len, 1)
frequencies: (d_model / 2)
angles:      (max_seq_len, d_model / 2)
table:       (max_seq_len, d_model)
```

The singleton position dimension and frequency vector broadcast to compute
every position-frequency combination. This is vectorized because tensor
operations process all indices without a Python scalar loop. Both the direct
power form and the equivalent exponential/logarithm form are vectorized:

```text
base ** (-even_indices / d_model)

exp(even_indices / d_model * -log(base))
```

The deterministic table should be registered as a buffer. It is not optimized,
but follows the module across devices and can participate in saved state. A
non-persistent buffer is also reasonable because the table can be reconstructed
from the constructor configuration. The selected slice should be cast to the
input dtype before addition.

At position zero every angle is zero, so all even dimensions are
`sin(0) = 0` and all odd dimensions are `cos(0) = 1`. This is a useful
independent test invariant.

### Additive sinusoidal encoding versus RoPE

Sinusoidal positional encoding adds the generated vector to token
representations before attention. It does not rotate those representations.
RoPE instead applies the corresponding pairwise rotations directly to
projected queries and keys.

## Rotary positional encoding (RoPE)

### Pairwise rotation

For one adjacent feature pair represented as a column vector, RoPE applies the
rotation matrix:

```text
R(theta) = [[cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]]
```

For an input pair `(x0, x1)`, the rotated coordinates are:

```text
y0 = x0 * cos(theta) - x1 * sin(theta)
y1 = x0 * sin(theta) + x1 * cos(theta)
```

RoPE applies this operation to projected queries and keys. Unlike absolute
sinusoidal positional encoding, it does not add the sine and cosine values to
the input.

### Shapes and vectorization

Let `B` be batch size, `H` the number of heads, `S` sequence length, and `D`
the head dimension. Because adjacent features form rotation pairs, `D` must be
positive and even.

```text
input x:       (B, H, S, D)
paired x:      (B, H, S, D/2, 2)
x0 and x1:     (B, H, S, D/2)
angle table:   (S, D/2)
rotated pairs: (B, H, S, D/2, 2)
output:        (B, H, S, D)
```

For example, a head vector `[a, b, c, d]` is interpreted as the pairs
`[[a, b], [c, d]]`. Selecting the final pair coordinate produces
`x0 = [a, c]` and `x1 = [b, d]`. The rotation equations are applied
elementwise, the results are stacked along the pair-coordinate dimension, and
the tensor is reshaped back to its original form.

The angle table shaped `(S, D/2)` broadcasts over the batch and head axes of
`x0` and `x1`. This avoids materializing block-diagonal `(D, D)` rotation
matrices. The vectorized implementation uses `O(SD)` cached trigonometric
values rather than `O(SD^2)` matrix entries.

### Frequency schedule

For pair index `i` and sequence position `p`, the inverse frequency and angle
are:

```text
inv_freq[i] = base^(-2i / D)
angle[p, i] = p * inv_freq[i]
```

Here `i` ranges from zero to `D/2 - 1`, so `2i/D` ranges from zero to just
below one. The denominator is `head_dim`, not `d_model`, because each query and
key head vector is rotated independently. Using `d_model` would make the
frequency range depend incorrectly on the number of attention heads.

### Why Q and K are rotated

Queries and keys determine attention routing: their dot products become the
attention logits. Applying position-dependent rotations to them makes the
resulting token interaction depend on relative position.

Values contain the payload selected by the attention probabilities. Standard
RoPE leaves them in a shared feature space so the attention output remains a
weighted combination of mutually aligned value vectors. Rotating every value
according to its source position would combine values expressed in different
position-dependent coordinate systems. Other positional architectures can
modify values, but this is not part of standard RoPE.

### Relative-position property

For one feature pair, let a query at position `m` and a key at position `n` be:

```text
q_tilde_m = R(m * theta) q
k_tilde_n = R(n * theta) k
```

Their dot product is:

```text
q_tilde_m^T k_tilde_n
    = (R(m * theta) q)^T R(n * theta) k
    = q^T R(m * theta)^T R(n * theta) k
    = q^T R(-m * theta) R(n * theta) k
    = q^T R((n - m) * theta) k
```

This uses the rotation identities:

```text
R(alpha)^T = R(alpha)^-1 = R(-alpha)
R(alpha) R(beta) = R(alpha + beta)
```

Queries and keys therefore receive absolute-position transformations, while
their attention score depends on relative displacement `n - m`.

### Orthogonality and preserved quantities

A rotation matrix is orthogonal, so `R(theta)^T R(theta) = I`.
Consequently, RoPE preserves the norm of every two-dimensional pair and thus
the norm of the complete query or key vector:

```text
||R(theta) x|| = ||x||
```

If Q and K are at the same position, both receive the same rotation. Their dot
product is unchanged:

```text
(R(theta) q)^T (R(theta) k)
    = q^T R(theta)^T R(theta) k
    = q^T k
```

If they are at different positions, their individual norms remain unchanged,
but their dot product generally changes according to the relative rotation:

```text
q^T R((n - m) * theta) k
```

This question is useful because it tests whether RoPE is understood as a true
rotation rather than an additive encoding. It also exposes the central design
property: RoPE changes attention logits according to relative position without
changing the scale of Q or K. Norm preservation and same-position dot-product
preservation are also useful test invariants, although a controlled-output
test is still needed to catch incorrect signs or angles.

### Dtype and buffers

The sine and cosine tables are deterministic, non-parameter state and should
be registered as buffers. Registration makes them follow the module across
devices and keeps them out of optimization.

Computing and storing the tables in `float32` gives more accurate
trigonometric values than computing them directly in a lower-precision dtype.
The selected table slice should then be cast to the Q and K dtype before the
rotation. Otherwise, operations with `float16` or `bfloat16` inputs can promote
the outputs to `float32`, increasing activation memory and bandwidth, losing
lower-precision accelerator performance, and causing incompatibilities with
later operations or fused kernels that expect a consistent dtype.

Using non-persistent buffers is reasonable because the tables can be
reconstructed deterministically from the constructor configuration and need
not increase checkpoint size.

### Sequence offsets and the KV cache

Without an offset, a one-token decoding call has `seq_len = 1` and selecting
`table[:1]` always applies the rotation for position zero. If 100 zero-indexed
positions are already cached, the next token must instead use position 100.

For a non-negative offset, the correct slice is:

```text
table[offset : offset + seq_len]
```

The implementation must enforce:

```text
offset >= 0
offset + seq_len <= max_seq_len
```

New keys should be rotated before insertion into the KV cache. Previously
cached keys are already in their position-dependent rotated form and must not
be rotated again. The basic API assumes Q and K describe the same newly
processed sequence chunk and therefore share a sequence length, position
offset, and dtype.

### Tests

The implementation is covered by focused tests suitable for an interview:

1. Output shapes for Q and K are preserved.
2. `bfloat16` inputs produce `bfloat16` outputs.
3. An odd head dimension is rejected.
4. Sequence lengths beyond the configured table are rejected.
5. Negative and excessively large offsets are rejected.
6. Rotating a suffix with an offset produces the same output as selecting that
   suffix after rotating the full sequence.
7. A controlled case checks position-zero identity, both signs in the rotation
   equations, multiple frequencies, broadcasting across batches and heads,
   and rotation of both Q and K.

The controlled test uses nonzero Q and K inputs. This distinguishes rotation
from either returning the input unchanged or incorrectly treating RoPE as an
additive positional encoding.
