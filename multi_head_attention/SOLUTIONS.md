# Multi-head attention interview practice: completed solutions

Completed reasoning and corrections will be recorded here as the interview
progresses. Implementation code will remain in its Python module.

## Forward shape flow

Let `B` be batch size, `S` sequence length, `E` model dimension, `H` number of
heads, and `D = E / H` the per-head dimension.

```text
input x:                   (B, S, E)
fused QKV projection:      (B, S, 3E)
Q, K, V after splitting:   (B, S, E) each
split features into heads: (B, S, H, D)
transpose sequence/heads:  (B, H, S, D)
attention output:          (B, H, S, D)
transpose heads/sequence:  (B, S, H, D)
recombine heads:           (B, S, E)
output projection:         (B, S, E)
```

A single fused linear projection to `3E` is mathematically equivalent to three
separate `E -> E` projections and is commonly more efficient. The final output
projection mixes information from the concatenated heads.

After a transpose, tensor storage may be non-contiguous. Recombining heads with
`view` can therefore fail; making the tensor contiguous first or using an
appropriate `reshape` handles the layout.

## Why multiple heads?

Each head receives its own learned query, key, and value projections and
therefore operates in a different feature subspace. Heads can learn different
token relationships or patterns, such as local, positional, syntactic, or
semantic interactions. Their outputs are concatenated and mixed by the output
projection.

Multiple heads make this specialization possible but do not guarantee that all
heads learn distinct behavior. With fixed model dimension `E`, increasing the
number of heads reduces each head dimension to `D = E / H`, so the leading
attention compute order remains similar rather than multiplying by `H` on top
of a full-`E` head.

## Score scaling

Attention scores are scaled by `sqrt(D)`, where `D` is the per-head dimension.
Within each head, a query/key dot product sums `D` products, so its variance
grows with `D`, not with the complete model dimension `E`.

## Separating QKV versus creating heads

The fused projection's last dimension contains three consecutive tensors, so
it is partitioned into three equal chunks along that existing dimension.
`chunk(3, dim=-1)` expresses the number of equal pieces; equivalently,
`split(E, dim=-1)` expresses each piece's size.

Creating heads is a reshape rather than a partition into a Python collection.
The `E` feature dimension is reinterpreted as `(H, D)`, producing
`(B, S, H, D)`. The number of elements is unchanged and `H * D = E`. The head
and sequence axes are then transposed for batched attention.

## Architectural validation and shape arithmetic

Tensor shape arguments must be integers. Computing `head_dim` with integer
division is valid only after checking that `d_model` is divisible by
`num_heads`.

That divisibility is an architectural invariant determined by constructor
arguments, so it should be validated in `__init__` rather than on every forward
call. Storing the resulting `head_dim` also avoids recomputing it and makes the
forward reshape express the module configuration directly.

## Why reshape precedes permutation

The fused projection initially has shape `(B, S, 3E)`. Its last dimension
contains the Q, K, and V feature groups, but separate QKV, head, and head-feature
axes do not exist yet. Viewing it as `(B, S, 3, H, D)` first factors the
contiguous `3E` dimension into those semantic axes. Only then can permutation
move the newly created QKV and head axes into the order `(3, B, H, S, D)`.

Permuting first cannot move a head axis that has not yet been created. Other
equivalent formulations can combine these transformations, but they still
conceptually create the axes before reordering them.

## Why `contiguous` comes before the final `view`

`permute` normally returns a view with changed strides rather than physically
reordering the tensor's storage. After changing `(B, H, S, D)` to
`(B, S, H, D)`, the logical `H` and `D` axes are not laid out compatibly for
merging with `view`.

Calling `contiguous()` materializes the data in the new `(B, S, H, D)` order;
then `view(B, S, E)` can safely merge adjacent `H` and `D`. Calling
`contiguous()` after `view` is too late because `view` must validate compatible
strides first and may already fail. `reshape(B, S, E)` is an alternative that
copies internally when a view is impossible.

## Attention dropout

The module's dropout parameter controls probability dropout inside scaled
dot-product attention. The module must propagate whether it is in training or
evaluation mode so the functional attention operation can enable or disable
dropout. Dropout after the output projection is usually owned by the enclosing
Transformer block as residual dropout rather than by this attention module.

A clean functional API stores the configured probability on
`MultiHeadAttention` and passes `dropout_p` to the lower-level attention
function. During training the effective probability is the configured value;
during evaluation it is zero. This keeps the function stateless while allowing
`module.train()` and `module.eval()` to control behavior.

## Equivalent QKV/head-layout strategies

A fused `(B, S, 3E)` projection can be reshaped directly into explicit QKV and
head axes before permutation, or first partitioned into three `(B, S, E)`
tensors and then reshape each tensor into heads. `unflatten` can express the
same dimension factorization more declaratively. Separate Q, K, and V linear
layers are also mathematically equivalent but give up the fused projection.

The choice affects readability and operation count, not the required final
layout: attention must receive each tensor as `(B, H, S, D)`. `reshape` is more
forgiving of non-contiguous inputs, while `view` requires compatible strides.

A tensor shaped `(B, S, E)` cannot generally be reshaped directly to
`(B, H, S, D)`. Splitting `E` creates `(B, S, H, D)` because each token's `E`
features are contiguous and subdivided into heads. Moving `H` before `S`
requires a transpose or permutation. A direct reshape would preserve flat
storage order and reinterpret portions of one token's features as other token
positions, silently producing the wrong semantic layout.

## Exception-message matching in pytest

The `match` argument of `pytest.raises` is a regular expression, not a literal
string comparison. Characters such as square brackets and periods have regex
meaning and must be escaped when they should be matched literally. `re.escape`
can escape a complete expected message. In most tests, matching a short stable
fragment such as the argument name is less brittle than matching the entire
error message.

## Reference testing against PyTorch MHA

A numerical comparison requires identical QKV and output-projection parameter
values in both modules. Directly assigning one module's `Parameter` objects to
another makes the modules share those objects. Instead, copy their tensor
values under `torch.no_grad()` so the modules remain structurally independent.

PyTorch's reference module also requires explicit query, key, and value inputs,
returns output and optional attention weights as a tuple, and defaults to
sequence-first layout unless `batch_first=True` is configured. Dropout
comparisons should be performed in evaluation mode or with zero probability.
