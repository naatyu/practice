# RMSNorm interview practice: completed solutions

## RMSNorm versus LayerNorm

RMSNorm rescales a vector using its root mean square and then applies a
learnable per-feature gain. It does not subtract the vector's mean.

LayerNorm subtracts the mean, divides by the standard deviation, and normally
applies learnable scale and bias parameters. RMSNorm removes the recentering
operation; the RMSNorm paper argues that recentering invariance is not necessary
to obtain LayerNorm's useful optimization behavior.

## Implementation details

RMS is computed independently for every vector over the final hidden
dimension. The reduced dimension is retained so the normalization factor
broadcasts over the corresponding vector.

Epsilon is added to the mean square before taking the square root. The learned
weight is a per-feature scale, so it is applied elementwise and broadcasts over
all preceding dimensions; it is not a matrix multiplication.

## Matrix multiplication versus elementwise multiplication

Elementwise multiplication pairs features with the same index and performs no
summation. For normalized input shaped `(..., D)` and a learned weight shaped
`(D,)`, broadcasting produces `(..., D)`: every feature is scaled independently.

Matrix multiplication contracts a shared dimension and sums over it. Multiplying
`(..., D) @ (D,)` produces `(...)`, collapsing the hidden dimension into one
dot product per vector. Multiplying by a matrix shaped `(D, D_out)` would mix
input features to produce new output features.

The deciding question is whether the operation should independently scale each
coordinate or sum/mix information across a dimension. RMSNorm uses independent
scaling. Attention uses matrix multiplication because query/key dot products sum
over the feature dimension, and probability/value multiplication sums over the
key-token dimension.

## PyTorch's `normalized_shape`

`normalized_shape` specifies the trailing input dimensions over which RMS is
computed. It also determines the shape of the optional learned weight. It does
not include batch, token, or other leading dimensions.

For transformer activations shaped `(B, S, D)`, using `normalized_shape=(D,)`
normalizes each token's length-`D` hidden vector independently and uses a
weight shaped `(D,)`. PyTorch accepts the integer `D` in the module constructor
as shorthand for the one-element tuple `(D,)`.

The argument is generalized to multiple trailing dimensions. For input shaped
`(B, C, H, W)`, `normalized_shape=(H, W)` computes one RMS over the final two
dimensions for each `(B, C)` position and uses a weight shaped `(H, W)`.

## Testing tradeoff

A randomized comparison with PyTorch's trusted implementation provides broad
coverage of the formula, broadcasting, weight application, and output shape.
A hand-computed case provides an independent oracle and documents the intended
mathematics, but it is not always worth the extra code in a short live
interview. It becomes more valuable when no trusted reference is available or
when the reference API could be configured incorrectly.

`torch.allclose(actual, expected)` returns a Boolean, so it must be used inside
a Python `assert`. In contrast, `torch.testing.assert_close(actual, expected)`
is itself an assertion: it raises an informative error when tensors differ and
returns `None` when they match. Wrapping it in `assert` turns a successful
comparison into `assert None`, which always fails.

## Backward-pass expectations

The forward implementation is composed entirely of differentiable PyTorch
operations, so autograd constructs its backward pass automatically. A general
ML coding interview is more likely to ask whether gradients work, or request a
gradient/reference test, than require a manual derivative.

A manual backward pass becomes more plausible for automatic-differentiation,
framework, compiler, or GPU-kernel roles. In that setting, the candidate may be
asked to derive gradients, implement a custom `autograd.Function`, decide which
forward values to save, or trade saved memory against recomputation.
