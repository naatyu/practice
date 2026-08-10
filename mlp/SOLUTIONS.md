# Transformer MLP interview practice: completed solutions

Completed reasoning and corrections will be recorded here as the interview
progresses. Implementation code will remain in its Python modules.

## Standard Transformer MLP

The MLP is the second major sublayer of a Transformer block, after attention
and its surrounding normalization/residual structure. For input shaped
`(B, S, E)`, its standard flow is:

```text
linear up:       (B, S, E) -> (B, S, F)
GELU:            (B, S, F) -> (B, S, F)
hidden dropout:  (B, S, F) -> (B, S, F)
linear down:     (B, S, F) -> (B, S, E)
output dropout:  (B, S, E) -> (B, S, E)
```

There is normally no second activation after the down projection. The
nonlinearity between the two linear maps is sufficient to prevent them from
collapsing into one linear transformation. Dropout placement varies by model;
the output dropout can also be described as residual-branch dropout owned by
the enclosing Transformer block.

The linear layers operate on the final feature dimension with the same weights
at every sequence position. They transform each token independently and do not
mix information across tokens. Attention performs token mixing; the MLP
performs feature mixing within each token. `F` is commonly larger than `E`,
often around `4E` in the original Transformer-style GELU MLP.

Each output coordinate of a linear layer is a weighted sum of all input
coordinates for the same token. Thus the projections mix features rather than
merely scaling each feature independently. GELU is the elementwise part: after
the up projection, it transforms each hidden scalar independently. Changing
one input feature can affect many output features at the same token position,
but cannot directly affect another token position.

## Module details

The constructor should reject non-positive model or hidden dimensions. PyTorch's
`nn.Dropout` validates its probability, but explicit validation can provide a
consistent module-level error contract.

One `nn.Dropout` instance may be called at both dropout locations when they use
the same probability. A dropout module has no learned parameters and samples a
fresh mask on every call during training; sharing the instance does not share
the mask. Separate instances can make the two locations easier to name or allow
different probabilities.

## Testing scope

PyTorch does not provide one canonical Transformer `MLP` module to use as a
direct oracle. A reference made from `linear -> GELU -> linear` with copied
weights largely repeats the implementation and therefore adds limited
independent coverage. For a live interview, constructor validation, output
shape, and train/eval dropout behavior are sufficient. A behavioral token-
independence test can provide more meaningful additional coverage if desired.

## SwiGLU

SwiGLU uses two parallel projections of the same input. One produces candidate
values and one produces an input-dependent gate:

```text
value = value_proj(x)                    # (B, S, F)
gate = SiLU(gate_proj(x))                # (B, S, F)
hidden = value * gate                    # (B, S, F)
output = down_proj(hidden)               # (B, S, E)
```

The multiplication is elementwise. The two input projections can be represented
by one `E -> 2F` linear layer and separated with
`torch.chunk(..., 2, dim=-1)`. Their independent weight matrices are merely
concatenated so one larger matrix multiplication can compute both.

Ignoring biases, a standard MLP has approximately `2EF` parameters, whereas
SwiGLU has `3EF`. Matching a standard hidden size of `4E` gives:

```text
3E * F_swiglu = 2E * 4E
F_swiglu = (8/3)E
```

The ratio is normally rounded to a hardware-friendly dimension and remains an
explicit architecture argument.

## SwiGLU testing lessons

A shape test cannot detect applying SiLU to the wrong branch, adding instead of
multiplying, or splitting the fused projection incorrectly. A stronger test
uses asymmetric hand-set weights, zero biases, and no dropout, then compares
with a separately calculated scalar result.

PyTorch linear weights have shape `(out_features, in_features)`, so a fused
`Linear(1, 2)` stores a `(2, 1)` weight. Parameters should be modified with
`copy_`, `fill_`, or `zero_` inside `torch.no_grad()`.

`torch.tensor(1.0)` creates a scalar containing one, while `torch.Tensor(1)`
allocates an uninitialized one-element tensor. Also,
`torch.testing.assert_close` performs the assertion itself and must not be
wrapped in a Python `assert`.
