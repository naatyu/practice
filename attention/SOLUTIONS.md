# Attention interview practice: completed solutions

## Exercise 1: Bidirectional scaled dot-product attention

### Shapes

For inputs shaped `(B, H, S, D)`:

```text
Q @ K.transpose(-2, -1): (B, H, S, S)
softmax scores:           (B, H, S, S)
probabilities @ V:        (B, H, S, D)
```

The first `S` in the score matrix indexes query rows. The second `S` indexes
key columns. Each query row is normalized across its key columns, so softmax is
applied with `dim=-1`.

### Scaling

The scale is `1 / sqrt(head_dim)`, not `1 / sqrt(model_dim)`. As the head
dimension grows, unscaled dot products tend to have larger variance. This can
make softmax saturate, producing overly peaked probabilities and weak
gradients.

### Why softmax?

For each query, softmax produces non-negative weights that sum to one. The
result is a differentiable weighted combination of the value vectors. The
normalization also prevents the output magnitude from growing directly with
the number of tokens or the raw score magnitude.

### Complexity

Per batch element and head:

```text
time:   O(S²D)
memory: O(S² + SD), dominated by O(S²)
```

Including batch size and heads, time is `O(BHS²D)`. Because
`model_dim = H * D`, it can also be written as `O(BS²model_dim)`.

The `(B, H, S, S)` scores and probabilities cause the quadratic memory cost.

### Tests

The implementation is tested using:

1. A hand-checkable case where equal scores must produce the mean of the value
   vectors.
2. A comparison with PyTorch's reference scaled dot-product attention.
3. An output-shape assertion.

Run:

```bash
uv run pytest -v
```

## Exercise 2: Causal attention

### Concept

In causal attention, a token may attend only to itself and earlier tokens. It
must not use future tokens because those tokens are unavailable when an
autoregressive model predicts the next token.

The allowed positions form a lower-triangular attention matrix. Future
positions are masked before softmax so that their resulting probabilities are
zero.

The mask only needs shape `(S, S)` because it broadcasts across batch elements
and heads. Two common representations are:

- A boolean lower-triangular mask, used to replace disallowed scores with
  negative infinity.
- An additive mask containing `0` for allowed positions and negative infinity
  for disallowed positions.

Masking logits is not done by multiplying them by zero. A zero logit still
contributes `exp(0) = 1` inside softmax, so the position would continue to
receive probability. Multiplication by negative infinity is also invalid:
negative logits become positive infinity, and a zero logit can produce `NaN`.

Additive masking behaves correctly for every finite score. Adding zero leaves
an allowed score unchanged, while adding negative infinity always makes a
disallowed score negative infinity. Softmax then assigns that position zero
probability.

Bidirectional and causal attention share the same computation apart from
masking, so a `causal` argument with a false default avoids duplicating the
function while preserving bidirectional behavior.

### Efficient causal masking

In an educational PyTorch implementation, use one boolean mask shaped `(S, S)`
on the same device as the scores. Broadcasting reuses it across batch elements
and heads; it should not be copied to `(B, H, S, S)`. Disallowed logits are
filled with negative infinity before softmax. If sequence lengths repeat, the
mask can be cached instead of reconstructed on every call.

In a fused production kernel, the full mask is usually not materialized.
Query/key positions determine whether an element is valid. Entire score tiles
above the causal diagonal can be skipped, and only the invalid elements in a
diagonal tile need to be excluded. PyTorch's optimized scaled dot-product
attention receives this intent through `is_causal=True`.

### PyTorch masking pitfalls

- `torch.ones` creates a floating-point tensor unless a dtype is specified.
  The `~` operator performs logical negation only on boolean tensors (or bitwise
  inversion on integer tensors), so a boolean mask must use `dtype=torch.bool`.
- The mask must be created on the same device as the attention scores.
- `masked_fill` returns a new tensor. Its result must be retained; alternatively,
  an explicitly in-place variant can modify the score tensor.
- A causal `(S, S)` mask broadcasts across the batch and head dimensions.

The causal implementation is validated against PyTorch's reference behavior
with `is_causal=True`.

An independent behavioral test can validate causality without using PyTorch as
an oracle: compute causal attention, change only a future value token, and
compute it again. Outputs for query positions before that token must remain
identical, while an output allowed to attend to the changed token should
change. This tests the causal invariant directly rather than comparing two
implementations.
