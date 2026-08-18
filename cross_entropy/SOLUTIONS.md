# Stable cross-entropy interview practice: completed solutions

Implementation code remains in its Python module.

## Cross-entropy for a one-hot target

Cross-entropy does not calculate a direct numerical difference between the
predicted and target probabilities. For target distribution `y` and predicted
distribution `p`, it is:

```text
loss = -sum_i y_i * log(p_i)
```

In language modeling, the target distribution is one-hot. Only the correct
class `y` has coefficient one, so the loss becomes:

```text
loss = -log(p_y)
```

If the model assigns high probability to the correct token, this loss is small.
If it assigns low probability, the negative logarithm produces a large loss.

## Expressing the loss with logits

For logits `z` over the vocabulary:

```text
p_y = exp(z_y) / sum_j exp(z_j)
```

Taking the negative logarithm gives:

```text
loss = -z_y + log(sum_j exp(z_j))
```

The first term uses only the correct-class logit. The second term normalizes
against every vocabulary class. Softmax probabilities do not need to be
materialized.

## Stable log-sum-exp

Directly computing `exp(z)` can overflow when a logit is large. Let:

```text
m = max_j z_j
```

After shifting, every `z_j - m` is non-positive, so every exponential is at
most one and the largest is exactly `exp(0) = 1`.

Factoring `exp(m)` out of the original sum shows why the maximum must be
restored:

```text
sum_j exp(z_j) = exp(m) * sum_j exp(z_j - m)

log(sum_j exp(z_j))
    = m + log(sum_j exp(z_j - m))
```

Forgetting `m` computes the true log-sum-exp minus `m`. The resulting loss
can become negative and incorrectly changes when the same constant is added to
every logit.

An equivalent form shifts the target logit as well:

```text
loss = log(sum_j exp(z_j - m)) - (z_y - m)
```

The two maximum terms then cancel, so `m` does not need to be added explicitly.

## Tensor shapes

For language-model tensors:

```text
logits:        (B, S, vocab_size)
targets:       (B, S)
maximum:       (B, S, 1)
shifted logits:(B, S, vocab_size)
log normalizer:(B, S)
target logits: (B, S)
token losses:  (B, S)
```

The maximum and exponential sum are computed over the final vocabulary
dimension. Keeping the maximum dimension allows it to broadcast when
subtracted from every vocabulary logit.

## Selecting target logits with gather

Cross-entropy needs a different correct-class logit for every token. `gather`
selects those entries without allocating a one-hot tensor shaped
`(B, S, vocab_size)`.

The target indices are expanded from `(B, S)` to `(B, S, 1)`. Gathering
along the vocabulary dimension returns the same shape as the index tensor:
`(B, S, 1)`. Squeezing the final singleton dimension produces `(B, S)`.

`gather` and loss reduction serve different purposes. Gathering selects the
correct vocabulary class. Reduction happens afterward across the per-token
losses:

- `none` returns `(B, S)`.
- `sum` returns the scalar total across all tokens.
- `mean` returns the scalar average and is the normal training default.

With padding or ignored labels, the mean must later divide by the number of
valid tokens rather than all tensor positions.

## Ignored targets

An ignored target such as `-100` is a sentinel rather than a valid vocabulary
index, so it cannot be passed directly to `gather`. A Boolean mask retains the
information about which token positions are valid, while a separate target
tensor replaces ignored entries with any safe class index:

```python
valid_mask = targets != ignore_index
safe_targets = targets.masked_fill(~valid_mask, 0)
```

Class zero is only a temporary valid address for gathering. The mask, not the
replacement value, records whether the position is ignored. After gathering,
ignored token losses are set to zero:

```python
token_loss = token_loss.masked_fill(~valid_mask, 0.0)
```

The reductions then have these semantics:

```text
none: masked per-token losses shaped (B, S)
sum:  sum of valid token losses
mean: sum of valid token losses / number of valid tokens
```

Using `token_loss.mean()` would incorrectly include padding positions in the
denominator. Calling `mean()` and then dividing by the valid count would divide
twice. If every target is ignored, the mean computes `0 / 0` and yields `NaN`,
matching PyTorch, while the sum is zero.

Inputs should not be modified in place. Mutating `targets` would unexpectedly
replace the caller's ignored labels and could corrupt later computations.
Functional operations are also easier to reason about with autograd and
compiler transformations; in-place operations should be reserved for cases
where profiling demonstrates a meaningful benefit and their safety is clear.

## Testing conclusions

The completed implementation is tested against PyTorch for `none`, `sum`, and
`mean` reductions. Tests also cover:

1. Extreme logits that overflow a naive exponential implementation.
2. Equality of gradients with PyTorch's reference loss.
3. Invalid reduction configuration.
4. Mixed valid and ignored targets, including a genuine class-zero target.
5. Preservation of the caller's target tensor.
6. The all-ignored behavior for every reduction.

Testing every reduction caught a broadcasting bug that the scalar mean could
hide accidentally. Testing target immutability separately caught an in-place
mutation even when the numerical loss matched.
