# FlashAttention interview practice: completed solutions

Completed reasoning and corrections will be recorded here as the interview
progresses. Implementation code remains in `flash-attention.py`.

## Bottleneck and tiling

Standard attention materializes the quadratic score or probability matrix in
high-bandwidth memory (HBM). Reading and writing this large intermediate causes
substantial memory traffic, and attention is often limited by that data
movement rather than arithmetic alone.

FlashAttention still computes the required pairwise scores, so its arithmetic
complexity remains quadratic. It processes query, key, and value tiles using
fast on-chip memory, fuses the attention operations, and avoids writing the
complete score and probability matrices to HBM. Only small score tiles and
per-query running statistics are needed at a time.

## Why block-local softmax is incorrect

Every probability in a score row shares one denominator containing all key
positions. Applying softmax independently to each key block gives each block
its own denominator. Every block would sum to one, so concatenating multiple
blocks would not produce one probability distribution over the full row.

Blockwise attention therefore needs an online method that updates global
normalization statistics as new key blocks arrive.

## Numerically stable softmax

Softmax subtracts the maximum score in each row before exponentiation.
Subtracting the same constant from every score does not change the resulting
probabilities because the common exponential factor cancels between numerator
and denominator.

After subtracting the maximum, the largest shifted score is zero and all
others are non-positive. Their exponentials are therefore at most one, which
prevents exponential overflow. Very small terms may underflow toward zero,
which is generally much safer than overflow.

## Online row maximum

For an existing running maximum `m_old` and a new block maximum `m_block`, the
updated maximum is the elementwise maximum of the two:

```text
m_new = max(m_old, m_block)
```

This is maintained independently for every query row.

## Rescaling the online normalization sum

Let the old running sum be:

```text
l_old = sum(exp(score_old - m_old))
```

To express the same old contributions relative to `m_new`, rewrite each
exponent:

```text
score_old - m_new
    = (score_old - m_old) + (m_old - m_new)
```

The second term is constant across the row, so it factors out of the sum:

```text
sum(exp(score_old - m_new))
    = exp(m_old - m_new) * l_old
```

After adding the new block, the updated normalization sum is:

```text
l_new = exp(m_old - m_new) * l_old
        + sum(exp(score_block - m_new))
```

If the maximum is unchanged, the rescaling factor is one. If the maximum
increases, the factor is less than one and the old contributions shrink. For
example, changing the maximum from `3` to `5` rescales the old sum by
`exp(-2)`, approximately `0.135`.

## Online weighted-value accumulator

For one query row, the attention output can be written as a ratio:

```text
output = sum(exp(score_j - m) * value_j)
         / sum(exp(score_j - m))
```

The denominator is the running normalization sum `l`. The numerator is a
running weighted-value accumulator:

```text
a = sum(exp(score_j - m) * value_j)
```

Unlike `m` and `l`, which are scalars per query row, `a` is a vector with the
same feature dimension as a value vector. Retaining `m`, `l`, and `a` is enough
to process score/value blocks without retaining the full score or probability
row.

### Scalar example

Consider one query, two score/value pairs, and scalar values:

```text
scores: [1, 2]
values: [10, 20]
```

Using the stable maximum `m = 2`, the unnormalized weights are:

```text
[exp(1 - 2), exp(2 - 2)] = [exp(-1), 1]
```

The denominator and weighted-value numerator are:

```text
l = exp(-1) + 1
a = exp(-1) * 10 + 1 * 20
```

The final output is `a / l`, which is the same weighted average that ordinary
softmax attention produces.

Processed blockwise, the first pair initially gives `m = 1`, `l = 1`, and
`a = 10`. When the second score raises the maximum to `2`, every old
unnormalized weight must be multiplied by `exp(1 - 2)`. Consequently both the
old denominator contribution and the old weighted-value contribution receive
the same correction factor before the new pair is added.

## Complete per-block update

For a score block `S_block` and its corresponding value block `V_block`, all
quantities below are maintained per query row:

```text
m_block = rowmax(S_block)
m_new = max(m_old, m_block)
alpha = exp(m_old - m_new)
P_block = exp(S_block - m_new)

l_new = alpha * l_old + rowsum(P_block)
a_new = alpha * a_old + P_block @ V_block
```

The exponentials come from the score block, not from the values. `rowsum`
reduces across keys in the block. The matrix multiplication with `V_block`
forms the new block's weighted-value contribution. Both old running quantities
are multiplied by `alpha` because both were expressed relative to the old
maximum.

After every key/value block has been processed:

```text
output = a / l
```

## Why values do not need to be retained

The weighted-value accumulator reduces the key/token dimension as each block
arrives. If block contributions are `a_1`, `a_2`, and `a_3`, the running state
contains their corrected sum rather than the individual contributions:

```text
a = corrected(a_1) + corrected(a_2) + a_3
```

Thus, for each query row, `a` is only one value-sized vector and `l` is one
scalar. Dividing that vector by the scalar produces the final output vector.
After a value block has contributed to `a`, the block does not need to be kept
for the forward result. This is analogous to computing a streaming weighted
average using a running weighted sum and a running total weight.

## Moving normalization after the weighted sum

FlashAttention computes the attention output, not the complete softmax matrix.
For one query row, define unnormalized weights
`w_j = exp(score_j - m)` and their sum `l = sum(w_j)`. Ordinary attention is:

```text
output = sum((w_j / l) * value_j)
```

Because `l` is the same scalar for every key in that query row, division can
move outside the sum:

```text
output = sum(w_j * value_j) / l
```

Therefore individual normalized probabilities never need to be stored.
Blockwise computation accumulates `sum(w_j * value_j)` in `a` and `sum(w_j)`
in `l`, then performs one row-wise division at the end. If the API had to
return the complete attention-probability matrix, that output would itself
require quadratic storage.

## Output-sized state is still required

The weighted-value accumulator has shape `(B, H, S, Dv)` if considered across
the complete sequence, and the normalization sum has shape `(B, H, S, 1)`.
This is linear in sequence length and is unavoidable because the final output
itself has shape `(B, H, S, Dv)`.

In a tiled implementation, only the accumulator for the active query block,
shaped `(B, H, Br, Dv)`, needs to remain in fast working memory. After all
key/value blocks have contributed, that block is divided by its normalization
sum and written to the corresponding output positions. The algorithm then
reuses the working buffers for the next query block.

The memory improvement is not the elimination of all intermediates. It is the
elimination of the quadratic `(B, H, S, S)` score and probability
intermediates, replacing them with output-sized storage and small tile-sized
working state.
