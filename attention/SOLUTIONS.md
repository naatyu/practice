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

## Attention dropout

Standard attention dropout is applied to probabilities after softmax and
before their matrix multiplication with values. Applying ordinary dropout to
logits before softmax would set dropped logits to zero rather than exclude
them, and softmax would still give those positions nonzero probability. Using
negative infinity instead would renormalize over the surviving positions and
define a different stochastic operation.

Post-softmax inverted dropout zeros sampled attention connections and scales
surviving weights by `1 / (1 - p)`. A sampled row therefore does not generally
sum to one, but each weight—and consequently the attention output—is preserved
in expectation. Dropout is active during training and disabled during
evaluation.

Because the attention implementation is stateless, it can accept an effective
dropout probability with a default of zero. It does not need to own a module
training flag if the caller is responsible for passing zero during evaluation.

## Dense causal local attention

For window size `W`, including the current token, a query at absolute position
`i` may attend to keys in the inclusive range:

```text
max(0, i - W + 1) through i
```

Using `i-W` as the lower boundary would expose `W+1` positions. For example,
position five with a window of three must see positions three, four, and five.

During cached attention, K contains the cached prefix followed by the current
keys, while Q contains only the current queries. If their lengths are `K` and
`Q`, query row `r` represents absolute position:

```text
K - Q + r
```

The dense reference constructs query positions shaped `[Q, 1]` and key
positions shaped `[1, K]`. Broadcasting forms a `[Q, K]` mask from two
conditions:

```text
key_position <= query_position
key_position >= query_position - W + 1
```

The first enforces causality and the second removes keys that are too old.
Disallowed scores become negative infinity before softmax. This implementation
is a useful correctness oracle, but it still materializes `[B, H, Q, K]`
scores and probabilities and therefore realizes no sparse-attention savings.

## Compact sliding-window representation

Each query needs a different overlapping K/V slice. For `S=4` and `W=3`, the
key windows are:

```text
query 0: [PAD, PAD, k0]
query 1: [PAD, k0,  k1]
query 2: [k0,  k1,  k2]
query 3: [k1,  k2,  k3]
```

Left padding makes every row rectangular. Unfolding a padded K tensor along its
sequence dimension initially produces `[B, H, Q, D, W]`; transposing the final
two dimensions gives `[B, H, Q, W, D]`. V follows the same process with its
possibly different feature dimension `Dv`. A Boolean vector is padded and
unfolded identically to identify placeholder positions before softmax.

The windows contain `Q * W` logical query-key interactions, but their source
range contains only `Q + W - 1` positions because adjacent windows share
`W-1` keys. `unfold` represents those overlapping slices as a view rather than
copying every shared key.

### Selecting relevant cached K/V

Let the cached-prefix length before the current queries be:

```text
C = K - Q
```

The earliest current query is at position `C`, so no current query can reach a
key before `C-W+1`. The relevant range and required left padding are:

```text
relevant_start = max(0, C - W + 1)
left_padding   = max(0, W - 1 - C)
```

Slicing before padding avoids copying and unfolding an old cached prefix that
lies outside every current window. After the optional padding, the relevant
sequence has exactly `Q + W - 1` positions and therefore unfolds directly into
`Q` windows.

### Matrix contractions

For scores, each query row `[1, D]` is multiplied by the transposed key window
`[D, W]`:

```python
scores = torch.matmul(
    q.unsqueeze(-2),
    k_windows.transpose(-1, -2),
).squeeze(-2)
```

This produces `[B, H, Q, W]`. It avoids explicitly writing the
`[B, H, Q, W, D]` elementwise-product tensor before reducing `D`.

After scaling, padding masking, and softmax, probabilities are combined with
the value windows using `[1, W] @ [W, Dv]`:

```python
output = torch.matmul(
    attention_probabilities.unsqueeze(-2),
    v_windows,
).squeeze(-2)
```

The result has shape `[B, H, Q, Dv]`. Equivalent `einsum` expressions can make
the contracted dimensions more explicit.

### Complexity and practical performance

Dense attention computes `O(BHQKD)` score work and stores `O(BHQK)` scores and
probabilities. Compact local attention computes `O(BHQWD)` score work and
stores `O(BHQW)` scores and probabilities. For self-attention, this replaces
the quadratic sequence factor with `S * W` when `W` is much smaller than `S`.

The plain-PyTorch implementation is algorithmically efficient but not
necessarily faster in wall-clock time. Dense attention benefits from large,
highly optimized matrix multiplications and fused kernels. Padding, strided
views, possible contiguous copies, and many logically small contractions can
reduce hardware utilization. Production sliding-window attention generally
uses a fused CUDA or Triton kernel, or block-sparse tiling that loads K/V blocks
directly without constructing explicit per-query windows.

## Bounded KV cache and absolute RoPE positions

After new K/V entries have participated in the current attention call, each
layer returns only its newest `W` positions. Trimming after attention matters
for multi-token prefill: every prefill query must first receive its correct
local context, even though only the final window is needed by future decode
calls.

Once eviction begins, cache length is a storage bound rather than a sequence
position. At absolute position 1,000, a window cache may still contain only 128
entries. Using 128 as the next RoPE offset would rotate new queries and keys at
the wrong position.

The decoder cache therefore stores:

```text
one shared absolute next-token position
one distinct (K, V) tensor pair per Transformer layer
```

The absolute position is shared because every layer processes the same token
positions. Duplicating it per layer would be redundant and could allow layers
to become inconsistent. The decoder passes the shared position to every block,
uses it as the RoPE offset, and advances it by `Q`. K/V eviction does not reduce
that counter.

The MHA retains the old length-derived offset only for unbounded backward-
compatible calls. A direct bounded-cache call with RoPE requires an explicit
position, preventing silent reuse of the fixed window length as an absolute
offset. Speculative cache rollback similarly updates both the retained tensor
views and the decoder-level logical position.

## Local-attention verification

Focused tests cover a hand-computed equal-score example, `W=1`, large-window
equivalence with dense causal attention, invalid arguments, and cached suffix
alignment. The compact implementation is compared with the dense reference
across several window sizes and cached query lengths. End-to-end decoder tests
verify MHA, GQA, and MQA parity between full local attention and token-by-token
decoding while cache tensors remain bounded and the absolute position continues
to increase.
