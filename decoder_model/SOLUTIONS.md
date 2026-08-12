# Decoder-only language model interview practice: completed solutions

## Architecture and shapes

The model transforms token IDs into vocabulary logits as follows:

```text
token IDs:          (B, S)
token embeddings:  (B, S, D)
Transformer blocks:(B, S, D)
final RMSNorm:      (B, S, D)
vocabulary logits: (B, S, V)
```

The final vocabulary operation is one linear projection, not another MLP. It
produces one unnormalized score for every vocabulary item at every sequence
position. Argmax or sampling is a separate operation used to choose token IDs.

Repeated blocks belong in `nn.ModuleList`. This registers them as child
modules, making their parameters visible to optimizers and `state_dict()`, and
allowing `.train()`, `.eval()`, and device or dtype conversions to propagate
through the model. An ordinary Python list does not provide this registration.

## Logits and probabilities

Logits are arbitrary real-valued scores. They need not be non-negative or sum
to one. Applying softmax over the final vocabulary dimension converts them to
probabilities:

```python
probabilities = torch.softmax(logits, dim=-1)
```

During training, logits should be passed directly to cross-entropy. PyTorch
combines log-softmax and negative log-likelihood using a numerically stable
implementation, so applying softmax first is unnecessary and undesirable.

## Next-token alignment and loss

For token IDs `[x0, x1, x2, x3]`, the logit produced at `x0` is trained to
predict `x1`, the logit at `x1` predicts `x2`, and so on. The final logit has no
next token in the provided sequence, while `x0` has no preceding prediction.

```python
prediction_logits = logits[:, :-1, :]  # (B, S - 1, V)
targets = input_ids[:, 1:]              # (B, S - 1)
```

`torch.nn.functional.cross_entropy` expects input shaped `(N, C, ...)`, where
dimension 1 is the class dimension. Decoder logits instead place vocabulary
classes last. A common solution is to flatten batch and sequence positions:

```python
loss = torch.nn.functional.cross_entropy(
    prediction_logits.reshape(-1, prediction_logits.shape[-1]),
    targets.reshape(-1),
)
```

The resulting shapes are `((B * (S - 1)), V)` for the logits and
`(B * (S - 1))` for the class-index targets. `reshape` is convenient because
slicing can produce non-contiguous tensors for which `view` may fail.

An equivalent formulation moves vocabulary to dimension 1:

```python
loss = torch.nn.functional.cross_entropy(
    prediction_logits.transpose(1, 2),  # (B, V, S - 1)
    targets,                            # (B, S - 1)
)
```

## Weight tying

PyTorch stores both relevant parameters with the same shape:

```text
nn.Embedding(V, D).weight:   (V, D)
nn.Linear(D, V).weight:      (V, D)
```

`nn.Linear` internally computes `input @ weight.T`, so no manual transpose is
needed. Tying directly shares the same `Parameter`:

```python
self.lm_head.weight = self.tok_emb.weight
```

Each vocabulary row then serves both as the input representation of a token
and as its output classifier vector. Tying removes approximately `V * D`
duplicate parameters and can provide useful regularization and parameter
efficiency.

A vocabulary bias is mathematically valid but is commonly omitted in tied and
modern decoder architectures. It would learn an additional context-independent
preference for every token. Bias usage is an architectural choice rather than
a correctness requirement; the same is true of QKV projection biases.

## Final normalization

In a pre-norm block, normalization prepares input for each residual branch:

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

The residual stream after the final addition has not itself been normalized.
A final RMSNorm controls the representation scale presented to the vocabulary
projection and improves training stability. It is distinct from the second
normalization inside the last block, which prepares that block's MLP input.

## SwiGLU hidden dimension

Ignoring biases, a conventional GELU MLP with hidden dimension `4D` has about:

```text
D * 4D + 4D * D = 8D^2
```

SwiGLU has two `D -> F` input projections and one `F -> D` output projection:

```text
2DF + FD = 3DF
```

Parameter matching gives:

```text
3DF = 8D^2
F = (8/3)D
```

The result is commonly rounded to a hardware-friendly multiple. Multiples of
eight can improve matrix-multiplication alignment in lower precision, but are
an efficiency recommendation rather than a universal requirement for Tensor
Core use.

## RoPE ownership and position offsets

The decoder shares one RoPE module across its blocks. This is safe here because
RoPE has no learned parameters or mutable forward state; its buffers are
deterministic and non-persistent. Separate per-layer instances would also be
correct and would make ownership more explicit at the cost of duplicating the
tables.

The decoder forwards `position_offset` through every Transformer block and
attention layer to RoPE. A focused propagation test uses a short sequence that
is valid by itself but whose offset makes its absolute positions exceed the
configured maximum. If the decoder accidentally drops the argument, the call
will succeed at offset zero; if propagation works, RoPE raises the expected
error.

## Testing scope

The decoder-level tests focus on composition rather than duplicating child
module tests:

1. Token IDs produce logits shaped `(B, S, V)`.
2. The requested number of Transformer blocks is registered.
3. Tied weights are the same `Parameter`, while untied weights are distinct.
4. An invalid absolute position proves that the offset reaches RoPE.

Attention causality, dropout, RoPE arithmetic, RMSNorm, and SwiGLU behavior are
tested in their respective lower-level modules.

