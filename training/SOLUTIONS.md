# Language-model training interview practice: completed solutions

## One optimizer step

The correct order for one ordinary training batch is:

```text
model.train()
optimizer.zero_grad()
forward pass and loss
loss.backward()
optional gradient clipping
optimizer.step()
```

PyTorch accumulates newly computed gradients into each parameter's `.grad`
buffer rather than overwriting it. This supports objectives with several
backward contributions and deliberate gradient accumulation. For an ordinary
one-batch-per-step loop, failing to clear old gradients mixes the current batch
with previous batches. `optimizer.zero_grad(set_to_none=True)` avoids filling
every buffer with explicit zeros and allows backward to allocate gradients as
needed.

## Causal alignment and labels

For `[A, B, C, D]`, the logit after `A` predicts `B`, the logit after `B`
predicts `C`, and the logit after `C` predicts `D`:

```python
shifted_logits = logits[:, :-1, :]  # (B, S - 1, V)
shifted_labels = labels[:, 1:]      # (B, S - 1)
```

This training API expects unshifted labels aligned with input IDs and performs
the shift internally. A dataset may instead return already shifted inputs and
targets, but the two conventions must not be mixed: shifting twice trains on
the wrong token pairs.

Labels remain separate from input IDs because input IDs must contain valid
embedding indices, while labels can contain `-100` to ignore padding, prompt
tokens, or other positions. A typical caller starts with
`labels = input_ids.clone()` and masks selected label positions afterward.

## Gradient clipping

`clip_grad_norm_` conceptually concatenates all parameter gradients, computes
their global norm, and rescales all gradients when that norm exceeds the
threshold. Rescaling preserves the aggregate gradient direction. It differs
from `clip_grad_value_`, which clamps individual elements.

Clipping must happen after backward has created the current gradients and
before the optimizer consumes them. With gradient accumulation, it happens
once after every microbatch has contributed.

## Detached reporting loss

The returned loss is for logging, not another backward pass. Returning
`loss.detach()` removes its autograd history and avoids retaining training
graphs when callers store losses. Returning `loss.item()` also removes graph
history but synchronizes a GPU to transfer the scalar immediately to the CPU.

## AdamW

Adam maintains exponential moving averages of the signed gradient and squared
gradient:

```text
m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2
```

After correcting the initialization bias, the adaptive update is proportional
to:

```text
m_hat / (sqrt(v_hat) + epsilon)
```

Adding L2 regularization to the loss adds a term proportional to the parameter
into the gradient. Adam's moments and adaptive denominator then transform that
term. AdamW instead applies parameter shrinkage separately from the adaptive
gradient update:

```text
theta <- theta - lr * adaptive_update - lr * weight_decay * theta
```

Biases and normalization gains are commonly placed in a zero-decay optimizer
group. Weight decay is primarily used to regularize learned transformation
matrices; shrinking normalization gains directly can undesirably suppress
representation scale. This is an empirical convention rather than a universal
mathematical requirement.

One common grouping heuristic places parameters with at least two dimensions
in the decay group and one-dimensional biases and normalization gains in the
no-decay group. Shared parameters such as tied embeddings must appear in
exactly one optimizer group.

## Gradient accumulation

For `K` equal-sized microbatches whose losses are means over equal numbers of
valid tokens:

```python
optimizer.zero_grad(set_to_none=True)

for input_ids, labels in microbatches:
    loss = causal_lm_loss(model(input_ids), labels)
    (loss / K).backward()

clip_if_configured()
optimizer.step()
```

Backward accumulates:

```text
(g_1 + g_2 + ... + g_K) / K
```

which matches the mean gradient of one concatenated batch under the equal-size
assumption. Reporting can either average detached unscaled losses or sum the
detached already-scaled losses. It must not divide twice.

If microbatches contain different numbers of valid tokens, equally averaging
their mean losses is incorrect because it gives each microbatch the same
weight. For exact equivalence, use summed token losses and divide every
accumulated contribution by the total valid-token count:

```text
global mean = sum of all valid token losses / total valid tokens
```

Equivalently, weight each microbatch mean by its fraction of the global valid
token count.

## Tiny-batch overfitting

Deliberately overfitting one sequence verifies that causal alignment, loss,
autograd, optimizer updates, and the model are connected well enough to
memorize. With dropout disabled and sufficient capacity, the loss should fall
close to zero. This is a powerful integration diagnostic, but it does not
prove generalization, realistic data correctness, or large-scale stability.

## Evaluation

`model.eval()` changes module behavior such as disabling dropout and making
BatchNorm use running statistics. It does not disable autograd.

`torch.no_grad()` disables gradient recording and reduces activation memory,
but it does not change dropout or any module's training flag. Validation
therefore normally uses both:

```python
model.eval()
with torch.no_grad():
    logits = model(input_ids)
```

`torch.inference_mode()` is a stronger inference-only context that can remove
additional autograd bookkeeping but places more restrictions on tensors
created within it.

## Tests

The training tests verify:

1. Next-token shifting and ignored-label propagation.
2. Parameter updates and detached loss reporting.
3. Clearing stale gradients.
4. Global gradient clipping.
5. Overfitting one short sequence with the complete decoder.
6. Equality between accumulated and concatenated-batch updates.
7. Rejection of an empty microbatch collection.

