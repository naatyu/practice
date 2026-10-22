# Transformer block interview practice

This file contains only the exercise prompt and interview questions. Completed
reasoning and corrections are kept separately in `SOLUTIONS.md`.

## Exercise

Implement a pre-norm decoder-only Transformer block in PyTorch using the
existing RMSNorm and multi-head attention implementations.

Use:

- Causal self-attention
- Two RMSNorm layers
- Two residual connections
- A position-wise SwiGLU feed-forward network (shared bias-free default)
- Feed-forward hidden dimension supplied to the constructor

Do not use `torch.nn.TransformerEncoderLayer` or
`torch.nn.TransformerDecoderLayer`.

Input and output shape:

```text
(batch_size, sequence_length, d_model)
```

The current API always returns `(hidden_states, kv_cache, aux_loss, z_loss)`.
Hidden states keep the shape above; cache is `None` unless requested, and
both router losses are `None` for dense FFNs or evaluation. Codex added this return
plumbing for the later MoE exercise; the block's tensor computation is unchanged.

For MoE, optional `shared_hidden_dim` and `shared_expert_gate` are forwarded to
`GroupedMoE`; defaults None/False keep the routed-only path. The shared branch
is combined inside MoE, before the block's existing residual addition.
`score_func` is also forwarded (default `"softmax"`); `"sigmoid"` disables the
router's z-loss but keeps the same block return signature.

### Discussion questions

1. What are the two main sublayers of a decoder Transformer block?
2. Where are normalization and residual connections placed in a pre-norm
   block?
3. Why must decoder self-attention be causal?
4. Along which dimension does the feed-forward network operate?
