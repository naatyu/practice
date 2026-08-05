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
- A position-wise feed-forward network with GELU
- Feed-forward hidden dimension supplied to the constructor

Do not use `torch.nn.TransformerEncoderLayer` or
`torch.nn.TransformerDecoderLayer`.

Input and output shape:

```text
(batch_size, sequence_length, d_model)
```

### Discussion questions

1. What are the two main sublayers of a decoder Transformer block?
2. Where are normalization and residual connections placed in a pre-norm
   block?
3. Why must decoder self-attention be causal?
4. Along which dimension does the feed-forward network operate?

