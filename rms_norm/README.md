# RMSNorm interview practice

This file contains only the exercise prompt and interview questions. Completed
reasoning and corrections are kept separately in `SOLUTIONS.md`.

## Exercise

Implement RMSNorm over the final dimension of a tensor without using
`torch.nn.RMSNorm` or `torch.nn.functional.rms_norm`.

```text
x:      (..., hidden_dim)
weight: (hidden_dim,)
output: (..., hidden_dim)
```

### Discussion questions

1. What does RMSNorm compute?
2. How does it differ from LayerNorm?
3. Which dimension is normalized, and why must it be retained during the
   reduction?
4. Where is epsilon placed in the RMS expression?
5. How is the learned per-feature weight applied?
6. How can tensor shapes and the intended operation distinguish matrix
   multiplication from elementwise multiplication?
7. What does PyTorch's `normalized_shape` argument represent?
8. Does this implementation need a manually written backward pass?
