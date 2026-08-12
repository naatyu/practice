# Decoder-only language model interview practice

This file contains only the exercise prompt and interview questions. Completed
answers and explanations are kept separately in `SOLUTIONS.md`.

## Exercise

Implement a decoder-only language model in PyTorch using the existing
Transformer block and rotary positional encoding implementations.

```text
input token IDs: (batch_size, sequence_length)
output logits:   (batch_size, sequence_length, vocab_size)
```

Use:

- Token embeddings
- A configurable stack of pre-norm causal Transformer blocks
- Rotary positional encoding
- A final RMSNorm
- A bias-free vocabulary projection
- Optional weight tying between the token embedding and vocabulary projection
- A position offset compatible with future KV-cached decoding

Do not implement the training loss or KV cache inside this exercise.

### Discussion questions

1. Why does a language model produce vocabulary logits at every sequence
   position instead of token IDs directly?
2. How are logits and targets aligned for next-token training?
3. What is the difference between logits and probabilities?
4. Why should repeated Transformer blocks be stored in `nn.ModuleList`?
5. What are the stored shapes of token-embedding and language-model-head
   weights?
6. How does weight tying work in PyTorch, and why can it be useful?
7. Why is the language-model head commonly bias-free?
8. Why does a pre-norm decoder apply a final normalization after all blocks?
9. Why is a SwiGLU hidden size near `(8/3) * d_model` often compared with a
   standard GELU hidden size of `4 * d_model`?
10. How should the next-token cross-entropy loss be computed from logits shaped
    `(B, S, V)`?
11. How can a test prove that the position offset reaches RoPE through every
    layer of the model?

