# Stable cross-entropy interview practice

This file contains only the exercise prompt and interview questions. Completed
reasoning and corrections are kept separately in `SOLUTIONS.md`.

## Exercise 1: Stable cross-entropy

Implement cross-entropy directly from logits without using:

- `torch.nn.functional.cross_entropy`
- `softmax`
- `log_softmax`
- `torch.logsumexp`

Input shapes:

```text
logits:  (batch_size, sequence_length, vocab_size)
targets: (batch_size, sequence_length)
```

The implementation should produce one loss per token and support `none`,
`sum`, and `mean` reductions. The default reduction should be `mean`.

### Discussion questions

1. What does cross-entropy measure between a target distribution and a
   predicted distribution?
2. Why does a one-hot target reduce cross-entropy to `-log(p_y)`?
3. How can cross-entropy be expressed directly in terms of logits?
4. Why can directly exponentiating a large logit overflow?
5. How does subtracting the maximum logit make the computation stable?
6. Why must the maximum be added back to the log-sum-exp expression?
7. How can shifting both the vocabulary logits and target logit avoid adding
   the maximum explicitly?
8. Along which dimension are the maximum and exponential sum computed for
   language-model logits?
9. Why should the maximum retain a singleton vocabulary dimension?
10. Why is `gather` used instead of constructing one-hot targets?
11. What are the index and output shapes of `gather`?
12. What do the `none`, `sum`, and `mean` reductions return?

## Later extensions

After the basic implementation and reference tests:

1. Add `ignore_index` and average only over valid tokens.
2. Add shifted causal language-model logits and targets.
3. Compare values and gradients with PyTorch.
4. Test extreme logits that overflow a naive implementation.
5. Use the loss in a minimal AdamW training loop.
6. Overfit a tiny batch as an end-to-end correctness test.
