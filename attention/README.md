# Attention interview practice

This file contains only exercise prompts and interview questions. Completed
answers and explanations are kept separately in `SOLUTIONS.md`.

## Exercise 1: Bidirectional scaled dot-product attention

Implement bidirectional scaled dot-product attention in PyTorch.

```text
q, k, v: (batch_size, num_heads, sequence_length, head_dim)
```

Do not use `torch.nn.functional.scaled_dot_product_attention`.

### Discussion questions

1. What does attention compute?
2. What is the shape of the token-to-token attention matrix?
3. How do `Q` and `K` produce that matrix? Which dimensions are transposed?
4. What is the difference between the attention matrix and the final output?
5. Why are the attention scores scaled?
6. Why does standard attention use softmax?
7. Which dimension does softmax normalize, and why?
8. What are the output shape and its intuitive meaning?
9. What are the time and memory complexities?
10. How would you test the implementation?

## Exercise 2: Causal attention

Extend the attention function to optionally support causal attention while
preserving bidirectional attention as the default.

### Discussion questions

1. What does causal attention change in the attention matrix?
2. Why is causal attention needed in autoregressive language models?
3. At what point in the computation should the causal mask be applied?

