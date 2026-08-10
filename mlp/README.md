# Transformer MLP interview practice

This file contains only the exercise prompt and interview questions. Completed
reasoning and corrections are kept separately in `SOLUTIONS.md`.

## Exercise 1: GELU MLP

Implement the position-wise feed-forward network used in a Transformer.

```text
input:  (batch_size, sequence_length, d_model)
output: (batch_size, sequence_length, d_model)
```

The hidden feed-forward dimension is provided to the constructor. Do not use a
prebuilt Transformer layer. Support optional dropout after the hidden
activation and after the down projection.

### Discussion questions

1. What operations make up a standard Transformer MLP?
2. What is the shape after each operation?
3. Along which axis does the MLP mix information?
4. Why is a nonlinear activation required between the projections?
5. Why is the hidden dimension commonly larger than `d_model`?
6. Where are activation and dropout placed in the standard MLP?

## Exercise 2: SwiGLU

After completing the GELU MLP, implement a SwiGLU feed-forward network with the
same input/output interface.

### Discussion questions

1. How does a gated linear unit differ from the standard GELU MLP?
2. Why are two projections from `d_model` to `hidden_dim` needed?
3. Which branch receives SiLU, and how are the branches combined?
4. What is the shape after every operation?
5. How can the two input projections be fused into one linear layer?
6. Why is the SwiGLU hidden dimension often smaller than the hidden dimension
   of a parameter-matched standard MLP?
