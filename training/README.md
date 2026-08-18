# Language-model training interview practice

This file contains only the exercise prompts and interview questions.
Completed answers and explanations are kept separately in `SOLUTIONS.md`.

## Exercise 1: One training step

Implement one decoder language-model training step using unshifted token IDs
and labels.

```text
input_ids: (batch_size, sequence_length)
labels:    (batch_size, sequence_length)
```

The labels are positionally aligned with the input IDs and may contain the
cross-entropy `ignore_index`. The function must:

- Enable training mode.
- Clear existing gradients.
- Run the decoder forward pass.
- Align logits and labels for next-token prediction.
- Compute cross-entropy.
- Run backward propagation.
- Optionally clip the global gradient norm.
- Perform one optimizer update.
- Return a detached loss for reporting.

### Discussion questions

1. Why does PyTorch accumulate gradients in `.grad`?
2. What is the correct order of zeroing gradients, forward, backward, clipping,
   and stepping the optimizer?
3. Why are input IDs and labels separate tensors?
4. Where should causal language-model shifting occur, and what API contract
   prevents shifting twice?
5. Why should gradient norm clipping occur after backward and before the
   optimizer step?
6. How does norm clipping differ from value clipping?
7. Why should a reported training loss be detached?
8. What does overfitting one tiny batch verify, and what does it not verify?
9. What first and second moments does Adam maintain?
10. How does AdamW differ from adding an L2 penalty to the loss?
11. Why are biases and normalization gains commonly excluded from weight
    decay?

## Exercise 2: Gradient accumulation

Implement one optimizer update from several equal-sized microbatches. Clear
gradients once, run backward once per microbatch, optionally clip once, and
step the optimizer once.

### Discussion questions

1. How does gradient accumulation create a larger effective batch?
2. Why is each mean microbatch loss divided by the number of microbatches
   before backward?
3. How should loss reporting differ from backward scaling?
4. When does averaging microbatch means fail to equal the full-batch mean?
5. How should variable numbers of valid tokens be handled exactly?
6. How can accumulated and concatenated-batch updates be tested for
   equivalence?
7. Why should clipping happen after every microbatch has accumulated?

## Exercise 3: Evaluation behavior

Explain and use evaluation mode without building an unnecessary validation
framework.

### Discussion questions

1. What behavior does `model.eval()` change?
2. What behavior does `torch.no_grad()` change?
3. Why are both normally used during validation?
4. How does `torch.inference_mode()` differ from `torch.no_grad()`?

