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
- Return detached total/task/balancing/z-loss details for reporting (`TrainingLoss`).

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

## Tiny MoE training run

```bash
uv run --extra plots python -m training.train_moe
```

`train_moe.py` reuses `train_step` on one fixed batch, without gradient accumulation.
It records loss, seven router metrics, and two drop-rate metrics per layer,
then resets both requested and dropped counts through `GroupedMoE.reset_metrics()`
once per optimizer step. `plot_metrics` appends the training setting to filenames:
`moe_metrics_no_aux_dropless.png` for the baseline, or
`moe_metrics_aux_0.05_capacity_1_probs.png` for auxiliary coefficient 0.05,
capacity factor 1, and score-priority dropping. Both have matching `_experts.png`
figures under `training/plots/`. Settings also appear in figure titles.
The expert figure has requested shares, mean probabilities, and drop rates;
colors identify experts and dashed lines mark equal shares or zero drops.
Use `--steps`, `--device`, or `--output` to change the run. `--output` specifies
the base path; the training-setting suffix is added even for custom names.

The decoder returns `DecoderOutput`; both training helpers consume its logits
and optional live balancing/z-losses. They backpropagate
`task_loss + aux_loss_coefficient * aux_loss + z_loss_coefficient * z_loss`
(omitting absent losses) and return detached `TrainingLoss` fields
(`loss`, `task_loss`, `aux_loss`, `z_loss`). Both coefficients default to zero.
For a balancing experiment, for example:

```bash
uv run --extra plots python -m training.train_moe --aux-loss-coefficient 0.01 --output training/plots/moe_balanced.png
```

Use `--z-loss-coefficient 0.001` to also study router stability. Nonzero z-loss
coefficients add a `_z_0.001` filename tag, preserving existing baseline names.
The plot distinguishes total, task, unweighted balancing loss, and unweighted z-loss. Compare with
the zero-coefficient baseline; an auxiliary reference of 1 is not a task-loss target.

Capacity settings pass from `ModelConfig` through the decoder and block to every
MoE layer. No `--capacity-factor` means dropless; otherwise choose a policy:

```bash
uv run --extra plots python -m training.train_moe --aux-loss-coefficient 0.05 --capacity-factor 1.0 --drop-policy probs
```

Use `--drop-policy position` for the keep-first comparison. Global drop-rate
curves and per-expert rates show overflow; requested-share curves remain pre-drop.
Codex added this configuration/reporting wiring and its tests. Capacity applies
per forward, so leave it disabled when checking full-versus-cached inference
equivalence: dropping decisions can depend on the other tokens in a chunk.

The learner wrote the initial script; Codex added logging, plotting, tests,
and training fixes. The default uses top-2 routing so normalized selected
weights can train the router through task loss. The learner implemented the
balancing formula; Codex added explicit loss returns, training integration,
and updated generation callers/tests. The tiny run still has no gradient accumulation.

