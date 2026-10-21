from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class TrainingLoss:
    """Detached total, task, and optional unweighted router losses for logging."""

    loss: torch.Tensor
    task_loss: torch.Tensor
    aux_loss: torch.Tensor | None
    z_loss: torch.Tensor | None = None


def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    max_grad_norm: float | None = None,
    *,
    aux_loss_coefficient: float = 0.0,
    z_loss_coefficient: float = 0.0,
) -> TrainingLoss:
    """Run one shifted language-model training step.

    Inputs: input_ids [B, S], labels [B, S]; the model returns
    DecoderOutput with logits [B, S, vocab_size]. Returns detached loss details.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = model(input_ids)
    shifted_logits = output.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    task_loss = loss_fn(shifted_logits, shifted_labels)
    loss = task_loss
    if output.aux_loss is not None:
        loss = loss + aux_loss_coefficient * output.aux_loss
    if output.z_loss is not None:
        loss = loss + z_loss_coefficient * output.z_loss
    loss.backward()

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return TrainingLoss(
        loss.detach(),
        task_loss.detach(),
        output.aux_loss.detach() if output.aux_loss is not None else None,
        output.z_loss.detach() if output.z_loss is not None else None,
    )


def train_accumulation_step(
    model: nn.Module,
    optimizer: Optimizer,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    max_grad_norm: float | None = None,
    *,
    aux_loss_coefficient: float = 0.0,
    z_loss_coefficient: float = 0.0,
) -> TrainingLoss:
    """Accumulate gradients over shifted language-model microbatches.

    Input: microbatches of (input_ids [B_i, S], labels [B_i, S]);
    the model returns DecoderOutput. Returns detached mean loss details.
    Auxiliary balancing is computed per microbatch, not across the full batch.
    """
    if not microbatches:
        raise ValueError("Expected at least one microbatch.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_microbatches = len(microbatches)

    microbatch_losses = []
    task_losses = []
    aux_losses = []
    z_losses = []

    for input_ids, labels in microbatches:
        output = model(input_ids)
        shifted_logits = output.logits[:, :-1, :]
        shifted_labels = labels[:, 1:]

        task_loss = loss_fn(shifted_logits, shifted_labels)
        loss = task_loss
        if output.aux_loss is not None:
            loss = loss + aux_loss_coefficient * output.aux_loss
            aux_losses.append(output.aux_loss.detach())
        if output.z_loss is not None:
            loss = loss + z_loss_coefficient * output.z_loss
            z_losses.append(output.z_loss.detach())
        scaled_loss = loss / num_microbatches
        scaled_loss.backward()
        microbatch_losses.append(scaled_loss.detach())
        task_losses.append(task_loss.detach())

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return TrainingLoss(
        torch.stack(microbatch_losses).sum(),
        torch.stack(task_losses).mean(),
        torch.stack(aux_losses).mean() if aux_losses else None,
        torch.stack(z_losses).mean() if z_losses else None,
    )
