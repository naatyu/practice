import torch
from torch import nn
from torch.optim import Optimizer


def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    max_grad_norm: float | None = None,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    loss = loss_fn(shifted_logits, shifted_labels)
    loss.backward()

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return loss.detach()


def train_accumulation_step(
    model: nn.Module,
    optimizer: Optimizer,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    max_grad_norm: float | None = None,
):
    if not microbatches:
        raise ValueError("Expected at least one microbatch.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_microbatches = len(microbatches)

    microbatch_loss = 0

    for input_ids, labels in microbatches:
        logits = model(input_ids)
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]

        loss = loss_fn(shifted_logits, shifted_labels)
        scaled_loss = loss / num_microbatches
        scaled_loss.backward()
        microbatch_loss += scaled_loss.detach()

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return microbatch_loss
