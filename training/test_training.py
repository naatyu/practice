import copy

import pytest
import torch
from torch import nn

from cross_entropy import CrossEntropy
from decoder_model import DecoderModel
from training import train_accumulation_step, train_step


class FixedLogitModel(nn.Module):
    """Small trainable model that makes training-step behavior easy to inspect."""

    def __init__(self, seq_len: int, vocab_size: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.randn(seq_len, vocab_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logits.unsqueeze(0).expand(input_ids.shape[0], -1, -1)


class RecordingCrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits_shape: torch.Size | None = None
        self.targets: torch.Tensor | None = None
        self.loss = CrossEntropy()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.logits_shape = logits.shape
        self.targets = targets.detach().clone()
        return self.loss(logits, targets)


def test_train_step_shifts_data_updates_parameters_and_detaches_loss() -> None:
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 4, 5
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[0, 2] = -100
    model = FixedLogitModel(seq_len, vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = RecordingCrossEntropy()
    parameters_before = model.logits.detach().clone()

    loss = train_step(model, optimizer, input_ids, labels, loss_fn)

    assert loss_fn.logits_shape == (batch_size, seq_len - 1, vocab_size)
    torch.testing.assert_close(loss_fn.targets, labels[:, 1:])
    assert not torch.equal(model.logits, parameters_before)
    assert loss.grad_fn is None
    assert not loss.requires_grad


def test_train_step_clears_stale_gradients() -> None:
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 4, 5
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    clean_model = FixedLogitModel(seq_len, vocab_size)
    dirty_model = copy.deepcopy(clean_model)
    dirty_model.logits.grad = torch.full_like(dirty_model.logits, 1_000.0)
    clean_optimizer = torch.optim.SGD(clean_model.parameters(), lr=0.1)
    dirty_optimizer = torch.optim.SGD(dirty_model.parameters(), lr=0.1)

    train_step(clean_model, clean_optimizer, input_ids, labels, CrossEntropy())
    train_step(dirty_model, dirty_optimizer, input_ids, labels, CrossEntropy())

    torch.testing.assert_close(dirty_model.logits, clean_model.logits)


def test_train_step_clips_global_gradient_norm() -> None:
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 4, 5
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    model = FixedLogitModel(seq_len, vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    max_grad_norm = 1e-3

    train_step(
        model,
        optimizer,
        input_ids,
        labels,
        CrossEntropy(),
        max_grad_norm=max_grad_norm,
    )

    gradient_norms = [
        parameter.grad.norm(2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    total_norm = torch.stack(gradient_norms).norm(2)
    assert total_norm <= max_grad_norm + 1e-6


def test_tiny_decoder_overfits_one_sequence() -> None:
    torch.manual_seed(0)
    model = DecoderModel(
        vocab_size=8,
        d_model=16,
        num_heads=2,
        max_seq_len=6,
        n_layers=1,
        hidden_dim=24,
        dropout_p=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    loss_fn = CrossEntropy()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    labels = input_ids.clone()

    first_loss = train_step(model, optimizer, input_ids, labels, loss_fn)
    for _ in range(59):
        final_loss = train_step(model, optimizer, input_ids, labels, loss_fn)

    assert final_loss < first_loss * 0.01
    assert final_loss < 0.02


def test_gradient_accumulation_matches_one_concatenated_batch() -> None:
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 8, 4, 5
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    full_batch_model = FixedLogitModel(seq_len, vocab_size)
    accumulated_model = copy.deepcopy(full_batch_model)
    full_batch_optimizer = torch.optim.SGD(full_batch_model.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
    microbatches = list(zip(input_ids.chunk(4), labels.chunk(4), strict=True))

    full_batch_loss = train_step(
        full_batch_model,
        full_batch_optimizer,
        input_ids,
        labels,
        CrossEntropy(),
    )
    accumulated_loss = train_accumulation_step(
        accumulated_model,
        accumulated_optimizer,
        microbatches,
        CrossEntropy(),
    )

    torch.testing.assert_close(accumulated_loss, full_batch_loss)
    torch.testing.assert_close(accumulated_model.logits, full_batch_model.logits)


def test_gradient_accumulation_rejects_empty_microbatches() -> None:
    model = FixedLogitModel(seq_len=4, vocab_size=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="microbatch"):
        train_accumulation_step(model, optimizer, [], CrossEntropy())
