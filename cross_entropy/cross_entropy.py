from typing import Literal

import torch
from torch import nn

Reductiontype = Literal["none", "mean", "sum"]


class CrossEntropy(nn.Module):
    def __init__(self, reduction: Reductiontype = "mean", ignore_index: int = -100):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"Expected reduction to be none, mean or sum, got: {reduction}"
            )
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        max_logits = torch.max(logits, dim=-1, keepdim=True).values
        shifted_logits = logits - max_logits

        log_normalizer = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

        valid_mask = targets != self.ignore_index
        targets = targets.masked_fill(~valid_mask, 0)

        correct_class_logits = torch.gather(
            shifted_logits, dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)

        token_loss = log_normalizer - correct_class_logits
        token_loss = token_loss.masked_fill(~valid_mask, 0.0)

        match self.reduction:
            case "none":
                return token_loss
            case "mean":
                return token_loss.sum() / valid_mask.sum()
            case "sum":
                return torch.sum(token_loss)
