import torch
from torch import nn


class CrossEntropy(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        max_logits = torch.max(logits, keepdim=True)
        stable_logits = logits - max_logits

        exp_logits = torch.exp(stable_logits)
        exp_logits = torch.sum(exp_logits, dim=-1)

        pred_probs = torch.log(exp_logits)
        correct_class = torch.gather(pred_probs, dim=-1, index=targets)
