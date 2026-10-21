import torch


def load_balancing_loss(
    probabilities: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """
    probabilities: [B, S, E]
    counts: [E], assignment counts from the current forward pass
    """
    mean_probabilities = probabilities.mean((0, 1))  # [E]
    assignment_fraction = counts / counts.sum()  # [E]

    return probabilities.shape[-1] * torch.sum(assignment_fraction * mean_probabilities)


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Keep the log-normalizer (log of the softmax denominator) near zero
    for router stability; [B, S, E] -> scalar.
    """
    return torch.logsumexp(logits.float(), dim=-1).square().mean()
