import torch


@torch.no_grad()
def maxvio(counts: torch.Tensor) -> torch.Tensor:
    """Relative overload of the busiest expert versus mean load; counts [E]."""
    target = counts.sum() / counts.shape[0]

    if target == 0:
        return counts.new_zeros((), dtype=torch.float32)

    return (counts.max() - target) / target


@torch.no_grad()
def assignment_fraction(counts: torch.Tensor) -> torch.Tensor:
    """Shares of actual top-k assignments, not router probabilities; [E] -> [E]."""
    total_assignments = counts.sum()
    if total_assignments == 0:
        return counts.new_zeros(counts.shape, dtype=torch.float32)

    return counts / total_assignments


@torch.no_grad()
def unused_expert_count(counts: torch.Tensor) -> torch.Tensor:
    """Count experts with no assignments in this window, not permanently dead experts."""
    return (counts == 0).sum()


@torch.no_grad()
def assignment_entropy(counts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Usage-balance entropy in bits from assignment counts [E].

    Higher means more evenly used experts, not less confident token routing.
    """
    fraction = assignment_fraction(counts)
    log_fraction = torch.log2(fraction.clamp_min(eps))

    return torch.sum(-fraction * log_fraction)


@torch.no_grad()
def router_entropy(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-token uncertainty in bits before top-k; [B, S, E] -> [B, S].

    Measures routing confidence, not aggregate expert-usage balance.
    """
    log_probabilities = torch.log2(probabilities.clamp_min(eps))

    return torch.sum(-probabilities * log_probabilities, dim=-1)
