"""Packed expert weights with looped or grouped-matmul execution."""

import math

import torch
from torch import nn


def _validate_softcap_betas(softcap_betas: tuple[float, float] | None) -> None:
    if softcap_betas is not None and (
        len(softcap_betas) != 2
        or any(not math.isfinite(beta) or beta <= 0 for beta in softcap_betas)
    ):
        raise ValueError(
            f"Expected two finite, strictly positive softcap betas "
            f"(gate, value), got: {softcap_betas}"
        )


def _expert_activation(
    value: torch.Tensor,
    gate: torch.Tensor,
    softcap_betas: tuple[float, float] | None,
) -> torch.Tensor:
    """Inputs/output [..., hidden_dim]; None selects SwiGLU.

    SiTU-GLU uses (beta_gate, beta_value), with softcap(x,beta)=beta*tanh(x/beta).
    The intermediate activation is bounded by beta_gate*beta_value in magnitude.
    """
    if softcap_betas is None:
        return value * nn.functional.silu(gate)

    gate_beta, value_beta = softcap_betas
    soft_value = value_beta * torch.tanh(value / value_beta)
    soft_gate = gate_beta * torch.tanh(gate / gate_beta)
    # The sigmoid sees the original gate, not the capped one.
    return soft_value * soft_gate * torch.sigmoid(gate)


class PackedExperts(nn.Module):
    """Packed bias-free experts with optional SiTU-GLU.

    softcap_betas: None for SwiGLU, or (beta_gate, beta_value) for SiTU-GLU.
    Each beta is a finite positive softcap magnitude, not a learned parameter.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int,
        softcap_betas: tuple[float, float] | None = None,
    ):
        super().__init__()
        _validate_softcap_betas(softcap_betas)
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.softcap_betas = softcap_betas
        self.up_weight = nn.Parameter(
            torch.empty(num_experts, d_model, 2 * hidden_dim)
        )  # [E, D, 2F]
        self.down_weight = nn.Parameter(
            torch.empty(num_experts, hidden_dim, d_model)
        )  # [E, F, D]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use zero-mean normal weights with std = 1/sqrt(projection fan-in)."""
        nn.init.normal_(self.up_weight, mean=0.0, std=1 / math.sqrt(self.d_model))
        nn.init.normal_(self.down_weight, mean=0.0, std=1 / math.sqrt(self.hidden_dim))

    def forward(
        self, grouped_tokens: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Process token assignments already grouped by expert.

        grouped_tokens: [N, D], N = retained assignments after any dropping.
        offsets: [E], cumulative exclusive ends; counts [2, 0, 3] -> [2, 2, 5].
        N_e: assignments for expert e; repeated offsets mean an empty group.

        Returns [N, D] in grouped order, before router weighting and combining.
        Assumes valid nondecreasing offsets ending at N.
        """
        output = torch.empty_like(grouped_tokens)  # [N, D]

        for expert_id in range(self.num_experts):
            start = offsets[expert_id - 1] if expert_id > 0 else 0
            end = offsets[expert_id]
            expert_tokens = grouped_tokens[start:end]  # [N_e, D]
            up = expert_tokens @ self.up_weight[expert_id]  # [N_e, 2F]
            value, gate = torch.chunk(up, 2, dim=-1)  # [N_e, F], [N_e, F]

            hidden = _expert_activation(value, gate, self.softcap_betas)  # [N_e, F]
            output[start:end] = hidden @ self.down_weight[expert_id]  # [N_e, D]

        return output


class GroupedExperts(nn.Module):
    """Grouped-MM experts with the same optional SiTU-GLU as PackedExperts.

    softcap_betas: None for SwiGLU, or (beta_gate, beta_value) for SiTU-GLU.
    Each beta is a finite positive softcap magnitude, not a learned parameter.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int,
        softcap_betas: tuple[float, float] | None = None,
    ):
        super().__init__()
        _validate_softcap_betas(softcap_betas)
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.softcap_betas = softcap_betas
        self.up_weight = nn.Parameter(
            torch.empty(num_experts, d_model, 2 * hidden_dim)
        )  # [E, D, 2F]
        self.down_weight = nn.Parameter(
            torch.empty(num_experts, hidden_dim, d_model)
        )  # [E, F, D]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use zero-mean normal weights with std = 1/sqrt(projection fan-in)."""
        nn.init.normal_(self.up_weight, mean=0.0, std=1 / math.sqrt(self.d_model))
        nn.init.normal_(self.down_weight, mean=0.0, std=1 / math.sqrt(self.hidden_dim))

    def forward(
        self, grouped_tokens: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Process token assignments already grouped by expert.

        grouped_tokens: [N, D], N = retained assignments after any dropping.
        offsets: [E], cumulative exclusive ends; counts [2, 0, 3] -> [2, 2, 5].

        Returns [N, D] in grouped order, before router weighting and combining.
        Assumes valid nondecreasing offsets ending at N.
        """
        up = nn.functional.grouped_mm(
            grouped_tokens, self.up_weight, offs=offsets
        )  # [N, 2F]
        value, gate = torch.chunk(up, 2, dim=-1)  # [N, F], [N, F]

        hidden = _expert_activation(value, gate, self.softcap_betas)  # [N, F]
        output = nn.functional.grouped_mm(
            hidden, self.down_weight, offs=offsets
        )  # [N, D]

        return output
