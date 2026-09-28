from dataclasses import dataclass

import torch
from torch import nn


class CausalDepthwiseConv1d(nn.Module):
    """Mix past tokens independently per channel; preserve [B, S, C]."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
        )
        self.left_padding = kernel_size - 1

    def forward(
        self, x: torch.Tensor, tail: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (output [B, S, C], new_tail [B, K-1, C]).

        The optional tail holds the previous K-1 projected tokens; a missing
        tail is treated as zeros for a fresh sequence.
        """
        history_size = self.conv.kernel_size[0] - 1

        if tail is None:
            tail = x.new_zeros(x.shape[0], history_size, x.shape[2])  # [B, K-1, C]

        combined = torch.cat((tail, x), dim=1)  # [B, S + K - 1, C]

        output = self.conv(combined.transpose(1, 2))  # [B, C, S]
        output = output.transpose(1, 2)  # [B, S, C]
        new_tail = (
            combined[:, -history_size:, :] if history_size else combined[:, :0, :]
        )  # [B, K - 1, C]

        return output, new_tail


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute channel-wise gated recurrent DeltaNet (Kimi delta attention).

    Inputs:
        q, k: [B, H, S, Dk]
        v: [B, H, S, Dv]
        beta: [B, H, S]
        alpha: [B, H, S, Dk]
        initial_state (optional): [B, H, Dk, Dv]
    """
    expected_state_shape = (*k.shape[:-2], k.shape[-1], v.shape[-1])
    if initial_state is None:
        initial_state = torch.zeros(
            expected_state_shape, device=k.device, dtype=k.dtype
        )  # [B, H , Dk, Dv]

    if initial_state.shape != expected_state_shape:
        raise ValueError(
            "Expected state and k/v dimensions to match, got: "
            f"S=[{initial_state.shape}], "
            f"k=[{k.shape}], "
            f"v=[{v.shape}]"
        )

    outputs = []
    state = initial_state  # [B, H, Dk, Dv]
    for t in range(k.shape[-2]):
        q_t = q[..., t, :]  # [B, H, Dk]
        k_t = k[..., t, :]  # [B, H, Dk]
        v_t = v[..., t, :]  # [B, H, Dv]
        beta_t = beta[..., t]  # [B, H]
        alpha_t = alpha[..., t, :]  # [B, H, Dk]

        state = alpha_t.unsqueeze(-1) * state  # [B, H, Dk, Dv]

        v_t_pred = (k_t.unsqueeze(-2) @ state).squeeze(-2)  # [B, H, Dv]
        error = v_t - v_t_pred  # [B, H, Dv]
        correction = k_t.unsqueeze(-1) @ error.unsqueeze(-2)  # [B, H, Dk, Dv]
        state = state + beta_t[..., None, None] * correction  # [B, H, Dk, Dv]
        outputs.append((q_t.unsqueeze(-2) @ state).squeeze(-2))  # [B, H, Dv]

    outputs = torch.stack(outputs, dim=-2)  # [B, H, S, Dv]

    return outputs, state if output_final_state else None


@dataclass
class KDACache:
    state: torch.Tensor  # [B, H, Dk, Dv]
    q_tail: torch.Tensor  # [B, K - 1, d_model]
    k_tail: torch.Tensor
    v_tail: torch.Tensor


class KimiDeltaAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, kernel_size: int):
        # C heck num heads
        if num_heads <= 0:
            raise ValueError(f"Expected num_heads to be > 0 got {num_heads} instead.")

        # Check for divisibility
        if d_model % num_heads != 0:
            raise ValueError(
                f"Expected d_model to be divisible by num_heads but got {d_model} and {num_heads} instead"
            )

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, num_heads)
        self.decay_input_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)

        self.q_conv = CausalDepthwiseConv1d(d_model, kernel_size=kernel_size)
        self.k_conv = CausalDepthwiseConv1d(d_model, kernel_size=kernel_size)
        self.v_conv = CausalDepthwiseConv1d(d_model, kernel_size=kernel_size)

        self.norm = nn.RMSNorm(self.head_dim)

        self.log_decay_rate = nn.Parameter(torch.zeros(self.num_heads))
        self.decay_input_bias = nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim)
        )

    def forward(
        self,
        x,
        cache: KDACache | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KDACache | None]:
        B, S, _ = x.shape

        decay_input = self.decay_input_proj(x)  # [B, S, d_model]
        decay_input = decay_input.view(
            B, S, self.num_heads, self.head_dim
        )  # [B, S, H, Dk]
        log_retention = -torch.exp(
            self.log_decay_rate[None, None, :, None]
        ) * nn.functional.softplus(
            decay_input + self.decay_input_bias[None, None, :, :]
        )  # [B, S, H, Dk]
        retention = torch.exp(log_retention)  # [B, S, H, Dk]
        retention = retention.transpose(1, 2)  # [B, H, S, Dk]

        beta = nn.functional.sigmoid(self.beta(x))  # [B, S, H]
        beta = beta.transpose(1, 2)  # [B, H, S]

        qkv = self.qkv(x)  # [B, S, d_model*3]
        q, k, v = qkv.split(self.d_model, dim=-1)  # [B, S, d_model] * 3

        q, q_tail = self.q_conv(q, None if cache is None else cache.q_tail)
        k, k_tail = self.k_conv(k, None if cache is None else cache.k_tail)
        v, v_tail = self.v_conv(v, None if cache is None else cache.v_tail)

        q = nn.functional.silu(q)  # [B, S, d_model]
        k = nn.functional.silu(k)  # [B, S, d_model]
        v = nn.functional.silu(v)  # [B, S, d_model]

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, Dk]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, Dk]
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, Dk]

        q_norm = nn.functional.normalize(q, dim=-1)  # [B, H, S, Dk]
        k_norm = nn.functional.normalize(k, dim=-1)  # [B, H, S, Dk]

        output, state = recurrent_kda(
            q_norm,
            k_norm,
            v,
            beta,
            retention,
            initial_state=None if cache is None else cache.state,
            output_final_state=use_cache,
        )  # [B, H, S, Dk]

        output = self.norm(output)
        output = output.transpose(1, 2)  # [B, S, H, Dk]
        output = output.contiguous().view(B, S, self.d_model)  # [B, S, d_model]
        output = nn.functional.sigmoid(self.gate(x)) * output  # [B, S, d_model]
        output = self.out_proj(output)  # [B, S, d_model]

        if use_cache:
            assert state is not None
            cache = KDACache(state, q_tail, k_tail, v_tail) if use_cache else None
            return output, cache

        return output, None
