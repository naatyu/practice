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

    def forward(self, x):
        """Apply left-padded Conv1d to x [B, S, C] -> [B, S, C]."""
        x = x.transpose(-1, -2)  # [B, C, S]
        x = nn.functional.pad(x, (self.left_padding, 0))  # [B, C, S + K - 1]
        x = self.conv(x)  # [B, C, S]

        return x.transpose(-1, -2)  # [B, S, C]


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
        self.alpha = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)

        self.q_conv = CausalDepthwiseConv1d(d_model, kernel_size=kernel_size)
        self.k_conv = CausalDepthwiseConv1d(d_model, kernel_size=kernel_size)
        self.v_conv = CausalDepthwiseConv1d(d_model, kernel_size=kernel_size)

        self.norm = nn.RMSNorm(self.head_dim)

    def forward(
        self,
        x,
        initial_state: torch.Tensor | None = None,
        *,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, S, _ = x.shape

        alpha = nn.functional.sigmoid(self.alpha(x))  # [B, S, d_model]
        alpha = alpha.view(B, S, self.num_heads, self.head_dim)  # [B, S, H, Dk]
        alpha = alpha.transpose(1, 2)  # [B, H, S, Dk]

        beta = nn.functional.sigmoid(self.beta(x))  # [B, S, H]
        beta = beta.transpose(1, 2)  # [B, H, S]

        qkv = self.qkv(x)  # [B, S, d_model*3]
        q, k, v = qkv.split(self.d_model, dim=-1)  # [B, S, d_model] * 3

        q = nn.functional.silu(self.q_conv(q))  # [B, S, d_model]
        k = nn.functional.silu(self.k_conv(k))  # [B, S, d_model]
        v = nn.functional.silu(self.v_conv(v))  # [B, S, d_model]

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
            alpha,
            initial_state,
            output_final_state=output_final_state,
        )  # [B, H, S, Dk]

        output = self.norm(output)
        output = output.transpose(1, 2)  # [B, S, H, Dk]
        output = output.contiguous().view(B, S, self.d_model)  # [B, S, d_model]
        output = nn.functional.sigmoid(self.gate(x)) * output  # [B, S, d_model]
        output = self.out_proj(output)  # [B, S, d_model]

        return output, state
