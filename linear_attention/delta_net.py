import torch


def naive_recurrent_delta_net(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,  # [B, H, S]
    initial_state: torch.Tensor | None = None,
    *,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
    for t in range(q.shape[-2]):
        k_t = k[..., t, :]  # [B, H, Dk]
        v_t = v[..., t, :]  # [B, H, Dv]
        q_t = q[..., t, :]  # [B, H, Dk]
        beta_t = beta[..., t]  # [B, H]

        v_t_pred = (k_t.unsqueeze(-2) @ state).squeeze(-2)  # [B, H, Dv]
        error = v_t - v_t_pred  # [B, H, Dv]
        correction = k_t.unsqueeze(-1) * error.unsqueeze(-2)  # [B, H, Dk, Dv]
        state = (
            state + beta_t.unsqueeze(-1).unsqueeze(-2) * correction
        )  # [B, H, Dk, Dv]
        outputs.append((q_t.unsqueeze(-2) @ state).squeeze(-2))  # [B, H, Dv]

    outputs = torch.stack(outputs, dim=-2)  # [B, H, S, Dv]

    return outputs, state if output_final_state else None
