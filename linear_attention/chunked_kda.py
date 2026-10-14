"""AI-authored worked example: chunkwise KDA with ordinary PyTorch operations."""

import torch


def chunked_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_retention: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 32,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Regroup the KDA recurrence into a triangular solve per chunk.

    Written by Codex at the learner's request, not by the learner.

    Inputs:
        q, k: [B, H, S, Dk]; the layer supplies normalized queries and keys.
        v: [B, H, S, Dv]
        beta: [B, H, S], write strengths (normally in [0, 1]).
        log_retention: [B, H, S, Dk], finite nonpositive log(alpha).
        initial_state (optional): [B, H, Dk, Dv]

    Unlike recurrent_kda, this accepts log(alpha), not alpha. Passing the
    layer's log gates directly avoids taking log of an underflowed zero.
    Arithmetic uses float32 (float64 if any input uses it), output uses v's
    dtype, and the returned state keeps the arithmetic dtype. Autograd is
    supported. Chunks are sequential, but there is no Python token loop.

    This deliberately materializes [B, H, C, C, Dk] surviving keys to make
    channel-wise decay visible. It is NOT the paper's optimized WY kernel
    and is not guaranteed faster or smaller than the recurrent operator.
    """
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")

    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError("q and k must have matching [B, H, S, Dk] shapes")
    if v.ndim != 4 or v.shape[:-1] != k.shape[:-1]:
        raise ValueError("v must have shape [B, H, S, Dv] matching k")
    if beta.shape != k.shape[:-1] or log_retention.shape != k.shape:
        raise ValueError("beta must be [B, H, S] and log_retention [B, H, S, Dk]")

    state_shape = (*k.shape[:2], k.shape[-1], v.shape[-1])
    if initial_state is not None and initial_state.shape != state_shape:
        raise ValueError("initial_state must have shape [B, H, Dk, Dv]")

    inputs = (q, k, v, beta, log_retention)
    if initial_state is not None:
        inputs = (*inputs, initial_state)
    if any(not x.is_floating_point() or x.device != q.device for x in inputs):
        raise ValueError("all inputs must be floating-point tensors on the same device")
    if not torch.isfinite(log_retention).all() or (log_retention > 0).any():
        raise ValueError("log_retention must be finite and nonpositive")

    output_dtype = v.dtype
    compute_dtype = (
        torch.float64
        if any(x.dtype == torch.float64 for x in inputs)
        else torch.float32
    )
    q, k, v, beta, log_retention = (
        x.to(compute_dtype) for x in (q, k, v, beta, log_retention)
    )
    state = (
        torch.zeros(state_shape, device=q.device, dtype=compute_dtype)
        if initial_state is None
        else initial_state.to(compute_dtype)
    )

    outputs = []
    # Explicit casts alone do not prevent AMP from downcasting the matmuls.
    with torch.autocast(device_type=q.device.type, enabled=False):
        for start in range(0, k.shape[-2], chunk_size):
            stop = start + chunk_size
            q_c, k_c, v_c = (x[..., start:stop, :] for x in (q, k, v))
            beta_c = beta[..., start:stop]  # [B, H, C]
            C = k_c.shape[-2]  # The last chunk may be shorter.

            # 1. Incoming state sees gates 0..t; a write at i sees gates i+1..t.
            cumulative_log = log_retention[..., start:stop, :].cumsum(dim=-2)
            incoming_decay = cumulative_log.exp()  # [B, H, C, Dk]
            pair_log = cumulative_log.unsqueeze(-2) - cumulative_log.unsqueeze(-3)
            causal = torch.ones(C, C, device=q.device, dtype=torch.bool).tril()
            # Mask BEFORE exp: noncausal log differences can be huge and positive.
            survival = pair_log.masked_fill(~causal[..., None], -torch.inf).exp()
            surviving_keys = survival * k_c.unsqueeze(-3)  # [B, H, t, i, Dk]

            # 2. e_t = base_t - sum_{i<t} beta_i * <k_t, surviving k_i> * e_i.
            #    Solve (I + A) E = base for all C errors, without a token loop.
            interactions = (k_c.unsqueeze(-2) * surviving_keys).sum(dim=-1)
            A = (interactions * beta_c.unsqueeze(-2)).tril(diagonal=-1)
            system = torch.eye(C, device=q.device, dtype=compute_dtype) + A
            base = v_c - (k_c * incoming_decay) @ state  # [B, H, C, Dv]
            errors = torch.linalg.solve_triangular(
                system, base, upper=False, unitriangular=True
            )  # [B, H, C, Dv]

            # 3. Queries read incoming state AND writes through their own token.
            reads = (q_c.unsqueeze(-2) * surviving_keys).sum(dim=-1)
            reads = reads * beta_c.unsqueeze(-2)  # [B, H, C, C], inclusive diagonal
            output = (q_c * incoming_decay) @ state + reads @ errors
            outputs.append(output)

            # 4. Carry only the final matrix state to the next chunk.
            final_keys = k_c * survival[..., -1, :, :] * beta_c.unsqueeze(-1)
            state = (
                incoming_decay[..., -1, :, None] * state
                + final_keys.transpose(-1, -2) @ errors
            )  # [B, H, Dk, Dv]

    output = torch.cat(outputs, dim=-2) if outputs else v
    return output.to(output_dtype), state if output_final_state else None
