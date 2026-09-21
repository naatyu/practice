import math

import torch
import torch.nn.functional as F


def local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """k, v: [B, H, S, D], q: [B, H, Q, D]"""
    if window_size <= 0:
        raise ValueError(
            f"Invalid window size, expected window size to be greater than 0 got: {window_size}"
        )

    attn_score = q @ k.transpose(-1, -2)  # [B, H, Q, S]
    scaled_attn_score = attn_score / math.sqrt(q.shape[-1])  # [B, H, Q, S]

    # Construct local window mask
    key_len = k.shape[-2]
    query_len = q.shape[-2]
    cached_len = key_len - query_len
    if cached_len < 0:
        raise ValueError(
            "Expected key squence length and query sequence length are not compatbile."
        )

    query_positions = torch.arange(
        start=cached_len, end=cached_len + query_len, device=q.device
    ).unsqueeze(1)  # [Q, 1]
    key_positions = torch.arange(key_len, device=q.device).unsqueeze(0)  # [1, S]
    causal_mask = query_positions >= key_positions  # [Q, S]
    window_mask = query_positions <= key_positions + window_size - 1  # [Q, S]
    mask = causal_mask & window_mask  # [Q, S]

    scaled_attn_score.masked_fill_(~mask, -torch.inf)  # [B, H, Q, S]

    attn_probs = F.dropout(
        torch.softmax(scaled_attn_score, dim=-1), p=dropout_p
    )  # [B, H, Q, S]

    return attn_probs @ v  # [B, H, Q, D]


def efficient_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """k, v: [B, H, S, D], q: [B, H, Q, D]"""
    if window_size <= 0:
        raise ValueError(
            f"Invalid window size, expected window size to be greater than 0 got: {window_size}"
        )

    key_len = k.shape[-2]
    query_len = q.shape[-2]
    cached_len = key_len - query_len
    if cached_len < 0:
        raise ValueError(
            "Expected key squence length and query sequence length are not compatbile."
        )

    # Only keep keys and values reachable by one of the Q queries. This avoids
    # padding and unfolding an old cached prefix that lies outside every window.
    relevant_start = max(0, cached_len - window_size + 1)
    relevant_k = k[..., relevant_start:, :]
    relevant_v = v[..., relevant_start:, :]

    # Pad only when the earliest query has fewer than W preceding positions.
    # The resulting sequence has Q + W - 1 entries and unfolds into Q windows.
    left_padding = max(0, window_size - 1 - cached_len)
    padded_key = F.pad(relevant_k, (0, 0, left_padding, 0))  # [B, H, Q + W - 1, D]
    padded_value = F.pad(relevant_v, (0, 0, left_padding, 0))  # [B, H, Q + W - 1, D]

    # 2. Create overlapping sequences
    k_windows = padded_key.unfold(-2, window_size, 1)  # [B, H, Q, D, W]
    v_windows = padded_value.unfold(-2, window_size, 1)  # [B, H, Q, D, W]
    k_windows = k_windows.transpose(-1, -2)  # [B, H, Q, W, D]
    v_windows = v_windows.transpose(-1, -2)  # [B, H, Q, W, D]

    # 3. Construct padding mask, padded keys are True, real keys are False
    padded_mask = torch.zeros(relevant_k.shape[-2], dtype=torch.bool, device=k.device)
    padded_mask = F.pad(padded_mask, (left_padding, 0), value=True)
    padded_windows_mask = padded_mask.unfold(0, window_size, 1)  # [Q, W]

    # 4. Compute local scores
    # Educational:
    # attn_scores = q.unsqueeze(-2) * k_windows  # [B, H, Q, W, D]
    # attn_scores = attn_scores.sum(dim=-1)  # [B, H, Q, W]

    attn_scores = torch.matmul(q.unsqueeze(-2), k_windows.transpose(-1, -2)).squeeze(
        -2
    )  # [B, H, Q, W], avoid to materialize [B, H, Q, W, D]
    scaled_attn_scores = attn_scores / math.sqrt(q.shape[-1])  # [B, H, Q, W]

    # 5. Remove padded positions
    scaled_attn_scores = scaled_attn_scores.masked_fill_(
        padded_windows_mask, -torch.inf
    )  # [B, H, Q, W]

    # 6. Apply softmax
    attn_probs = F.dropout(
        torch.softmax(scaled_attn_scores, dim=-1), p=dropout_p
    )  # [B, H, Q, W]

    # 7. Combine windows
    # Educational:
    # output = attn_probs.unsqueeze(-1) * v_windows  # [B, H, Q, W, D]
    # output = output.sum(dim=-2)  # [B, H, Q, D]

    output = torch.matmul(attn_probs.unsqueeze(-2), v_windows).squeeze(
        -2
    )  # [B, H, Q, D]

    return output
