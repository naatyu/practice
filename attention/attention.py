import math

import torch


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Implement bidirectional scaled dot-product attention.

    q, k, v:
        (batch_size, num_heads, sequence_length, head_dim)

    Do not use torch.nn.functional.scaled_dot_product_attention.
    """
    attention_scores = q @ k.transpose(-2, -1)
    scaled_attention_scores = attention_scores / math.sqrt(q.shape[-1])

    if causal:
        mask = torch.ones(
            (scaled_attention_scores.shape[-2], scaled_attention_scores.shape[-1]),
            dtype=torch.bool,
            device=q.device,
        ).tril()
        scaled_attention_scores = scaled_attention_scores.masked_fill(~mask, -torch.inf)

    attention_probabilities = torch.nn.functional.dropout(
        torch.softmax(scaled_attention_scores, dim=-1), p=dropout_p
    )

    return attention_probabilities @ v
