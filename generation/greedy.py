import torch

from decoder_model import DecoderModel


@torch.inference_mode()
def generate_greedy_naive(
    model: DecoderModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """input_ids: [B, S]"""
    current_generation = input_ids  # Safe since cat will create a copy later , no risk of in place change

    finished = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )  # [B]

    for _ in range(max_new_tokens):
        logits = model(current_generation)  # [B, S, V]
        generated_logits = logits[:, -1, :]  # [B, V], select only the last token
        generated_tokens = torch.argmax(
            generated_logits, dim=-1, keepdim=True
        )  # [B, 1]

        if eos_token_id is not None:
            finished = finished | generated_tokens.squeeze(1).eq(eos_token_id)  # [B]
            generated_tokens = torch.where(
                finished.unsqueeze(1),
                torch.full_like(generated_tokens, eos_token_id),
                generated_tokens,
            )  # [B,  1]

        current_generation = torch.cat(
            (current_generation, generated_tokens), dim=-1
        )  # [B, S+1]

        if eos_token_id is not None and finished.all():
            break

    return current_generation


@torch.inference_mode()
def generate_greedy(
    model: DecoderModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    if max_new_tokens == 0:
        return input_ids

    # Prefill phase
    prefill_logits, kv_caches = model(
        input_ids, use_cache=True
    )  # [B, S, V], [([B, H, S, D], [B, H, S, D])] * n_layer
    first_token = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True)  # [B, 1]
    current_generation = torch.cat((input_ids, first_token), dim=-1)  # [B, S+1]

    # Decode phase
    if eos_token_id is not None:
        finished = first_token.squeeze(1).eq(eos_token_id)  # [B]
        if finished.all():
            return current_generation

    for _ in range(max_new_tokens - 1):
        logits, kv_caches = model(
            current_generation[:, -1].unsqueeze(-1), use_cache=True, kv_caches=kv_caches
        )  # [B, 1, V]
        next_tokens = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # [B, 1]

        if eos_token_id is not None:
            finished = finished | next_tokens.squeeze(1).eq(eos_token_id)  # [B]
            next_tokens = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_tokens, eos_token_id),
                next_tokens,
            )  # [B, 1]

        current_generation = torch.cat(
            (current_generation, next_tokens), dim=-1
        )  # [B, S+1]

        if eos_token_id is not None and finished.all():
            break

    return current_generation
