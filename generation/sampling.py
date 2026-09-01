import torch

from decoder_model import DecoderModel


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0 but got: {temperature}")
    vocab_size = logits.shape[-1]
    if top_k is not None and (top_k <= 0 or top_k > vocab_size):
        raise ValueError(
            f"Expected top_k to be in 0 < top_k <= {vocab_size}: got {top_k}"
        )
    if top_p is not None and not (0 < top_p <= 1):
        raise ValueError(f"Expected top_p to be in 0 < top_p <= 1: got {top_p}")

    scaled_logits = logits / temperature  # [B, V]

    if top_k is not None:
        _, indices = torch.topk(scaled_logits, k=top_k, dim=-1)  # [B, top_k]
        keep_mask = torch.zeros_like(
            scaled_logits, dtype=torch.bool, device=scaled_logits.device
        ).scatter_(dim=-1, index=indices, value=True)  # [B, V]
        scaled_logits.masked_fill_(~keep_mask, value=float("-inf"))

    if top_p is not None and top_p != 1.0:
        sorted_values, sorted_indices = torch.sort(
            scaled_logits, descending=True
        )  #  [B, V], [B, V]
        probabilities = torch.softmax(sorted_values, dim=-1)  #  [B, V]

        cumsum = torch.cumsum(probabilities, dim=-1)  #  [B, V]
        sorted_remove_mask = cumsum > top_p  #  [B, V]
        sorted_remove_mask[..., 1:] = sorted_remove_mask[
            ..., :-1
        ].clone()  # [B, V] To keep crossing token
        sorted_remove_mask[..., 0] = (
            False  # Always make sure to have at least one token
        )

        remove_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)  #  [B, V]
        remove_mask.scatter_(
            dim=-1, index=sorted_indices, src=sorted_remove_mask
        )  #  [B, V]

        scaled_logits = scaled_logits.masked_fill_(remove_mask, float("-inf"))

    probabilities = torch.softmax(scaled_logits, dim=-1)  #  [B, V]
    selected_token = torch.multinomial(probabilities, 1, generator=generator)  # [B, 1]

    return selected_token


@torch.inference_mode()
def generate_sampled(
    model: DecoderModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if max_new_tokens == 0:
        return input_ids

    # Prefill phase
    prefill_logits, kv_caches = model(
        input_ids, use_cache=True
    )  # [B, S, V], [([B, H, S, D], [B, H, S, D])] * n_layer
    first_token = sample_next_token(
        prefill_logits[:, -1, :],
        temperature=temperature,
        generator=generator,
        top_k=top_k,
        top_p=top_p,
    )  # [B, 1]
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
        next_tokens = sample_next_token(
            logits.squeeze(1),
            temperature=temperature,
            generator=generator,
            top_k=top_k,
            top_p=top_p,
        )  # [B, 1]

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
