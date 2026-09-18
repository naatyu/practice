import torch

from generation.sampling import (
    apply_repetition_penalty,
    logits_to_probabilities,
    sample_next_token,
)


def verify_speculative_tokens(
    draft_tokens: torch.Tensor,  # [N]
    draft_probs: torch.Tensor,  # [N, V]
    target_probs: torch.Tensor,  # [N+1, V] N is the number of draft proposals
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    for i, token in enumerate(draft_tokens):
        p_y, q_y = target_probs[i, token], draft_probs[i, token]  # [], []
        accept_probs = torch.clamp(p_y / q_y, max=1)  # []
        u = torch.rand((), generator=generator, device=draft_probs.device)
        accepted = u < accept_probs  # []

        # Token is accepted, go to next token
        if accepted:
            continue

        # Token is refused, sample from missing probabilities
        missing_probs = torch.clamp(target_probs[i] - draft_probs[i], min=0)  # [V]
        missing_probs /= missing_probs.sum()  # [V]
        corrected_token = torch.multinomial(
            missing_probs, 1, generator=generator
        )  # [1]

        return torch.cat((draft_tokens[:i], corrected_token), dim=-1)  # [i+1]

    bonus_token = torch.multinomial(target_probs[-1], 1, generator=generator)  # [1]

    return torch.cat((draft_tokens, bonus_token), dim=-1)  # [N+1]


def verify_speculative_tokens_batched(
    draft_tokens: torch.Tensor,  # [B, N]
    draft_probs: torch.Tensor,  # [B, N, V]
    target_probs: torch.Tensor,  # [B, N+1, V]
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Verify a batch and return the longest common committed prefix [B, C]."""
    batch_size, num_draft_tokens = draft_tokens.shape
    proposed_target_probs = torch.gather(
        target_probs[:, :num_draft_tokens],
        dim=-1,
        index=draft_tokens.unsqueeze(-1),
    ).squeeze(-1)  # [B, N]
    proposed_draft_probs = torch.gather(
        draft_probs,
        dim=-1,
        index=draft_tokens.unsqueeze(-1),
    ).squeeze(-1)  # [B, N]

    acceptance_probs = torch.clamp(
        proposed_target_probs / proposed_draft_probs, max=1
    )  # [B, N]
    accepted = torch.rand(
        acceptance_probs.shape,
        generator=generator,
        device=draft_tokens.device,
    ) < acceptance_probs  # [B, N]
    rejected = ~accepted
    has_rejection = rejected.any(dim=-1)  # [B]
    first_rejection = rejected.to(torch.int64).argmax(dim=-1)  # [B]
    accepted_lengths = torch.where(
        has_rejection,
        first_rejection + 1,
        torch.full_like(first_rejection, num_draft_tokens + 1),
    )  # [B]

    # Rectangular KV caches require every batch row to commit the same number
    # of positions. Longer valid rows are safely truncated to the shortest one.
    commit_length = int(accepted_lengths.min().item())

    if commit_length == num_draft_tokens + 1:
        bonus_tokens = torch.multinomial(
            target_probs[:, -1], 1, generator=generator
        )  # [B, 1]
        return torch.cat((draft_tokens, bonus_tokens), dim=-1)  # [B, N+1]

    committed_tokens = draft_tokens[:, :commit_length].clone()  # [B, C]
    correction_rows = accepted_lengths.eq(commit_length)  # [B]
    correction_index = commit_length - 1
    correction_probs = torch.clamp(
        target_probs[correction_rows, correction_index]
        - draft_probs[correction_rows, correction_index],
        min=0,
    )  # [R, V]
    correction_probs /= correction_probs.sum(dim=-1, keepdim=True)
    corrected_tokens = torch.multinomial(
        correction_probs, 1, generator=generator
    ).squeeze(-1)  # [R]
    committed_tokens[correction_rows, -1] = corrected_tokens

    return committed_tokens


def _truncate_kv_caches(
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]], cache_length: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return cache views restricted to a shared logical sequence length."""
    return [
        (
            key_cache[..., :cache_length, :],
            value_cache[..., :cache_length, :],
        )
        for key_cache, value_cache in kv_caches
    ]


@torch.inference_mode()
def generate_single_sampled_speculative(
    model: torch.nn.Module,
    draft_model: torch.nn.Module,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    repetition_penalty: float = 1.0,
    num_speculative_token: int = 4,
) -> torch.Tensor:
    """input_ids = [S]"""
    if num_speculative_token <= 0:
        raise ValueError("num_speculative_token must be greater than 0")

    draft_sequence = input_ids
    draft_tokens = []
    draft_probs = []

    # Generate draft tokens
    for _ in range(num_speculative_token):
        logits = draft_model(draft_sequence)  # [S + N, V]
        next_logits = logits[-1, :]  # [V]
        next_logits = apply_repetition_penalty(
            next_logits, draft_sequence, repetition_penalty
        )  # [V]
        next_token, next_probs = sample_next_token(
            next_logits,
            temperature=temperature,
            generator=generator,
            top_k=top_k,
            top_p=top_p,
        )  # [1]; [V]
        draft_tokens.append(next_token)
        draft_probs.append(next_probs)
        draft_sequence = torch.cat((draft_sequence, next_token), dim=-1)

    draft_tokens = torch.cat(draft_tokens, dim=-1)
    draft_probs = torch.stack(draft_probs, dim=0)

    target_logits = model(draft_sequence)  # [S + N, V]
    verified_logits = target_logits[input_ids.shape[-1] - 1 :, :]  # [N + 1, V]

    penalized_target_logits = []

    for i, row_logits in enumerate(verified_logits):
        history = draft_sequence[: input_ids.shape[-1] + i]
        penalized_row = apply_repetition_penalty(
            row_logits,
            history,
            repetition_penalty,
        )
        penalized_target_logits.append(penalized_row)

    penalized_target_logits = torch.stack(penalized_target_logits)  # [N+1, V]
    target_probs = logits_to_probabilities(
        penalized_target_logits, temperature=temperature, top_k=top_k, top_p=top_p
    )

    verified_tokens = verify_speculative_tokens(
        draft_tokens,
        draft_probs,
        target_probs,
        generator=generator,
    )  # between [1] and [N + 1]

    return torch.cat((input_ids, verified_tokens), dim=-1)


@torch.inference_mode()
def generate_sampled_speculative(
    model: torch.nn.Module,
    draft_model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
    num_speculative_token: int = 4,
) -> torch.Tensor:
    """Generate one sequence through repeated uncached speculative rounds."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be greater than or equal to 0")
    if num_speculative_token <= 0:
        raise ValueError("num_speculative_token must be greater than 0")

    current_generation = input_ids
    prompt_length = input_ids.shape[-1]

    while current_generation.shape[-1] - prompt_length < max_new_tokens:
        generated_count = current_generation.shape[-1] - prompt_length
        remaining = max_new_tokens - generated_count
        draft_count = min(num_speculative_token, remaining)

        round_output = generate_single_sampled_speculative(
            model,
            draft_model,
            current_generation,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
            repetition_penalty=repetition_penalty,
            num_speculative_token=draft_count,
        )

        new_tokens = round_output[current_generation.shape[-1] :]
        new_tokens = new_tokens[:remaining]

        if eos_token_id is not None:
            eos_positions = torch.nonzero(
                new_tokens == eos_token_id,
                as_tuple=False,
            )

            if eos_positions.numel() > 0:
                first_eos_position = eos_positions[0, 0].item()
                new_tokens = new_tokens[: first_eos_position + 1]

                current_generation = torch.cat(
                    (current_generation, new_tokens),
                    dim=-1,
                )
                break

        current_generation = torch.cat((current_generation, new_tokens), dim=-1)

    return current_generation


@torch.inference_mode()
def generate_sampled_speculative_cached(
    model: torch.nn.Module,
    draft_model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
    num_speculative_token: int = 4,
) -> torch.Tensor:
    """Batched speculative decoding using each model's independent KV cache.

    Batch rows advance in lockstep so their rectangular KV-cache tensors keep
    one shared sequence length. A row with a longer accepted prefix is safely
    truncated to the shortest accepted prefix in that round.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch_size, sequence_length]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be greater than or equal to 0")
    if num_speculative_token <= 0:
        raise ValueError("num_speculative_token must be greater than 0")
    if max_new_tokens == 0:
        return input_ids

    target_prefill_logits, target_caches = model(input_ids, use_cache=True)
    draft_prefill_logits, draft_caches = draft_model(input_ids, use_cache=True)
    if target_prefill_logits.shape[-1] != draft_prefill_logits.shape[-1]:
        raise ValueError("target and draft models must use the same vocabulary")

    current_generation = input_ids
    prompt_length = input_ids.shape[-1]
    batch_size = input_ids.shape[0]
    target_next_logits = target_prefill_logits[:, -1]  # [B, V]
    draft_next_logits = draft_prefill_logits[:, -1]  # [B, V]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    while current_generation.shape[-1] - prompt_length < max_new_tokens:
        generated_count = current_generation.shape[-1] - prompt_length
        remaining = max_new_tokens - generated_count
        draft_count = min(num_speculative_token, remaining)
        round_prefix_length = current_generation.shape[-1]

        proposed_tokens = []
        proposed_probs = []
        draft_round_caches = draft_caches
        round_draft_next_logits = draft_next_logits

        for _ in range(draft_count):
            draft_history = (
                current_generation
                if not proposed_tokens
                else torch.cat((current_generation, *proposed_tokens), dim=-1)
            )  # [B, current_length + proposed_count]
            penalized_logits = apply_repetition_penalty(
                round_draft_next_logits,
                draft_history,
                repetition_penalty,
            )  # [B, V]
            next_tokens, next_probs = sample_next_token(
                penalized_logits,
                temperature=temperature,
                generator=generator,
                top_k=top_k,
                top_p=top_p,
            )  # [B, 1], [B, V]

            if eos_token_id is not None:
                next_tokens = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_tokens, eos_token_id),
                    next_tokens,
                )
                eos_probs = torch.zeros_like(next_probs)
                eos_probs[:, eos_token_id] = 1
                next_probs = torch.where(
                    finished.unsqueeze(-1), eos_probs, next_probs
                )

            proposed_tokens.append(next_tokens)
            proposed_probs.append(next_probs)
            decoded_logits, draft_round_caches = draft_model(
                next_tokens,
                kv_caches=draft_round_caches,
                use_cache=True,
            )
            round_draft_next_logits = decoded_logits[:, -1]

        draft_tokens = torch.cat(proposed_tokens, dim=-1)  # [B, N]
        draft_probs = torch.stack(proposed_probs, dim=1)  # [B, N, V]

        target_block_logits, target_round_caches = model(
            draft_tokens,
            kv_caches=target_caches,
            use_cache=True,
        )  # [B, N, V]
        verification_logits = torch.cat(
            (target_next_logits.unsqueeze(1), target_block_logits), dim=1
        )  # [B, N+1, V]

        penalized_target_logits = []
        for i in range(draft_count + 1):
            target_history = torch.cat(
                (current_generation, draft_tokens[:, :i]), dim=-1
            )
            penalized_target_logits.append(
                apply_repetition_penalty(
                    verification_logits[:, i],
                    target_history,
                    repetition_penalty,
                )
            )
        target_probs = logits_to_probabilities(
            torch.stack(penalized_target_logits, dim=1),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )  # [B, N+1, V]

        if eos_token_id is not None:
            eos_probs = torch.zeros_like(target_probs)
            eos_probs[..., eos_token_id] = 1
            target_probs = torch.where(
                finished[:, None, None], eos_probs, target_probs
            )

        committed_tokens = verify_speculative_tokens_batched(
            draft_tokens,
            draft_probs,
            target_probs,
            generator=generator,
        )
        committed_tokens = committed_tokens[:, :remaining]

        # Stop a round at the earliest newly generated EOS. This keeps every
        # row's cache rectangular while avoiding invalid tokens after EOS.
        if eos_token_id is not None:
            new_eos = committed_tokens.eq(eos_token_id) & ~finished.unsqueeze(-1)
            positions = torch.arange(
                committed_tokens.shape[-1], device=input_ids.device
            ).expand(batch_size, -1)
            sentinel = torch.full_like(positions, committed_tokens.shape[-1])
            first_eos = torch.where(new_eos, positions, sentinel).amin(dim=-1)
            earliest_eos = int(first_eos.min().item())
            if earliest_eos < committed_tokens.shape[-1]:
                committed_tokens = committed_tokens[:, : earliest_eos + 1]

            committed_tokens = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(committed_tokens, eos_token_id),
                committed_tokens,
            )
            finished = finished | committed_tokens.eq(eos_token_id).any(dim=-1)

        commit_length = committed_tokens.shape[-1]
        reusable_cache_length = round_prefix_length + commit_length - 1
        draft_reusable_caches = _truncate_kv_caches(
            draft_round_caches, reusable_cache_length
        )
        target_reusable_caches = _truncate_kv_caches(
            target_round_caches, reusable_cache_length
        )

        # Recompute the final committed position. It may be a correction token
        # rather than the draft proposal stored in the speculative caches.
        draft_last_logits, draft_caches = draft_model(
            committed_tokens[:, -1:],
            kv_caches=draft_reusable_caches,
            use_cache=True,
        )
        target_last_logits, target_caches = model(
            committed_tokens[:, -1:],
            kv_caches=target_reusable_caches,
            use_cache=True,
        )
        draft_next_logits = draft_last_logits[:, -1]
        target_next_logits = target_last_logits[:, -1]
        current_generation = torch.cat(
            (current_generation, committed_tokens), dim=-1
        )

        if eos_token_id is not None and finished.all():
            break

    return current_generation
