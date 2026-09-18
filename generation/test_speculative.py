import torch
from torch import nn

from decoder_model import DecoderModel
from generation import generate_sampled
from generation.speculative import (
    generate_sampled_speculative,
    generate_sampled_speculative_cached,
    generate_single_sampled_speculative,
    verify_speculative_tokens,
)


class OffsetDecoder(nn.Module):
    """Predict the token at a fixed offset from each input token."""

    def __init__(self, vocab_size: int, offset: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.offset = offset
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.seen_inputs.append(input_ids.clone())
        next_ids = (input_ids + self.offset) % self.vocab_size
        return torch.zeros(
            *input_ids.shape,
            self.vocab_size,
            device=input_ids.device,
        ).scatter_(-1, next_ids.unsqueeze(-1), 1.0)


class CachedBatchOffsetDecoder(nn.Module):
    """Cached decoder with a configurable deterministic offset per batch row."""

    def __init__(self, vocab_size: int, offsets: list[int]) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.register_buffer("offsets", torch.tensor(offsets))
        self.seen_input_lengths: list[int] = []
        self.incoming_cache_lengths: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        assert use_cache
        self.seen_input_lengths.append(input_ids.shape[-1])
        next_ids = (input_ids + self.offsets[:, None]) % self.vocab_size
        logits = torch.zeros(
            *input_ids.shape,
            self.vocab_size,
            device=input_ids.device,
        ).scatter_(-1, next_ids.unsqueeze(-1), 1.0)

        if kv_caches is None:
            self.incoming_cache_lengths.append(0)
            cached_ids = input_ids
        else:
            cached_tensor = kv_caches[0][0]
            self.incoming_cache_lengths.append(cached_tensor.shape[-2])
            previous_ids = cached_tensor[:, 0, :, 0]
            cached_ids = torch.cat((previous_ids, input_ids), dim=-1)

        cache = cached_ids[:, None, :, None]
        return logits, [(cache, cache.clone())]


def test_single_speculative_round_accepts_all_proposals_and_bonus() -> None:
    target_model = OffsetDecoder(vocab_size=8, offset=1)
    draft_model = OffsetDecoder(vocab_size=8, offset=1)
    input_ids = torch.tensor([0, 1])

    generated = generate_single_sampled_speculative(
        target_model,
        draft_model,
        input_ids,
        num_speculative_token=2,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(generated, torch.tensor([0, 1, 2, 3, 4]))
    assert len(draft_model.seen_inputs) == 2
    torch.testing.assert_close(draft_model.seen_inputs[0], torch.tensor([0, 1]))
    torch.testing.assert_close(draft_model.seen_inputs[1], torch.tensor([0, 1, 2]))
    assert len(target_model.seen_inputs) == 1
    torch.testing.assert_close(target_model.seen_inputs[0], torch.tensor([0, 1, 2, 3]))


def test_single_speculative_round_replaces_first_rejected_proposal() -> None:
    target_model = OffsetDecoder(vocab_size=8, offset=2)
    draft_model = OffsetDecoder(vocab_size=8, offset=1)
    input_ids = torch.tensor([0, 1])

    generated = generate_single_sampled_speculative(
        target_model,
        draft_model,
        input_ids,
        num_speculative_token=2,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )

    # The draft proposes token 2, but the target assigns all probability to token 3.
    torch.testing.assert_close(generated, torch.tensor([0, 1, 3]))


def test_speculative_generation_repeats_rounds_and_respects_length() -> None:
    target_model = OffsetDecoder(vocab_size=16, offset=1)
    draft_model = OffsetDecoder(vocab_size=16, offset=1)
    input_ids = torch.tensor([0, 1])

    generated = generate_sampled_speculative(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=7,
        num_speculative_token=2,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(generated, torch.arange(9))
    assert len(target_model.seen_inputs) == 3
    assert len(draft_model.seen_inputs) == 5


def test_speculative_generation_with_zero_new_tokens_skips_models() -> None:
    target_model = OffsetDecoder(vocab_size=8, offset=1)
    draft_model = OffsetDecoder(vocab_size=8, offset=1)
    input_ids = torch.tensor([0, 1])

    generated = generate_sampled_speculative(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=0,
    )

    torch.testing.assert_close(generated, input_ids)
    assert target_model.seen_inputs == []
    assert draft_model.seen_inputs == []


def test_speculative_generation_stops_at_first_eos_in_verified_block() -> None:
    target_model = OffsetDecoder(vocab_size=16, offset=1)
    draft_model = OffsetDecoder(vocab_size=16, offset=1)

    generated = generate_sampled_speculative(
        target_model,
        draft_model,
        torch.tensor([0, 1]),
        max_new_tokens=10,
        eos_token_id=3,
        num_speculative_token=4,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(generated, torch.tensor([0, 1, 2, 3]))


def test_speculative_generation_ignores_eos_beyond_length_limit() -> None:
    target_model = OffsetDecoder(vocab_size=16, offset=1)
    draft_model = OffsetDecoder(vocab_size=16, offset=1)

    generated = generate_sampled_speculative(
        target_model,
        draft_model,
        torch.tensor([0, 1]),
        max_new_tokens=1,
        eos_token_id=3,
        num_speculative_token=4,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )

    # The bonus token is EOS, but it lies beyond the requested one-token output.
    torch.testing.assert_close(generated, torch.tensor([0, 1, 2]))


def test_cached_batched_speculative_generation_accepts_multiple_rounds() -> None:
    target_model = CachedBatchOffsetDecoder(vocab_size=32, offsets=[1, 1])
    draft_model = CachedBatchOffsetDecoder(vocab_size=32, offsets=[1, 1])
    input_ids = torch.tensor([[0, 1], [4, 5]])

    generated = generate_sampled_speculative_cached(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=7,
        num_speculative_token=2,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )

    expected = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [4, 5, 6, 7, 8, 9, 10, 11, 12],
        ]
    )
    torch.testing.assert_close(generated, expected)
    assert max(target_model.seen_input_lengths[1:]) <= 2
    assert max(draft_model.seen_input_lengths[1:]) == 1


def test_cached_batched_speculative_generation_synchronizes_rejections() -> None:
    target_model = CachedBatchOffsetDecoder(vocab_size=32, offsets=[1, 2])
    draft_model = CachedBatchOffsetDecoder(vocab_size=32, offsets=[1, 1])
    input_ids = torch.tensor([[0, 1], [4, 5]])

    generated = generate_sampled_speculative_cached(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=3,
        num_speculative_token=2,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )

    # Row 0 could accept longer blocks, but advances one position at a time
    # because row 1 rejects its first proposal in every lockstep round.
    expected = torch.tensor([[0, 1, 2, 3, 4], [4, 5, 7, 9, 11]])
    torch.testing.assert_close(generated, expected)


def test_cached_batched_speculative_generation_tracks_finished_rows() -> None:
    target_model = CachedBatchOffsetDecoder(vocab_size=32, offsets=[1, 1])
    draft_model = CachedBatchOffsetDecoder(vocab_size=32, offsets=[1, 1])
    input_ids = torch.tensor([[0, 1], [4, 5]])

    generated = generate_sampled_speculative_cached(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=4,
        eos_token_id=3,
        num_speculative_token=3,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )

    expected = torch.tensor([[0, 1, 2, 3, 3, 3], [4, 5, 6, 7, 8, 9]])
    torch.testing.assert_close(generated, expected)


def test_cached_batched_speculative_generation_works_with_real_decoder() -> None:
    torch.manual_seed(0)
    model = DecoderModel(
        vocab_size=17,
        d_model=16,
        num_heads=4,
        max_seq_len=16,
        n_layers=2,
        hidden_dim=32,
        dropout_p=0.0,
    ).eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    expected = generate_sampled(
        model,
        input_ids,
        max_new_tokens=4,
        top_k=1,
        generator=torch.Generator().manual_seed(4),
    )
    actual = generate_sampled_speculative_cached(
        model,
        model,
        input_ids,
        max_new_tokens=4,
        num_speculative_token=2,
        top_k=1,
        generator=torch.Generator().manual_seed(4),
    )

    torch.testing.assert_close(actual, expected)


def test_verify_speculative_tokens_returns_bonus_when_all_are_accepted() -> None:
    draft_tokens = torch.tensor([0, 1])
    draft_probs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    target_probs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    verified = verify_speculative_tokens(
        draft_tokens,
        draft_probs,
        target_probs,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(verified, torch.tensor([0, 1, 2]))


def test_verify_speculative_tokens_stops_at_first_rejection() -> None:
    draft_tokens = torch.tensor([0, 1, 2])
    draft_probs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target_probs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    verified = verify_speculative_tokens(
        draft_tokens,
        draft_probs,
        target_probs,
        generator=torch.Generator().manual_seed(1),
    )

    # Proposal 0 is accepted. Proposal 1 is rejected and replaced by token 2;
    # proposal 2 and the bonus distribution must not affect the result.
    torch.testing.assert_close(verified, torch.tensor([0, 2]))


def test_verify_speculative_tokens_preserves_probability_inputs() -> None:
    draft_tokens = torch.tensor([0])
    draft_probs = torch.tensor([[0.8, 0.2]])
    target_probs = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
    original_draft_probs = draft_probs.clone()
    original_target_probs = target_probs.clone()

    verify_speculative_tokens(
        draft_tokens,
        draft_probs,
        target_probs,
        generator=torch.Generator().manual_seed(4),
    )

    torch.testing.assert_close(draft_probs, original_draft_probs)
    torch.testing.assert_close(target_probs, original_target_probs)


def test_verify_speculative_tokens_is_reproducible() -> None:
    draft_tokens = torch.tensor([0, 1])
    draft_probs = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    target_probs = torch.tensor([[0.4, 0.6], [0.8, 0.2], [0.3, 0.7]])

    first = verify_speculative_tokens(
        draft_tokens,
        draft_probs,
        target_probs,
        generator=torch.Generator().manual_seed(9),
    )
    second = verify_speculative_tokens(
        draft_tokens,
        draft_probs,
        target_probs,
        generator=torch.Generator().manual_seed(9),
    )

    torch.testing.assert_close(first, second)


def test_speculative_verification_preserves_target_distribution() -> None:
    draft_distribution = torch.tensor([0.8, 0.2])
    target_distribution = torch.tensor([0.5, 0.5])
    draft_probs = draft_distribution.unsqueeze(0)
    target_probs = torch.stack((target_distribution, target_distribution))
    generator = torch.Generator().manual_seed(17)
    sample_count = 10_000
    counts = torch.zeros(2)

    for _ in range(sample_count):
        proposal = torch.multinomial(
            draft_distribution,
            num_samples=1,
            generator=generator,
        )
        verified = verify_speculative_tokens(
            proposal,
            draft_probs,
            target_probs,
            generator=generator,
        )
        counts[verified[0]] += 1

    empirical_distribution = counts / sample_count
    torch.testing.assert_close(
        empirical_distribution,
        target_distribution,
        atol=0.02,
        rtol=0.0,
    )
