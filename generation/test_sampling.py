import pytest
import torch
from torch import nn

from decoder_model import DecoderModel
from generation import generate_greedy, generate_sampled, sample_next_token
from generation.sampling import apply_repetition_penalty


class RecordingSamplingDecoder(DecoderModel):
    """Predict `(current_token + 1) % vocab_size` and record cache usage."""

    def __init__(self, vocab_size: int) -> None:
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.seen_inputs: list[torch.Tensor] = []
        self.grad_enabled: list[bool] = []
        self.incoming_cache_lengths: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[
        torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]
    ]:
        self.seen_inputs.append(input_ids.clone())
        self.grad_enabled.append(torch.is_grad_enabled())

        next_ids = (input_ids + 1) % self.vocab_size
        logits = torch.zeros(
            *input_ids.shape,
            self.vocab_size,
            device=input_ids.device,
        ).scatter_(-1, next_ids.unsqueeze(-1), 1.0)

        if not use_cache:
            assert kv_caches is None
            return logits

        if kv_caches is None:
            self.incoming_cache_lengths.append(0)
            cached_ids = input_ids.clone()
        else:
            previous_ids = kv_caches[0][0]
            self.incoming_cache_lengths.append(previous_ids.shape[-1])
            cached_ids = torch.cat((previous_ids, input_ids), dim=-1)

        return logits, [(cached_ids, cached_ids.clone())]


class FixedPenaltyDecoder(nn.Module):
    """Return fixed logits that expose which token IDs receive a penalty."""

    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = 4

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        assert use_cache
        logits = torch.zeros(
            *input_ids.shape,
            self.vocab_size,
            device=input_ids.device,
        )
        logits[..., 1] = 4.0
        logits[..., 2] = 3.0
        logits[..., 3] = 2.5

        if kv_caches is None:
            cached_ids = input_ids.clone()
        else:
            cached_ids = torch.cat((kv_caches[0][0], input_ids), dim=-1)

        return logits, [(cached_ids, cached_ids.clone())]


def test_repetition_penalty_is_sign_aware_and_batch_specific() -> None:
    logits = torch.tensor(
        [
            [4.0, -3.0, 2.0, -1.0, 0.0],
            [-5.0, 6.0, -2.0, 3.0, 1.0],
        ]
    )
    input_ids = torch.tensor(
        [
            [0, 1, 1],
            [4, 2, 2],
        ]
    )

    actual = apply_repetition_penalty(logits, input_ids, penalty=2.0)

    expected = torch.tensor(
        [
            [2.0, -6.0, 2.0, -1.0, 0.0],
            [-5.0, 6.0, -4.0, 3.0, 0.5],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_repetition_penalty_preserves_input_and_penalizes_duplicate_once() -> None:
    logits = torch.tensor([[1.0, -2.0, 6.0]])
    original = logits.clone()
    input_ids = torch.tensor([[2, 2, 2]])

    penalized = apply_repetition_penalty(logits, input_ids, penalty=3.0)

    torch.testing.assert_close(logits, original)
    torch.testing.assert_close(penalized, torch.tensor([[1.0, -2.0, 2.0]]))


@pytest.mark.parametrize("penalty", [0.0, -1.0, 0.5, float("nan")])
def test_repetition_penalty_rejects_values_below_one_or_nan(
    penalty: float,
) -> None:
    logits = torch.tensor([[1.0, 2.0]])
    input_ids = torch.tensor([[0]])

    with pytest.raises(ValueError, match="(?i)penalty"):
        apply_repetition_penalty(logits, input_ids, penalty)


def test_repetition_penalty_one_is_identity() -> None:
    logits = torch.tensor([[1.0, -2.0, 3.0]])
    input_ids = torch.tensor([[0, 2]])

    actual = apply_repetition_penalty(logits, input_ids, penalty=1.0)

    torch.testing.assert_close(actual, logits)


def test_generate_sampled_penalizes_prompt_and_growing_history() -> None:
    model = FixedPenaltyDecoder()
    input_ids = torch.tensor([[1]])

    generated = generate_sampled(
        model,
        input_ids,
        max_new_tokens=2,
        top_k=1,
        penalty=2.0,
    )

    # Token 1 is penalized from the prompt, so token 2 is selected first.
    # Then tokens 1 and 2 are penalized, so token 3 is selected next.
    torch.testing.assert_close(generated, torch.tensor([[1, 2, 3]]))


@pytest.mark.parametrize("temperature", [0.0, -0.5])
def test_sample_next_token_rejects_non_positive_temperature(
    temperature: float,
) -> None:
    logits = torch.tensor([[1.0, 2.0]])

    with pytest.raises(ValueError, match="(?i)temperature"):
        sample_next_token(logits, temperature=temperature)


def test_sample_next_token_matches_seeded_reference() -> None:
    logits = torch.tensor(
        [
            [-2.0, 0.0, 1.0],
            [3.0, -1.0, 0.0],
        ]
    )
    temperature = 0.7
    expected_generator = torch.Generator().manual_seed(42)
    actual_generator = torch.Generator().manual_seed(42)
    probabilities = torch.softmax(logits / temperature, dim=-1)

    expected = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=expected_generator,
    )
    actual = sample_next_token(
        logits,
        temperature=temperature,
        generator=actual_generator,
    )

    torch.testing.assert_close(actual, expected)


def test_sample_next_token_returns_one_long_id_per_batch_row() -> None:
    logits = torch.randn(5, 11)

    token_ids = sample_next_token(
        logits,
        generator=torch.Generator().manual_seed(0),
    )

    assert token_ids.shape == (5, 1)
    assert token_ids.dtype == torch.long
    assert token_ids.device == logits.device


def test_sample_next_token_is_reproducible_with_seeded_generators() -> None:
    logits = torch.zeros(20, 8)
    first_generator = torch.Generator().manual_seed(7)
    second_generator = torch.Generator().manual_seed(7)

    first = sample_next_token(logits, generator=first_generator)
    second = sample_next_token(logits, generator=second_generator)

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("top_k", [0, -1, 5])
def test_sample_next_token_rejects_invalid_top_k(top_k: int) -> None:
    logits = torch.randn(2, 4)

    with pytest.raises(ValueError, match="top_k"):
        sample_next_token(logits, top_k=top_k)


def test_top_k_one_is_equivalent_to_greedy_selection() -> None:
    logits = torch.tensor(
        [
            [-1.0, 3.0, 2.0],
            [4.0, 0.0, 1.0],
        ]
    )

    sampled = sample_next_token(
        logits,
        temperature=2.0,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(sampled, logits.argmax(dim=-1, keepdim=True))


def test_top_k_matches_seeded_masked_logit_reference() -> None:
    logits = torch.tensor(
        [
            [-2.0, 0.0, 4.0, 1.0],
            [3.0, -1.0, 2.0, 0.0],
        ]
    )
    temperature = 0.8
    top_k = 2
    expected_generator = torch.Generator().manual_seed(11)
    actual_generator = torch.Generator().manual_seed(11)
    scaled_logits = logits / temperature
    top_indices = torch.topk(scaled_logits, k=top_k, dim=-1).indices
    keep_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
    keep_mask.scatter_(dim=-1, index=top_indices, value=True)
    probabilities = torch.softmax(
        scaled_logits.masked_fill(~keep_mask, float("-inf")),
        dim=-1,
    )

    expected = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=expected_generator,
    )
    actual = sample_next_token(
        logits,
        temperature=temperature,
        top_k=top_k,
        generator=actual_generator,
    )

    torch.testing.assert_close(actual, expected)


def test_top_k_vocab_size_matches_unfiltered_sampling() -> None:
    logits = torch.randn(8, 5)
    unfiltered_generator = torch.Generator().manual_seed(3)
    top_k_generator = torch.Generator().manual_seed(3)

    unfiltered = sample_next_token(logits, generator=unfiltered_generator)
    top_k_all = sample_next_token(
        logits,
        top_k=logits.shape[-1],
        generator=top_k_generator,
    )

    torch.testing.assert_close(top_k_all, unfiltered)


def test_top_k_does_not_modify_input_logits() -> None:
    logits = torch.randn(3, 7)
    original = logits.clone()

    sample_next_token(
        logits,
        top_k=3,
        generator=torch.Generator().manual_seed(5),
    )

    torch.testing.assert_close(logits, original)


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1, float("nan")])
def test_sample_next_token_rejects_invalid_top_p(top_p: float) -> None:
    logits = torch.randn(2, 4)

    with pytest.raises(ValueError, match="top_p"):
        sample_next_token(logits, top_p=top_p)


def test_top_p_matches_seeded_nucleus_reference() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [0.50, 0.30, 0.15, 0.05],
                [0.05, 0.15, 0.30, 0.50],
            ]
        )
    )
    top_p = 0.7
    expected_generator = torch.Generator().manual_seed(13)
    actual_generator = torch.Generator().manual_seed(13)
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    sorted_remove_mask = sorted_probabilities.cumsum(dim=-1) > top_p
    sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
    sorted_remove_mask[..., 0] = False
    remove_mask = torch.zeros_like(logits, dtype=torch.bool)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove_mask)
    probabilities = torch.softmax(
        logits.masked_fill(remove_mask, float("-inf")),
        dim=-1,
    )

    expected = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=expected_generator,
    )
    actual = sample_next_token(
        logits,
        top_p=top_p,
        generator=actual_generator,
    )

    torch.testing.assert_close(actual, expected)


def test_very_small_top_p_keeps_highest_logit() -> None:
    logits = torch.tensor(
        [
            [1.0, 4.0, 2.0],
            [3.0, 0.0, 1.0],
        ]
    )

    sampled = sample_next_token(
        logits,
        top_p=1e-6,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(sampled, logits.argmax(dim=-1, keepdim=True))


def test_top_p_one_matches_unfiltered_sampling() -> None:
    logits = torch.randn(8, 5)
    unfiltered_generator = torch.Generator().manual_seed(17)
    top_p_generator = torch.Generator().manual_seed(17)

    unfiltered = sample_next_token(logits, generator=unfiltered_generator)
    top_p_one = sample_next_token(
        logits,
        top_p=1.0,
        generator=top_p_generator,
    )

    torch.testing.assert_close(top_p_one, unfiltered)


def test_top_p_composes_with_top_k_and_preserves_input() -> None:
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]]).repeat(50, 1)
    original = logits.clone()

    samples = sample_next_token(
        logits,
        top_k=3,
        top_p=0.8,
        generator=torch.Generator().manual_seed(23),
    )

    assert set(samples.squeeze(-1).tolist()) <= {0, 1}
    torch.testing.assert_close(logits, original)


def test_generate_sampled_top_k_one_matches_cached_greedy() -> None:
    torch.manual_seed(0)
    model = DecoderModel(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=8,
        n_layers=2,
        hidden_dim=24,
        dropout_p=0.0,
    ).eval()
    input_ids = torch.randint(0, model.vocab_size, (2, 3))

    greedy = generate_greedy(model, input_ids, max_new_tokens=3)
    sampled = generate_sampled(
        model,
        input_ids,
        max_new_tokens=3,
        temperature=2.0,
        top_k=1,
        generator=torch.Generator().manual_seed(4),
    )

    torch.testing.assert_close(sampled, greedy)


def test_generate_sampled_is_reproducible_with_seeded_generators() -> None:
    torch.manual_seed(1)
    model = DecoderModel(
        vocab_size=24,
        d_model=16,
        num_heads=4,
        max_seq_len=8,
        n_layers=1,
        hidden_dim=24,
        dropout_p=0.0,
    ).eval()
    input_ids = torch.randint(0, model.vocab_size, (3, 3))

    first = generate_sampled(
        model,
        input_ids,
        max_new_tokens=4,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        generator=torch.Generator().manual_seed(19),
    )
    second = generate_sampled(
        model,
        input_ids,
        max_new_tokens=4,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        generator=torch.Generator().manual_seed(19),
    )

    torch.testing.assert_close(first, second)


def test_generate_sampled_prefills_then_uses_single_token_decode() -> None:
    model = RecordingSamplingDecoder(vocab_size=10)
    input_ids = torch.tensor([[2, 4]])

    generated = generate_sampled(
        model,
        input_ids,
        max_new_tokens=4,
        top_k=1,
    )

    torch.testing.assert_close(generated, torch.tensor([[2, 4, 5, 6, 7, 8]]))
    assert [call.shape[-1] for call in model.seen_inputs] == [2, 1, 1, 1]
    assert model.incoming_cache_lengths == [0, 2, 3, 4]
    assert model.grad_enabled == [False, False, False, False]


def test_generate_sampled_handles_staggered_eos() -> None:
    model = RecordingSamplingDecoder(vocab_size=10)
    input_ids = torch.tensor([[1, 4], [1, 2]])

    generated = generate_sampled(
        model,
        input_ids,
        max_new_tokens=6,
        top_k=1,
        eos_token_id=5,
    )

    torch.testing.assert_close(
        generated,
        torch.tensor(
            [
                [1, 4, 5, 5, 5],
                [1, 2, 3, 4, 5],
            ]
        ),
    )
    assert [call.shape[-1] for call in model.seen_inputs] == [2, 1, 1]


def test_generate_sampled_zero_new_tokens_skips_model_and_rng() -> None:
    model = RecordingSamplingDecoder(vocab_size=10)
    input_ids = torch.tensor([[2, 4]])
    generator = torch.Generator().manual_seed(29)
    initial_state = generator.get_state().clone()

    generated = generate_sampled(
        model,
        input_ids,
        max_new_tokens=0,
        generator=generator,
    )

    torch.testing.assert_close(generated, input_ids)
    torch.testing.assert_close(generator.get_state(), initial_state)
    assert model.seen_inputs == []
