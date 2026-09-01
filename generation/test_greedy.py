import torch
from torch import nn

from decoder_model import DecoderModel
from generation import generate_greedy, generate_greedy_naive


class IncrementingDecoder(DecoderModel):
    """Predict token `(current_token + 1) % vocab_size` at every position."""

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
        )
        logits = logits.scatter_(-1, next_ids.unsqueeze(-1), 1.0)

        if not use_cache:
            assert kv_caches is None
            return logits

        if kv_caches is None:
            self.incoming_cache_lengths.append(0)
            cached_ids = input_ids.clone()
        else:
            assert len(kv_caches) == 1
            previous_ids = kv_caches[0][0]
            self.incoming_cache_lengths.append(previous_ids.shape[-1])
            cached_ids = torch.cat((previous_ids, input_ids), dim=-1)

        return logits, [(cached_ids, cached_ids.clone())]


def test_generate_greedy_naive_returns_prompt_and_new_tokens() -> None:
    model = IncrementingDecoder(vocab_size=6)
    input_ids = torch.tensor([[1, 4], [2, 0]])

    generated = generate_greedy_naive(model, input_ids, max_new_tokens=3)

    torch.testing.assert_close(
        generated,
        torch.tensor(
            [
                [1, 4, 5, 0, 1],
                [2, 0, 1, 2, 3],
            ]
        ),
    )


def test_generate_greedy_naive_forwards_complete_growing_sequence() -> None:
    model = IncrementingDecoder(vocab_size=10)
    input_ids = torch.tensor([[2, 4]])

    generate_greedy_naive(model, input_ids, max_new_tokens=3)

    assert len(model.seen_inputs) == 3
    torch.testing.assert_close(model.seen_inputs[0], torch.tensor([[2, 4]]))
    torch.testing.assert_close(model.seen_inputs[1], torch.tensor([[2, 4, 5]]))
    torch.testing.assert_close(model.seen_inputs[2], torch.tensor([[2, 4, 5, 6]]))


def test_generate_greedy_naive_preserves_input() -> None:
    model = IncrementingDecoder(vocab_size=10)
    input_ids = torch.tensor([[2, 4]])
    original = input_ids.clone()

    generate_greedy_naive(model, input_ids, max_new_tokens=2)

    torch.testing.assert_close(input_ids, original)


def test_generate_greedy_naive_disables_gradient_tracking() -> None:
    model = IncrementingDecoder(vocab_size=10)
    input_ids = torch.tensor([[2, 4]])

    generate_greedy_naive(model, input_ids, max_new_tokens=2)

    assert model.grad_enabled == [False, False]


def test_generate_greedy_naive_with_zero_new_tokens_skips_model() -> None:
    model = IncrementingDecoder(vocab_size=10)
    input_ids = torch.tensor([[2, 4]])

    generated = generate_greedy_naive(model, input_ids, max_new_tokens=0)

    torch.testing.assert_close(generated, input_ids)
    assert model.seen_inputs == []


def test_generate_greedy_matches_naive_output() -> None:
    input_ids = torch.tensor([[1, 4], [2, 0]])

    naive = generate_greedy_naive(
        IncrementingDecoder(vocab_size=6),
        input_ids,
        max_new_tokens=3,
    )
    cached = generate_greedy(
        IncrementingDecoder(vocab_size=6),
        input_ids,
        max_new_tokens=3,
    )

    torch.testing.assert_close(cached, naive)


def test_generate_greedy_prefills_once_then_decodes_one_token_at_a_time() -> None:
    model = IncrementingDecoder(vocab_size=10)
    input_ids = torch.tensor([[2, 4]])

    generate_greedy(model, input_ids, max_new_tokens=4)

    assert [call.shape[-1] for call in model.seen_inputs] == [2, 1, 1, 1]
    assert model.incoming_cache_lengths == [0, 2, 3, 4]
    torch.testing.assert_close(model.seen_inputs[0], torch.tensor([[2, 4]]))
    torch.testing.assert_close(model.seen_inputs[1], torch.tensor([[5]]))
    torch.testing.assert_close(model.seen_inputs[2], torch.tensor([[6]]))
    torch.testing.assert_close(model.seen_inputs[3], torch.tensor([[7]]))


def test_generate_greedy_with_one_new_token_only_prefills() -> None:
    model = IncrementingDecoder(vocab_size=10)
    input_ids = torch.tensor([[2, 4]])

    generated = generate_greedy(model, input_ids, max_new_tokens=1)

    torch.testing.assert_close(generated, torch.tensor([[2, 4, 5]]))
    assert [call.shape[-1] for call in model.seen_inputs] == [2]


def test_generate_greedy_with_zero_new_tokens_skips_model() -> None:
    model = IncrementingDecoder(vocab_size=10)
    input_ids = torch.tensor([[2, 4]])

    generated = generate_greedy(model, input_ids, max_new_tokens=0)

    torch.testing.assert_close(generated, input_ids)
    assert model.seen_inputs == []


def test_cached_greedy_matches_naive_with_real_decoder() -> None:
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

    naive = generate_greedy_naive(model, input_ids, max_new_tokens=3)
    cached = generate_greedy(model, input_ids, max_new_tokens=3)

    torch.testing.assert_close(cached, naive)


def test_greedy_eos_keeps_finished_rows_rectangular_until_all_finish() -> None:
    input_ids = torch.tensor([[1, 4], [1, 2]])
    expected = torch.tensor(
        [
            [1, 4, 5, 5, 5],
            [1, 2, 3, 4, 5],
        ]
    )
    naive_model = IncrementingDecoder(vocab_size=10)
    cached_model = IncrementingDecoder(vocab_size=10)

    naive = generate_greedy_naive(
        naive_model,
        input_ids,
        max_new_tokens=6,
        eos_token_id=5,
    )
    cached = generate_greedy(
        cached_model,
        input_ids,
        max_new_tokens=6,
        eos_token_id=5,
    )

    torch.testing.assert_close(naive, expected)
    torch.testing.assert_close(cached, expected)
    assert len(naive_model.seen_inputs) == 3
    assert [call.shape[-1] for call in cached_model.seen_inputs] == [2, 1, 1]


def test_greedy_eos_stops_after_prefill_when_every_row_finishes() -> None:
    input_ids = torch.tensor([[1, 4], [2, 4]])
    expected = torch.tensor([[1, 4, 5], [2, 4, 5]])
    naive_model = IncrementingDecoder(vocab_size=10)
    cached_model = IncrementingDecoder(vocab_size=10)

    naive = generate_greedy_naive(
        naive_model,
        input_ids,
        max_new_tokens=4,
        eos_token_id=5,
    )
    cached = generate_greedy(
        cached_model,
        input_ids,
        max_new_tokens=4,
        eos_token_id=5,
    )

    torch.testing.assert_close(naive, expected)
    torch.testing.assert_close(cached, expected)
    assert len(naive_model.seen_inputs) == 1
    assert len(cached_model.seen_inputs) == 1


def test_eos_in_prompt_does_not_mark_sequence_finished() -> None:
    input_ids = torch.tensor([[1, 5]])
    expected = torch.tensor([[1, 5, 6, 7]])

    naive = generate_greedy_naive(
        IncrementingDecoder(vocab_size=10),
        input_ids,
        max_new_tokens=2,
        eos_token_id=5,
    )
    cached = generate_greedy(
        IncrementingDecoder(vocab_size=10),
        input_ids,
        max_new_tokens=2,
        eos_token_id=5,
    )

    torch.testing.assert_close(naive, expected)
    torch.testing.assert_close(cached, expected)


def test_eos_token_id_zero_still_stops_early() -> None:
    input_ids = torch.tensor([[1, 3], [1, 2]])
    expected = torch.tensor([[1, 3, 0, 0], [1, 2, 3, 0]])

    naive = generate_greedy_naive(
        IncrementingDecoder(vocab_size=4),
        input_ids,
        max_new_tokens=5,
        eos_token_id=0,
    )
    cached = generate_greedy(
        IncrementingDecoder(vocab_size=4),
        input_ids,
        max_new_tokens=5,
        eos_token_id=0,
    )

    torch.testing.assert_close(naive, expected)
    torch.testing.assert_close(cached, expected)
