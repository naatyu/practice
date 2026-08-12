import torch
from torch import nn

from positional_encoding import RotaryPositionalEncoding
from rms_norm import RMSNorm
from transformer_block import TransformerBlock


class DecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        n_layers: int,
        hidden_dim: int,
        dropout_p: float = 0.0,
        rope_base: float = 10_000,
        *,
        tie_weights: bool = True,
    ):
        super().__init__()

        # Args
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.rope_base = rope_base
        self.dropout_p = dropout_p
        self.d_head = self.d_model // self.num_heads

        # Layers
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEncoding(
            d_head=self.d_head, max_seq_len=self.max_seq_len, base=self.rope_base
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    hidden_dim=self.hidden_dim,
                    dropout_p=self.dropout_p,
                    rope=self.rope,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model=self.d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = (
                self.tok_emb.weight
            )  # weight matrix is already transposed when stored for a linear layer

    def forward(
        self, input_ids: torch.Tensor, position_offset: int = 0
    ) -> torch.Tensor:
        # Tokens embeddings, [B, S] -> [B, S, d_model]
        x = self.tok_emb(input_ids)

        # Loop through transformer blocks
        for block in self.blocks:
            x = block(x, position_offset)

        # Final norm
        x = self.final_norm(x)

        # Output logits
        return self.lm_head(x)
