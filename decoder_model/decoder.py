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
        num_kv_heads: int | None = None,
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
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
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
                    num_kv_heads=self.num_kv_heads,
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
        self,
        input_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if use_cache and kv_caches is not None and len(kv_caches) != len(self.blocks):
            raise ValueError(
                "KV cache number of layers does not match the model number of layers."
            )
        updated_caches = []
        # Tokens embeddings, [B, S] -> [B, S, d_model]
        x = self.tok_emb(input_ids)

        # Loop through transformer blocks
        for i, block in enumerate(self.blocks):
            if use_cache and kv_caches is not None:
                x, updated_cache = block(x, kv_caches[i], use_cache=True)
                updated_caches.append(updated_cache)
            elif use_cache and kv_caches is None:
                x, updated_cache = block(x, use_cache=True)
                updated_caches.append(updated_cache)
            else:
                x = block(x)

        # Final norm
        x = self.final_norm(x)

        # Output logits
        if use_cache:
            return self.lm_head(x), updated_caches
        return self.lm_head(x)
