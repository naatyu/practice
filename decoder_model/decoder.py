from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from normalization import RMSNorm
from positional_encoding import RotaryPositionalEncoding
from transformer_block import TransformerBlock

from .cache import DecoderCache


@dataclass
class DecoderOutput:
    """Logits [B, S, V], optional KV cache, and mean per-MoE-layer router losses."""

    logits: torch.Tensor
    kv_caches: DecoderCache | None = None
    aux_loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None


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
        window_size: int | None = None,
        num_experts: int | None = None,
        top_k: int | None = None,
        *,
        tie_weights: bool = True,
        capacity_factor: float | None = None,
        drop_policy: Literal["position", "probs"] = "probs",
        shared_hidden_dim: int | None = None,
        shared_expert_gate: bool = False,
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
        self.window_size = window_size
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
                    window_size=self.window_size,
                    num_experts=num_experts,
                    top_k=top_k,
                    capacity_factor=capacity_factor,
                    drop_policy=drop_policy,
                    shared_hidden_dim=shared_hidden_dim,
                    shared_expert_gate=shared_expert_gate,
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
        kv_caches: DecoderCache | None = None,
        *,
        use_cache: bool = False,
    ) -> DecoderOutput:
        """Compute decoder logits for a token chunk.

        Inputs: input_ids [B, S]; optional cache with per-layer K/V
        [B, num_kv_heads, K, d_head] and an absolute next-token position.
        Returns DecoderOutput; router losses are None for dense models or eval.
        """
        if use_cache and kv_caches is not None and len(kv_caches) != len(self.blocks):
            raise ValueError(
                "KV cache number of layers does not match the model number of layers."
            )
        position_offset = kv_caches.position if kv_caches is not None else 0
        updated_caches = []
        aux_losses = []
        z_losses = []
        # Tokens embeddings, [B, S] -> [B, S, d_model]
        x = self.tok_emb(input_ids)

        # Loop through transformer blocks
        for i, block in enumerate(self.blocks):
            x, updated_cache, aux_loss, z_loss = block(
                x,
                kv_caches[i] if use_cache and kv_caches is not None else None,
                use_cache=use_cache,
                position_offset=position_offset,
            )
            if updated_cache is not None:
                updated_caches.append(updated_cache)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
            if z_loss is not None:
                z_losses.append(z_loss)

        # Final norm
        x = self.final_norm(x)

        # Output logits
        cache = (
            DecoderCache(
                layers=updated_caches,
                position=position_offset + input_ids.shape[-1],
            )
            if use_cache
            else None
        )
        # Balance each layer independently, then average scalar losses (not counts).
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        z_loss = torch.stack(z_losses).mean() if z_losses else None
        return DecoderOutput(
            logits=self.lm_head(x), kv_caches=cache, aux_loss=aux_loss, z_loss=z_loss
        )
