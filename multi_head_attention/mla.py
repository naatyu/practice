import torch
from torch import nn

from rms_norm import RMSNorm


def precompute_freqs_cis(seq_len: int, base: int, head_dim: int) -> torch.Tensor:
    dim_positions = torch.arange(0, head_dim, 2)  # [head_dim / 2]
    token_positions = torch.arange(0, seq_len)  # [seq_len]
    freqs = 1 / (base ** (dim_positions / head_dim))  # [head_dim / 2]

    freqs = torch.outer(token_positions, freqs)  # [seq_len, head_dim / 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [seq_len, head_dim / 2]

    return freqs_cis


def apply_rope(x: torch.Tensor, freq_cis: torch.Tensor):
    # This version for MLA expect [B, S, H, D] as input
    dtype = x.dtype

    x = x.float()  # Because for complex operations we need FP32
    x = x.view(*x.shape[:-1], -1, 2)  # [B, S, H, D/2, 2]
    x = torch.view_as_complex(x)  # [B, S, H, D/2]
    freq_cis = freq_cis.unsqueeze(0).unsqueeze(2)  # [1, S, 1, D / 2]
    rot_x = x * freq_cis  # [B, S, H, D/2]
    rot_x = torch.view_as_real(rot_x)  # [B, S, H, D/2, 2]
    rot_x = rot_x.flatten(3)

    return rot_x.to(dtype)


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # Using lora for Q only help for activation memory but has no effect on kv-cache
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(
                self.d_model, self.n_heads * self.qk_head_dim, bias=False
            )
        else:
            self.wq_a = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
            self.q_norm = RMSNorm(self.q_lora_rank)

        self.wkv_a = nn.Linear(
            self.d_model, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.out = nn.Linear(self.n_heads * self.v_head_dim, self.d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        # B = batch_size, S = sequence_length, D = model dimension, d = head dimension, H = number of heads
        batch_size, seq_len, _ = x.shape  # [B, S, D]

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)  # [B, S, H * d_qk]
        else:
            q = self.wq_a(x)  # [B, S, q_lora_rank]
            q = self.wq_b(self.q_norm(q))  # [B, S, H * d_qk]

        q = q.view(
            batch_size, seq_len, self.n_heads, self.qk_head_dim
        )  # [B, S, H, d_qk]
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )  # [B, S, H, d_qk_nope], [B, S, H, d_qk_rope]
        q_rope = apply_rope(q_rope, freqs_cis)  # [B, S, H, d_qk_rope]
        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, S, H, d_qk]

        # Key and Value down projection
        kv = self.wkv_a(x)  # [B, S, kv_lora_rank + d_qk_rope]

        kv, k_rope = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )  # [B, S, kv_lora_rank], [B, S, d_qk_rope]

        k_rope = k_rope.unsqueeze(2)  # [B, S, 1, d_qk_rope]
        k_rope = apply_rope(k_rope, freqs_cis)  # [B, S, 1, d_qk_rope]

        # Key and Value Up projection
        kv = self.wkv_b(self.kv_norm(kv))  # [B, S, H * (d_qk_nope + d_v)]
        kv = kv.view(
            batch_size, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
        )  # [B, S, H, (d_qk_nope + d_v)]
        k_nope, v = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )  # [B, S, H, d_qk_nope], [B, S, H, d_v]

        k_rope = k_rope.expand((-1, -1, self.n_heads, -1))  # [B, S, H, d_qk_rope]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B, S, H, d_qk]

        # Tranpose
        q = q.transpose(1, 2)  # [B, H, S, d_qk]
        k = k.transpose(1, 2)  # [B, H, S, d_qk]
        v = v.transpose(1, 2)  # [B, H, S, d_v]

        attn_out = self.inner_attention(q, k, v)  # [B, H, S, d_v]

        attn_out = attn_out.transpose(1, 2).contiguous()  # [B, S, H, d_v]
        attn_out = attn_out.view(
            batch_size, seq_len, self.n_heads * self.v_head_dim
        )  # [B, S, H * d_v]

        return self.out(attn_out)
