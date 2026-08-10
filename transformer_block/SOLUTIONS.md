# Transformer block interview practice: completed solutions

Implementation code remains in its Python module.

## Block structure

A decoder Transformer block has two main sublayers: causal self-attention and a
position-wise feed-forward network. Attention mixes information across token
positions; SwiGLU mixes and gates features independently within each token.

This exercise uses pre-normalization and gives each sublayer its own residual:

```text
x_attn = x + attention(norm1(x), causal=True)
output = x_attn + swiglu(norm2(x_attn))
```

There is not one residual connection around attention and the MLP together.
The second norm and SwiGLU must receive `x_attn`, not the original `x`, so
the feed-forward branch sees the newly added contextual information.

## Independent normalization layers

The block uses two separate `RMSNorm` instances. Reusing one would tie its
learned scale across two locations whose inputs can develop different feature
statistics. Token and positional representations are formed before the stack;
they are not recreated inside every block.

## Causality and testing

Multi-head attention defaults to non-causal attention, so the decoder block
must explicitly call it with `causal=True`. Shape-only tests cannot detect a
missing causal flag.

An end-to-end causality test constructs two inputs identical through position
`t` and different afterward. With dropout disabled, both outputs must be equal
through and including `t`; later positions may differ.

The block constructor needs `d_model`, `num_heads`, the explicit SwiGLU
`hidden_dim`, and optional dropout and RMSNorm epsilon values. A causal
argument is unnecessary because causality is part of this decoder-only block's
contract.
