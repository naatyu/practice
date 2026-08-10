# Deep learning implementation practice

Interview-style PyTorch exercises for implementing core Transformer operations
from first principles. Each exercise keeps the prompt and completed discussion
separate:

- `README.md`: exercise requirements and interview questions
- `SOLUTIONS.md`: reasoning, corrections, and testing lessons
- Python modules: the implementations themselves

## Exercises

1. `attention/`: scaled dot-product attention, causal masking, and dropout
2. `rms_norm/`: RMSNorm as a trainable PyTorch module
3. `multi_head_attention/`: multi-head self-attention
4. `mlp/`: a GELU Transformer MLP and fused-projection SwiGLU
5. `transformer_block/`: a pre-norm causal decoder block
6. `flash-attention/`: online-softmax and tiled-attention reasoning in progress

Run the complete test suite from the repository root with:

```bash
uv run pytest
```
