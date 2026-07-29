# FlashAttention interview practice: completed solutions

Completed reasoning and corrections will be recorded here as the interview
progresses. Implementation code remains in `flash-attention.py`.

## Bottleneck and tiling

Standard attention materializes the quadratic score or probability matrix in
high-bandwidth memory (HBM). Reading and writing this large intermediate causes
substantial memory traffic, and attention is often limited by that data
movement rather than arithmetic alone.

FlashAttention still computes the required pairwise scores, so its arithmetic
complexity remains quadratic. It processes query, key, and value tiles using
fast on-chip memory, fuses the attention operations, and avoids writing the
complete score and probability matrices to HBM. Only small score tiles and
per-query running statistics are needed at a time.
