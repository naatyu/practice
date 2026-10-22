"""CPU MoE comparison: run python -m moe.benchmark_moe --threads 1."""

import argparse
import platform

import torch
from torch import nn
from torch.utils.benchmark import Timer

from mlp import SwiGLU
from moe.expert_batched_moe import ExpertBatchedMoe
from moe.grouped_moe import GroupedMoE
from moe.naive_moe import NaiveMoe


def make_models(d_model: int, hidden_dim: int, num_experts: int, top_k: int):
    reference = ExpertBatchedMoe(d_model, hidden_dim, num_experts, top_k)
    naive = NaiveMoe(d_model, hidden_dim, num_experts, top_k)
    naive.load_state_dict(reference.state_dict())
    packed = GroupedMoE(d_model, hidden_dim, num_experts, top_k, expert_backend="loop")
    grouped = GroupedMoE(
        d_model, hidden_dim, num_experts, top_k, expert_backend="grouped_mm"
    )
    packed.router.load_state_dict(reference.router.state_dict())
    with torch.no_grad():
        for expert_id, expert in enumerate(reference.experts):
            assert isinstance(expert, SwiGLU)
            packed.experts.up_weight[expert_id].copy_(expert.up_proj.weight.T)
            packed.experts.down_weight[expert_id].copy_(expert.down_proj.weight.T)
    grouped.load_state_dict(packed.state_dict())
    return {
        "naive": naive,
        "expert_batched": reference,
        "packed_loop": packed,
        "grouped_mm": grouped,
    }


def measure(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    backward: bool,
    threads: int,
    min_run_time: float,
) -> tuple[float, float]:
    model.train(backward)
    x = tokens.detach().clone().requires_grad_(backward)

    @torch.inference_mode()
    def forward():
        return model(x)

    def forward_backward():
        model.zero_grad(set_to_none=True)
        x.grad = None
        output, _, _ = model(x)
        output.square().mean().backward()

    run = forward_backward if backward else forward
    run()  # Warm up before Timer's own calibration.
    timing = Timer(
        stmt="run()", globals={"run": run}, num_threads=threads
    ).blocked_autorange(min_run_time=min_run_time)
    return timing.median * 1e3, timing.iqr / timing.median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=0.2)
    args = parser.parse_args()
    if args.threads < 1 or args.min_run_time <= 0:
        parser.error("threads and min-run-time must be positive")
    torch.set_num_threads(args.threads)

    print(
        f"PyTorch {torch.__version__}; {platform.platform()}; "
        f"CPU float32; {args.threads} thread(s); seed 42",
        flush=True,
    )
    print(
        "End-to-end medians in ms; routing, dispatch, weighting, and combine included."
    )
    print("Backward includes forward, squared-mean loss, gradient reset, and backward.")
    print(
        "| T | D | F | E | K | Mode | Naive | Expert-batched | Packed loop | Grouped MM |"
    )
    print("|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|")
    largest_relative_iqr = 0.0
    cases = [(1, 64, 128), (32, 64, 128), (256, 128, 256)]
    for token_count, d_model, hidden_dim in cases:
        torch.manual_seed(42)
        models = make_models(d_model, hidden_dim, num_experts=8, top_k=2)
        tokens = torch.randn(1, token_count, d_model)
        with torch.inference_mode():
            expected = models["expert_batched"](tokens)
            for model in models.values():
                torch.testing.assert_close(model(tokens), expected)

        for backward in (False, True):
            medians = []
            for model in models.values():
                median, relative_iqr = measure(
                    model,
                    tokens,
                    backward=backward,
                    threads=args.threads,
                    min_run_time=args.min_run_time,
                )
                medians.append(f"{median:.3f}")
                largest_relative_iqr = max(largest_relative_iqr, relative_iqr)
            mode = "forward + backward" if backward else "inference"
            print(
                f"| {token_count} | {d_model} | {hidden_dim} | 8 | 2 | {mode} | "
                + " | ".join(medians)
                + " |",
                flush=True,
            )
    print(f"Largest relative IQR: {largest_relative_iqr:.1%} (quick-run timing noise).")
    print("CPU results do not measure an optimized CUDA grouped-GEMM kernel.")


if __name__ == "__main__":
    main()
