"""AI-authored two-rank forward/backward checks against independent dense math."""

import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from moe.expert_parallel_backward import expert_parallel_forward
from moe.experts import GroupedExperts


def _inputs(rank: int, case: str):
    generator = torch.Generator().manual_seed(100 + rank)
    tokens = torch.randn(3, 16, generator=generator)
    upstream = torch.randn(3, 16, generator=generator)
    if case == "mixed":
        ids = [[3, 0], [2, 1], [3, 2]] if rank == 0 else [[0, 1], [0, 2], [1, 3]]
    elif case == "local_only":
        ids = [[0, 1], [1, 0], [0, 1]] if rank == 0 else [[2, 3], [3, 2], [2, 3]]
    elif case == "empty_receiver":
        ids = [[0, 1], [1, 0], [0, 1]]  # Rank 1 receives no expert assignments.
    else:
        assert case == "empty_source"
        ids = [[3, 0], [2, 1], [3, 2]]
        if rank == 1:
            tokens, upstream, ids = tokens[:0], upstream[:0], []
    return (
        tokens.requires_grad_(),
        torch.tensor(ids, dtype=torch.int64).reshape(-1, 2),
        upstream,
    )


def _router():
    torch.manual_seed(123)
    return torch.nn.Linear(16, 4, bias=False)


def _experts(rank: int):
    torch.manual_seed(42 + rank)
    return GroupedExperts(16, 32, 2)


def _check_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        for case in ("mixed", "local_only", "empty_receiver", "empty_source"):
            tokens, ids, upstream = _inputs(rank, case)
            router, experts = _router(), _experts(rank)
            weights = router(tokens).gather(1, ids).softmax(-1)
            weights.retain_grad()
            actual = expert_parallel_forward(tokens, ids, weights, experts)
            (actual * upstream).sum().backward()
            assert router.weight.grad is not None
            dist.all_reduce(router.weight.grad)  # Global SUM, not a mean.

            # No collectives/grouped MM in the reference. One global autograd graph
            # contains both source losses and identical copies of each owner's weights.
            ref_router = _router()
            ref_experts = [_experts(owner) for owner in range(2)]
            ref_inputs, ref_weights, ref_outputs, losses = [], [], [], []
            for source in range(2):
                x, selected_ids, g = _inputs(source, case)
                mixing = ref_router(x).gather(1, selected_ids).softmax(-1)
                mixing.retain_grad()
                rows = []
                for token, token_ids, token_weights in zip(
                    x, selected_ids.tolist(), mixing, strict=True
                ):
                    contributions = []
                    for expert_id, weight in zip(token_ids, token_weights, strict=True):
                        owner, local_id = divmod(expert_id, 2)
                        selected = ref_experts[owner]
                        value, gate = (token @ selected.up_weight[local_id]).chunk(2)
                        y = (
                            value * torch.nn.functional.silu(gate)
                        ) @ selected.down_weight[local_id]
                        contributions.append(weight * y)
                    rows.append(torch.stack(contributions).sum(0))
                # Preserve empty-source input/router paths in the reference graph.
                out = torch.stack(rows) if rows else x + mixing.sum() * 0
                ref_inputs.append(x)
                ref_weights.append(mixing)
                ref_outputs.append(out)
                losses.append((out * g).sum())
            sum(losses).backward()

            torch.testing.assert_close(actual, ref_outputs[rank], rtol=1e-4, atol=1e-5)
            for observed, expected in (
                (tokens.grad, ref_inputs[rank].grad),
                (weights.grad, ref_weights[rank].grad),
                (router.weight.grad, ref_router.weight.grad),
                (experts.up_weight.grad, ref_experts[rank].up_weight.grad),
                (experts.down_weight.grad, ref_experts[rank].down_weight.grad),
            ):
                assert observed is not None
                # An entirely unused reference owner has no graph edge; EP may
                # produce an explicit zero gradient through its empty grouped MM.
                if expected is None:
                    expected = torch.zeros_like(observed)
                torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_two_rank_outputs_and_gradients_match_reference(tmp_path):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    # Parent-enforced timeout also covers failures before the process group exists.
    # Kill both children on timeout/failure, so pytest cannot leave a hung peer.
    processes = []
    logs = []
    try:
        for rank in range(2):
            log = (tmp_path / f"rank_{rank}.log").open("w+")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "moe.tests.test_expert_parallel_backward",
                        str(rank),
                        rendezvous,
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        for rank, process in enumerate(processes):
            try:
                code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pytest.fail(f"EP rank {rank} exceeded the 60-second timeout")
            if code:
                for log in logs:
                    log.seek(0)
                pytest.fail("\n".join(log.read() for log in logs))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.wait(timeout=5)
        for log in logs:
            log.close()


if __name__ == "__main__":
    _check_worker(int(sys.argv[1]), sys.argv[2])
