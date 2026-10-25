"""AI-authored two-rank histogram pooling and empty-rank lifecycle checks."""

import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from moe.router import TopKRouter


def _router(num_bins=32):
    router = TopKRouter(4, 4, 2, use_quantile_balancing=True, num_bins=num_bins)
    router.expert_bias.copy_(torch.tensor([0.15, -0.2, 0.05, 0.0]))
    return router


@pytest.mark.parametrize("has_scores", [False, True])
def test_distributed_exact_rejected_without_mutating_window(has_scores):
    router = _router(num_bins=None)
    if has_scores:
        router(torch.randn(1, 3, 4))
    bias = router.expert_bias.clone()
    chunks = list(router._quantile_score_chunks)
    # The unsupported branch rejects before attempting any collective.
    with pytest.raises(NotImplementedError, match="Distributed exact QB"):
        router.update_quantile_bias(process_group=Mock(spec=dist.ProcessGroup))
    torch.testing.assert_close(router.expert_bias, bias)
    assert len(router._quantile_score_chunks) == len(chunks)
    assert all(
        a is b for a, b in zip(router._quantile_score_chunks, chunks, strict=True)
    )


def _check_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        generator = torch.Generator().manual_seed(51)
        shards = [
            torch.rand(3, 4, generator=generator),
            torch.rand(7, 4, generator=generator),
        ]
        for num_bins in (1, 32):
            router = _router(num_bins)
            assert router.bias_histogram is not None
            bias_object = router.expert_bias
            # Successive windows also verify reset and recomputed old-bias bounds.
            for lengths in ((3, 7), (0, 7), (3, 0), (0, 0)):
                old_bias = router.expert_bias.clone()
                local_scores = shards[rank][: lengths[rank]]
                router.accumulate_bias_histogram(
                    local_scores, old_bias, 2, router.bias_histogram
                )
                # Independent pooling: concatenate both shards, never average quantiles.
                global_scores = torch.cat(
                    [s[:n] for s, n in zip(shards, lengths, strict=True)]
                )
                expected = old_bias
                if len(global_scores):
                    pooled = torch.zeros_like(router.bias_histogram)
                    lower, width = router.accumulate_bias_histogram(
                        global_scores, old_bias, 2, pooled
                    )
                    expected = router.estimate_bias_from_histogram(
                        pooled, lower, width, 2
                    )
                router.update_quantile_bias(process_group=dist.group.WORLD)
                assert router.expert_bias is bias_object
                torch.testing.assert_close(router.expert_bias, expected)
                assert router.bias_histogram.count_nonzero() == 0

        # Default None remains local even with a live world process group.
        router = _router()
        assert router.bias_histogram is not None
        lower, width = router.accumulate_bias_histogram(
            shards[rank], router.expert_bias, 2, router.bias_histogram
        )
        expected = router.estimate_bias_from_histogram(
            router.bias_histogram, lower, width, 2
        )
        router.update_quantile_bias()
        torch.testing.assert_close(router.expert_bias, expected)
        assert router.bias_histogram.count_nonzero() == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_two_rank_histogram_updates_match_pooled_reference(tmp_path):
    processes, logs = [], []
    try:
        for rank in range(2):
            log = (tmp_path / f"rank_{rank}.log").open("w+")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "moe.tests.test_distributed_quantile_update",
                        str(rank),
                        (tmp_path / "rendezvous").as_uri(),
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        for process in processes:
            try:
                code = process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pytest.fail("Distributed QB exceeded its 45-second timeout")
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
