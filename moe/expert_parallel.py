"""Educational CPU expert parallelism, built one step at a time."""

from datetime import timedelta

import torch.distributed as dist


def main() -> None:
    # torchrun supplies each process's rank and the rendezvous settings.
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Rank {rank}/{world_size} ready", flush=True)

        # Next: add the all-to-all warm-up here.
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
