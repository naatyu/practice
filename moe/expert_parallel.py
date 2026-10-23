"""Educational CPU expert parallelism, built one step at a time."""

from datetime import timedelta

import torch
import torch.distributed as dist

from .experts import GroupedExperts


def prepare_dispatch(
    tokens: torch.Tensor, expert_ids: torch.Tensor, num_experts: int, world_size: int
):
    """Group local token assignments by destination rank.

    tokens: [T, D]; expert_ids: [T, K], global IDs.
    Assumes num_experts is divisible by world_size, with consecutive IDs per rank.
    Returns token rows [T*K, D], expert IDs [T*K], counts [world_size],
    and the assignment permutation [T*K] needed to restore order after return.
    """
    k = expert_ids.shape[1]
    experts_per_rank = num_experts // world_size

    # Map each assignment to its owner and count rows to send to each rank.
    flat_expert_ids = expert_ids.flatten()  # [T * K]
    dest_rank = flat_expert_ids // experts_per_rank  # [T * K]
    send_counts = torch.bincount(dest_rank, minlength=world_size)  # [W]

    # Make one contiguous chunk per destination; preserve order within each chunk.
    rank_permutation = torch.argsort(dest_rank, stable=True)  # [T * K]

    # Flat assignment i belongs to token i // k. Gather whole rows, repeating as needed.
    token_permutation = rank_permutation // k  # [T * K]
    send_tokens = tokens[token_permutation]  # [T * K, D]
    # Apply the same permutation so each token row stays paired with its expert ID.
    send_expert_ids = flat_expert_ids[rank_permutation]  # [T * K]

    return send_tokens, send_expert_ids, send_counts, rank_permutation


def compute_local_experts(
    recv_tokens: torch.Tensor,
    recv_expert_ids: torch.Tensor,
    local_experts: GroupedExperts,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return expert-grouped outputs [N, D] and the received-row permutation [N].

    recv_tokens: [N, D]; recv_expert_ids: [N], global IDs owned by this rank.
    local_experts stores only this rank's experts. N is received assignments.
    """
    # Translate model-wide IDs into indices in this rank's expert weights.
    experts_per_rank = local_experts.num_experts
    local_expert_ids = recv_expert_ids - rank * experts_per_rank  # [N]

    # Group received rows by expert; keep the permutation to undo this later.
    expert_permutation = torch.argsort(local_expert_ids, stable=True)  # [N]
    grouped_tokens = recv_tokens[expert_permutation]  # [N, D]

    # Cumulative ends describe each local expert's rows, including empty groups.
    local_assignments = torch.bincount(
        local_expert_ids, minlength=experts_per_rank
    )  # [Ne]
    offsets = local_assignments.cumsum(dim=0, dtype=torch.int32)  # [Ne]

    # Outputs remain expert-grouped, not in the order required for return.
    grouped_output = local_experts(grouped_tokens, offsets)

    return grouped_output, expert_permutation


def all_to_all_warmup(rank: int) -> None:
    """Completed one-dimensional exchange, kept as a reference."""
    send_counts = torch.tensor([5, 1] if rank == 0 else [2, 4])
    send_input = torch.arange(6) * (rank + 1)

    received_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(received_counts, send_counts)

    received_input = send_input.new_empty((int(received_counts.sum().item()),))
    dist.all_to_all_single(
        received_input,
        send_input,
        input_split_sizes=send_counts.tolist(),
        output_split_sizes=received_counts.tolist(),
    )
    print(received_input)


def dispatch_demo(rank: int, world_size: int) -> None:
    """Exercise: send token rows and aligned global expert IDs to their owners."""
    if world_size != 2:
        raise ValueError("This example requires exactly two ranks")

    # D=16 fits grouped-MM alignment; distinct values identify each source rank.
    tokens = torch.arange(32, dtype=torch.float32).reshape(2, 16) + 100 * rank
    expert_ids = torch.tensor(
        [[3, 0], [2, 1]] if rank == 0 else [[0, 1], [0, 2]]
    )  # [T, K]
    # Fixed mixing weights in original token/slot order; each row sums to one.
    expert_weights = tokens.new_tensor(
        [[0.75, 0.25], [0.4, 0.6]] if rank == 0 else [[0.2, 0.8], [0.65, 0.35]]
    )  # [T, K]
    num_experts = 4

    send_tokens, send_expert_ids, send_counts, permutation = prepare_dispatch(
        tokens, expert_ids, num_experts, world_size
    )

    # TODO 1: Allocate received_counts [world_size] and exchange the counts.
    received_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(received_counts, send_counts)

    # TODO 2: Compute num_received and create the send/receive split lists.
    num_received = int(received_counts.sum())
    send_split = send_counts.tolist()
    recv_split = received_counts.tolist()

    # TODO 3: Allocate received_tokens [num_received, D] and received_expert_ids
    # [num_received], preserving each corresponding send tensor's dtype/device.
    recv_tokens = send_tokens.new_empty((num_received, send_tokens.shape[-1]))
    recv_expert_ids = send_expert_ids.new_empty((num_received,))

    # TODO 4: Exchange token rows, then expert IDs, using the same split lists.
    dist.all_to_all_single(
        recv_tokens,
        send_tokens,
        input_split_sizes=send_split,
        output_split_sizes=recv_split,
    )
    dist.all_to_all_single(
        recv_expert_ids,
        send_expert_ids,
        input_split_sizes=send_split,
        output_split_sizes=recv_split,
    )

    # TODO 5: Print rank, received counts, tokens, and IDs to inspect alignment.
    # Keep permutation on this source rank: later it will restore returned outputs.
    print(
        f"Rank {rank}: received counts={received_counts.tolist()}, "
        f"tokens={recv_tokens.tolist()}, expert_ids={recv_expert_ids.tolist()}, "
        f"permutation={permutation.tolist()}",
        flush=True,
    )

    local_experts = GroupedExperts(
        tokens.shape[-1], hidden_dim=32, num_experts=num_experts // world_size
    )
    grouped_output, expert_permutation = compute_local_experts(
        recv_tokens, recv_expert_ids, local_experts, rank
    )

    received_order_output = torch.empty_like(grouped_output)
    received_order_output[expert_permutation] = grouped_output

    returned_output = received_order_output.new_empty(send_tokens.shape)
    dist.all_to_all_single(
        returned_output,
        received_order_output,
        input_split_sizes=recv_split,
        output_split_sizes=send_split,
    )
    assignment_output = torch.empty_like(returned_output)
    assignment_output[permutation] = returned_output
    expert_output = assignment_output.reshape(tokens.shape[0], expert_ids.shape[1], -1)

    output = (expert_output * expert_weights.unsqueeze(-1)).sum(dim=1)
    print(f"Rank {rank}: combined output shape={tuple(output.shape)}", flush=True)


def main() -> None:
    # torchrun supplies each process's rank and the rendezvous settings.
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Rank {rank}/{world_size} ready", flush=True)
        dispatch_demo(rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
