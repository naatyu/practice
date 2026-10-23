"""AI-authored teaching example: autograd through two EP all-to-all exchanges.

Run with two CPU processes:
python -m torch.distributed.run --master-addr=127.0.0.1 --nproc-per-node=2 \
    --module moe.expert_parallel_backward
"""

from datetime import timedelta

import torch
import torch.distributed as dist

from .expert_parallel import compute_local_experts, prepare_dispatch
from .experts import GroupedExperts


class _AllToAll(torch.autograd.Function):
    """Exchange rows [N_send, D] -> [N_recv, D]; reverse that exchange in backward.

    Split lists contain row counts per peer in the default process group.
    Every rank must participate in forward AND backward in the same order.
    This educational implementation supports first-order gradients only.
    """

    @staticmethod
    def forward(
        ctx,
        rows: torch.Tensor,
        send_splits: list[int],
        recv_splits: list[int],
    ) -> torch.Tensor:
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        received = rows.new_empty((sum(recv_splits), rows.shape[-1]))
        dist.all_to_all_single(
            received,
            rows.contiguous(),
            input_split_sizes=send_splits,
            output_split_sizes=recv_splits,
        )
        return received

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_received: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        # A forward row travelled source -> destination. Its gradient travels back.
        # Send one gradient per RECEIVED row and receive one per originally SENT row.
        grad_rows = grad_received.new_empty(
            (sum(ctx.send_splits), grad_received.shape[-1])
        )
        dist.all_to_all_single(
            grad_rows,
            grad_received.contiguous(),
            input_split_sizes=ctx.recv_splits,
            output_split_sizes=ctx.send_splits,
        )
        # Only rows are differentiable; the two split lists are integer metadata.
        return grad_rows, None, None


def expert_parallel_forward(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    local_experts: GroupedExperts,
) -> torch.Tensor:
    """Dropless EP: tokens [T,D], IDs/weights [T,K] -> output [T,D].

    Consecutive, equally many experts per rank; initialized default group.
    IDs/counts are nondifferentiable. Gradients flow to tokens, mixing weights,
    and local expert parameters.
    For this backward demo, tokens require grad on every rank, including empty ones,
    so every rank traverses both communication nodes during loss.backward().
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    send_tokens, send_ids, send_counts, permutation = prepare_dispatch(
        tokens, expert_ids, local_experts.num_experts * world_size, world_size
    )

    # Counts and IDs carry routing metadata, not differentiable activations.
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    send_splits, recv_splits = send_counts.tolist(), recv_counts.tolist()
    recv_ids = send_ids.new_empty((sum(recv_splits),))
    dist.all_to_all_single(
        recv_ids,
        send_ids,
        input_split_sizes=send_splits,
        output_split_sizes=recv_splits,
    )

    # Dispatch: backward returns expert-input gradients to the token owners.
    recv_tokens = _AllToAll.apply(send_tokens, send_splits, recv_splits)
    grouped_output, expert_permutation = compute_local_experts(
        recv_tokens, recv_ids, local_experts, rank
    )
    # Ordinary autograd differentiates expert MLPs and these permutations.
    # Expert parameter gradients stay on their owner rank.
    received_order_output = torch.empty_like(grouped_output)
    received_order_output[expert_permutation] = grouped_output

    # Return: backward sends weighted output gradients BACK to expert owners.
    # This node is visited first in backward, before the expert MLPs/dispatch node.
    returned = _AllToAll.apply(received_order_output, recv_splits, send_splits)
    assignment_output = torch.empty_like(returned)
    assignment_output[permutation] = returned
    expert_output = assignment_output.reshape(
        tokens.shape[0], expert_ids.shape[1], tokens.shape[-1]
    )  # [T,K,D]; use explicit D so an empty token batch also works.

    # On the source rank: dL/dy_k = weight_k * g; dL/dweight_k = dot(y_k, g).
    # Autograd also sums repeated token-row gradients from prepare_dispatch's gather.
    return (expert_output * expert_weights.unsqueeze(-1)).sum(dim=1)


def backward_demo() -> None:
    """Learned router [D,E], local experts, and input gradients on two CPU ranks."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise ValueError("This example requires exactly two ranks")

    # Replicated routers start identically; expert weights differ across owners.
    torch.manual_seed(123)
    router = torch.nn.Linear(16, 4, bias=False)
    torch.manual_seed(42 + rank)
    local_experts = GroupedExperts(16, 32, 2)
    tokens = torch.randn(3, 16, requires_grad=True)

    selected_logits, expert_ids = router(tokens).topk(2, dim=-1)
    # Gradients flow through selected weights, NOT the discrete top-k indices.
    weights = selected_logits.softmax(dim=-1)  # [T,K]
    weights.retain_grad()  # Inspection only: non-leaf tensors normally discard .grad.
    output = expert_parallel_forward(tokens, expert_ids, weights, local_experts)
    loss = output.square().mean()
    loss.backward()  # Both ranks must call this, even with zero received assignments.

    # Separate from EP exchanges: SUM local router gradients for the global objective
    # loss_rank0 + loss_rank1. Divide by world_size for their average instead.
    # No expert-gradient reduction: these expert parameters are NOT replicated here.
    assert router.weight.grad is not None
    dist.all_reduce(router.weight.grad, op=dist.ReduceOp.SUM)
    assert tokens.grad is not None and weights.grad is not None
    assert local_experts.up_weight.grad is not None
    print(
        f"Rank {rank}: token grad={tuple(tokens.grad.shape)}, "
        f"mixing-weight grad={tuple(weights.grad.shape)}, "
        f"local expert grad={tuple(local_experts.up_weight.grad.shape)}, "
        f"synchronized router grad={tuple(router.weight.grad.shape)}",
        flush=True,
    )


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        backward_demo()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
