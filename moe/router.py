from typing import Literal

import torch
import torch.distributed as dist
from torch import nn

from moe.losses import load_balancing_loss, router_z_loss

from .metrics import (
    assignment_entropy,
    assignment_fraction,
    maxvio,
    router_entropy,
    unused_expert_count,
)


class TopKRouter(nn.Module):
    """Top-k routing with detached statistics and an optional-use QB workspace.

    QB + num_bins=None uses exact quantiles (stores detached score chunks).
    QB + positive num_bins uses an int32 histogram [num_experts,num_bins].
    Explicit num_bins requires QB. Disabled/exact mode allocates no histogram.
    Accumulation/update/reset lifecycle is separate from diagnostic metrics.
    """

    tokens_per_expert: torch.Tensor
    router_prob_sum: torch.Tensor
    num_tokens: torch.Tensor
    router_entropy_sum: torch.Tensor
    selected_prob_mass_sum: torch.Tensor
    expert_bias: torch.Tensor
    bias_histogram: torch.Tensor | None

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        *,
        use_quantile_balancing: bool = False,
        num_bins: int | None = None,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.score_func = score_func
        self.use_quantile_balancing = use_quantile_balancing
        self.num_bins = num_bins
        # Exact-mode temporary state: append cheaply, concatenate once per step.
        # No checkpoint/autograd state; finish/reset a window before moving devices.
        self._quantile_score_chunks: list[torch.Tensor] = []

        if not (1 <= top_k <= num_experts):
            raise ValueError(f"Top K must be between 1 and {num_experts}, got: {top_k}")
        if use_quantile_balancing and top_k == num_experts:
            raise ValueError("Quantile balancing requires top_k < num_experts")
        if num_bins is not None and not use_quantile_balancing:
            raise ValueError("Explicit num_bins requires use_quantile_balancing=True")
        if num_bins is not None and num_bins <= 0:
            raise ValueError(
                f"Expected num_bins to be strictly positive, got: {num_bins}"
            )
        if score_func not in ["softmax", "sigmoid"]:
            raise ValueError(
                f'score_func must be either "softmax" or "sigmoid", got: {score_func}'
            )

        self.router = nn.Linear(d_model, num_experts, bias=False)

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "router_prob_sum",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "num_tokens", torch.zeros((), dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "router_entropy_sum", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "selected_prob_mass_sum",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts, dtype=torch.float32),
        )
        self.register_buffer(
            "bias_histogram",
            torch.zeros(num_experts, num_bins, dtype=torch.int32)
            if num_bins is not None
            else None,  # [E,num_bins] only in histogram mode
            persistent=False,
        )

    @torch.no_grad()
    def reset_metrics(self) -> None:
        self.tokens_per_expert.zero_()
        self.router_prob_sum.zero_()
        self.num_tokens.zero_()
        self.router_entropy_sum.zero_()
        self.selected_prob_mass_sum.zero_()

    @torch.no_grad()
    def compute_metrics(self) -> dict[str, torch.Tensor]:
        """Read statistics from the full normalized affinity view; means are [E]."""
        metrics = {}

        metrics["maxvio"] = maxvio(self.tokens_per_expert)
        metrics["assignment_fraction"] = assignment_fraction(self.tokens_per_expert)
        metrics["unused_expert_count"] = unused_expert_count(self.tokens_per_expert)
        metrics["assignment_entropy"] = assignment_entropy(self.tokens_per_expert)
        metrics["mean_router_probability"] = (
            self.router_prob_sum / self.num_tokens.clamp_min(1)
        )
        metrics["mean_router_entropy"] = (
            self.router_entropy_sum / self.num_tokens.clamp_min(1)
        )
        metrics["mean_selected_probability_mass"] = (
            self.selected_prob_mass_sum / self.num_tokens.clamp_min(1)
        )

        return metrics

    @torch.no_grad()
    def update_expert_bias(self, rate: float) -> None:
        target_count = self.tokens_per_expert.sum() / self.num_experts
        self.expert_bias += rate * torch.sign(
            target_count - self.tokens_per_expert
        )  # [E]

    @torch.no_grad()
    def update_quantile_bias(
        self, process_group: dist.ProcessGroup | None = None
    ) -> None:
        """Replace biases once per optimizer step; clear its QB window.

        Exact mode pools detached [T_i,E] scores before taking a quantile.
        Histogram mode interprets counts with the OLD biases' bin boundaries.
        None keeps updates local, even when distributed is initialized. An explicit
        group sums histograms [E,num_bins] in place; exact distributed mode is unsupported.
        All group members must call with matching modes, old biases and bins,
        including ranks with zero tokens. Caller initializes the process group.
        Disabled/eval/globally empty windows are no-ops.
        """
        if not self.training or not self.use_quantile_balancing:
            return

        histogram = self.bias_histogram
        if histogram is None:
            if process_group is not None:
                raise NotImplementedError("Distributed exact QB is not implemented")
            if not self._quantile_score_chunks:
                return
            scores = torch.cat(self._quantile_score_chunks, dim=0)  # [sum(T_i),E]
            new_biases = self.estimate_quantile_bias(
                scores, self.expert_bias, self.top_k
            )
        else:
            if process_group is not None:
                # SUM preserves counts, not rank averages; even empty ranks participate.
                dist.all_reduce(histogram, op=dist.ReduceOp.SUM, group=process_group)
            # Each row counts all tokens. One host check per step, not per forward.
            if histogram[0].sum().item() == 0:
                return
            lower_bound = self.expert_bias.min() - 1  # [] from the old biases
            upper_bound = self.expert_bias.max() + 1  # []
            bin_width = (upper_bound - lower_bound) / histogram.shape[1]  # []
            new_biases = self.estimate_bias_from_histogram(
                histogram, lower_bound, bin_width, self.top_k
            )
        self.expert_bias.copy_(new_biases)
        if histogram is None:
            self._quantile_score_chunks.clear()
        else:
            histogram.zero_()

    @torch.no_grad()
    def estimate_quantile_bias(
        self,
        router_scores: torch.Tensor,  # [T, E], raw sigmoid scores
        expert_bias: torch.Tensor,  # [E], current selection biases
        top_k: int,
    ) -> torch.Tensor:  # [E], new centered biases
        """Estimate replacement biases, without mutating inputs or router state.

        Scores [T,E], current biases [E] -> centered new biases [E].
        Assumes nonempty raw sigmoid scores and 1 <= top_k < E.
        Exact in-memory quantiles; no training-step accumulation or synchronization.
        """
        # The strongest rejected expert defines each token's frozen cutoff.
        biased_scores = router_scores + expert_bias  # [T, E]
        top_scores = biased_scores.topk(top_k + 1, dim=-1).values  # [T, K+1]
        cutoff = top_scores[:, -1:]  # [T, 1]; broadcast over experts.

        # Bias needed by expert e to reach token t's cutoff, using RAW scores.
        required_bias = cutoff - router_scores  # [T, E]
        # Each expert targets K/E of tokens; reduce tokens, not experts.
        quantiles = torch.quantile(
            required_bias, q=top_k / expert_bias.shape[0], dim=0
        )  # [E]

        # A common offset cannot change top-k. Return targets, not increments.
        return quantiles - quantiles.mean()

    @torch.no_grad()
    def accumulate_bias_histogram(
        self,
        router_scores: torch.Tensor,  # [T,E]
        expert_bias: torch.Tensor,  # [E]
        top_k: int,
        histogram: torch.Tensor,  # [E,num_bins], updated in place
    ) -> tuple[torch.Tensor, torch.Tensor]:  # Two scalar tensors: [], []
        """Add one microbatch's required-bias counts to histogram IN PLACE.

        Scores [T,E]: finite sigmoid values in [0,1]; current bias [E]: finite.
        histogram [E,num_bins]: int32/int64 counts, initially zero, on the same device.
        Caller ensures counts (including pooled ranks) fit the counter dtype.
        T = local token count; E = expert count; K = top_k.
        Returns lower_bound [] and bin_width [] for later quantile extraction.
        Keep biases/bin count fixed across all accumulated microbatches/ranks;
        reset counts when starting a new window. No bias update or collective.
        """
        # Avoid low-precision rounding of bin boundaries; preserve float64 references.
        scores = (
            router_scores
            if router_scores.dtype == torch.float64
            else router_scores.float()
        )  # [T,E]
        bias = expert_bias.to(dtype=scores.dtype)  # [E]

        # 1. Every required bias lies in [b_min-1, b_max+1]. Use common bins.
        lower_bound = bias.amin() - 1
        upper_bound = bias.amax() + 1
        num_bins = histogram.shape[1]
        bin_width = (upper_bound - lower_bound) / num_bins

        # 2. One strongest-rejected cutoff per token, just like the exact estimator.
        biased_scores = scores + bias  # [T,E]
        # Top-(K+1) values are [T,K+1]; keep their last column as [T,1].
        cutoff = biased_scores.topk(top_k + 1, dim=-1).values[:, -1:]  # [T,1]

        # 3. Each token contributes a required bias for EVERY expert, selected or not.
        required_bias = cutoff - scores  # [T,E]

        # 4. Distance from the left edge / bin width gives the zero-based bin.
        bin_indices = (
            ((required_bias - lower_bound) / bin_width).floor().long()
        )  # [T,E]
        bin_indices.clamp_(
            0, num_bins - 1
        )  # [T,E]; include the upper endpoint in the last bin.

        # 5. Each expert is a row. Every observation adds 1 to that row's chosen bin.
        indices = bin_indices.transpose(0, 1)  # [E,T]
        # Indices stay int64; source counts match the histogram's counter dtype.
        counts = torch.ones_like(indices, dtype=histogram.dtype)  # [E,T]
        histogram.scatter_add_(dim=1, index=indices, src=counts)  # [E,num_bins]
        return lower_bound, bin_width

    @torch.no_grad()
    def estimate_bias_from_histogram(
        self,
        histogram: torch.Tensor,  # [num_experts, num_bins]
        lower_bound: torch.Tensor,  # scalar
        bin_width: torch.Tensor,  # scalar
        top_k: int,
    ) -> torch.Tensor:  # [num_experts], centered biases
        """Read approximate replacement biases from pooled counts, without mutation.

        histogram [E,num_bins] has equal positive row totals T; 1 <= top_k < E.
        Bounds are scalar tensors from accumulation with identical fixed bins.
        Assumes valid inputs; validation/window management belong to the caller.
        """
        hist_cumsum = torch.cumsum(histogram, dim=-1)  # [E, num_bins]
        target = top_k * hist_cumsum[0, -1] / histogram.shape[0]  # [] = T*K/E

        # First bin whose cumulative count reaches the target, for each expert.
        bin_idx = torch.searchsorted(
            hist_cumsum, target.expand(histogram.shape[0], 1).contiguous()
        )  # [E, 1]

        # Count inside the chosen bin and before it (also works for bin zero).
        bin_count = torch.gather(histogram, index=bin_idx, dim=1)  # [E, 1]
        bin_cumsum = torch.gather(hist_cumsum, index=bin_idx, dim=1)  # [E, 1]
        before_cumsum = bin_cumsum - bin_count  # [E, 1]

        # Assume samples are uniform within the bin: left edge + fraction * width.
        fraction = (target - before_cumsum) / bin_count  # [E, 1]
        bin_lower_bound = lower_bound + bin_idx * bin_width  # [E, 1]
        bias_estimate = bin_lower_bound + fraction * bin_width  # [E, 1]
        # Remove the common offset; these are replacement biases, not increments.
        mean_centered_bias_estimate = (
            bias_estimate.squeeze(1) - bias_estimate.mean()
        )  # [E]

        return mean_centered_bias_estimate

    def forward(  # ty: ignore[invalid-mutable-override]
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        E = number of experts
        K = number of selected experts
        x: [B, S, D]
        Returns IDs, weights [B, S, K], balancing loss, and z-loss (None in eval).
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = self.router(x.float())  # [B, S, E]

            if self.score_func == "softmax":
                router_scores = nn.functional.softmax(
                    router_logits, dim=-1
                )  # [B, S, E]
                normalized_scores = router_scores
            elif self.score_func == "sigmoid":
                router_scores = nn.functional.sigmoid(router_logits)  # [B, S, E]
                normalized_scores = router_scores / router_scores.sum(
                    -1, keepdim=True
                )  # [B, S, E]

        # Accumulate all raw scores; keep the bias fixed until the step boundary.
        if self.training and self.use_quantile_balancing:
            flat_scores = router_scores.detach().reshape(-1, self.num_experts)  # [T,E]
            if self.bias_histogram is None:
                if flat_scores.shape[0] > 0:
                    self._quantile_score_chunks.append(flat_scores)
            else:
                self.accumulate_bias_histogram(
                    flat_scores, self.expert_bias, self.top_k, self.bias_histogram
                )

        _, experts = torch.topk(
            router_scores + self.expert_bias, k=self.top_k, dim=-1
        )  # [B, S, K]
        scores = router_scores.gather(dim=-1, index=experts)  # [B, S, K]

        counts = torch.bincount(experts.flatten(), minlength=self.num_experts)  # [E]
        weights = scores / torch.sum(scores, dim=-1, keepdim=True)  # [B, S, K]

        aux_loss = (
            load_balancing_loss(normalized_scores, counts) if self.training else None
        )
        z_loss = (
            router_z_loss(router_logits)
            if self.training and self.score_func != "sigmoid"
            else None
        )

        if self.training:
            with torch.no_grad():
                self.tokens_per_expert += counts
                self.router_prob_sum += normalized_scores.float().sum(dim=(0, 1))  # [E]
                self.num_tokens += x.shape[0] * x.shape[1]  # B*S
                self.router_entropy_sum += router_entropy(
                    normalized_scores.float()
                ).sum()
                self.selected_prob_mass_sum += (
                    normalized_scores.gather(-1, experts).float().sum()
                )

        return experts, weights, aux_loss, z_loss


if __name__ == "__main__":
    x = torch.randn((4, 6, 5))
    router = TopKRouter(d_model=5, num_experts=4, top_k=2)
    router(x)
