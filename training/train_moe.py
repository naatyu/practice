"""Tiny MoE overfit run"""

import argparse
from dataclasses import asdict, dataclass
from math import isfinite, log2
from pathlib import Path
from typing import Literal

import torch

from cross_entropy import CrossEntropy
from decoder_model import DecoderModel
from moe import GroupedMoE

from .training import train_step


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 16
    num_heads: int = 4
    max_seq_len: int = 64
    n_layers: int = 6
    hidden_dim: int = 3 * d_model
    num_kv_heads: int | None = 2
    dropout_p: float = 0.0
    rope_base: float = 11
    window_size: int | None = None
    num_experts: int | None = 4
    top_k: int | None = 2
    tie_weights: bool = True
    capacity_factor: float | None = None
    drop_policy: Literal["position", "probs"] = "probs"
    shared_hidden_dim: int | None = None
    shared_expert_gate: bool = False
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    use_quantile_balancing: bool = False
    num_bins: int | None = None  # QB: None = exact; positive count = histogram.


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    beta1: float = 0.99
    beta2: float = 0.95
    weight_decay: float = 0.1
    fused: bool = True


def train_moe(
    model_config: ModelConfig,
    optim_config: OptimizerConfig,
    *,
    steps: int = 100,
    device: str = "cpu",
    aux_loss_coefficient: float = 0.0,
    z_loss_coefficient: float = 0.0,
    bias_update_rate: float = 0.0,
) -> list[dict]:
    """Overfit one fixed batch; record metrics/update bias once per optimizer step.

    Optional QB is local only: exact when num_bins=None, otherwise histogram.
    QB, sign-based updates and auxiliary balancing are mutually exclusive here.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if not isfinite(bias_update_rate) or bias_update_rate < 0:
        raise ValueError("bias_update_rate must be finite and non-negative")
    if model_config.use_quantile_balancing and bias_update_rate > 0:
        raise ValueError(
            "Quantile balancing and sign-based bias updates cannot both be enabled"
        )
    if (
        bias_update_rate > 0 or model_config.use_quantile_balancing
    ) and aux_loss_coefficient != 0:
        raise ValueError(
            "Auxiliary balancing loss and expert bias updates cannot both be enabled"
        )
    if model_config.score_func == "sigmoid" and z_loss_coefficient != 0:
        raise ValueError(
            "z-loss is disabled for sigmoid routing; use z_loss_coefficient=0"
        )
    torch.manual_seed(42)
    model = DecoderModel(**asdict(model_config)).to(device)
    moes = []
    for block in model.blocks:
        assert isinstance(block.ffn, GroupedMoE)
        moes.append(block.ffn)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_config.lr,
        betas=(optim_config.beta1, optim_config.beta2),
        weight_decay=optim_config.weight_decay,
        fused=optim_config.fused and torch.device(device).type == "cuda",
    )
    loss_fn = CrossEntropy()
    # Keep the data independent of random draws during model initialization.
    batch_generator = torch.Generator(device=device).manual_seed(42)
    batch = torch.randint(
        model_config.vocab_size,
        (8, model_config.max_seq_len),
        device=device,
        generator=batch_generator,
    )  # [B, S] token IDs, not embeddings.

    for moe in moes:
        moe.reset_metrics()

    history = []
    for step in range(steps):
        losses = train_step(
            model,
            optimizer,
            batch,
            batch,
            loss_fn,
            max_grad_norm=1.0,
            aux_loss_coefficient=aux_loss_coefficient,
            z_loss_coefficient=z_loss_coefficient,
        )

        # Update once per optimizer step, before resetting assignment counts.
        for moe in moes:
            if moe.router.use_quantile_balancing:
                moe.router.update_quantile_bias()
            elif bias_update_rate > 0.0:
                moe.router.update_expert_bias(rate=bias_update_rate)

        layer_metrics = [
            {
                name: value.detach().cpu().clone()
                for name, value in moe.compute_metrics().items()
            }
            for moe in moes
        ]
        history.append(
            {
                "loss": losses.loss.item(),
                "task_loss": losses.task_loss.item(),
                "aux_loss": losses.aux_loss.item()
                if losses.aux_loss is not None
                else 0.0,
                "aux_loss_coefficient": aux_loss_coefficient,
                "z_loss": losses.z_loss.item() if losses.z_loss is not None else None,
                "z_loss_coefficient": z_loss_coefficient,
                "bias_update_rate": bias_update_rate,
                "use_quantile_balancing": model_config.use_quantile_balancing,
                "num_bins": model_config.num_bins,
                "score_func": model_config.score_func,
                "capacity_factor": model_config.capacity_factor,
                "drop_policy": model_config.drop_policy,
                "shared_hidden_dim": model_config.shared_hidden_dim,
                "shared_expert_gate": model_config.shared_expert_gate,
                "layers": layer_metrics,
            }
        )
        for moe in moes:
            moe.reset_metrics()

        if step == 0 or (step + 1) % 10 == 0 or step == steps - 1:
            z_label = (
                f"{losses.z_loss.item():.4f}" if losses.z_loss is not None else "n/a"
            )
            loads = [round(layer["maxvio"].item(), 2) for layer in layer_metrics]
            drops = [
                round(layer["dropped_assignment_rate"].item(), 3)
                for layer in layer_metrics
            ]
            print(
                f"step {step + 1:3d} | task {losses.task_loss.item():.4f} "
                f"| aux {history[-1]['aux_loss']:.4f} | z {z_label} "
                f"| maxvio per layer {loads} "
                f"| drop per layer {drops}"
            )

    return history


def plot_metrics(history: list[dict], output_path: str | Path) -> None:
    """Save summary curves and a separate per-layer expert-usage figure."""
    import matplotlib.pyplot as plt

    coefficient = history[0]["aux_loss_coefficient"]
    run_tag = f"aux_{coefficient:g}" if coefficient != 0 else "no_aux"
    run_label = (
        f"auxiliary coefficient = {coefficient:g}"
        if coefficient != 0
        else "no auxiliary loss"
    )
    score_func = history[0].get("score_func", "softmax")
    sigmoid_scoring = score_func == "sigmoid"
    if sigmoid_scoring:
        run_tag += "_sigmoid"
    run_label += f", scoring = {score_func}"
    z_coefficient = history[0].get("z_loss_coefficient", 0.0)
    if z_coefficient != 0:
        run_tag += f"_z_{z_coefficient:g}"
    run_label += f", z-loss coefficient = {z_coefficient:g}"
    bias_rate = history[0].get("bias_update_rate", 0.0)
    if bias_rate > 0:
        run_tag += f"_bias_{bias_rate:g}"
        run_label += f", bias update rate = {bias_rate:g}"
    if history[0].get("use_quantile_balancing", False):
        bins = history[0].get("num_bins")
        mode = "exact" if bins is None else f"hist_{bins}"
        run_tag += f"_qb_{mode}"
        run_label += f", quantile balancing = {mode}"
    capacity = history[0]["capacity_factor"]
    policy = history[0]["drop_policy"]
    capacity_tag = (
        f"capacity_{capacity:g}_{policy}" if capacity is not None else "dropless"
    )
    run_tag = f"{run_tag}_{capacity_tag}"
    run_label += (
        f", capacity factor = {capacity:g} ({policy})"
        if capacity is not None
        else ", dropless"
    )
    shared_width = history[0].get("shared_hidden_dim")
    if shared_width is not None:
        gate_tag = "gated" if history[0].get("shared_expert_gate", False) else "ungated"
        run_tag += f"_shared_{shared_width}_{gate_tag}"
        run_label += f", shared width = {shared_width} ({gate_tag})"
    output_path = Path(output_path)
    output_path = output_path.with_name(
        f"{output_path.stem}_{run_tag}{output_path.suffix}"
    )

    fig = plt.figure(figsize=(14, 16), layout="constrained")
    grid = fig.add_gridspec(4, 2)
    panels = [fig.add_subplot(grid[row, col]) for row in range(3) for col in range(2)]
    panels.append(fig.add_subplot(grid[3, :]))  # Full-width drop-rate panel.
    fig.suptitle(f"MoE training ({run_label})")
    steps = range(1, len(history) + 1)
    num_layers = len(history[0]["layers"])
    num_experts = history[0]["layers"][0]["assignment_fraction"].numel()
    panels[0].plot(steps, [record["loss"] for record in history], label="Total")
    panels[0].plot(steps, [record["task_loss"] for record in history], label="Task")
    panels[0].plot(
        steps, [record["aux_loss"] for record in history], label="Aux (unweighted)"
    )
    if not sigmoid_scoring:
        panels[0].plot(
            steps,
            [record.get("z_loss", 0.0) for record in history],
            label="Z (unweighted)",
        )
    panels[0].legend(fontsize="small")
    panels[0].set_title(
        "Training loss (fixed batch)\n"
        "Task near zero = memorization; auxiliary loss has a balanced reference of 1.",
        fontsize=10,
    )

    scalar_metrics = [
        (
            "maxvio",
            "MaxVio (relative overload)",
            "Lower is more balanced; 0 = perfect balance, 0.2 = 20% overload.",
        ),
        (
            "unused_expert_count",
            "Unused experts",
            "0 means every expert received assignments during this step.",
        ),
        (
            "assignment_entropy",
            "Assignment entropy (bits): usage balance",
            "Higher is more balanced; log2(E) = uniform usage, 0 = one expert.",
        ),
        (
            "mean_router_entropy",
            "Router entropy (bits): normalized sigmoid affinities"
            if sigmoid_scoring
            else "Router entropy (bits): token uncertainty",
            f"Lower = stronger relative token preferences; {log2(num_experts):g} bits = equal normalized affinities.",
        ),
        (
            "mean_selected_probability_mass",
            "Selected normalized affinity mass"
            if sigmoid_scoring
            else "Selected probability mass (before normalization)",
            "Share of full normalized affinities retained by selection, before top-k renormalization; not a quality score.",
        ),
        (
            "dropped_assignment_rate",
            "Dropped assignment rate",
            "Fraction of requested expert assignments dropped; 0 = no overflow.",
        ),
    ]
    for ax, (name, title, guide) in zip(panels[1:7], scalar_metrics):
        for layer in range(num_layers):
            values = [record["layers"][layer][name].item() for record in history]
            ax.plot(steps, values, label=f"Layer {layer}")
        ax.set_title(f"{title}\n{guide}", fontsize=10)
        ax.legend(fontsize="small")
        if name == "dropped_assignment_rate":
            ax.set_ylim(0, 1)

    for ax in panels:
        ax.set_xlabel("Optimizer step")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved metrics plot to {output_path}")

    fig, axes = plt.subplots(
        num_layers,
        3,
        figsize=(21, 3 * num_layers),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )
    fig.suptitle(
        f"Expert routing by layer ({run_label})\n"
        "Each color is one expert; dashed lines mark equal share (1/E) or zero drops.",
        fontsize=12,
    )
    expert_metrics = [
        (
            "assignment_fraction",
            "Requested assignment share",
            "Before capacity; equal shares mean balanced router requests.",
        ),
        (
            "mean_router_probability",
            "Mean normalized sigmoid affinity (before top-k)"
            if sigmoid_scoring
            else "Mean router probability (before top-k)",
            "Average full-expert-normalized sigmoid score; preference, not workload."
            if sigmoid_scoring
            else "Average softmax score over tokens; preference, not actual workload.",
        ),
        (
            "dropped_assignment_rate_per_expert",
            "Dropped assignment rate per expert",
            "Fraction of this expert's requests dropped; 0 = no overflow.",
        ),
    ]
    peak = max(
        layer[name].max().item()
        for record in history
        for layer in record["layers"]
        for name, _, _ in expert_metrics[:2]
    )
    upper = min(1.0, max(1.5 / num_experts, peak * 1.1))
    for column, (name, title, guide) in enumerate(expert_metrics):
        values = torch.stack(
            [
                torch.stack([layer[name] for layer in record["layers"]])
                for record in history
            ]
        )  # [steps, layers, experts]
        for layer in range(num_layers):
            ax = axes[layer, column]
            for expert in range(num_experts):
                ax.plot(
                    steps, values[:, layer, expert].numpy(), label=f"Expert {expert}"
                )
            is_drop_rate = name == "dropped_assignment_rate_per_expert"
            ax.axhline(
                0 if is_drop_rate else 1 / num_experts,
                color="black",
                linestyle="--",
                linewidth=1,
            )
            ax.set_ylim(0, 1 if is_drop_rate else upper)
            subtitle = f"\n{guide}" if layer == 0 else ""
            ax.set_title(f"Layer {layer}: {title}{subtitle}", fontsize=10)
            ax.grid(alpha=0.2)
            if layer == 0:
                ax.legend(fontsize="small", ncol=2)
            if layer == num_layers - 1:
                ax.set_xlabel("Optimizer step")

    expert_path = output_path.with_name(
        f"{output_path.stem}_experts{output_path.suffix}"
    )
    fig.savefig(expert_path, dpi=150)
    plt.close(fig)
    print(f"Saved expert routing plot to {expert_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--aux-loss-coefficient", type=float, default=0.0)
    parser.add_argument("--z-loss-coefficient", type=float, default=0.0)
    parser.add_argument("--bias-update-rate", type=float, default=0.0)
    parser.add_argument(
        "--use-quantile-balancing",
        action="store_true",
        help="Use a local QB replacement once per optimizer step instead of sign updates.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=None,
        help="QB histogram bin count; omit for exact quantiles. Requires QB.",
    )
    parser.add_argument("--capacity-factor", type=float, default=None)
    parser.add_argument("--drop-policy", choices=["position", "probs"], default="probs")
    parser.add_argument("--shared-hidden-dim", type=int, default=None)
    parser.add_argument("--shared-expert-gate", action="store_true")
    parser.add_argument(
        "--score-func", choices=["softmax", "sigmoid"], default="softmax"
    )
    parser.add_argument(
        "--output",
        default="training/plots/moe_metrics.png",
        help="Base plot path; active balancing, z-loss, capacity, and shared settings are appended.",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    history = train_moe(
        ModelConfig(
            capacity_factor=args.capacity_factor,
            drop_policy=args.drop_policy,
            shared_hidden_dim=args.shared_hidden_dim,
            shared_expert_gate=args.shared_expert_gate,
            score_func=args.score_func,
            use_quantile_balancing=args.use_quantile_balancing,
            num_bins=args.num_bins,
        ),
        OptimizerConfig(),
        steps=args.steps,
        device=args.device,
        aux_loss_coefficient=args.aux_loss_coefficient,
        z_loss_coefficient=args.z_loss_coefficient,
        bias_update_rate=args.bias_update_rate,
    )
    plot_metrics(history, args.output)
