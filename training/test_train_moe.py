"""AI-authored checks for the tiny MoE training/logging experiment."""

import math
from dataclasses import replace

import pytest
import torch

from decoder_model import DecoderModel
from mlp import SwiGLU
from moe import GroupedMoE
from moe.router import TopKRouter
from training.train_moe import ModelConfig, OptimizerConfig, plot_metrics, train_moe


def test_batch_is_fixed_across_steps_and_architecture_changes(
    small_config, monkeypatch
):
    batches = []
    original = DecoderModel.forward

    def capture(self, input_ids, *args, **kwargs):
        batches.append(input_ids.detach().clone())
        return original(self, input_ids, *args, **kwargs)

    monkeypatch.setattr(DecoderModel, "forward", capture)
    configs = [
        small_config,
        replace(small_config, shared_hidden_dim=24),
        replace(small_config, shared_hidden_dim=24, shared_expert_gate=True),
    ]
    for config in configs:
        train_moe(config, OptimizerConfig(), steps=2)
    assert len(batches) == 6
    for batch in batches[1:]:
        torch.testing.assert_close(batch, batches[0])


@pytest.mark.parametrize("gated", [False, True])
def test_training_config_enables_shared_expert_in_every_layer(
    small_config, monkeypatch, gated
):
    calls = []
    original = SwiGLU.forward

    def capture(self, x):
        calls.append(x.shape)
        assert self.down_proj.in_features == 24
        return original(self, x)

    monkeypatch.setattr(SwiGLU, "forward", capture)
    small_config.shared_hidden_dim = 24
    small_config.shared_expert_gate = gated
    history = train_moe(small_config, OptimizerConfig(), steps=2)
    assert calls == [torch.Size([8, 8, 16])] * (2 * small_config.n_layers)
    for record in history:
        assert record["shared_hidden_dim"] == 24
        assert record["shared_expert_gate"] == gated
        assert math.isfinite(record["task_loss"])


def test_shared_expert_plot_names_include_width_and_gate(small_config, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    small_config.shared_hidden_dim = 24
    history = train_moe(small_config, OptimizerConfig(), steps=1)
    output = tmp_path / "metrics.png"
    for gated in (False, True):
        history[0]["shared_expert_gate"] = gated
        plot_metrics(history, output)
        gate_tag = "gated" if gated else "ungated"
        tag = f"no_aux_dropless_shared_24_{gate_tag}"
        assert (tmp_path / f"metrics_{tag}.png").is_file()
        assert (tmp_path / f"metrics_{tag}_experts.png").is_file()


@pytest.fixture
def small_config():
    return ModelConfig(
        vocab_size=16,
        d_model=16,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        n_layers=2,
        hidden_dim=32,
    )


def test_fixed_batch_loss_decreases_and_history_is_detached(small_config):
    history = train_moe(small_config, OptimizerConfig(lr=0.01), steps=20)
    assert len(history) == 20
    assert history[-1]["loss"] < history[0]["loss"]
    for record in history:
        assert isinstance(record["loss"], float) and math.isfinite(record["loss"])
        assert len(record["layers"]) == 2
        for metrics in record["layers"]:
            assert len(metrics) == 9
            for value in metrics.values():
                assert value.device.type == "cpu"
                assert not value.requires_grad and value.grad_fn is None
                assert torch.isfinite(value).all()
            torch.testing.assert_close(
                metrics["assignment_fraction"].sum(), torch.tensor(1.0)
            )


def test_metrics_cover_full_batch_and_reset_each_step(small_config, monkeypatch):
    observations = []
    forward_calls = []
    original = TopKRouter.compute_metrics
    original_forward = TopKRouter.forward

    def capture_forward(self, x):
        forward_calls.append(x.shape[:2])
        return original_forward(self, x)

    def capture(self):
        observations.append(
            (self.num_tokens.item(), self.tokens_per_expert.sum().item())
        )
        return original(self)

    monkeypatch.setattr(TopKRouter, "compute_metrics", capture)
    monkeypatch.setattr(TopKRouter, "forward", capture_forward)
    train_moe(small_config, OptimizerConfig(), steps=3)
    # 8 sequences * 8 tokens, K=2; no leakage from previous optimizer steps.
    assert observations == [(64, 128)] * (3 * small_config.n_layers)
    assert forward_calls == [(8, 8)] * (3 * small_config.n_layers)


def test_auxiliary_coefficient_is_applied_and_losses_logged_separately(small_config):
    history = train_moe(
        small_config, OptimizerConfig(), steps=2, aux_loss_coefficient=0.01
    )
    for record in history:
        assert record["loss"] == pytest.approx(
            record["task_loss"] + 0.01 * record["aux_loss"]
        )
        assert math.isfinite(record["aux_loss"]) and record["aux_loss"] > 0


def test_z_loss_coefficient_and_history(small_config):
    history = train_moe(
        small_config,
        OptimizerConfig(),
        steps=2,
        aux_loss_coefficient=0.05,
        z_loss_coefficient=0.001,
    )
    for record in history:
        assert record["z_loss_coefficient"] == 0.001
        assert math.isfinite(record["z_loss"]) and record["z_loss"] > 0
        assert record["loss"] == pytest.approx(
            record["task_loss"] + 0.05 * record["aux_loss"] + 0.001 * record["z_loss"]
        )


@pytest.mark.parametrize("policy", ["position", "probs"])
def test_capacity_metrics_are_reported_and_reset_at_moe_level(
    small_config, monkeypatch, policy
):
    small_config.capacity_factor = 0.25
    small_config.drop_policy = policy
    original = GroupedMoE.compute_metrics
    observations = []

    def capture(self):
        requested = self.router.tokens_per_expert.sum().item()
        dropped = self.dropped_assignments_per_expert.sum().item()
        observations.append((requested, dropped))
        return original(self)

    monkeypatch.setattr(GroupedMoE, "compute_metrics", capture)
    history = train_moe(
        small_config, OptimizerConfig(), steps=3, aux_loss_coefficient=0.01
    )
    assert len(observations) == 3 * small_config.n_layers
    for requested, dropped in observations:
        assert requested == 128
        assert (
            96 <= dropped <= requested
        )  # At most 4 experts * capacity 8 are retained.
    for record in history:
        assert record["capacity_factor"] == 0.25 and record["drop_policy"] == policy
        for metrics in record["layers"]:
            assert 0.75 <= metrics["dropped_assignment_rate"] <= 1
            assert metrics["dropped_assignment_rate_per_expert"].shape == (4,)
            assert not metrics["dropped_assignment_rate"].requires_grad


@pytest.mark.parametrize(
    "coefficient,z_coefficient,capacity,policy,tag",
    [
        (0.0, 0.0, None, "probs", "no_aux_dropless"),
        (0.01, 0.0, 0.5, "position", "aux_0.01_capacity_0.5_position"),
        (0.01, 0.0, 0.5, "probs", "aux_0.01_capacity_0.5_probs"),
        (0.01, 0.001, None, "probs", "aux_0.01_z_0.001_dropless"),
    ],
)
def test_plot_saves_all_metrics_without_modifying_history(
    small_config, tmp_path, coefficient, z_coefficient, capacity, policy, tag
):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    small_config.capacity_factor = capacity
    small_config.drop_policy = policy
    history = train_moe(
        small_config,
        OptimizerConfig(),
        steps=2,
        aux_loss_coefficient=coefficient,
        z_loss_coefficient=z_coefficient,
    )
    saved = history[0]["layers"][0]["assignment_fraction"].clone()
    output = tmp_path / "plots" / "metrics.png"
    plot_metrics(history, output)
    for name in (f"metrics_{tag}.png", f"metrics_{tag}_experts.png"):
        path = output.with_name(name)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1000
    assert not output.exists()  # Keep different training settings in separate files.
    assert not plt.get_fignums()
    torch.testing.assert_close(history[0]["layers"][0]["assignment_fraction"], saved)
