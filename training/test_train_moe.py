"""AI-authored checks for the tiny MoE training/logging experiment."""

import math

import pytest
import torch

from moe.router import TopKRouter
from training.train_moe import ModelConfig, OptimizerConfig, plot_metrics, train_moe


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
            assert len(metrics) == 7
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


@pytest.mark.parametrize("coefficient, tag", [(0.0, "no_aux"), (0.01, "aux_0.01")])
def test_plot_saves_all_metrics_without_modifying_history(
    small_config, tmp_path, coefficient, tag
):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    history = train_moe(
        small_config, OptimizerConfig(), steps=2, aux_loss_coefficient=coefficient
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
