"""AI-authored tests for QB configuration forwarding and optimizer-step wiring."""

from dataclasses import replace

import pytest
import torch

from moe.router import TopKRouter
from training.train_moe import ModelConfig, OptimizerConfig, plot_metrics, train_moe


@pytest.fixture
def qb_config():
    return ModelConfig(
        vocab_size=16,
        d_model=16,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        n_layers=2,
        hidden_dim=32,
        score_func="sigmoid",
        use_quantile_balancing=True,
    )


@pytest.mark.parametrize("num_bins", [None, 32])
@pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
def test_qb_updates_once_per_layer_after_optimizer_before_metric_reset(
    qb_config, monkeypatch, num_bins, score_func
):
    config = replace(qb_config, num_bins=num_bins, score_func=score_func)
    events, routers = [], {}
    original_step = torch.optim.AdamW.step
    original_update = TopKRouter.update_quantile_bias

    def step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        events.append("step")
        return result

    def update(self):
        assert self.use_quantile_balancing and self.num_bins == num_bins
        assert self.training
        assert events[-1] in ("step", "qb")
        assert self.tokens_per_expert.sum() == 8 * 8 * 2
        if self.bias_histogram is None:
            assert sum(chunk.shape[0] for chunk in self._quantile_score_chunks) == 64
        else:
            assert self.bias_histogram.shape == (4, num_bins)
            torch.testing.assert_close(
                self.bias_histogram.sum(-1), torch.full((4,), 64)
            )
        events.append("qb")
        original_update(self)
        assert torch.isfinite(self.expert_bias).all()
        torch.testing.assert_close(
            self.expert_bias.mean(), self.expert_bias.new_zeros(()), atol=1e-6, rtol=0
        )
        assert not self._quantile_score_chunks
        if self.bias_histogram is not None:
            assert self.bias_histogram.count_nonzero() == 0
        routers[id(self)] = self

    def unexpected_sign(*args, **kwargs):
        pytest.fail("QB must never also execute the sign update")

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(TopKRouter, "update_quantile_bias", update)
    monkeypatch.setattr(TopKRouter, "update_expert_bias", unexpected_sign)
    history = train_moe(config, OptimizerConfig(), steps=3)
    assert events == (["step"] + ["qb"] * config.n_layers) * 3
    assert len(routers) == config.n_layers
    for record in history:
        assert record["use_quantile_balancing"] is True
        assert record["num_bins"] == num_bins
        assert record["score_func"] == score_func
        assert record["bias_update_rate"] == 0
        # Logging still observes routing diagnostics from the completed step.
        assert len(record["layers"]) == config.n_layers
        for layer in record["layers"]:
            torch.testing.assert_close(
                layer["assignment_fraction"].sum(), torch.tensor(1.0)
            )


@pytest.mark.parametrize(
    "kwargs", [{"bias_update_rate": 0.001}, {"aux_loss_coefficient": 0.05}]
)
def test_qb_rejects_competing_balancing_strategies(qb_config, kwargs):
    with pytest.raises(ValueError, match="cannot both be enabled"):
        train_moe(qb_config, OptimizerConfig(), steps=1, **kwargs)


def test_default_runner_never_calls_quantile_update(qb_config, monkeypatch):
    def unexpected_update(*args, **kwargs):
        pytest.fail("Default runner must leave QB disabled")

    monkeypatch.setattr(TopKRouter, "update_quantile_bias", unexpected_update)
    history = train_moe(
        replace(qb_config, use_quantile_balancing=False), OptimizerConfig(), steps=1
    )
    assert history[0]["use_quantile_balancing"] is False


@pytest.mark.parametrize("num_bins", [None, 32])
def test_qb_plot_tags_identify_exact_and_histogram_runs(qb_config, tmp_path, num_bins):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    history = train_moe(
        replace(qb_config, num_bins=num_bins), OptimizerConfig(), steps=1
    )
    plot_metrics(history, tmp_path / "metrics.png")
    mode = "exact" if num_bins is None else "hist_32"
    for suffix in (".png", "_experts.png"):
        assert (
            tmp_path / f"metrics_no_aux_sigmoid_qb_{mode}_dropless{suffix}"
        ).is_file()


def test_runner_rejects_num_bins_without_qb(qb_config):
    with pytest.raises(ValueError, match="requires use_quantile_balancing"):
        train_moe(
            replace(qb_config, use_quantile_balancing=False, num_bins=32),
            OptimizerConfig(),
            steps=1,
        )
