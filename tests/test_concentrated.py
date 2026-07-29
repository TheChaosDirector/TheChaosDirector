"""Concentrated-mode mechanics: top-k weights, excess reward, reality gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CostsCfg, RiskCfg, load_config
from src.env.portfolio_env import PortfolioEnv, action_to_weights
from src.features.factory import build_features, warmup_days
from src.modes import is_allocation, is_concentrated, is_daytrade
from src.referee.checks import check_reality_gate
from src.registry.store import save_experiments


def test_topk_weights_respect_max_names_and_caps():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = 10
        action = rng.normal(0, 2, n + 1)
        valid = np.ones(n)
        w = action_to_weights(action, valid, max_weight=0.35, max_gross=1.0, max_names=4)
        assert (w >= 0).all()
        assert (w <= 0.35 + 1e-12).all()
        assert w.sum() <= 1.0 + 1e-9
        assert int((w > 1e-12).sum()) <= 4


def test_min_gross_blocks_all_cash():
    n = 8
    # Huge cash logit — without min_gross this is ~100% cash.
    action = np.full(n + 1, -5.0)
    action[-1] = 10.0
    valid = np.ones(n)
    w0 = action_to_weights(action, valid, max_weight=0.35, max_gross=1.0, max_names=4)
    assert w0.sum() < 0.2
    w = action_to_weights(
        action, valid, max_weight=0.35, max_gross=1.0, max_names=4, min_gross=0.80
    )
    assert w.sum() >= 0.80 - 1e-9
    assert int((w > 1e-12).sum()) <= 4
    assert (w <= 0.35 + 1e-12).all()


def test_concentrated_config_loads():
    cfg = load_config("configs/concentrated-smoke.yaml", mode="concentrated")
    assert is_concentrated(cfg)
    assert is_allocation(cfg)
    assert not is_daytrade(cfg)
    assert cfg.concentrated.max_names == 4
    assert cfg.concentrated.min_gross_exposure == 0.80
    assert cfg.concentrated.excess_reward_weight == 1.5
    assert cfg.concentrated.learner == "supervised"
    assert cfg.concentrated.train_cost_multiplier == 3.0
    assert cfg.risk.max_weight_per_name == 0.35
    assert cfg.risk.drawdown_penalty == 0.08
    assert cfg.artifacts_dir.as_posix().endswith("concentrated")


def test_excess_reward_journal(fake_market, cfg):
    close, volume = fake_market
    panel = build_features(close, volume, cfg.features, benchmark="SPY")
    start = warmup_days(cfg.features)
    # Synthetic bench returns aligned to close length.
    br = np.zeros(len(close), dtype=np.float64)
    br[:-1] = 0.001
    env = PortfolioEnv(
        close=close,
        panel=panel,
        costs=CostsCfg(),
        risk=RiskCfg(max_weight_per_name=0.5, drawdown_penalty=0.1),
        start=start,
        end=len(close) - 1,
        benchmark_returns=br,
        max_names=3,
        min_gross=0.80,
        excess_reward_weight=1.5,
        absolute_reward_weight=0.05,
    )
    obs, _ = env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[-1] = 5.0  # prefer cash — min_gross should still force investment
    done = False
    while not done:
        obs, reward, done, _, _ = env.step(action)
        assert np.isfinite(reward)
    journal = env.results()
    assert "excess_return" in journal.columns
    assert "benchmark_return" in journal.columns
    assert "gross_exposure" in journal.columns
    assert float(journal["gross_exposure"].min()) >= 0.80 - 1e-6
    assert abs(journal["benchmark_return"].iloc[0] - 0.001) < 1e-12


def test_concentrated_reality_gate_rules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(
        str(Path(__file__).resolve().parents[1] / "configs" / "concentrated-smoke.yaml"),
        mode="concentrated",
    )
    # Fake fold metrics: 2/5 positive excess, mean excess negative → FAIL
    weak = [
        {
            "fold": i,
            "out_of_sample": {"total_return": 0.01, "sharpe": 0.5},
            "benchmark_oos": {"total_return": 0.02, "sharpe": 0.6},
            "oos_excess_mean": 0.001 if i < 2 else -0.002,
        }
        for i in range(5)
    ]
    save_experiments(cfg, weak)
    report = check_reality_gate(cfg)
    assert report["name"] == "reality_gate"
    assert report["passed"] is False

    # 3/5 positive excess and positive mean → PASS (>=40% and mean>0)
    strong = [
        {
            "fold": i,
            "out_of_sample": {"total_return": 0.05, "sharpe": 1.0},
            "benchmark_oos": {"total_return": 0.02, "sharpe": 0.6},
            "oos_excess_mean": 0.002 if i < 3 else -0.0005,
        }
        for i in range(5)
    ]
    save_experiments(cfg, strong)
    report = check_reality_gate(cfg)
    assert report["passed"] is True
    assert report["details"]["positive_excess_fraction"] == 0.6
