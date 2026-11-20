"""Sanity checks on the trading gym: legal weights, costs, accounting."""

from __future__ import annotations

import numpy as np

from src.config import CostsCfg, RiskCfg
from src.env.portfolio_env import PortfolioEnv, action_to_weights
from src.features.factory import build_features, warmup_days


def _make_env(fake_market, cfg, **kwargs):
    close, volume = fake_market
    panel = build_features(close, volume, cfg.features, benchmark="SPY")
    start = warmup_days(cfg.features)
    return PortfolioEnv(
        close=close,
        panel=panel,
        costs=CostsCfg(),
        risk=RiskCfg(max_weight_per_name=0.5),
        start=start,
        end=len(close) - 1,
        **kwargs,
    )


def test_weights_are_always_legal():
    rng = np.random.default_rng(1)
    for _ in range(200):
        action = rng.normal(0, 3, 6)
        valid = rng.integers(0, 2, 5).astype(float)
        w = action_to_weights(action, valid, max_weight=0.25, max_gross=1.0)
        assert (w >= 0).all(), "short position slipped through"
        assert (w <= 0.25 + 1e-12).all(), "per-name cap violated"
        assert w.sum() <= 1.0 + 1e-9, "gross exposure cap violated"
        assert (w[valid == 0] == 0).all(), "traded a name with no price"


def test_episode_runs_and_accounts_add_up(fake_market, cfg):
    env = _make_env(fake_market, cfg)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape

    rng = np.random.default_rng(0)
    done = False
    while not done:
        action = rng.normal(0, 1, env.action_space.shape).astype(np.float32)
        obs, reward, done, _, _ = env.step(action)
        assert np.isfinite(reward)

    journal = env.results()
    assert len(journal) == env.end - env.start
    # Equity must equal compounded net returns.
    np.testing.assert_allclose(
        journal["equity"].iloc[-1],
        (1 + journal["net_return"]).prod(),
        rtol=1e-9,
    )
    assert (journal["cost"] >= 0).all()


def test_all_cash_agent_never_loses_money(fake_market, cfg):
    env = _make_env(fake_market, cfg)
    env.reset()
    # Huge logit on the cash slot => ~100% cash forever.
    action = np.full(env.action_space.shape, -5.0, dtype=np.float32)
    action[-1] = 5.0
    done = False
    while not done:
        _, _, done, _, _ = env.step(action)
    journal = env.results()
    assert abs(journal["equity"].iloc[-1] - 1.0) < 0.01, "cash-only account should stay flat"
