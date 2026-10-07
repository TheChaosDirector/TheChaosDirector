"""Day-trade gym: must pick one, open→close accounting, forced signal."""

from __future__ import annotations

import numpy as np

from src.config import CostsCfg, DaytradeCfg, RiskCfg
from src.env.daytrade_env import DaytradeEnv, resolve_pick
from src.features.factory import build_features, warmup_days


def _make_env(fake_market, cfg, **kwargs):
    close, volume = fake_market
    open_ = close.shift(1).fillna(close)
    panel = build_features(close, volume, cfg.features, benchmark="SPY")
    start = warmup_days(cfg.features)
    return DaytradeEnv(
        open_=open_,
        close=close,
        panel=panel,
        costs=CostsCfg(commission_bps=1.0, slippage_bps=5.0),
        risk=RiskCfg(),
        daytrade=DaytradeCfg(forced_threshold=0.5),
        start=start,
        end=len(close),
        benchmark="SPY",
        **kwargs,
    )


def test_resolve_pick_falls_back_from_invalid():
    valid = np.array([1.0, 0.0, 1.0, 1.0])
    idx, forced = resolve_pick(np.array([1, 1]), valid)  # 1 is invalid
    assert idx in (0, 2, 3)
    assert forced is True

    idx2, forced2 = resolve_pick(np.array([2, 0]), valid)
    assert idx2 == 2
    assert forced2 is False


def test_episode_always_one_ticker(fake_market, cfg):
    env = _make_env(fake_market, cfg)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape

    rng = np.random.default_rng(0)
    done = False
    while not done:
        action = env.action_space.sample()
        # re-seed sample via rng for determinism-ish
        action = np.array([rng.integers(0, env.n_assets), rng.integers(0, 2)])
        obs, reward, done, _, _ = env.step(action)
        assert np.isfinite(reward)

    journal = env.results()
    assert len(journal) == env.end - env.start
    assert journal["ticker"].notna().all()
    np.testing.assert_allclose(journal["cost"], 0.0012, rtol=1e-9)


def test_open_close_accounting(fake_market, cfg):
    env = _make_env(fake_market, cfg)
    env.reset()
    action = np.array([0, 0])  # ticker 0, not forced
    env.step(action)
    row = env.history[0]
    expected_gross = row["close"] / row["open"] - 1.0
    expected_net = expected_gross - 0.0012
    assert abs(row["gross_return"] - expected_gross) < 1e-12
    assert abs(row["net_return"] - expected_net) < 1e-12
    assert row["forced"] is False


def test_forced_flag_from_action(fake_market, cfg):
    env = _make_env(fake_market, cfg)
    env.reset()
    env.step(np.array([0, 1]))
    assert env.history[0]["forced"] is True
