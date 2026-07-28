"""Day-trade gym: must pick one, open→close accounting, forced signal."""

from __future__ import annotations

import numpy as np

from src.config import CostsCfg, DaytradeCfg, RiskCfg
from src.env.daytrade_env import DaytradeEnv, pick_one
from src.features.factory import build_features, warmup_days


def _make_env(fake_market, cfg, **kwargs):
    close, volume = fake_market
    # Synthetic opens: a tiny gap from prior close, first day = close.
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
        **kwargs,
    )


def test_pick_one_always_valid_and_forced_threshold():
    valid = np.array([1.0, 0.0, 1.0, 1.0])
    # Prefer invalid index 1 — must be ignored.
    action = np.array([0.0, 9.0, 1.0, 0.5, 5.0])  # last = reluctance logit
    idx, reluctance, forced = pick_one(action, valid, forced_threshold=0.5)
    assert idx == 2
    assert forced is True
    assert reluctance > 0.5

    action2 = np.array([2.0, 9.0, 0.0, 0.0, -5.0])
    idx2, reluctance2, forced2 = pick_one(action2, valid, forced_threshold=0.5)
    assert idx2 == 0
    assert forced2 is False
    assert reluctance2 < 0.5


def test_episode_always_one_ticker(fake_market, cfg):
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
    assert journal["ticker"].notna().all()
    assert (journal["ticker"].astype(str).str.len() > 0).all()
    # Costs = 2 * (1+5) bps = 12 bps every day.
    np.testing.assert_allclose(journal["cost"], 0.0012, rtol=1e-9)


def test_open_close_accounting(fake_market, cfg):
    env = _make_env(fake_market, cfg)
    env.reset()
    # Force pick of ticker 0 by huge score; low reluctance.
    action = np.full(env.action_space.shape, -5.0, dtype=np.float32)
    action[0] = 5.0
    action[-1] = -5.0
    obs, reward, done, _, info = env.step(action)
    row = env.history[0]
    expected_gross = row["close"] / row["open"] - 1.0
    expected_net = expected_gross - 0.0012
    assert abs(row["gross_return"] - expected_gross) < 1e-12
    assert abs(row["net_return"] - expected_net) < 1e-12
    assert row["forced"] is False


def test_high_reluctance_marks_forced(fake_market, cfg):
    env = _make_env(fake_market, cfg)
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 5.0
    action[-1] = 5.0  # high reluctance
    env.step(action)
    assert env.history[0]["forced"] is True
