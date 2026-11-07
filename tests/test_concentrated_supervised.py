"""Concentrated smarter path: rank-then-size, baseline blend, harsh costs."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import load_config
from src.env.portfolio_env import action_to_weights, apply_hold_deadband
from src.train.concentrated_supervised import (
    ConcentratedRankerAdapter,
    _zscore_1d,
    prune_ensemble_members,
    sticky_top_k,
)


def test_zscore_and_ranker_forces_investment():
    rng = np.random.default_rng(0)
    n_assets, n_features = 6, 4
    # Toy regressor: fit tiny data so predict works.
    X = rng.normal(size=(40, n_features))
    y = rng.normal(size=40)
    model = HistGradientBoostingRegressor(max_iter=20, max_depth=2, random_state=0).fit(X, y)
    names = [f"f{i}" for i in range(n_features)]
    names[0] = "mom_63d"
    ad = ConcentratedRankerAdapter(
        model,
        n_assets=n_assets,
        n_features=n_features,
        feature_names=names,
        momentum_feature="mom_63d",
        baseline_mix=0.5,
        max_names=3,
        sticky_rank_buffer=2,
    )
    obs = rng.normal(size=n_assets * n_features + n_assets + 2).astype(np.float32)
    action, _ = ad.predict(obs)
    assert action.shape == (n_assets + 1,)
    assert action[-1] < -5
    w = action_to_weights(
        action,
        np.ones(n_assets),
        max_weight=0.35,
        max_gross=1.0,
        max_names=3,
        min_gross=0.80,
    )
    assert w.sum() >= 0.80 - 1e-9
    assert int((w > 1e-12).sum()) <= 3


def test_sticky_top_k_prefers_incumbents_near_elite():
    scores = np.array([0.1, 0.9, 0.8, 0.7, 0.05, 0.4])
    prev = np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0])  # holds 0 and 3
    keep = sticky_top_k(scores, prev, max_names=2, sticky_rank_buffer=2)
    assert len(keep) == 2
    # 3 is within elite of k+buffer=4; should be sticky-kept with best name 1
    assert 3 in set(keep.tolist())
    assert 1 in set(keep.tolist())


def test_hold_deadband_skips_tiny_fidgets():
    prev = np.array([0.4, 0.4, 0.0, 0.0])
    target = np.array([0.41, 0.39, 0.0, 0.0])  # tiny shuffle
    out = apply_hold_deadband(
        prev,
        target,
        deadband=0.03,
        max_weight=0.5,
        max_gross=1.0,
        min_gross=0.80,
    )
    np.testing.assert_allclose(out, prev, atol=1e-12)


def test_prune_ensemble_keeps_median_or_better_capped():
    green = [
        {
            "fold": i,
            "oos_excess_mean": ex,
            "out_of_sample": {"total_return": 0.1, "sharpe": 1.0},
        }
        for i, ex in enumerate([0.001, 0.005, 0.003, 0.0005, 0.004, 0.002])
    ]
    kept = prune_ensemble_members(green, max_members=3)
    assert len(kept) <= 3
    # Median of six excesses is 0.0025; keep those >= median, then cap to 3.
    assert all(float(r["oos_excess_mean"]) >= 0.0025 for r in kept)
    assert [r["fold"] for r in kept] == [1, 4, 2]


def test_concentrated_supervised_config_defaults():
    cfg = load_config("configs/concentrated-smoke.yaml", mode="concentrated")
    assert cfg.concentrated.learner == "supervised"
    assert cfg.concentrated.train_cost_multiplier == 3.0
    assert cfg.concentrated.baseline_mix == 0.45
    assert cfg.concentrated.hold_deadband == 0.03
    assert cfg.concentrated.sticky_rank_buffer == 2
    assert cfg.concentrated.ensemble_max_members == 5
    assert _zscore_1d(np.array([1.0, 1.0, 1.0])).sum() == 0.0
