"""Concentrated smarter path: rank-then-size, baseline blend, harsh costs."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import load_config
from src.env.portfolio_env import action_to_weights
from src.train.concentrated_supervised import ConcentratedRankerAdapter, _zscore_1d


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


def test_concentrated_supervised_config_defaults():
    cfg = load_config("configs/concentrated-smoke.yaml", mode="concentrated")
    assert cfg.concentrated.learner == "supervised"
    assert cfg.concentrated.train_cost_multiplier == 3.0
    assert cfg.concentrated.baseline_mix == 0.45
    assert _zscore_1d(np.array([1.0, 1.0, 1.0])).sum() == 0.0
