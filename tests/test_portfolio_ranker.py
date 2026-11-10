"""The ranker's labels stop at the training window, and the default book stays PPO."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.factory import FeaturePanel
from src.train.portfolio_ranker import PortfolioRankerAdapter, decision_rows


class _Echo:
    def predict(self, rows):
        return np.asarray(rows, dtype=float)[:, 0]


def test_training_rows_do_not_read_the_day_the_window_ends_on():
    dates = pd.bdate_range("2020-01-02", periods=8)
    close = pd.DataFrame({"SPY": np.linspace(100, 107, 8), "AAA": np.linspace(50, 57, 8)}, index=dates)
    eligible = pd.DataFrame(True, index=dates, columns=close.columns)
    values = np.ones((8, 2, 1), dtype=np.float32)
    panel = FeaturePanel(dates=dates, tickers=list(close.columns), names=["mom"], values=values)
    end = 6
    close.iloc[end] = 1_000_000.0
    _, first = decision_rows(close, panel, eligible, 1, end, "SPY")
    close.iloc[end] = 1.0
    _, second = decision_rows(close, panel, eligible, 1, end, "SPY")
    np.testing.assert_allclose(first, second)
    assert len(first) > 0
    assert np.max(np.abs(first)) < 1.0


def test_cash_logit_stays_at_zero():
    adapter = PortfolioRankerAdapter(_Echo(), n_assets=2, n_features=1)
    action, _ = adapter.predict(np.array([1.0, 3.0, 9.0, 9.0, 9.0]))
    assert action.shape == (3,)
    assert float(action[-1]) == 0.0
    assert float(action[1]) > float(action[0])


def test_ranker_config_is_not_the_saved_ppo_book():
    cfg = load_config("configs/delisted-ranker.yaml")
    assert cfg.training.learner == "ranker"
    assert cfg.features.include_macro is True
    assert cfg.compete.spy_anchor == 0.0
    assert cfg.paths.artifacts_dir == "artifacts/delisted-ranker"
    saved = load_config("configs/yield-compete-robust.yaml")
    assert saved.training.learner == "ppo"
    assert saved.features.include_macro is False
