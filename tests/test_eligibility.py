"""Point-in-time membership: later volume cannot put a name in an earlier book."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.data.download import _apply_filters
from src.data.eligibility import point_in_time_eligible
from src.env.portfolio_env import action_to_weights


def _frames(n=40):
    dates = pd.bdate_range("2000-01-03", periods=n)
    close = pd.DataFrame(
        {
            "SPY": np.full(n, 100.0),
            "OLD": np.full(n, 50.0),
            "NEW": np.full(n, 50.0),
        },
        index=dates,
    )
    volume = pd.DataFrame(index=dates, columns=close.columns, dtype=float)
    volume["SPY"] = 1_000_000.0
    volume["OLD"] = 1_000_000.0
    volume["NEW"] = 1_000.0
    volume.iloc[-5:, volume.columns.get_loc("NEW")] = 50_000_000.0
    return close, volume


def test_future_volume_does_not_make_a_name_eligible_early():
    close, volume = _frames()
    mask = point_in_time_eligible(
        close, volume,
        min_history_days=10,
        min_dollar_volume=1_000_000,
        max_names=2,
        benchmark="SPY",
    )
    early = mask.index[15]
    assert bool(mask.loc[early, "OLD"])
    assert not bool(mask.loc[early, "NEW"])
    # The spike is only at the end. A day before it, NEW still fails the trailing test.
    before_spike = mask.index[-6]
    assert not bool(mask.loc[before_spike, "NEW"])


def test_deleting_the_future_does_not_change_today():
    close, volume = _frames()
    kwargs = dict(min_history_days=10, min_dollar_volume=1_000_000, max_names=2, benchmark="SPY")
    full = point_in_time_eligible(close, volume, **kwargs)
    cut_at = 20
    trimmed = point_in_time_eligible(close.iloc[: cut_at + 1], volume.iloc[: cut_at + 1], **kwargs)
    pd.testing.assert_series_equal(
        full.iloc[cut_at].astype(bool),
        trimmed.iloc[cut_at].astype(bool),
        check_names=False,
    )


def test_missing_listing_does_not_become_cash():
    # Equal scores. One of two names cannot be traded. Its share moves to the
    # name that can, and cash stays at the explicit cash slot (1/3).
    weights = action_to_weights(np.zeros(3), np.array([1.0, 0.0]), max_weight=1.0, max_gross=1.0)
    assert weights[1] == 0.0
    np.testing.assert_allclose(weights[0], 2.0 / 3.0)
    np.testing.assert_allclose(1.0 - weights.sum(), 1.0 / 3.0)


def test_download_keeps_names_the_full_file_ranking_would_have_dropped():
    cfg = Config()
    cfg.universe.max_tickers = 1
    cfg.universe.min_history_days = 5
    cfg.data.min_dollar_volume = 1e12
    dates = pd.bdate_range("2000-01-03", periods=8)

    def frame(volume):
        return pd.DataFrame(
            {
                "date": dates,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": volume,
            }
        )

    kept = _apply_filters(
        cfg,
        {
            "SPY": frame(np.full(8, 100.0)),
            "QUIET": frame(np.full(8, 1.0)),
            "LOUD": frame(np.full(8, 1_000_000.0)),
        },
    )
    assert set(kept) == {"SPY", "QUIET", "LOUD"}
