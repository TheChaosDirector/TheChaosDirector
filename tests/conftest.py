"""Shared fixtures: a small deterministic fake market."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config


@pytest.fixture
def fake_market():
    """500 business days, 4 tickers (incl. SPY benchmark), reproducible."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=500)
    tickers = ["SPY", "AAA", "BBB", "CCC"]
    market = rng.normal(0.0004, 0.01, len(dates))

    close = {}
    volume = {}
    for i, t in enumerate(tickers):
        beta = 1.0 if t == "SPY" else 0.8 + 0.3 * i
        idio = 0.0 if t == "SPY" else 0.012
        rets = beta * market + rng.normal(0, idio, len(dates))
        close[t] = 100 * np.exp(np.cumsum(rets))
        volume[t] = rng.integers(1_000_000, 50_000_000, len(dates)).astype(float)

    return (
        pd.DataFrame(close, index=dates),
        pd.DataFrame(volume, index=dates),
    )


@pytest.fixture
def cfg():
    c = Config()
    c.features.momentum_windows = [5, 21]
    c.features.vol_window = 21
    c.features.volume_window = 21
    c.features.ma_windows = [10, 50]
    c.features.regime_ma = 50
    return c
