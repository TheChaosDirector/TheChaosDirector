"""Dead names stay in the even-split book until their prices stop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.honest import replay_even_split
from src.data.delisted import listed_through, load_price_frames
from src.data.eligibility import point_in_time_eligible


def _book(n=40):
    dates = pd.bdate_range("2001-01-02", periods=n)
    dead = np.full(n, 100.0)
    dead[15:] = 100.0 * np.power(0.7, np.arange(1, n - 14))
    close = pd.DataFrame(
        {"SPY": np.full(n, 100.0), "LIVE": np.full(n, 100.0), "DEAD": dead},
        index=dates,
    )
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)
    return close, volume


def test_a_name_that_goes_to_zero_reduces_the_book():
    close, volume = _book()
    kwargs = dict(
        benchmark="SPY",
        min_history_days=5,
        min_dollar_volume=1_000,
        max_names=3,
        cost_rate=0.0,
        rebalance_every=5,
    )
    with_dead = replay_even_split(close, volume, **kwargs)
    without = replay_even_split(close.drop(columns=["DEAD"]), volume.drop(columns=["DEAD"]), **kwargs)
    wealth_with = float((1.0 + with_dead["net_return"]).prod())
    wealth_without = float((1.0 + without["net_return"]).prod())
    assert wealth_with < wealth_without
    mask = point_in_time_eligible(
        close, volume,
        min_history_days=5,
        min_dollar_volume=1_000,
        max_names=3,
        benchmark="SPY",
    )
    assert bool(mask["DEAD"].iloc[10])
    assert close["DEAD"].iloc[-1] < 1.0


def test_raw_dollar_volume_keeps_a_split_adjusted_name():
    dates = pd.bdate_range("2004-01-02", periods=20)
    # Adjusted close is tiny after a later split. Raw dollars traded were large.
    close = pd.DataFrame({"SPY": 100.0, "SPLIT": 0.5}, index=dates)
    volume = pd.DataFrame({"SPY": 1_000_000.0, "SPLIT": 1_000.0}, index=dates)
    dollar = pd.DataFrame({"SPY": 1e8, "SPLIT": 5_000_000.0}, index=dates)
    missed = point_in_time_eligible(
        close, volume,
        min_history_days=5,
        min_dollar_volume=1_000_000,
        max_names=2,
        benchmark="SPY",
    )
    kept = point_in_time_eligible(
        close, volume,
        min_history_days=5,
        min_dollar_volume=1_000_000,
        max_names=2,
        benchmark="SPY",
        dollar_volume=dollar,
    )
    assert not bool(missed["SPLIT"].iloc[-1])
    assert bool(kept["SPLIT"].iloc[-1])


def test_rename_duplicate_is_not_loaded_twice(tmp_path: Path):
    path = tmp_path / "prices.csv"
    rows = []
    for day in pd.bdate_range("2012-05-18", periods=3):
        for symbol, close in (("FB", 38.0), ("META", 38.0), ("AAPL", 20.0)):
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "symbol": symbol,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 1000,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    close, _, _ = load_price_frames(path)
    assert "META" in close.columns
    assert "FB" not in close.columns
    assert "AAPL" in close.columns


def test_a_vendor_spike_is_not_kept(tmp_path: Path):
    rows = []
    for i, day in enumerate(pd.bdate_range("2015-11-02", periods=30)):
        rows.append(
            {
                "date": day.date().isoformat(),
                "symbol": "SPIKE",
                "close": 100.0 if i % 2 == 0 else 0.01,
                "adjusted_close": 100.0 if i % 2 == 0 else 0.01,
                "volume": 0 if i % 2 == 0 else 1000,
            }
        )
    for i, day in enumerate(pd.bdate_range("2001-06-01", periods=40)):
        price = 100.0 * (0.85 ** i)
        rows.append(
            {
                "date": day.date().isoformat(),
                "symbol": "FALL",
                "close": price,
                "adjusted_close": price,
                "volume": 1_000_000,
            }
        )
    path = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    close, _, _ = load_price_frames(path)
    kept = close["SPIKE"].dropna()
    if len(kept) >= 2:
        assert float(kept.max() / kept.min()) < 8.0
    adjacent = close["SPIKE"].pct_change().dropna()
    if len(adjacent):
        assert float(adjacent.abs().max()) < 1.0
    fell = close["FALL"].dropna()
    assert float(fell.iloc[0]) == 100.0
    assert float(fell.iloc[-1]) < 1.0
    assert len(fell) >= 35


def _row(day, symbol, close, adjusted, volume):
    return {
        "date": day,
        "symbol": symbol,
        "close": close,
        "adjusted_close": adjusted,
        "volume": volume,
    }


def test_a_column_swap_is_not_a_crash(tmp_path: Path):
    # Traded close is the smooth series, then the adjusted column takes over.
    rows = []
    for i, day in enumerate(pd.bdate_range("2003-08-01", periods=12)):
        rows.append(_row(day.date().isoformat(), "SWAP", 80 - i, (80 - i) * 10, 1_000_000))
    rows.append(_row("2003-08-19", "SWAP", 4.6, 69.0, 1_000_000))
    for i, day in enumerate(pd.bdate_range("2003-08-20", periods=8)):
        rows.append(_row(day.date().isoformat(), "SWAP", 4.5, 68 - i, 1_000_000))
    path = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    close, _, _ = load_price_frames(path)
    series = close["SWAP"].dropna()
    joined = float(series.loc[pd.Timestamp("2003-08-19")] / series.loc[pd.Timestamp("2003-08-18")] - 1)
    assert joined > -0.2
    assert float(series.iloc[-1]) < float(series.iloc[0])


def test_a_dead_tape_jump_is_dropped(tmp_path: Path):
    rows = []
    for i, day in enumerate(pd.bdate_range("2005-02-01", periods=10)):
        rows.append(_row(day.date().isoformat(), "TAPE", 3.0, 30.0, 1_000_000))
    rows.append(_row("2005-02-15", "TAPE", 80.0, 800.0, 100))
    rows.append(_row("2005-02-16", "TAPE", 80.0, 800.0, 100))
    path = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    close, _, _ = load_price_frames(path)
    assert pd.isna(close.loc[pd.Timestamp("2005-02-15"), "TAPE"])


def test_a_real_collapse_is_kept(tmp_path: Path):
    rows = []
    for i, day in enumerate(pd.bdate_range("2008-09-01", periods=8)):
        price = 8.0 if i < 5 else 2.0
        rows.append(_row(day.date().isoformat(), "DEAD", price, price, 80_000_000))
    path = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    close, _, _ = load_price_frames(path)
    series = close["DEAD"].dropna()
    assert float(series.iloc[4]) == 8.0
    assert float(series.iloc[5]) == 2.0


def test_a_vendor_ceiling_is_not_a_price(tmp_path: Path):
    rows = []
    for i, day in enumerate(pd.bdate_range("2003-09-01", periods=8)):
        price = 20.0 if i < 4 else 999_999.9999
        rows.append(_row(day.date().isoformat(), "CAP", price, price, 1_000_000))
    path = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    close, _, _ = load_price_frames(path)
    assert float(close["CAP"].dropna().max()) == 20.0


def test_a_zero_price_is_not_a_trade(tmp_path: Path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        [
            {"date": "2008-09-25", "symbol": "WAMUQ", "close": 1.0, "adjusted_close": 1.0, "volume": 10},
            {"date": "2008-09-26", "symbol": "WAMUQ", "close": 0.0, "adjusted_close": 0.0, "volume": 10},
        ]
    ).to_csv(path, index=False)
    close, _, _ = load_price_frames(path)
    assert pd.isna(close.loc[pd.Timestamp("2008-09-26"), "WAMUQ"])


def test_names_that_ended_early_are_not_still_listed():
    dates = pd.bdate_range("2000-01-03", periods=30)
    close = pd.DataFrame(
        {
            "SPY": np.full(30, 100.0),
            "LIVE": np.full(30, 50.0),
            "DEAD": np.concatenate([np.full(10, 40.0), np.full(20, np.nan)]),
        },
        index=dates,
    )
    keep = listed_through(close, "SPY", tail=5)
    assert "DEAD" not in keep
    assert "LIVE" in keep
    assert "SPY" in keep
