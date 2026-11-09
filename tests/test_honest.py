"""Dead names stay in the even-split book until their prices stop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.honest import replay_even_split
from src.data.delisted import (
    apply_known_splits,
    listed_through,
    load_gap_names,
    load_price_frames,
)
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
    close, _, _ = load_price_frames(path, include_gaps=False)
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
    close, _, _ = load_price_frames(path, include_gaps=False)
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
    close, _, _ = load_price_frames(path, include_gaps=False)
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
    close, _, _ = load_price_frames(path, include_gaps=False)
    assert pd.isna(close.loc[pd.Timestamp("2005-02-15"), "TAPE"])


def test_a_real_collapse_is_kept(tmp_path: Path):
    rows = []
    for i, day in enumerate(pd.bdate_range("2008-09-01", periods=8)):
        price = 8.0 if i < 5 else 2.0
        rows.append(_row(day.date().isoformat(), "DEAD", price, price, 80_000_000))
    path = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    close, _, _ = load_price_frames(path, include_gaps=False)
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
    close, _, _ = load_price_frames(path, include_gaps=False)
    assert float(close["CAP"].dropna().max()) == 20.0


def test_a_zero_price_is_not_a_trade(tmp_path: Path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        [
            {"date": "2008-09-25", "symbol": "WAMUQ", "close": 1.0, "adjusted_close": 1.0, "volume": 10},
            {"date": "2008-09-26", "symbol": "WAMUQ", "close": 0.0, "adjusted_close": 0.0, "volume": 10},
        ]
    ).to_csv(path, index=False)
    close, _, _ = load_price_frames(path, include_gaps=False)
    assert pd.isna(close.loc[pd.Timestamp("2008-09-26"), "WAMUQ"])


def test_a_known_split_leaves_only_that_days_move():
    frame = pd.DataFrame(
        [
            {"date": "2006-04-27", "symbol": "LEH", "close": 156.01, "volume": 1_000_000},
            {"date": "2006-04-28", "symbol": "LEH", "close": 151.15, "volume": 1_000_000},
            {"date": "2006-05-01", "symbol": "LEH", "close": 71.75, "volume": 2_000_000},
            {"date": "2006-05-02", "symbol": "LEH", "close": 72.83, "volume": 2_000_000},
        ]
    )
    out = apply_known_splits(frame).set_index("date")
    before = float(out.loc["2006-04-28", "close"])
    after = float(out.loc["2006-05-01", "close"])
    assert abs(before - 151.15 / 2) < 1e-9
    move = after / before - 1
    assert -0.10 < move < 0
    dollars = before * float(out.loc["2006-04-28", "volume"])
    assert abs(dollars - 151.15 * 1_000_000) < 1


def test_a_crash_is_not_rewritten_as_a_split():
    frame = pd.DataFrame(
        [
            {"date": "2008-09-11", "symbol": "LEH", "close": 4.22, "volume": 100_000_000},
            {"date": "2008-09-12", "symbol": "LEH", "close": 3.65, "volume": 300_000_000},
            {"date": "2008-09-15", "symbol": "LEH", "close": 0.21, "volume": 400_000_000},
            {"date": "2008-09-16", "symbol": "LEH", "close": 0.30, "volume": 140_000_000},
        ]
    )
    out = apply_known_splits(frame)
    assert float(out["close"].iloc[-2]) == 0.21
    assert float(out["close"].iloc[0]) == 4.22


def test_earlier_prices_are_added_without_a_second_copy(tmp_path: Path):
    main = tmp_path / "prices.csv"
    gap = tmp_path / "gap.csv"
    pd.DataFrame(
        [
            _row("2008-03-17", "BSC", 4.81, 4.81, 166_000_000),
            _row("2008-03-18", "BSC", 5.91, 5.91, 166_000_000),
        ]
    ).to_csv(main, index=False)
    pd.DataFrame(
        [
            {"date": "2008-03-13", "symbol": "BSC", "close": 57.0, "volume": 20_000_000},
            {"date": "2008-03-14", "symbol": "BSC", "close": 30.0, "volume": 80_000_000},
            {"date": "2008-03-17", "symbol": "BSC", "close": 4.81, "volume": 1_000},
        ]
    ).to_csv(gap, index=False)
    close, volume, _ = load_price_frames(main, gap_file=gap)
    assert float(close.loc[pd.Timestamp("2008-03-14"), "BSC"]) == 30.0
    assert float(close.loc[pd.Timestamp("2008-03-17"), "BSC"]) == 4.81
    assert float(volume.loc[pd.Timestamp("2008-03-17"), "BSC"]) == 166_000_000
    crash = float(close.loc[pd.Timestamp("2008-03-17"), "BSC"] / close.loc[pd.Timestamp("2008-03-14"), "BSC"] - 1)
    assert crash < -0.80


def test_old_gm_does_not_become_the_new_listing(tmp_path: Path):
    main = tmp_path / "prices.csv"
    gap = tmp_path / "gap.csv"
    rows = [
        _row("2010-11-16", "KEEP", 10.0, 10.0, 1_000_000),
        _row("2010-11-17", "GM", 33.0, 25.0, 0),
        _row("2010-11-17", "KEEP", 10.0, 10.0, 1_000_000),
        _row("2010-11-18", "GM", 34.0, 25.0, 40_000_000),
        _row("2010-11-18", "KEEP", 10.0, 10.0, 1_000_000),
    ]
    pd.DataFrame(rows).to_csv(main, index=False)
    pd.DataFrame(
        [
            {"date": "2008-12-31", "symbol": "GM", "close": 3.20, "volume": 20_000_000},
            {"date": "2009-06-01", "symbol": "MTLQQ", "close": 0.75, "volume": 30_000_000},
            {"date": "2010-11-16", "symbol": "MTLQQ", "close": 0.10, "volume": 5_000_000},
            {"date": "2010-11-18", "symbol": "MTLQQ", "close": 0.05, "volume": 5_000_000},
        ]
    ).to_csv(gap, index=False)
    close, _, _ = load_price_frames(main, gap_file=gap)
    assert float(close.loc[pd.Timestamp("2010-11-16"), "GM"]) == 0.10
    assert pd.isna(close.loc[pd.Timestamp("2010-11-17"), "GM"])
    assert float(close.loc[pd.Timestamp("2010-11-18"), "GM"]) == 25.0
    # The new listing is the next row after a blank day, so it is not a return.
    assert pd.isna(close["GM"].pct_change().loc[pd.Timestamp("2010-11-18")])


def test_a_missing_share_count_does_not_skip_the_next_crash(tmp_path: Path):
    main = tmp_path / "prices.csv"
    gap = tmp_path / "gap.csv"
    pd.DataFrame([_row("2002-07-02", "KEEP", 10.0, 10.0, 1_000)]).to_csv(main, index=False)
    pd.DataFrame(
        [
            {"date": "2002-06-25", "symbol": "WCOM", "close": 0.83, "volume": 180_000_000},
            {"date": "2002-06-26", "symbol": "WCOM", "close": 0.83, "volume": 0},
            {"date": "2002-06-28", "symbol": "WCOM", "close": 0.83, "volume": 0},
            {"date": "2002-07-01", "symbol": "WCOM", "close": 0.06, "volume": 1_500_000_000},
        ]
    ).to_csv(gap, index=False)
    close, volume, _ = load_price_frames(main, gap_file=gap)
    assert float(close.loc[pd.Timestamp("2002-06-28"), "WCOM"]) == 0.83
    assert float(volume.loc[pd.Timestamp("2002-06-28"), "WCOM"]) == 180_000_000
    crash = float(close.loc[pd.Timestamp("2002-07-01"), "WCOM"] / close.loc[pd.Timestamp("2002-06-28"), "WCOM"] - 1)
    assert crash < -0.90


def test_holding_the_crash_is_worse_than_starting_at_the_stub():
    dates = pd.bdate_range("2008-03-03", periods=12)
    stub = np.full(len(dates), 4.81)
    full = stub.copy()
    full[:8] = 30.0
    spy = np.full(len(dates), 100.0)
    kwargs = dict(
        benchmark="SPY",
        min_history_days=3,
        min_dollar_volume=1_000,
        max_names=2,
        cost_rate=0.0,
        rebalance_every=5,
    )
    crashed = pd.DataFrame({"SPY": spy, "BSC": full}, index=dates)
    late = pd.DataFrame({"SPY": spy, "BSC": stub}, index=dates)
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=["SPY", "BSC"])
    with_crash = replay_even_split(crashed, volume, **kwargs)
    without = replay_even_split(late, volume, **kwargs)
    wealth_crash = float((1.0 + with_crash["net_return"]).prod())
    wealth_stub = float((1.0 + without["net_return"]).prod())
    assert wealth_crash < wealth_stub


def test_the_saved_gap_file_contains_the_four_collapses():
    gaps = load_gap_names()
    leh = gaps[gaps["symbol"] == "LEH"].sort_values("date")
    wcom = gaps[gaps["symbol"] == "WCOM"].sort_values("date")
    bear = gaps[gaps["symbol"] == "BSC"].sort_values("date")
    old_gm = gaps[gaps["symbol"].isin(["GM", "MTLQQ"])].sort_values("date")
    assert float(leh["close"].iloc[-1]) < 1.0
    assert pd.Timestamp("2008-09-15") in set(leh["date"])
    assert float(wcom["close"].iloc[-1]) < 0.1
    assert float(bear.loc[bear["date"] == "2008-03-14", "close"].iloc[0]) == 30.0
    assert float(old_gm.loc[old_gm["date"] == "2009-06-01", "close"].iloc[0]) < 1.0
    # The May 2006 morning is a split, so the adjusted path does not fall by half.
    before = float(leh.loc[leh["date"] == "2006-04-28", "close"].iloc[0])
    after = float(leh.loc[leh["date"] == "2006-05-01", "close"].iloc[0])
    assert -0.10 < after / before - 1 < 0


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
