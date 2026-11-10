"""Cash dividends raise the total return and leave dollar volume on the traded close."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.delisted import load_gap_names, load_price_frames, total_return_close
from src.data.eligibility import point_in_time_eligible
from src.data.gap_dividends import cash_dividends, write_dividend_file


def test_a_dividend_offsets_the_ex_day_drop():
    close = np.array([100.0, 99.0, 100.0])
    cash = np.array([0.0, 1.0, 0.0])
    adjusted = total_return_close(close, cash)
    assert adjusted[-1] == 100.0
    assert abs(adjusted[1] / adjusted[0] - 1.0) < 1e-12
    assert abs(close[1] / close[0] - 1.0 - (-0.01)) < 1e-12


def test_cash_is_in_the_return_and_not_in_dollar_volume(tmp_path: Path):
    main = tmp_path / "prices.csv"
    gap = tmp_path / "gap.csv"
    dividends = tmp_path / "dividends.csv"
    pd.DataFrame(
        [
            {"date": "2008-01-02", "symbol": "KEEP", "close": 10.0, "adjusted_close": 10.0, "volume": 1000},
            {"date": "2008-01-03", "symbol": "KEEP", "close": 10.0, "adjusted_close": 10.0, "volume": 1000},
            {"date": "2008-01-04", "symbol": "KEEP", "close": 10.0, "adjusted_close": 10.0, "volume": 1000},
        ]
    ).to_csv(main, index=False)
    pd.DataFrame(
        [
            {"date": "2008-01-02", "symbol": "LEH", "close": 100.0, "volume": 10},
            {"date": "2008-01-03", "symbol": "LEH", "close": 99.0, "volume": 10},
            {"date": "2008-01-04", "symbol": "LEH", "close": 100.0, "volume": 10},
        ]
    ).to_csv(gap, index=False)
    pd.DataFrame([{"date": "2008-01-03", "symbol": "LEH", "cash": 1.0}]).to_csv(dividends, index=False)
    close, _, dollar = load_price_frames(main, gap_file=gap, dividend_file=dividends)
    returned = float(close.loc[pd.Timestamp("2008-01-03"), "LEH"] / close.loc[pd.Timestamp("2008-01-02"), "LEH"] - 1)
    assert abs(returned) < 1e-9
    assert abs(float(dollar.loc[pd.Timestamp("2008-01-02"), "LEH"]) - 1000.0) < 1e-6


def _tiny_book(tmp_path: Path, dividends: list[dict]):
    main = tmp_path / "prices.csv"
    gap = tmp_path / "gap.csv"
    div_path = tmp_path / "dividends.csv"
    rows = []
    for day in pd.bdate_range("2008-01-02", periods=8):
        for symbol in ("KEEP", "SPY"):
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "symbol": symbol,
                    "close": 10.0,
                    "adjusted_close": 10.0,
                    "volume": 1000,
                }
            )
    pd.DataFrame(rows).to_csv(main, index=False)
    gap_rows = []
    for day in pd.bdate_range("2008-01-02", periods=8):
        gap_rows.append({"date": day.date().isoformat(), "symbol": "LEH", "close": 50.0, "volume": 100_000})
    pd.DataFrame(gap_rows).to_csv(gap, index=False)
    pd.DataFrame(dividends or [{"date": "1990-01-01", "symbol": "NONE", "cash": 0.0}]).to_csv(div_path, index=False)
    return load_price_frames(main, gap_file=gap, dividend_file=div_path)


def test_a_later_dividend_does_not_change_an_earlier_eligibility_row(tmp_path: Path):
    plain_dir = tmp_path / "plain"
    paid_dir = tmp_path / "paid"
    plain_dir.mkdir()
    paid_dir.mkdir()
    close_a, volume_a, dollar_a = _tiny_book(plain_dir, [])
    close_b, volume_b, dollar_b = _tiny_book(
        paid_dir, [{"date": "2008-01-11", "symbol": "LEH", "cash": 10.0}]
    )
    early = pd.Timestamp("2008-01-04")
    assert abs(float(dollar_a.loc[early, "LEH"]) - float(dollar_b.loc[early, "LEH"])) < 1e-6
    assert float(close_b.loc[early, "LEH"]) < float(close_a.loc[early, "LEH"])
    kwargs = dict(min_history_days=2, min_dollar_volume=1_000, max_names=2, benchmark="SPY")
    before = point_in_time_eligible(close_a, volume_a, dollar_volume=dollar_a, **kwargs)
    after = point_in_time_eligible(close_b, volume_b, dollar_volume=dollar_b, **kwargs)
    assert bool(before.loc[early, "LEH"]) == bool(after.loc[early, "LEH"])


def test_the_filed_dividends_match_the_years_that_were_read():
    write_dividend_file()
    table = cash_dividends()
    saved = pd.read_csv(Path(__file__).resolve().parents[1] / "src/data/gap_dividends.csv")
    assert len(saved) == len(table)
    gaps = load_gap_names()
    leh = gaps[gaps["symbol"] == "LEH"]
    bear = gaps[gaps["symbol"] == "BSC"]
    gm = gaps[gaps["symbol"].isin(["GM", "MTLQQ"])]
    assert "WCOM" not in set(gaps.loc[gaps["cash"] > 0, "symbol"])
    assert abs(float(leh.loc[leh["date"].dt.year == 2007, "cash"].sum()) - 0.60) < 1e-9
    assert abs(float(leh.loc[leh["date"].dt.year == 2006, "cash"].sum()) - 0.48) < 1e-9
    assert float(leh.loc[leh["date"] > "2008-05-31", "cash"].sum()) == 0.0
    assert abs(float(bear.loc[bear["date"].dt.year == 2007, "cash"].sum()) - 1.28) < 1e-9
    assert float(bear.loc[bear["date"] > "2008-01-31", "cash"].sum()) == 0.0
    assert abs(float(gm.loc[gm["date"].dt.year == 2005, "cash"].sum()) - 2.00) < 1e-9
    assert abs(float(gm.loc[gm["date"].dt.year == 2008, "cash"].sum()) - 0.50) < 1e-9
    assert float(gm.loc[gm["date"] > "2008-06-30", "cash"].sum()) == 0.0
    for symbol, group in gaps.groupby("symbol"):
        group = group.sort_values("date")
        assert abs(float(group["adjusted_close"].iloc[-1]) - float(group["close"].iloc[-1])) < 1e-9, symbol
        paid_rows = group[group["cash"] > 0]
        if paid_rows.empty:
            continue
        paid = paid_rows.iloc[0]
        prev = group[group["date"] < paid["date"]].iloc[-1]
        total = float(paid["adjusted_close"] / prev["adjusted_close"] - 1)
        raw = float(paid["close"] / prev["close"] - 1)
        assert total > raw
