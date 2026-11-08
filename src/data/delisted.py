"""Prices for companies that later died, were bought, or changed ticker.

The curated list in ``universe.py`` is companies that still trade. A replay
built only from that list never holds Enron, Washington Mutual, or the other
names that went to about zero. This module reads a daily S&P 500 file that
keeps those names through their last trade.

Source: FINSABER (Apache-2.0), file
``all_sp500_prices_2000_2024_delisted_include.csv``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_URL = (
    "https://huggingface.co/datasets/finsaber-team/FINSABER-reproduce/"
    "resolve/main/data/price/all_sp500_prices_2000_2024_delisted_include.csv"
)

# The short ticker is already covered by the long one: META includes the years
# Facebook traded as FB, and GOOG covers the same days as the GOOGL share class.
# Holding both would count one company twice.
DROP_WHEN_ALSO_PRESENT = {"FB": "META", "GOOGL": "GOOG"}


def _repair_one_symbol(group: pd.DataFrame) -> pd.DataFrame:
    """Join days where the adjusted column and the traded column change places.

    On some rows the file's adjusted close is the split-adjusted price, and on
    the next row the traded close is. A real crash moves the price people paid.
    A column swap does not: one of the two columns still lines up with
    yesterday. A jump with almost no shares traded is not a price either.
    """
    group = group.copy()
    close = group["close"].to_numpy(dtype=np.float64, copy=True)
    adj = group["adjusted_close"].to_numpy(dtype=np.float64, copy=True)
    volume = group["volume"].to_numpy(dtype=np.float64, copy=True)
    dates = pd.to_datetime(group["date"]).to_numpy()
    for i in range(1, len(group)):
        gap_days = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D")
        c0, c1 = close[i - 1], close[i]
        a0, a1 = adj[i - 1], adj[i]
        v0, v1 = volume[i - 1], volume[i]
        if (
            gap_days > 7
            or not np.isfinite(gap_days)
            or c0 <= 0
            or c1 <= 0
            or a0 <= 0
            or a1 <= 0
        ):
            continue
        adj_ratio = a1 / a0
        if 1.0 / 3.0 < adj_ratio < 3.0:
            continue
        # A dead tape: the price jumps and almost nobody traded. Not a return.
        if v0 > 0 and np.isfinite(v1) and v1 < 0.05 * v0 and (adj_ratio > 4.0 or adj_ratio < 0.25):
            close[i] = np.nan
            adj[i] = np.nan
            volume[i] = np.nan
            continue
        # Only stitch when the two columns have been different, parallel series.
        # If they print the same number, this is a real price change.
        if i < 6:
            continue
        recent = close[i - 6 : i] / adj[i - 6 : i]
        recent = recent[np.isfinite(recent) & (recent > 0)]
        if len(recent) < 4:
            continue
        usual = float(np.median(recent))
        if not (0.8 < (c0 / a0) / usual < 1.25):
            continue
        if 0.8 <= usual <= 1.25:
            continue
        # An oscillation comes back. A real join stays at the new price.
        later = [
            float(adj[j])
            for j in range(i + 1, min(i + 8, len(group)))
            if np.isfinite(adj[j]) and adj[j] > 0
        ][:4]
        near_new = sum(1 for value in later if 1.0 / 1.5 < value / a1 < 1.5)
        near_old = sum(1 for value in later if 1.0 / 1.5 < value / a0 < 1.5)
        if near_new < 3 or near_new <= near_old:
            continue
        if 1.0 / 1.5 < (a1 / c0) < 1.5:
            # Today's adjusted close continues yesterday's traded close.
            adj[:i] *= c0 / a0
        elif 1.0 / 1.5 < (c1 / c0) < 1.5:
            # The traded price did not jump. Keep that return.
            adj[i:] *= (c1 / c0) / adj_ratio
        elif 1.0 / 1.5 < (c1 / a0) < 1.5:
            # Today's traded close continues yesterday's adjusted close.
            adj[i:] *= c1 / a1
    group["close"] = close
    group["adjusted_close"] = adj
    group["volume"] = volume
    return group


def _repair_scale_breaks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["symbol", "date"])
    parts = [_repair_one_symbol(group) for _, group in frame.groupby("symbol", sort=False)]
    return pd.concat(parts, ignore_index=True)


def _neighbor_spike(prices: pd.Series, factor: float = 8.0) -> np.ndarray:
    """True where a print leaves the last price and the next print comes back.

    A collapse that stays down is kept. A dead ticker that flips between a
    penny and a normal price comes back on the next print, and that print is
    dropped.
    """
    values = prices.to_numpy(dtype=np.float64)
    n = len(values)
    previous = np.full(n, -1, dtype=int)
    last = -1
    for i, value in enumerate(values):
        previous[i] = last
        if np.isfinite(value) and value > 0:
            last = i
    nxt = np.full(n, -1, dtype=int)
    last = -1
    for i in range(n - 1, -1, -1):
        nxt[i] = last
        if np.isfinite(values[i]) and values[i] > 0:
            last = i
    bad = np.zeros(n, dtype=bool)
    for i, value in enumerate(values):
        before_i, after_i = previous[i], nxt[i]
        if before_i < 0 or after_i < 0 or not np.isfinite(value) or value <= 0:
            continue
        before, after = values[before_i], values[after_i]
        jumped = value > before * factor or value * factor < before
        came_back = 0.5 < (after / before) < 2.0
        bad[i] = jumped and came_back
    return bad


def _blank_vendor_spikes(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop prints that sit far from the prices around them.

    A few dead tickers in this file flip between a penny and a normal price.
    A real collapse moves over many sessions and stays near its own recent
    prices, so those days are kept. The check looks at neighboring prints only
    to throw out a vendor error, and the thrown-out day is never traded.
    """
    frame = frame.sort_values(["symbol", "date"])
    bad = np.zeros(len(frame), dtype=bool)
    for _, prices in frame.groupby("symbol", sort=False)["adjusted_close"]:
        bad[frame.index.get_indexer(prices.index)] = _neighbor_spike(prices)
    frame.loc[bad, ["close", "adjusted_close", "volume"]] = np.nan
    return frame


def ensure_price_file(dest: Path) -> Path:
    """Download the delisted-inclusive price file if it is not already here."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(PRICE_URL, dest)
    return dest


def load_price_frames(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return adjusted close, volume, and raw dollar volume.

    Adjusted close is the return series, so splits and dividends stay in the
    result. Volume in this file is already split-adjusted, so dollar volume is
    adjusted close times volume. Raw close times that volume would count one
    trade again for every later split. A non-positive price is treated as
    "no longer trading."
    """
    frame = pd.read_csv(
        path,
        usecols=["date", "symbol", "close", "adjusted_close", "volume"],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    present = set(frame["symbol"].unique())
    drop = [
        short
        for short, long in DROP_WHEN_ALSO_PRESENT.items()
        if short in present and long in present
    ]
    if drop:
        frame = frame[~frame["symbol"].isin(drop)]
    # A non-positive price, or a row with no shares traded, is not a price
    # the book can use. Several dead tickers print a normal price on a day
    # when the recorded volume is zero.
    # 999,999.9999 is a vendor ceiling, not a price. No name in this file
    # trades there; the print repeats day after day once the series overflows.
    sentinel = frame["adjusted_close"] >= 999_999
    frame.loc[sentinel, ["close", "adjusted_close", "volume"]] = np.nan
    dead = (frame["close"] <= 0) | (frame["volume"].fillna(0) <= 0)
    frame.loc[dead, ["close", "adjusted_close", "volume"]] = np.nan
    frame = _repair_scale_breaks(frame)
    frame = _blank_vendor_spikes(frame)
    frame["dollar"] = frame["adjusted_close"] * frame["volume"]

    def _wide(column: str) -> pd.DataFrame:
        wide = frame.pivot(index="date", columns="symbol", values=column)
        wide = wide.sort_index()
        wide.columns.name = None
        return wide

    return _wide("adjusted_close"), _wide("volume"), _wide("dollar")


def with_benchmark(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    spy_close: pd.Series,
    spy_volume: pd.Series,
    benchmark: str = "SPY",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add the benchmark and keep only days it traded.

    The delisted file has no SPY. The benchmark series is the adjusted close
    already stored for the index, so the comparison is total return against
    total return.
    """
    spy_close = spy_close.dropna().sort_index()
    spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)
    spy_volume = spy_volume.sort_index()
    spy_volume.index = pd.to_datetime(spy_volume.index).tz_localize(None)
    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    idx = close.index.intersection(spy_close.index)
    close = close.reindex(idx)
    volume = volume.reindex(idx)
    dollar_volume = dollar_volume.reindex(idx)
    close[benchmark] = spy_close.reindex(idx).to_numpy()
    volume[benchmark] = spy_volume.reindex(idx).to_numpy()
    dollar_volume[benchmark] = close[benchmark] * volume[benchmark]
    return close, volume, dollar_volume


def listed_through(close: pd.DataFrame, benchmark: str, tail: int = 5) -> list[str]:
    """Symbols that still have a price at the end of the file, plus the benchmark."""
    cutoff = close.index[max(len(close) - tail, 0)]
    keep: list[str] = []
    for name in close.columns:
        if name == benchmark:
            keep.append(name)
            continue
        last = close[name].last_valid_index()
        if last is not None and last >= cutoff:
            keep.append(name)
    return keep
