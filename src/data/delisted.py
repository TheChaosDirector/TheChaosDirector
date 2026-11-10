"""Prices for companies that later died, were bought, or changed ticker.

The curated list in ``universe.py`` is companies that still trade. A replay
built only from that list never holds Enron, Washington Mutual, or the other
names that went to about zero. This module reads a daily S&P 500 file that
keeps those names through their last trade.

Source: FINSABER (Apache-2.0), file
``all_sp500_prices_2000_2024_delisted_include.csv``.

That file never traded Lehman, WorldCom, Bear Stearns before the weekend it
was sold, or General Motors before the 2009 bankruptcy. Those four series are
read from ``gap_names.csv`` and added for the years they actually traded.
Lehman, Bear Stearns, and old General Motors also get the cash dividends
listed in ``gap_dividends.csv``. WorldCom's common stock does not: the $0.60
figure in its 2002 filing was a cancelled MCI tracking-stock dividend.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.gap_dividends import DIVIDEND_FILE

PRICE_URL = (
    "https://huggingface.co/datasets/finsaber-team/FINSABER-reproduce/"
    "resolve/main/data/price/all_sp500_prices_2000_2024_delisted_include.csv"
)

# The short ticker is already covered by the long one: META includes the years
# Facebook traded as FB, and GOOG covers the same days as the GOOGL share class.
# Holding both would count one company twice.
DROP_WHEN_ALSO_PRESENT = {"FB": "META", "GOOGL": "GOOG"}

# Daily traded close and share volume for the four names missing above.
# Closes are the prices people paid. Cash dividends are a separate file and
# are applied as a total-return series. Dollar volume stays on the traded close.
GAP_FILE = Path(__file__).with_name("gap_names.csv")

# Old GM kept trading under MTLQQ after the ticker change, through the
# bankruptcy. Those rows are the same company as GM. They are not the
# company that started trading in November 2010.
OLD_GM_TICKER = "MTLQQ"

# Lehman split 2-for-1 on both of these mornings. The close falls by about
# half and the next days stay there. Scaling the earlier prices takes the
# split out and leaves the ordinary move from that day.
KNOWN_SPLITS = {
    ("LEH", "2000-10-23"): 2.0,
    ("LEH", "2006-05-01"): 2.0,
}


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


def total_return_close(close: np.ndarray, cash: np.ndarray) -> np.ndarray:
    """Backward dividend adjustment. The last traded close stays put.

    Walking backward, a dividend lowers every earlier price by
    ``close / (close + cash)``. The return into the payable date is then
    ``(close + cash) / previous close - 1``. A day with no cash is unchanged.
    """
    adjusted = np.array(close, dtype=np.float64, copy=True)
    paid = np.asarray(cash, dtype=np.float64)
    factor = 1.0
    for i in range(len(close) - 1, -1, -1):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            adjusted[i] = np.nan
            continue
        adjusted[i] = price * factor
        dividend = paid[i] if i < len(paid) and np.isfinite(paid[i]) else 0.0
        if dividend > 0:
            factor *= price / (price + dividend)
    return adjusted


def apply_known_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Take the recorded stock splits out of a raw close series.

    Earlier prices, share counts, and cash dividends are scaled so a 2-for-1
    is not a loss. Dollars traded stay the same, because the price is divided
    by the same factor the volume is multiplied by. The move left on the split
    morning stays in the series. Dividend cash is divided by the same factor,
    so the yield is unchanged.
    """
    frame = frame.sort_values(["symbol", "date"]).copy()
    close = frame["close"].to_numpy(dtype=np.float64, copy=True)
    volume = frame["volume"].to_numpy(dtype=np.float64, copy=True)
    has_cash = "cash" in frame.columns
    cash = frame["cash"].to_numpy(dtype=np.float64, copy=True) if has_cash else None
    symbols = frame["symbol"].astype(str).to_numpy()
    dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    for (symbol, day), factor in KNOWN_SPLITS.items():
        hits = np.flatnonzero((symbols == symbol) & (dates == day))
        if len(hits) != 1:
            continue
        i = int(hits[0])
        if i == 0 or not (close[i - 1] > 0 and close[i] > 0):
            continue
        # Refuse a date that is not actually a split in this file.
        if abs((close[i - 1] / close[i]) / factor - 1.0) > 0.15:
            continue
        earlier = (symbols == symbol) & (np.arange(len(frame)) < i)
        close[earlier] = close[earlier] / factor
        volume[earlier] = volume[earlier] * factor
        if cash is not None:
            cash[earlier] = cash[earlier] / factor
    frame["close"] = close
    frame["volume"] = volume
    if cash is not None:
        frame["cash"] = cash
    return frame


def _keep_flat_closes(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep a close that repeats the prior close when no share count was recorded.

    WorldCom printed $0.83 for three sessions with a zero share count, then
    $0.06 on a session with more than a billion shares. Dropping the $0.83
    days would skip that loss. The prior session's share count keeps the
    close. A day the price actually changed is not filled in.
    """
    frame = frame.sort_values(["symbol", "date"]).copy()
    volume = frame["volume"].to_numpy(dtype=np.float64, copy=True)
    close = frame["close"].to_numpy(dtype=np.float64)
    symbols = frame["symbol"].astype(str).to_numpy()
    for i in range(1, len(frame)):
        if symbols[i] != symbols[i - 1] or volume[i] > 0 or not close[i] > 0:
            continue
        if close[i] == close[i - 1] and volume[i - 1] > 0:
            volume[i] = volume[i - 1]
    frame["volume"] = volume
    return frame


def _snap_payment(pay: pd.Timestamp, traded: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Move a payable date onto a session the name actually traded.

    The payable date is kept when it is already a session. A market holiday
    uses the next session, up to five calendar days later, and otherwise the
    previous session inside that same window.
    """
    traded = pd.DatetimeIndex(traded).sort_values()
    if pay in traded:
        return pay
    later = traded[traded > pay]
    if len(later) and (later[0] - pay).days <= 5:
        return later[0]
    earlier = traded[traded < pay]
    if len(earlier) and (pay - earlier[-1]).days <= 5:
        return earlier[-1]
    return None


def _attach_dividends(frame: pd.DataFrame, dividend_file: Path | None) -> pd.DataFrame:
    """Add a cash column. Payments land on a session the name traded."""
    frame = frame.copy()
    frame["cash"] = 0.0
    path = DIVIDEND_FILE if dividend_file is None else Path(dividend_file)
    if not path.exists():
        return frame
    dividends = pd.read_csv(path)
    if dividends.empty:
        return frame
    dividends["date"] = pd.to_datetime(dividends["date"])
    dividends["cash"] = dividends["cash"].astype(float)
    snapped: list[dict] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        payments = dividends[dividends["symbol"] == symbol]
        if payments.empty:
            continue
        traded = pd.DatetimeIndex(group["date"])
        for pay, cash in zip(payments["date"], payments["cash"]):
            if not cash > 0:
                continue
            day = _snap_payment(pd.Timestamp(pay), traded)
            if day is None:
                continue
            snapped.append({"date": day, "symbol": symbol, "paid": float(cash)})
    if not snapped:
        return frame
    paid = pd.DataFrame(snapped).groupby(["date", "symbol"], as_index=False)["paid"].sum()
    frame = frame.merge(paid, on=["date", "symbol"], how="left")
    frame["cash"] = frame["paid"].fillna(0.0)
    return frame.drop(columns=["paid"])


def _with_total_return(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        cash = group["cash"].to_numpy(dtype=np.float64) if "cash" in group.columns else np.zeros(len(group))
        group["adjusted_close"] = total_return_close(group["close"].to_numpy(dtype=np.float64), cash)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def load_gap_names(
    path: Path | None = None,
    dividend_file: Path | None = None,
) -> pd.DataFrame:
    """The four missing names, with splits taken out and cash dividends applied.

    ``close`` is the traded close after the split scale. ``adjusted_close`` is
    the total-return series: the last traded close is unchanged, and earlier
    levels are a little lower so a payable date earns the cash. ``cash`` is
    the dividend in those same split-adjusted share units.
    """
    path = GAP_FILE if path is None else Path(path)
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = frame["close"].astype(float)
    frame["volume"] = frame["volume"].astype(float)
    frame = frame.dropna(subset=["date", "symbol", "close", "volume"])
    frame = _keep_flat_closes(frame)
    frame = frame[(frame["close"] > 0) & (frame["volume"] > 0)]
    frame = _attach_dividends(frame, dividend_file)
    frame = apply_known_splits(frame)
    return _with_total_return(frame)


def _append_gap_names(frame: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """Add gap rows on dates the main file does not already cover.

    Old GM's later ticker is renamed to GM, and any of its rows on or after
    the new company's first day are left out. A gap of missing days stays
    between the two, so the replay does not earn a return from the penny
    stock into the new listing.
    """
    extra = gaps.copy()
    extra.loc[extra["symbol"] == OLD_GM_TICKER, "symbol"] = "GM"
    pieces: list[pd.DataFrame] = []
    for symbol, group in extra.groupby("symbol", sort=False):
        existing = frame.loc[frame["symbol"] == symbol, "date"]
        if len(existing):
            group = group[group["date"] < existing.min()]
        if group.empty:
            continue
        pieces.append(group)
    if not pieces:
        return frame
    add = pd.concat(pieces, ignore_index=True)
    if "adjusted_close" not in add.columns:
        add["adjusted_close"] = add["close"]
    # Dollar volume uses the traded close. The total-return level is lower
    # in the past, and using it would understate how much stock changed hands.
    add["dollar_price"] = add["close"]
    add = add[["date", "symbol", "close", "adjusted_close", "volume", "dollar_price"]]
    return pd.concat([frame, add], ignore_index=True)


def ensure_price_file(dest: Path) -> Path:
    """Download the delisted-inclusive price file if it is not already here."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(PRICE_URL, dest)
    return dest


def load_price_frames(
    path: Path,
    *,
    include_gaps: bool = True,
    gap_file: Path | None = None,
    dividend_file: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return adjusted close, volume, and dollar volume.

    Adjusted close is the return series, so splits and dividends stay in the
    result. Volume in the main file is already split-adjusted, so dollar
    volume there is adjusted close times volume. Raw close times that volume
    would count one trade again for every later split. A non-positive price
    is treated as "no longer trading."

    Lehman, WorldCom, Bear Stearns before the sale, and old General Motors are
    read from ``gap_names.csv`` and added on the days this file does not
    already have them. Their return series includes the cash dividends in
    ``gap_dividends.csv``. Dollar volume for those rows uses the traded close,
    not the dividend-adjusted level.
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
    if include_gaps:
        frame = _append_gap_names(frame, load_gap_names(gap_file, dividend_file=dividend_file))
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
    if "dollar_price" in frame.columns:
        frame.loc[frame["adjusted_close"].isna(), "dollar_price"] = np.nan
        basis = frame["dollar_price"].where(frame["dollar_price"].notna(), frame["adjusted_close"])
    else:
        basis = frame["adjusted_close"]
    frame["dollar"] = basis * frame["volume"]

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
