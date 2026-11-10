"""Cash dividends for the four names that were spliced into the long replay.

Amounts are dollars per share on the day they were paid, in the share units
that traded that day. Splits are taken out later, with the prices. The cash
is placed on the payable date from the filing, which is after the ex-date, so
the return shows up a few sessions late.

Lehman Brothers (fiscal year ends November 30). Common dividends were paid on
the last weekday of February, May, August, and November. Annual rates are the
ones declared in the 10-Ks, restated for the October 2000 split until the
April 2006 split, and restated for both splits after that. Pre-split payments
are stored at the cash a holder of the old share received. The February and
May 2008 payments are the $0.17 in the 2008 10-Qs. The September 10, 2008
release cut the dividend and the firm filed for bankruptcy on September 15,
so nothing after May 2008 is included.

Bear Stearns (fiscal year ends November 30). The annual declared amounts are
from the 10-K highlights: $0.55, $0.60, $0.62, $0.74, $0.85, $1.00, $1.12,
and $1.28 for fiscal 2000 through 2007. Fiscal 2006 and 2007 were $0.28 and
$0.32 each quarter. Earlier years are the annual amount split into four equal
payments. Payable dates that an 8-K names are used. The other payments use
the 27th, which is the middle of the named dates (the 25th through the 29th).
The January 25, 2008 payment was declared on December 20, 2007. Nothing after
the March 2008 sale is included. The January 15 dates in the 10-K are the
preferred-stock dates, not the common stock.

General Motors (the company that went through bankruptcy, not the 2010
listing). The 10-Ks show $0.50 a quarter on the $1-2/3 par common stock for
2000 through 2005, and $0.25 a quarter for 2006 and 2007. The board approved
the $0.25 payment for the quarters ended March 31 and June 30, 2008. On
July 15, 2008 it suspended later common dividends. Payable dates follow the
2000 filings, which paid on June 10 and December 9: the 10th of March, June,
September, and December. A weekend moves to the next session when the prices
are joined.

WorldCom common stock is not in this table. The $0.60 figure in the 2002
10-K was an MCI tracking-stock dividend, and the board cancelled it. No
separate common dividend for these years was in the filings that were read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DIVIDEND_FILE = Path(__file__).with_name("gap_dividends.csv")

# Lehman cash per traded share, keyed by (year, month) of the payable month.
_LEHMAN = {
    (2000, 2): 0.11,
    (2000, 5): 0.11,
    (2000, 8): 0.11,
    (2000, 11): 0.055,
}
for _year, _quarter in (
    (2001, 0.07),
    (2002, 0.09),
    (2003, 0.12),
    (2004, 0.16),
    (2005, 0.20),
):
    for _month in (2, 5, 8, 11):
        _LEHMAN[(_year, _month)] = _quarter
_LEHMAN[(2006, 2)] = 0.24
for _month in (5, 8, 11):
    _LEHMAN[(2006, _month)] = 0.12
for _month in (2, 5, 8, 11):
    _LEHMAN[(2007, _month)] = 0.15
_LEHMAN[(2008, 2)] = 0.17
_LEHMAN[(2008, 5)] = 0.17

# Bear fiscal-year quarterly cash. Calendar year of the payment matches the
# fiscal year that ends that November.
_BEAR_QUARTER = {
    2000: 0.55 / 4,
    2001: 0.60 / 4,
    2002: 0.62 / 4,
    2003: 0.74 / 4,
    2004: 0.85 / 4,
    2005: 0.25,
    2006: 0.28,
    2007: 0.32,
}
_BEAR_NAMED = {
    "2005-04-29": 0.25,
    "2005-10-28": 0.25,
    "2006-01-27": 0.28,
    "2007-01-26": 0.32,
    "2007-07-27": 0.32,
    "2008-01-25": 0.32,
}


def _last_weekday(year: int, month: int) -> pd.Timestamp:
    day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    while day.day_of_week >= 5:
        day -= pd.Timedelta(days=1)
    return day


def _bear_rows() -> list[tuple[pd.Timestamp, str, float]]:
    rows: list[tuple[pd.Timestamp, str, float]] = []
    named_months = {(pd.Timestamp(day).year, pd.Timestamp(day).month) for day in _BEAR_NAMED}
    for day, cash in _BEAR_NAMED.items():
        rows.append((pd.Timestamp(day), "BSC", float(cash)))
    for year, cash in _BEAR_QUARTER.items():
        for month in (1, 4, 7, 10):
            if (year, month) in named_months:
                continue
            rows.append((pd.Timestamp(year, month, 27), "BSC", float(cash)))
    return rows


def _gm_rows() -> list[tuple[pd.Timestamp, str, float]]:
    rows: list[tuple[pd.Timestamp, str, float]] = []
    for year in range(2000, 2008):
        cash = 0.50 if year <= 2005 else 0.25
        for month in (3, 6, 9, 12):
            rows.append((pd.Timestamp(year, month, 10), "GM", cash))
    rows.append((pd.Timestamp(2008, 3, 10), "GM", 0.25))
    rows.append((pd.Timestamp(2008, 6, 10), "GM", 0.25))
    return rows


def cash_dividends() -> pd.DataFrame:
    """One row per payment: date, symbol, cash."""
    rows: list[tuple[pd.Timestamp, str, float]] = []
    for (year, month), cash in _LEHMAN.items():
        rows.append((_last_weekday(year, month), "LEH", float(cash)))
    rows.extend(_bear_rows())
    rows.extend(_gm_rows())
    frame = pd.DataFrame(rows, columns=["date", "symbol", "cash"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    if frame.duplicated(["date", "symbol"]).any():
        raise RuntimeError("Two dividends landed on the same symbol and day.")
    return frame


def write_dividend_file(path: Path | None = None) -> Path:
    path = DIVIDEND_FILE if path is None else Path(path)
    frame = cash_dividends()
    out = frame.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return path
