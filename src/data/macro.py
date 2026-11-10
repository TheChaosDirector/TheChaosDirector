"""Inflation and unemployment, released on a delay.

The monthly file stores the CPI index and the unemployment rate. A feature
row for a session only sees a release dated before that session:

- CPI for a month is treated as public on the 16th of the next month.
- Unemployment for a month is treated as public on the 8th of the next month
  (the first Friday of the next month is never later than the 7th).

The bill minus inflation uses a Treasury quote from before that same session.
Nothing here is shifted a second time by the feature factory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MACRO_FILE = Path(__file__).with_name("macro_monthly.csv")


def load_macro_monthly(path: Path | None = None) -> pd.DataFrame:
    path = MACRO_FILE if path is None else Path(path)
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def _asof(index: pd.DatetimeIndex, known_on, values) -> pd.Series:
    """Latest value whose release date is strictly before each index date."""
    quotes = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(np.asarray(known_on)).reset_index(drop=True)),
            "value": np.asarray(values, dtype=float),
        }
    ).dropna()
    quotes["date"] = quotes["date"].astype("datetime64[ns]")
    quotes = quotes.sort_values("date").drop_duplicates("date", keep="last")
    wanted = pd.DataFrame({"date": pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None)})
    wanted["date"] = wanted["date"].astype("datetime64[ns]")
    if quotes.empty:
        return pd.Series(index=pd.DatetimeIndex(wanted["date"]), dtype=float)
    merged = pd.merge_asof(
        wanted.sort_values("date"),
        quotes,
        on="date",
        direction="backward",
        allow_exact_matches=False,
    )
    return merged.set_index("date").reindex(wanted["date"].to_numpy())["value"]


def macro_known_before(
    index: pd.DatetimeIndex,
    monthly: pd.DataFrame | None = None,
    bill_annual_percent: pd.Series | None = None,
) -> pd.DataFrame:
    """Columns ``cpi_yoy``, ``unrate``, and ``real_bill``, one row per session.

    ``cpi_yoy`` and ``real_bill`` are decimals (0.03 is 3 percent).
    ``unrate`` is the unemployment rate as a decimal. ``real_bill`` is the
    lagged bill minus lagged inflation. Days before the first release are
    missing; the feature factory turns those into zeros.
    """
    monthly = load_macro_monthly() if monthly is None else monthly.copy()
    monthly = monthly.sort_values("date")
    measured = pd.to_datetime(monthly["date"])
    next_month = measured + pd.offsets.DateOffset(months=1)
    cpi = monthly["cpi"].astype(float)
    yoy = cpi / cpi.shift(12) - 1.0
    unrate = monthly["unrate_percent"].astype(float) / 100.0
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None))
    out["cpi_yoy"] = _asof(out.index, next_month + pd.Timedelta(days=15), yoy).to_numpy()
    out["unrate"] = _asof(out.index, next_month + pd.Timedelta(days=7), unrate).to_numpy()
    if bill_annual_percent is None:
        out["real_bill"] = float("nan")
    else:
        bill = bill_annual_percent.dropna().sort_index()
        lagged = _asof(out.index, bill.index, bill.to_numpy(dtype=float))
        out["real_bill"] = lagged.to_numpy() / 100.0 - out["cpi_yoy"].to_numpy()
    out.index.name = None
    return out
