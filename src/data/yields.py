"""Short-term US Treasury yield: the hurdle a competing copy has to beat.

``^IRX`` is the 13-week T-bill discount rate, quoted in percent. One day of
cash is credited ``yield / 100 / 252``. That is the simple daily hurdle, not
a traded bond price (no duration, no mark-to-market). Sitting in cash earns
this rate. Taking stock risk is only rewarded when the book clears it.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config

TRADING_DAYS = 252


def tbill_path(cfg: Config) -> Path:
    return cfg.lake_dir / "yields" / "tbill.parquet"


def synthetic_tbill_percent(dates: pd.DatetimeIndex) -> pd.Series:
    """Deterministic yield path so offline runs still see a changing hurdle."""
    n = len(dates)
    x = np.linspace(0.0, np.pi, n)
    annual = 2.5 + 2.0 * np.sin(x - 0.5)
    return pd.Series(annual, index=pd.DatetimeIndex(dates), name="annual_percent")


def download_tbill_percent(symbol: str, start: str, end: str | None) -> pd.Series:
    """Download the bill yield. Values are annualized percent, not decimals."""
    import yfinance as yf

    raw = yf.download(
        symbol,
        start=start,
        end=end or date.today().isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            f"No Treasury yield data for {symbol}. The competing copy needs a "
            "short-term US yield (^IRX, the 13-week bill) before it can train."
        )
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    series = pd.Series(close, copy=False).astype(float)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "annual_percent"
    if series.empty:
        raise RuntimeError(f"Treasury series {symbol} was empty after cleaning.")
    return series


def write_tbill_series(cfg: Config, annual: pd.Series, source: str) -> Path:
    path = tbill_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(annual.index).tz_localize(None),
            "annual_percent": np.asarray(annual, dtype=float),
        }
    )
    df.to_parquet(path, index=False)
    (path.parent / "tbill.json").write_text(
        json.dumps(
            {
                "source": source,
                "symbol": cfg.compete.symbol,
                "n": int(len(df)),
                "note": "Annualized percent. Daily hurdle is percent/100/252.",
            },
            indent=2,
        )
    )
    return path


def align_tbill_daily(cfg: Config, index: pd.DatetimeIndex) -> pd.Series:
    """Daily simple T-bill return aligned to ``index``.

    Quotes are forward-filled onto the requested dates. Days before the first
    quote earn zero rather than borrowing a future rate.
    """
    path = tbill_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"No Treasury yield file at {path}. Download with the yield-compete "
            "config so the copy can train against short-term US yields."
        )
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    quoted = (
        frame.drop_duplicates("date")
        .set_index("date")["annual_percent"]
        .astype(float)
        .sort_index()
    )
    wanted = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None)
    combined_index = quoted.index.union(wanted)
    filled = quoted.reindex(combined_index).sort_index().ffill()
    annual = filled.reindex(wanted)
    # Leading gap: no quote yet. Do not back-fill from the future.
    annual = annual.fillna(0.0)
    daily = annual / 100.0 / TRADING_DAYS
    daily.index = wanted
    daily.name = "tbill_daily"
    return daily


def annualized_observation(daily_yield: float) -> float:
    """Scale the daily hurdle back to an annualized decimal for the observation."""
    if not np.isfinite(daily_yield):
        return 0.0
    return float(daily_yield) * TRADING_DAYS
