"""Fill the data lake: bulk-download daily OHLCV bars to parquet files.

One parquet file per ticker under ``<lake_dir>/prices/<TICKER>.parquet`` with
columns: date, open, high, low, close, volume (close is dividend/split
adjusted). A ``manifest.json`` records what was downloaded and with which
filters, so later stages know exactly what data exists.

If the network is unavailable, ``--synthetic`` builds a statistically
plausible fake market (geometric random walks with a shared market factor)
so every downstream stage can still run.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.universe import get_universe

log = logging.getLogger(__name__)

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def prices_dir(cfg: Config) -> Path:
    return cfg.lake_dir / "prices"


def manifest_path(cfg: Config) -> Path:
    return cfg.lake_dir / "manifest.json"


def _write_ticker(cfg: Config, ticker: str, df: pd.DataFrame) -> None:
    out = prices_dir(cfg) / f"{ticker}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_parquet(out, index=False)


def _download_yahoo(cfg: Config, tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    end = cfg.data.end or date.today().isoformat()
    frames: dict[str, pd.DataFrame] = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        log.info("Downloading batch %d-%d of %d tickers", i + 1, i + len(batch), len(tickers))
        raw = yf.download(
            batch,
            start=cfg.data.start,
            end=end,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            continue
        for t in batch:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(sub.index).tz_localize(None),
                    "open": sub["Open"].values,
                    "high": sub["High"].values,
                    "low": sub["Low"].values,
                    "close": sub["Close"].values,
                    "volume": sub["Volume"].values,
                }
            )
            frames[t] = df
    return frames


def _synthetic_market(cfg: Config, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Fake but realistic market: shared market factor + per-name noise.

    The benchmark ticker gets a low-volatility 'index-like' path so
    buy-and-hold comparisons stay meaningful.
    """
    rng = np.random.default_rng(7)
    end = cfg.data.end or date.today().isoformat()
    dates = pd.bdate_range(cfg.data.start, end)
    n = len(dates)
    market = rng.normal(0.0004, 0.010, n)  # ~10% annual drift, ~16% vol

    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        beta = 1.0 if t == cfg.universe.benchmark else float(rng.uniform(0.5, 1.6))
        idio = 0.0 if t == cfg.universe.benchmark else float(rng.uniform(0.008, 0.02))
        rets = beta * market + rng.normal(0, idio, n)
        close = 100.0 * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.normal(0, 0.008, n))
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close * (1 - rng.normal(0, 0.004, n)),
                "high": close * (1 + intraday),
                "low": close * (1 - intraday),
                "close": close,
                "volume": rng.integers(2_000_000, 80_000_000, n).astype(float),
            }
        )
        frames[t] = df
    return frames


def _apply_filters(cfg: Config, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Keep tickers with enough history and enough traded dollar volume."""
    kept: dict[str, pd.DataFrame] = {}
    for t, df in frames.items():
        if len(df) < cfg.universe.min_history_days:
            log.info("Dropping %s: only %d days of history", t, len(df))
            continue
        med_dollar = float((df["close"] * df["volume"]).median())
        if med_dollar < cfg.data.min_dollar_volume and t != cfg.universe.benchmark:
            log.info("Dropping %s: median dollar volume %.0f too low", t, med_dollar)
            continue
        kept[t] = df

    if cfg.universe.benchmark not in kept and cfg.universe.benchmark in frames:
        kept[cfg.universe.benchmark] = frames[cfg.universe.benchmark]

    if len(kept) > cfg.universe.max_tickers:
        # Keep the most liquid names (benchmark always survives).
        liquidity = {
            t: float((df["close"] * df["volume"]).median()) for t, df in kept.items()
        }
        ranked = sorted(liquidity, key=liquidity.get, reverse=True)
        selected = set(ranked[: cfg.universe.max_tickers]) | {cfg.universe.benchmark}
        kept = {t: kept[t] for t in kept if t in selected}
    return kept


def run_download(cfg: Config, synthetic: bool = False) -> dict:
    tickers = get_universe(cfg.universe.name)
    if cfg.universe.benchmark not in tickers:
        tickers.append(cfg.universe.benchmark)

    if synthetic:
        frames = _synthetic_market(cfg, tickers)
        source = "synthetic"
    else:
        frames = _download_yahoo(cfg, tickers)
        source = "yahoo"

    frames = _apply_filters(cfg, frames)
    if cfg.universe.benchmark not in frames:
        raise RuntimeError(
            f"Benchmark {cfg.universe.benchmark} missing from downloaded data; "
            "cannot continue without a buy-and-hold comparison."
        )

    for t, df in frames.items():
        _write_ticker(cfg, t, df.sort_values("date").reset_index(drop=True))

    manifest = {
        "source": source,
        "universe": cfg.universe.name,
        "benchmark": cfg.universe.benchmark,
        "tickers": sorted(frames.keys()),
        "n_tickers": len(frames),
        "start": cfg.data.start,
        "end": cfg.data.end or date.today().isoformat(),
        "min_dollar_volume": cfg.data.min_dollar_volume,
    }
    manifest_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(cfg).write_text(json.dumps(manifest, indent=2))
    log.info("Lake ready: %d tickers from %s", len(frames), source)
    return manifest
