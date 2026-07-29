"""Supervised day-trade ranker: pick the name with the best predicted open→close.

Plain English:
- For every past morning, we know yesterday's clues (lagged features).
- After the day ends, we know what each stock actually did open→close.
- The model learns: "given yesterday's clues, which name tends to have a
  better open→close today?"
- At decision time it ranks names and must pick the top one (no cash).
- Forced flag: True when the top pick's predicted edge is weak (below a
  small threshold) — still trades, just admits the hand was forced.

Walk-forward folds, costs, and the referee still apply. This path exists
because pure RL was memorizing the practice window (huge practice Sharpe,
awful exam Sharpe). A ranker with the same anti-cheat timing is harder to
fool that way.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import Config
from src.data.lake import Lake
from src.env.daytrade_env import DaytradeEnv
from src.features.factory import build_features, warmup_days
from src.metrics.scorecard import scorecard
from src.registry.store import fold_dir, save_champion, save_experiments
from src.train.walkforward import (
    Fold,
    _daytrade_benchmark_returns,
    _subset_daytrade_universe,
    make_folds,
)

log = logging.getLogger(__name__)


class DaytradeRankerAdapter:
    """Duck-types a Stable-Baselines model enough for rollout / paper / backtest."""

    def __init__(self, model, n_assets: int, n_features: int, weak_edge: float = 0.0005):
        self.model = model
        self.n_assets = n_assets
        self.n_features = n_features
        self.weak_edge = weak_edge

    def predict(self, obs, deterministic: bool = True):
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        flat = obs[: self.n_assets * self.n_features]
        X = flat.reshape(self.n_assets, self.n_features)
        scores = self.model.predict(X)
        idx = int(np.argmax(scores))
        forced = 1 if float(scores[idx]) < self.weak_edge else 0
        return np.array([idx, forced], dtype=np.int64), None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "n_assets": self.n_assets,
                "n_features": self.n_features,
                "weak_edge": self.weak_edge,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DaytradeRankerAdapter":
        blob = joblib.load(path)
        return cls(blob["model"], blob["n_assets"], blob["n_features"], blob.get("weak_edge", 0.0005))


def _build_xy(panel, opens, closes, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack (day, ticker) rows: features → that ticker's open→close return."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    n_assets = len(panel.tickers)
    for t in range(start, end):
        feats = panel.values[t]  # (N, F), already lagged
        o = opens[t]
        c = closes[t]
        valid = np.isfinite(o) & np.isfinite(c) & (o > 0) & (c > 0)
        if not valid.any():
            continue
        rets = np.where(valid, c / o - 1.0, np.nan)
        for i in range(n_assets):
            if not valid[i]:
                continue
            xs.append(feats[i])
            ys.append(float(rets[i]))
    if not xs:
        raise RuntimeError("No training rows for supervised daytrade fold")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float64)


def _rollout_ranker(adapter: DaytradeRankerAdapter, env: DaytradeEnv) -> pd.DataFrame:
    from src.train.walkforward import rollout

    return rollout(adapter, env)


def train_supervised_fold(cfg: Config, fold: Fold) -> dict:
    t0 = time.time()
    lake = Lake(cfg)
    close = lake.close
    volume = lake.volume
    open_ = lake.open.reindex_like(close)
    close, volume, open_ = _subset_daytrade_universe(
        close, volume, open_, lake.benchmark, cfg.daytrade.max_names
    )
    panel = build_features(close, volume, cfg.features, lake.benchmark)
    opens = open_.values.astype(np.float64)
    closes = close.values.astype(np.float64)

    X, y = _build_xy(panel, opens, closes, fold.train_start, fold.train_end)
    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=cfg.training.seed + fold.index,
    )
    model.fit(X, y)
    adapter = DaytradeRankerAdapter(
        model,
        n_assets=len(panel.tickers),
        n_features=panel.n_features,
        weak_edge=0.0005,
    )

    is_env = DaytradeEnv(
        open_=open_,
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        daytrade=cfg.daytrade,
        start=fold.train_start,
        end=fold.train_end,
        benchmark=lake.benchmark,
    )
    oos_env = DaytradeEnv(
        open_=open_,
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        daytrade=cfg.daytrade,
        start=fold.test_start,
        end=fold.test_end,
        benchmark=lake.benchmark,
    )
    is_journal = _rollout_ranker(adapter, is_env)
    oos_journal = _rollout_ranker(adapter, oos_env)
    is_card = scorecard(is_journal["net_return"], is_journal.get("turnover"))
    oos_card = scorecard(oos_journal["net_return"], oos_journal.get("turnover"))
    bench = _daytrade_benchmark_returns(open_, close, lake.benchmark, fold.test_start, fold.test_end)
    bench_card = scorecard(bench)

    metrics = {
        "fold": fold.index,
        "mode": cfg.mode,
        "learner": "supervised",
        "train_start": str(close.index[fold.train_start].date()),
        "train_end": str(close.index[fold.train_end - 1].date()),
        "test_start": str(close.index[fold.test_start].date()),
        "test_end": str(close.index[fold.test_end - 1].date()),
        "embargo_days": cfg.training.embargo_days,
        "train_start_idx": fold.train_start,
        "train_end_idx": fold.train_end,
        "test_start_idx": fold.test_start,
        "test_end_idx": fold.test_end,
        "in_sample": is_card,
        "out_of_sample": oos_card,
        "benchmark_oos": bench_card,
        "train_seconds": round(time.time() - t0, 1),
        "n_tickers": len(panel.tickers),
        "forced_rate_oos": float(oos_journal["forced"].mean()) if "forced" in oos_journal.columns else None,
        "oos_excess_mean": float(oos_journal["excess_return"].mean()) if "excess_return" in oos_journal.columns else None,
    }

    d = fold_dir(cfg, fold.index)
    d.mkdir(parents=True, exist_ok=True)
    adapter.save(d / "model.joblib")
    # Marker so loaders know this isn't an SB3 zip.
    (d / "learner.txt").write_text("supervised")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    oos_journal.to_csv(d / "oos_journal.csv")
    # Keep a tiny stub zip path unused; champion points at joblib.
    return metrics


def _supervised_worker(payload: dict) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from src.config import load_config

    cfg = load_config(payload["config_path"], mode=payload["mode"])
    fold = Fold(**payload["fold"])
    metrics = train_supervised_fold(cfg, fold)
    log.info(
        "Supervised fold %d done in %.0fs — exam Sharpe %.2f (practice %.2f), bench %.2f",
        metrics["fold"],
        metrics["train_seconds"],
        metrics["out_of_sample"]["sharpe"],
        metrics["in_sample"]["sharpe"],
        metrics["benchmark_oos"]["sharpe"],
    )
    return metrics


def run_supervised_training(cfg: Config) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from dataclasses import asdict
    import os

    lake = Lake(cfg)
    close = lake.close
    volume = lake.volume
    open_ = lake.open.reindex_like(close)
    close, volume, open_ = _subset_daytrade_universe(
        close, volume, open_, lake.benchmark, cfg.daytrade.max_names
    )
    panel = build_features(close, volume, cfg.features, lake.benchmark)
    warmup = warmup_days(cfg.features)
    folds = make_folds(len(close), cfg, warmup, need_next_day=False)
    if not folds:
        raise RuntimeError("Not enough history for supervised daytrade folds")

    n_workers = max(1, min(int(cfg.training.n_workers), len(folds), os.cpu_count() or 1))
    log.info(
        "Supervised daytrade plan: %d folds | %d tickers | workers=%d",
        len(folds),
        len(panel.tickers),
        n_workers,
    )

    records: list[dict] = []
    if n_workers == 1:
        for fold in folds:
            records.append(train_supervised_fold(cfg, fold))
    else:
        payloads = [
            {"config_path": cfg.config_path, "mode": cfg.mode, "fold": asdict(fold)}
            for fold in folds
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_supervised_worker, p) for p in payloads]
            for fut in as_completed(futs):
                records.append(fut.result())
        records.sort(key=lambda r: r["fold"])

    save_experiments(cfg, records)

    positive = [
        r
        for r in records
        if r["out_of_sample"]["total_return"] > 0 and r["out_of_sample"]["sharpe"] > 0
    ]
    pool = positive or records
    best = max(pool, key=lambda r: (r["out_of_sample"]["sharpe"], r["out_of_sample"]["total_return"]))
    selected_by = "best_positive_oos_sharpe" if positive else "best_oos_sharpe_no_positive_folds"

    champion = {
        "fold": best["fold"],
        "mode": "daytrade",
        "learner": "supervised",
        "model_path": str(fold_dir(cfg, best["fold"]) / "model.joblib"),
        "tickers": panel.tickers,
        "feature_names": panel.names,
        "selected_by": selected_by,
        "positive_oos_folds": len(positive),
        "total_folds": len(records),
        "metrics": best,
        "config_path": cfg.config_path,
    }
    save_champion(cfg, champion)
    log.info(
        "Supervised champion: fold %d | green folds %d/%d | exam Sharpe %.2f return %.1f%%",
        best["fold"],
        len(positive),
        len(records),
        best["out_of_sample"]["sharpe"],
        best["out_of_sample"]["total_return"] * 100,
    )
    return champion
