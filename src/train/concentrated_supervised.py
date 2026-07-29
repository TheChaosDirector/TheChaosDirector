"""Concentrated supervised learner: rank names, then size a small book.

Plain English:
- Predict next-day excess return vs SPY from lagged clues (ranker).
- Blend those scores with a simple momentum baseline (dumb-but-strong prior).
- Keep top-k names, force ≥ min_gross invested, cap per-name weight.
- Grade folds under *harsher* trading costs so churn looks expensive.
- If several exam folds are green, deploy an ensemble of them.
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
from src.features.factory import build_features, warmup_days
from src.metrics.scorecard import scorecard
from src.registry.store import fold_dir, registry_dir, save_champion, save_experiments
from src.train.walkforward import (
    Fold,
    _benchmark_returns,
    _make_portfolio_env,
    make_folds,
    rollout,
)

log = logging.getLogger(__name__)


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x)
    out = (x - m) / s
    return np.where(np.isfinite(out), out, 0.0)


class ConcentratedRankerAdapter:
    """Produces portfolio action logits from per-name predicted excess + momentum."""

    def __init__(
        self,
        model,
        n_assets: int,
        n_features: int,
        feature_names: list[str],
        *,
        momentum_feature: str = "mom_63d",
        baseline_mix: float = 0.45,
        max_names: int = 6,
    ):
        self.model = model
        self.n_assets = n_assets
        self.n_features = n_features
        self.feature_names = list(feature_names)
        self.momentum_feature = momentum_feature
        self.baseline_mix = float(baseline_mix)
        self.max_names = int(max_names)
        if momentum_feature in self.feature_names:
            self._mom_idx = self.feature_names.index(momentum_feature)
        else:
            # Fall back to first momentum-like feature if window missing.
            self._mom_idx = next(
                (i for i, n in enumerate(self.feature_names) if n.startswith("mom_")),
                0,
            )

    def score_names(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        flat = obs[: self.n_assets * self.n_features]
        X = flat.reshape(self.n_assets, self.n_features)
        pred = np.asarray(self.model.predict(X), dtype=np.float64)
        mom = X[:, self._mom_idx].astype(np.float64)
        mix = np.clip(self.baseline_mix, 0.0, 1.0)
        return (1.0 - mix) * _zscore_1d(pred) + mix * _zscore_1d(mom)

    def predict(self, obs, deterministic: bool = True):
        scores = self.score_names(obs)
        action = np.full(self.n_assets + 1, -8.0, dtype=np.float32)
        # Keep only top-k logits high; cash stays very low (min_gross forces invest).
        k = min(self.max_names, self.n_assets)
        keep = np.argpartition(scores, -k)[-k:]
        action[keep] = (scores[keep] * 4.0).astype(np.float32)
        action[-1] = -10.0
        return action, None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "concentrated_ranker",
                "model": self.model,
                "n_assets": self.n_assets,
                "n_features": self.n_features,
                "feature_names": self.feature_names,
                "momentum_feature": self.momentum_feature,
                "baseline_mix": self.baseline_mix,
                "max_names": self.max_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConcentratedRankerAdapter":
        blob = joblib.load(path)
        return cls(
            blob["model"],
            blob["n_assets"],
            blob["n_features"],
            blob["feature_names"],
            momentum_feature=blob.get("momentum_feature", "mom_63d"),
            baseline_mix=blob.get("baseline_mix", 0.45),
            max_names=blob.get("max_names", 6),
        )


class ConcentratedEnsembleAdapter:
    """Average ranker scores across green folds, then same logit mapping."""

    def __init__(self, adapters: list[ConcentratedRankerAdapter]):
        if not adapters:
            raise ValueError("ensemble needs at least one adapter")
        self.adapters = adapters
        self.n_assets = adapters[0].n_assets
        self.n_features = adapters[0].n_features
        self.max_names = adapters[0].max_names
        self.baseline_mix = adapters[0].baseline_mix
        self.momentum_feature = adapters[0].momentum_feature
        self.feature_names = adapters[0].feature_names
        self._mom_idx = adapters[0]._mom_idx

    def score_names(self, obs: np.ndarray) -> np.ndarray:
        scores = None
        for ad in self.adapters:
            s = ad.score_names(obs)
            scores = s if scores is None else scores + s
        return scores / len(self.adapters)

    def predict(self, obs, deterministic: bool = True):
        scores = self.score_names(obs)
        action = np.full(self.n_assets + 1, -8.0, dtype=np.float32)
        k = min(self.max_names, self.n_assets)
        keep = np.argpartition(scores, -k)[-k:]
        action[keep] = (scores[keep] * 4.0).astype(np.float32)
        action[-1] = -10.0
        return action, None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": "concentrated_ensemble", "adapters": self.adapters}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ConcentratedEnsembleAdapter":
        blob = joblib.load(path)
        return cls(blob["adapters"])


def _build_xy(panel, closes: np.ndarray, start: int, end: int, bench_idx: int):
    """(day, ticker) rows: lagged features → next-day excess vs SPY."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    n_assets = len(panel.tickers)
    for t in range(start, end):
        if t + 1 >= len(closes):
            break
        feats = panel.values[t]
        px0 = closes[t]
        px1 = closes[t + 1]
        valid = np.isfinite(px0) & np.isfinite(px1) & (px0 > 0)
        if not valid.any() or not valid[bench_idx]:
            continue
        rets = np.where(valid, px1 / px0 - 1.0, np.nan)
        bench_ret = float(rets[bench_idx])
        for i in range(n_assets):
            if not valid[i]:
                continue
            xs.append(feats[i])
            ys.append(float(rets[i]) - bench_ret)
    if not xs:
        raise RuntimeError("No training rows for concentrated supervised fold")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float64)


def _mom_feature_name(cfg: Config, panel_names: list[str]) -> str:
    want = f"mom_{cfg.concentrated.momentum_window}d"
    if want in panel_names:
        return want
    moms = [n for n in panel_names if n.startswith("mom_")]
    return moms[-1] if moms else panel_names[0]


def train_supervised_fold(cfg: Config, fold: Fold) -> dict:
    t0 = time.time()
    lake = Lake(cfg)
    close = lake.close
    volume = lake.volume
    panel = build_features(close, volume, cfg.features, lake.benchmark)
    closes = close.values.astype(np.float64)
    bench_idx = panel.tickers.index(lake.benchmark)
    mom_feat = _mom_feature_name(cfg, panel.names)

    X, y = _build_xy(panel, closes, fold.train_start, fold.train_end, bench_idx)
    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=180,
        min_samples_leaf=80,
        l2_regularization=2.0,
        random_state=cfg.training.seed + fold.index,
    )
    model.fit(X, y)

    adapter = ConcentratedRankerAdapter(
        model,
        n_assets=len(panel.tickers),
        n_features=panel.n_features,
        feature_names=panel.names,
        momentum_feature=mom_feat,
        baseline_mix=cfg.concentrated.baseline_mix,
        max_names=cfg.concentrated.max_names,
    )

    harsh = float(cfg.concentrated.train_cost_multiplier)
    is_env = _make_portfolio_env(
        close,
        panel,
        cfg,
        fold.train_start,
        fold.train_end,
        cost_multiplier=harsh,
        benchmark=lake.benchmark,
    )
    oos_env = _make_portfolio_env(
        close,
        panel,
        cfg,
        fold.test_start,
        fold.test_end,
        cost_multiplier=harsh,
        benchmark=lake.benchmark,
    )
    is_journal = rollout(adapter, is_env)
    oos_journal = rollout(adapter, oos_env)
    is_card = scorecard(is_journal["net_return"], is_journal.get("turnover"))
    oos_card = scorecard(oos_journal["net_return"], oos_journal.get("turnover"))
    bench = _benchmark_returns(close, lake.benchmark, fold.test_start, fold.test_end)
    bench_card = scorecard(bench)

    metrics = {
        "fold": fold.index,
        "mode": cfg.mode,
        "learner": "concentrated_supervised",
        "train_cost_multiplier": harsh,
        "baseline_mix": cfg.concentrated.baseline_mix,
        "momentum_feature": mom_feat,
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
        "oos_excess_mean": float(oos_journal["excess_return"].mean())
        if "excess_return" in oos_journal.columns
        else float(oos_card["total_return"] - bench_card["total_return"]),
        "avg_gross_exposure_oos": float(oos_journal["gross_exposure"].mean())
        if "gross_exposure" in oos_journal.columns
        else None,
        "avg_turnover_oos": float(oos_journal["turnover"].mean())
        if "turnover" in oos_journal.columns
        else None,
    }

    d = fold_dir(cfg, fold.index)
    d.mkdir(parents=True, exist_ok=True)
    adapter.save(d / "model.joblib")
    (d / "learner.txt").write_text("concentrated_supervised")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    oos_journal.to_csv(d / "oos_journal.csv")
    return metrics


def _worker(payload: dict) -> dict:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from src.config import load_config

    cfg = load_config(payload["config_path"], mode=payload["mode"])
    fold = Fold(**payload["fold"])
    metrics = train_supervised_fold(cfg, fold)
    log.info(
        "Concentrated supervised fold %d done in %.0fs — exam Sharpe %.2f turnover %.2f",
        metrics["fold"],
        metrics["train_seconds"],
        metrics["out_of_sample"]["sharpe"],
        metrics.get("avg_turnover_oos") or 0.0,
    )
    return metrics


def _is_green(r: dict) -> bool:
    return (
        float(r.get("oos_excess_mean") or 0.0) > 0
        and r["out_of_sample"]["total_return"] > 0
        and r["out_of_sample"]["sharpe"] > 0
    )


def run_supervised_training(cfg: Config) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from dataclasses import asdict
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    lake = Lake(cfg)
    close = lake.close
    volume = lake.volume
    panel = build_features(close, volume, cfg.features, lake.benchmark)
    warmup = warmup_days(cfg.features)
    folds = make_folds(len(close), cfg, warmup, need_next_day=True)
    if not folds:
        raise RuntimeError("Not enough history for concentrated supervised folds")

    n_workers = max(1, min(int(cfg.training.n_workers), len(folds), os.cpu_count() or 1))
    log.info(
        "Concentrated supervised plan: %d folds | %d tickers | workers=%d | train_cost_x=%.1f | baseline_mix=%.2f",
        len(folds),
        len(panel.tickers),
        n_workers,
        cfg.concentrated.train_cost_multiplier,
        cfg.concentrated.baseline_mix,
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
            futs = [pool.submit(_worker, p) for p in payloads]
            for fut in as_completed(futs):
                records.append(fut.result())
        records.sort(key=lambda r: r["fold"])

    save_experiments(cfg, records)

    green = [r for r in records if _is_green(r)]
    pool = green or [
        r
        for r in records
        if float(r.get("oos_excess_mean") or 0.0) > 0 and r["out_of_sample"]["total_return"] > 0
    ] or records
    best = max(
        pool,
        key=lambda r: (
            float(r.get("oos_excess_mean") or 0.0),
            r["out_of_sample"]["total_return"],
            r["out_of_sample"]["sharpe"],
        ),
    )

    if len(green) >= 2:
        adapters = [
            ConcentratedRankerAdapter.load(fold_dir(cfg, r["fold"]) / "model.joblib")
            for r in green
        ]
        ensemble = ConcentratedEnsembleAdapter(adapters)
        model_path = str(registry_dir(cfg) / "ensemble.joblib")
        ensemble.save(model_path)
        selected_by = f"ensemble_of_{len(green)}_green_folds"
        champ_fold = best["fold"]
        learner = "concentrated_supervised_ensemble"
    else:
        model_path = str(fold_dir(cfg, best["fold"]) / "model.joblib")
        selected_by = "best_green_or_fallback"
        champ_fold = best["fold"]
        learner = "concentrated_supervised"

    champion = {
        "fold": champ_fold,
        "mode": "concentrated",
        "learner": learner,
        "model_path": model_path,
        "tickers": panel.tickers,
        "feature_names": panel.names,
        "selected_by": selected_by,
        "positive_oos_folds": len(green),
        "total_folds": len(records),
        "ensemble_folds": [r["fold"] for r in green] if len(green) >= 2 else None,
        "max_names": cfg.concentrated.max_names,
        "min_gross_exposure": cfg.concentrated.min_gross_exposure,
        "baseline_mix": cfg.concentrated.baseline_mix,
        "train_cost_multiplier": cfg.concentrated.train_cost_multiplier,
        "metrics": best,
        "config_path": cfg.config_path,
    }
    save_champion(cfg, champion)
    log.info(
        "Concentrated supervised champion: %s | green %d/%d | best excess %.5f exam Sharpe %.2f",
        selected_by,
        len(green),
        len(records),
        float(best.get("oos_excess_mean") or 0.0),
        best["out_of_sample"]["sharpe"],
    )
    return champion
