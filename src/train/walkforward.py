"""Walk-forward self-training with optional parallel folds.

The timeline is chopped into repeating folds:

    [ train ~N years ][ embargo gap ][ exam ~M months ]
                       [ train ~N years ][ embargo gap ][ exam ... ]

Each fold trains ONLY on its train window, freezes, and is graded on the exam
window it has never seen. Folds are independent, so they can run in parallel
via ProcessPoolExecutor when training.n_workers > 1.

Supports two modes:
- portfolio — multi-name allocation brain (existing)
- daytrade  — must-pick one name, buy open / sell close, with a "forced" signal
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import pandas as pd

from src.config import Config, load_config
from src.data.lake import Lake
from src.env.daytrade_env import DaytradeEnv
from src.env.portfolio_env import PortfolioEnv
from src.features.factory import build_features, warmup_days
from src.metrics.scorecard import scorecard
from src.registry.store import fold_dir, save_champion, save_experiments, save_fold

log = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21


@dataclass
class Fold:
    index: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive


def make_folds(
    n_dates: int,
    cfg: Config,
    warmup: int,
    *,
    need_next_day: bool = True,
) -> list[Fold]:
    """Build walk-forward folds.

    ``need_next_day=True`` (portfolio): env needs prices at t+1, so the last
    usable index is n_dates-2.
    ``need_next_day=False`` (daytrade): open/close are same-day, last usable
    index is n_dates-1.
    """
    train_len = int(cfg.training.train_years * TRADING_DAYS_PER_YEAR)
    test_len = int(cfg.training.test_months * TRADING_DAYS_PER_MONTH)
    embargo = cfg.training.embargo_days
    # Maximum exclusive end index the env will accept.
    max_end = (n_dates - 1) if need_next_day else n_dates

    folds: list[Fold] = []
    cursor = warmup
    i = 0
    while cursor + train_len + embargo + test_len <= max_end:
        folds.append(
            Fold(
                index=i,
                train_start=cursor,
                train_end=cursor + train_len,
                test_start=cursor + train_len + embargo,
                test_end=cursor + train_len + embargo + test_len,
            )
        )
        cursor += test_len
        i += 1
    return folds


def rollout(model, env) -> pd.DataFrame:
    """Play the frozen policy through the env once, deterministically."""
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    return env.results()


def _make_portfolio_env(close, panel, cfg: Config, start: int, end: int, cost_multiplier: float = 1.0):
    return PortfolioEnv(
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        start=start,
        end=end,
        cost_multiplier=cost_multiplier,
    )


def _make_daytrade_env(open_, close, panel, cfg: Config, start: int, end: int, cost_multiplier: float = 1.0):
    return DaytradeEnv(
        open_=open_,
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        daytrade=cfg.daytrade,
        start=start,
        end=end,
        cost_multiplier=cost_multiplier,
    )


def _benchmark_returns(close: pd.DataFrame, benchmark: str, start: int, end: int) -> pd.Series:
    # end is exclusive for the env; benchmark series uses closes over the window.
    bench = close[benchmark].iloc[start:end]
    if len(bench) < 2:
        # Daytrade grades same-day open→close; approximate bench with close-to-close
        # over the same dates by extending one bar when available.
        bench = close[benchmark].iloc[start : min(end + 1, len(close))]
    return bench.pct_change().dropna()


def train_one_fold(cfg: Config, fold: Fold) -> dict:
    """Train + grade a single fold. Safe to call from a worker process."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    t0 = time.time()
    lake = Lake(cfg)
    close = lake.close
    panel = build_features(close, lake.volume, cfg.features, lake.benchmark)

    if cfg.mode == "daytrade":
        open_ = lake.open.reindex_like(close)
        train_env = Monitor(_make_daytrade_env(open_, close, panel, cfg, fold.train_start, fold.train_end))
        is_env = _make_daytrade_env(open_, close, panel, cfg, fold.train_start, fold.train_end)
        oos_env = _make_daytrade_env(open_, close, panel, cfg, fold.test_start, fold.test_end)
    else:
        train_env = Monitor(_make_portfolio_env(close, panel, cfg, fold.train_start, fold.train_end))
        is_env = _make_portfolio_env(close, panel, cfg, fold.train_start, fold.train_end)
        oos_env = _make_portfolio_env(close, panel, cfg, fold.test_start, fold.test_end)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.training.learning_rate,
        n_steps=cfg.training.n_env_steps,
        seed=cfg.training.seed + fold.index,
        verbose=0,
    )
    model.learn(total_timesteps=cfg.training.total_timesteps)

    is_journal = rollout(model, is_env)
    oos_journal = rollout(model, oos_env)
    turnover = is_journal["turnover"] if "turnover" in is_journal.columns else None
    oos_turnover = oos_journal["turnover"] if "turnover" in oos_journal.columns else None
    is_card = scorecard(is_journal["net_return"], turnover)
    oos_card = scorecard(oos_journal["net_return"], oos_turnover)

    bench = _benchmark_returns(close, lake.benchmark, fold.test_start, fold.test_end)
    bench_card = scorecard(bench)

    metrics = {
        "fold": fold.index,
        "mode": cfg.mode,
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
    }
    if cfg.mode == "daytrade" and "forced" in oos_journal.columns:
        metrics["forced_rate_oos"] = float(oos_journal["forced"].mean())

    save_fold(cfg, fold.index, model, metrics)
    oos_journal.to_csv(fold_dir(cfg, fold.index) / "oos_journal.csv")
    return metrics


def _worker(payload: dict) -> dict:
    """Process entrypoint: reload config and train one fold."""
    import os

    # Keep each parallel worker on one BLAS/torch thread so N workers don't
    # thrash each other into a slower-than-serial mess.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(payload["config_path"], mode=payload["mode"])
    fold = Fold(**payload["fold"])
    metrics = train_one_fold(cfg, fold)
    log.info(
        "Fold %d done in %.0fs — exam Sharpe %.2f (practice %.2f), bench %.2f",
        metrics["fold"],
        metrics["train_seconds"],
        metrics["out_of_sample"]["sharpe"],
        metrics["in_sample"]["sharpe"],
        metrics["benchmark_oos"]["sharpe"],
    )
    return metrics


def run_training(cfg: Config) -> dict:
    # Parent process: also stay polite if serial, and set a default before spawn.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    lake = Lake(cfg)
    close = lake.close
    panel = build_features(close, lake.volume, cfg.features, lake.benchmark)
    warmup = warmup_days(cfg.features)
    need_next_day = cfg.mode != "daytrade"

    folds = make_folds(len(close), cfg, warmup, need_next_day=need_next_day)
    if not folds:
        raise RuntimeError(
            "Not enough history for even one train/test fold. "
            "Download more history or shrink training.train_years / test_months."
        )

    n_workers = max(1, int(cfg.training.n_workers))
    n_workers = min(n_workers, len(folds), os.cpu_count() or 1)
    log.info(
        "Walk-forward plan: %d folds over %d days | mode=%s | workers=%d",
        len(folds),
        len(close),
        cfg.mode,
        n_workers,
    )

    records: list[dict] = []
    if n_workers == 1:
        for fold in folds:
            metrics = train_one_fold(cfg, fold)
            records.append(metrics)
            log.info(
                "Fold %d done in %.0fs — exam Sharpe %.2f (practice %.2f), bench %.2f",
                metrics["fold"],
                metrics["train_seconds"],
                metrics["out_of_sample"]["sharpe"],
                metrics["in_sample"]["sharpe"],
                metrics["benchmark_oos"]["sharpe"],
            )
    else:
        payloads = [
            {"config_path": cfg.config_path, "mode": cfg.mode, "fold": asdict(fold)}
            for fold in folds
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, p): p["fold"]["index"] for p in payloads}
            for fut in as_completed(futures):
                metrics = fut.result()
                records.append(metrics)
                log.info(
                    "Fold %d finished (worker) in %.0fs — exam Sharpe %.2f",
                    metrics["fold"],
                    metrics["train_seconds"],
                    metrics["out_of_sample"]["sharpe"],
                )
        records.sort(key=lambda r: r["fold"])

    save_experiments(cfg, records)

    best = max(
        records,
        key=lambda r: (r["out_of_sample"]["sharpe"], r["out_of_sample"]["total_return"]),
    )
    champion = {
        "fold": best["fold"],
        "mode": cfg.mode,
        "model_path": str(fold_dir(cfg, best["fold"]) / "model.zip"),
        "tickers": panel.tickers,
        "feature_names": panel.names,
        "selected_by": "out_of_sample.sharpe",
        "metrics": best,
        "config_path": cfg.config_path,
    }
    save_champion(cfg, champion)
    log.info(
        "Champion (%s): fold %d (exam Sharpe %.2f)",
        cfg.mode,
        best["fold"],
        best["out_of_sample"]["sharpe"],
    )
    return champion
