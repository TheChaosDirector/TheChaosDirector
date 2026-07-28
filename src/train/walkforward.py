"""Walk-forward self-training: the agent's practice-then-exam loop.

The timeline is chopped into repeating folds:

    [ train ~3 years ][ embargo gap ][ exam ~6 months ]
                       [ train ~3 years ][ embargo gap ][ exam ... ]

For each fold the agent trains ONLY on the train window, gets frozen, and is
graded on the exam window it has never seen. The embargo gap between them
stops information from bleeding across the boundary (labels near the edge
overlap in time). The fold whose *exam* grade is best becomes the champion.

Grades earned on the train window ("in-sample") are reported too, but only as
a sanity check — a huge gap between in-sample and exam scores is the classic
signature of overfitting, and the referee looks at exactly that.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.data.lake import Lake
from src.env.portfolio_env import PortfolioEnv
from src.features.factory import FeaturePanel, build_features, warmup_days
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


def make_folds(n_dates: int, cfg: Config, warmup: int) -> list[Fold]:
    train_len = int(cfg.training.train_years * TRADING_DAYS_PER_YEAR)
    test_len = int(cfg.training.test_months * TRADING_DAYS_PER_MONTH)
    embargo = cfg.training.embargo_days

    folds: list[Fold] = []
    cursor = warmup
    i = 0
    # Reserve one final day because the env needs t+1 prices.
    while cursor + train_len + embargo + test_len < n_dates - 1:
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


def rollout(model, env: PortfolioEnv) -> pd.DataFrame:
    """Play the frozen policy through the env once, deterministically."""
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    return env.results()


def _make_env(close, panel, cfg: Config, start: int, end: int, cost_multiplier: float = 1.0):
    return PortfolioEnv(
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        start=start,
        end=end,
        cost_multiplier=cost_multiplier,
    )


def _benchmark_returns(close: pd.DataFrame, benchmark: str, start: int, end: int) -> pd.Series:
    bench = close[benchmark].iloc[start : end + 1]
    return bench.pct_change().dropna()


def run_training(cfg: Config) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    lake = Lake(cfg)
    close = lake.close
    panel = build_features(close, lake.volume, cfg.features, lake.benchmark)
    warmup = warmup_days(cfg.features)

    folds = make_folds(len(close), cfg, warmup)
    if not folds:
        raise RuntimeError(
            "Not enough history for even one train/test fold. "
            "Download more history or shrink training.train_years / test_months."
        )
    log.info("Walk-forward plan: %d folds over %d trading days", len(folds), len(close))

    records: list[dict] = []
    for fold in folds:
        t0 = time.time()
        train_env = Monitor(_make_env(close, panel, cfg, fold.train_start, fold.train_end))

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=cfg.training.learning_rate,
            n_steps=cfg.training.n_env_steps,
            seed=cfg.training.seed + fold.index,
            verbose=0,
        )
        model.learn(total_timesteps=cfg.training.total_timesteps)

        # Practice-test grade (in-sample) — for the overfitting gap check only.
        is_journal = rollout(model, _make_env(close, panel, cfg, fold.train_start, fold.train_end))
        is_card = scorecard(is_journal["net_return"], is_journal["turnover"])

        # Exam grade (out-of-sample) — the one that counts.
        oos_journal = rollout(model, _make_env(close, panel, cfg, fold.test_start, fold.test_end))
        oos_card = scorecard(oos_journal["net_return"], oos_journal["turnover"])

        bench = _benchmark_returns(close, lake.benchmark, fold.test_start, fold.test_end - 1)
        bench_card = scorecard(bench)

        metrics = {
            "fold": fold.index,
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
        save_fold(cfg, fold.index, model, metrics)
        records.append(metrics)
        log.info(
            "Fold %d done in %.0fs — exam Sharpe %.2f (practice %.2f), bench %.2f",
            fold.index,
            metrics["train_seconds"],
            oos_card["sharpe"],
            is_card["sharpe"],
            bench_card["sharpe"],
        )

        oos_journal.to_csv(fold_dir(cfg, fold.index) / "oos_journal.csv")

    save_experiments(cfg, records)

    # Champion = best exam (out-of-sample) Sharpe, ties broken by return.
    best = max(
        records,
        key=lambda r: (r["out_of_sample"]["sharpe"], r["out_of_sample"]["total_return"]),
    )
    champion = {
        "fold": best["fold"],
        "model_path": str(fold_dir(cfg, best["fold"]) / "model.zip"),
        "tickers": panel.tickers,
        "feature_names": panel.names,
        "selected_by": "out_of_sample.sharpe",
        "metrics": best,
        "config_path": cfg.config_path,
    }
    save_champion(cfg, champion)
    log.info("Champion: fold %d (exam Sharpe %.2f)", best["fold"], best["out_of_sample"]["sharpe"])
    return champion
