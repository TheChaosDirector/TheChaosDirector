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

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.lake import Lake
from src.env.daytrade_env import DaytradeEnv
from src.env.portfolio_env import PortfolioEnv
from src.features.factory import build_daytrade_features, build_features, warmup_days
from src.metrics.scorecard import scorecard, total_return
from src.registry.protect import assert_training_copy_is_isolated
from src.registry.store import fold_dir, registry_dir, save_champion, save_experiments, save_fold
from src.train.compete import compete_env_kwargs, write_diary_grade
from src.train.experience_blend import (
    build_experience_split,
    should_promote_experience_tune,
)

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


def _make_portfolio_env(
    close,
    panel,
    cfg: Config,
    start: int,
    end: int,
    cost_multiplier: float = 1.0,
    experience: dict | None = None,
    volume=None,
):
    eligible = None
    if volume is not None and cfg.mode == "portfolio":
        from src.data.eligibility import point_in_time_eligible

        eligible = point_in_time_eligible(
            close,
            volume.reindex(index=close.index, columns=close.columns),
            min_history_days=cfg.universe.min_history_days,
            min_dollar_volume=cfg.data.min_dollar_volume,
            max_names=cfg.universe.max_tickers,
            benchmark=cfg.universe.benchmark,
        )
    return PortfolioEnv(
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        start=start,
        end=end,
        cost_multiplier=cost_multiplier,
        eligible=eligible,
        **compete_env_kwargs(cfg, close.index, experience, close=close),
    )


def _hints_in_window(hints: dict | None, index, start: int, end: int) -> dict | None:
    if not hints:
        return None
    window = {}
    for i in range(start, end):
        key = pd.Timestamp(index[i]).date()
        if key in hints:
            window[key] = hints[key]
    return window or None


def _excess_total(journal: pd.DataFrame, dates: list | None) -> float:
    if journal.empty or "excess_return" not in journal.columns:
        return 0.0
    series = journal["excess_return"]
    if dates:
        want = set(dates)
        mask = [pd.Timestamp(i).date() in want for i in series.index]
        series = series.loc[mask]
    if len(series) == 0:
        return 0.0
    return total_return(series)


def _make_daytrade_env(
    open_,
    close,
    panel,
    cfg: Config,
    start: int,
    end: int,
    cost_multiplier: float = 1.0,
    randomize_episodes: bool = False,
):
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
        benchmark=cfg.universe.benchmark,
        randomize_episodes=randomize_episodes,
    )


def _benchmark_returns(close: pd.DataFrame, benchmark: str, start: int, end: int) -> pd.Series:
    # end is exclusive for the env; benchmark series uses closes over the window.
    bench = close[benchmark].iloc[start:end]
    if len(bench) < 2:
        bench = close[benchmark].iloc[start : min(end + 1, len(close))]
    return bench.pct_change().dropna()


def _daytrade_benchmark_returns(open_: pd.DataFrame, close: pd.DataFrame, benchmark: str, start: int, end: int) -> pd.Series:
    """Same-day open→close returns for the benchmark (fair daytrade yardstick)."""
    o = open_[benchmark].iloc[start:end]
    c = close[benchmark].iloc[start:end]
    return (c / o - 1.0).replace([np.inf, -np.inf], np.nan).dropna()


def _subset_daytrade_universe(close, volume, open_, benchmark: str, max_names: int | None):
    """Keep benchmark + the most liquid names so the daytrader can actually learn."""
    if not max_names or max_names >= close.shape[1]:
        return close, volume, open_
    dollar = (close * volume).median()
    ranked = [t for t in dollar.sort_values(ascending=False).index if t != benchmark]
    keep = [benchmark] + ranked[: max(0, max_names - 1)]
    keep = [t for t in keep if t in close.columns]
    return close[keep], volume[keep], open_[keep]


def train_one_fold(cfg: Config, fold: Fold) -> dict:
    """Train + grade a single fold. Safe to call from a worker process."""
    from stable_baselines3.common.monitor import Monitor

    t0 = time.time()
    lake = Lake(cfg)
    close = lake.close
    volume = lake.volume
    panel = None
    experience = None

    if cfg.mode == "daytrade":
        open_ = lake.open.reindex_like(close)
        close, volume, open_ = _subset_daytrade_universe(
            close, volume, open_, lake.benchmark, cfg.daytrade.max_names
        )
        panel = build_daytrade_features(close, volume, open_, cfg.features, lake.benchmark)
        train_env = Monitor(
            _make_daytrade_env(
                open_, close, panel, cfg, fold.train_start, fold.train_end, randomize_episodes=True
            )
        )
        is_env = _make_daytrade_env(open_, close, panel, cfg, fold.train_start, fold.train_end)
        oos_env = _make_daytrade_env(open_, close, panel, cfg, fold.test_start, fold.test_end)
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    else:
        panel = build_features(close, volume, cfg.features, lake.benchmark)
        if cfg.compete.enabled and cfg.compete.replay_experience:
            from src.data.yields import align_tbill_daily

            daily = align_tbill_daily(cfg, close.index)
            split = build_experience_split(cfg, list(panel.tickers), daily)
            experience = _hints_in_window(
                split.train_hints, close.index, fold.train_start, fold.train_end
            )
        train_env = Monitor(
            _make_portfolio_env(
                close, panel, cfg, fold.train_start, fold.train_end,
                experience=experience, volume=volume,
            )
        )
        # Grade on the raw hurdle. The diary bonus is a training nudge, not part of the exam.
        is_env = _make_portfolio_env(close, panel, cfg, fold.train_start, fold.train_end, volume=volume)
        oos_env = _make_portfolio_env(close, panel, cfg, fold.test_start, fold.test_end, volume=volume)
        if cfg.compete.enabled and cfg.compete.repeated_edge:
            policy_kwargs = dict(net_arch=[64, 64])
        elif cfg.compete.enabled and (cfg.compete.reward == "both" or cfg.compete.focus == "market"):
            policy_kwargs = dict(net_arch=[128, 128])
        else:
            policy_kwargs = dict(net_arch=[64, 64])
        open_ = None

    model = _new_ppo(cfg, train_env, seed=cfg.training.seed + fold.index, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=cfg.training.total_timesteps)

    is_journal = rollout(model, is_env)
    oos_journal = rollout(model, oos_env)
    turnover = is_journal["turnover"] if "turnover" in is_journal.columns else None
    oos_turnover = oos_journal["turnover"] if "turnover" in oos_journal.columns else None
    is_card = scorecard(is_journal["net_return"], turnover)
    oos_card = scorecard(oos_journal["net_return"], oos_turnover)

    if cfg.mode == "daytrade":
        bench = _daytrade_benchmark_returns(open_, close, lake.benchmark, fold.test_start, fold.test_end)
    else:
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
        "n_tickers": len(panel.tickers),
    }
    if cfg.mode == "daytrade" and "forced" in oos_journal.columns:
        metrics["forced_rate_oos"] = float(oos_journal["forced"].mean())
    if cfg.mode == "daytrade" and "excess_return" in oos_journal.columns:
        metrics["oos_excess_mean"] = float(oos_journal["excess_return"].mean())
    if cfg.compete.enabled and "excess_return" in oos_journal.columns:
        metrics["excess_oos"] = scorecard(oos_journal["excess_return"])
        metrics["treasury_oos"] = scorecard(oos_journal["rf"])
        metrics["experience_hints_in_train"] = 0 if experience is None else len(experience)
    if "benchmark_return" in oos_journal.columns and float(oos_journal["benchmark_return"].abs().sum()) > 0:
        metrics["spy_oos"] = scorecard(oos_journal["benchmark_return"])
        metrics["spy_excess_oos"] = scorecard(oos_journal["net_return"] - oos_journal["benchmark_return"])

    save_fold(cfg, fold.index, model, metrics)
    oos_journal.to_csv(fold_dir(cfg, fold.index) / "oos_journal.csv")
    return metrics


def _new_ppo(cfg: Config, env, *, seed: int, policy_kwargs: dict):
    """One PPO learner. The dual-hurdle copy uses a wider net and a real batch."""
    from stable_baselines3 import PPO

    n_steps = max(64, cfg.training.n_env_steps)
    if cfg.compete.enabled and cfg.compete.repeated_edge:
        batch = min(128, max(64, n_steps // 4))
        ent = 0.01
    elif cfg.compete.enabled and (cfg.compete.reward == "both" or cfg.compete.focus == "market"):
        batch = min(256, max(64, n_steps // 4))
        ent = 0.005
    else:
        batch = min(64, max(32, n_steps // 4))
        ent = 0.02 if cfg.mode == "daytrade" else 0.0
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.training.learning_rate,
        n_steps=n_steps,
        batch_size=batch,
        ent_coef=ent,
        seed=seed,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )


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
    assert_training_copy_is_isolated(cfg)
    if cfg.compete.enabled and cfg.mode != "portfolio":
        raise RuntimeError("Treasury competition is wired for portfolio mode only.")

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
    volume = lake.volume
    if cfg.mode == "daytrade":
        open_ = lake.open.reindex_like(close)
        close, volume, open_ = _subset_daytrade_universe(
            close, volume, open_, lake.benchmark, cfg.daytrade.max_names
        )
        panel = build_daytrade_features(close, volume, open_, cfg.features, lake.benchmark)
    else:
        panel = build_features(close, volume, cfg.features, lake.benchmark)
    warmup = warmup_days(cfg.features)
    need_next_day = cfg.mode != "daytrade"

    from src.train.holdout import describe_seal, fold_history_length

    if cfg.compete.enabled and cfg.compete.final_holdout_days > 0 and cfg.mode == "portfolio":
        log.info("%s", describe_seal(close.index, cfg))
        if cfg.compete.repeated_edge:
            log.info(
                "Repeated-edge rule is locked before this run: the stitched exams must beat "
                "SPY after costs, and at least half of those years must finish ahead. "
                "The sealed year is not scored."
            )

    folds = make_folds(
        fold_history_length(len(close), cfg),
        cfg,
        warmup,
        need_next_day=need_next_day,
    )
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

    if cfg.compete.enabled and cfg.compete.repeated_edge and cfg.mode == "portfolio":
        from src.train.robust import apply_repeated_edge_gate

        diary = write_diary_grade(cfg, close.index)
        if diary and diary.get("plain_english"):
            log.info("%s", diary["plain_english"])
        return apply_repeated_edge_gate(cfg, close, panel, records, volume=volume)

    # Consistency-first champion for daytrade: prefer exam windows that finished
    # green (positive return + positive Sharpe). Fall back to best Sharpe if none.
    if cfg.mode == "daytrade":
        positive = [
            r
            for r in records
            if r["out_of_sample"]["total_return"] > 0 and r["out_of_sample"]["sharpe"] > 0
        ]
        if positive:
            best = max(
                positive,
                key=lambda r: (r["out_of_sample"]["sharpe"], r["out_of_sample"]["total_return"]),
            )
            selected_by = "best_positive_oos_sharpe"
        else:
            best = max(
                records,
                key=lambda r: (r["out_of_sample"]["sharpe"], r["out_of_sample"]["total_return"]),
            )
            selected_by = "best_oos_sharpe_no_positive_folds"
        n_pos = len(positive)
        log.info(
            "Daytrade consistency: %d/%d exam folds finished green",
            n_pos,
            len(records),
        )
    elif cfg.compete.enabled and cfg.compete.focus == "market" and "spy_oos" in records[0]:
        def _gap(row: dict) -> float:
            return row["out_of_sample"]["total_return"] - row["spy_oos"]["total_return"]

        ahead = [r for r in records if _gap(r) > 0]
        pool = ahead or records
        best = max(pool, key=lambda r: (_gap(r), r["out_of_sample"]["sharpe"]))
        selected_by = (
            "pre_holdout_exam_beat_spy" if ahead else "pre_holdout_exam_closest_to_spy_none_ahead"
        )
        n_pos = len(ahead)
        log.info("Pre-holdout exams ahead of SPY after costs: %d/%d", len(ahead), len(records))
    elif cfg.compete.enabled and cfg.compete.reward == "both" and "excess_oos" in records[0]:
        both = [
            r
            for r in records
            if r["out_of_sample"]["total_return"] > r["benchmark_oos"]["total_return"]
            and r["out_of_sample"]["total_return"] > r.get("treasury_oos", {}).get("total_return", 0.0)
        ]
        pool = both or records
        best = max(
            pool,
            key=lambda r: (r["excess_oos"]["sharpe"], r["out_of_sample"]["total_return"]),
        )
        selected_by = (
            "pre_holdout_exam_beat_spy_and_tbill"
            if both
            else "pre_holdout_exam_excess_vs_tbill_none_beat_both"
        )
        n_pos = len(both)
        log.info(
            "Pre-holdout exams that beat both SPY and the bill: %d/%d",
            len(both),
            len(records),
        )
    elif cfg.compete.enabled and "excess_oos" in records[0]:
        best = max(
            records,
            key=lambda r: (r["excess_oos"]["sharpe"], r["excess_oos"]["total_return"]),
        )
        selected_by = "out_of_sample.excess_vs_tbill.sharpe"
        n_pos = None
    else:
        best = max(
            records,
            key=lambda r: (r["out_of_sample"]["sharpe"], r["out_of_sample"]["total_return"]),
        )
        selected_by = "out_of_sample.sharpe"
        n_pos = None

    champion = {
        "fold": best["fold"],
        "mode": cfg.mode,
        "model_path": str(fold_dir(cfg, best["fold"]) / "model.zip"),
        "tickers": panel.tickers,
        "feature_names": panel.names,
        "selected_by": selected_by,
        "positive_oos_folds": n_pos,
        "total_folds": len(records),
        "metrics": best,
        "config_path": cfg.config_path,
    }
    if cfg.compete.enabled:
        champion["objective"] = "beat_spy" if cfg.compete.focus == "market" else "beat_tbill"
        champion["yield_weight"] = cfg.compete.yield_weight
        champion["focus"] = cfg.compete.focus
        champion["hurdle_symbol"] = cfg.compete.symbol
        champion["learner"] = "ppo"
    save_champion(cfg, champion)
    log.info(
        "Champion (%s): fold %d (exam Sharpe %.2f, return %.1f%%) via %s",
        cfg.mode,
        best["fold"],
        best["out_of_sample"]["sharpe"],
        best["out_of_sample"]["total_return"] * 100,
        selected_by,
    )
    if cfg.compete.enabled:
        import json

        (registry_dir(cfg) / "champion_walkforward.json").write_text(
            json.dumps(champion, indent=2, default=str)
        )
        diary = write_diary_grade(cfg, close.index)
        if diary and diary.get("plain_english"):
            log.info("%s", diary["plain_english"])
        champion = _maybe_experience_tune(cfg, close, panel, champion, volume=volume)
        if cfg.compete.final_holdout_days > 0 and cfg.mode == "portfolio":
            from src.train.holdout import train_and_score_holdout

            holdout_champion = train_and_score_holdout(cfg, close, panel, volume=volume)
            if holdout_champion is not None:
                champion = holdout_champion
    return champion


def _maybe_experience_tune(cfg: Config, close: pd.DataFrame, panel, champion: dict, volume=None) -> dict:
    """Continue a copy of the walk-forward champion on saved decisions.

    The walk-forward zip and ``champion_walkforward.json`` stay in place. This
    pass replaces ``champion.json`` only when the held-out diary days improve
    and the full-sample excess versus bills does not fall apart.
    """
    import json

    from stable_baselines3 import PPO

    report: dict = {"attempted": False, "promoted": False}
    if cfg.compete.final_holdout_days > 0:
        report["reason"] = "locked holdout protocol does not replay saved decisions"
        _write_tune_report(cfg, report)
        return champion
    if cfg.compete.experience_timesteps <= 0 or not cfg.compete.replay_experience:
        report["reason"] = "experience replay pass is off"
        _write_tune_report(cfg, report)
        return champion

    from src.data.yields import align_tbill_daily

    daily = align_tbill_daily(cfg, close.index)
    split = build_experience_split(cfg, list(panel.tickers), daily)
    report["split"] = split.report
    if not split.train_hints:
        report["reason"] = "no saved decisions had enough weight inside this universe"
        _write_tune_report(cfg, report)
        log.info("Experience tune skipped: %s", report["reason"])
        return champion

    norm = pd.DatetimeIndex(pd.to_datetime(close.index)).tz_localize(None)
    hint_days = set(split.train_hints)
    positions = [i for i, ts in enumerate(norm) if ts.date() in hint_days]
    if not positions:
        report["reason"] = "training hints fall outside this price history"
        _write_tune_report(cfg, report)
        log.info("Experience tune skipped: %s", report["reason"])
        return champion

    start = max(int(np.min(positions)) - 5, 0)
    end = min(int(np.max(positions)) + 2, len(close) - 1)
    if end - start < 10:
        report["reason"] = "hint window is too short to tune"
        _write_tune_report(cfg, report)
        return champion

    hints = _hints_in_window(split.train_hints, close.index, start, end)
    if not hints:
        report["reason"] = "training hints fell outside the tunable window"
        _write_tune_report(cfg, report)
        return champion
    train_env = _make_portfolio_env(close, panel, cfg, start, end, experience=hints, volume=volume)
    model = PPO.load(champion["model_path"])
    model.set_env(train_env)
    log.info(
        "Experience tune: %d steps on %d saved decisions (%s to %s)",
        cfg.compete.experience_timesteps,
        len(hints or {}),
        close.index[start].date(),
        close.index[end - 1].date(),
    )
    model.learn(total_timesteps=int(cfg.compete.experience_timesteps))
    tune_dir = registry_dir(cfg) / "experience_tune"
    tune_dir.mkdir(parents=True, exist_ok=True)
    tuned_path = tune_dir / "model.zip"
    model.save(tuned_path)

    warmup = warmup_days(cfg.features)
    exam = _make_portfolio_env(close, panel, cfg, warmup, len(close) - 1, volume=volume)
    base_journal = rollout(PPO.load(champion["model_path"]), exam)
    tuned_journal = rollout(
        PPO.load(tuned_path),
        _make_portfolio_env(close, panel, cfg, warmup, len(close) - 1, volume=volume),
    )
    pretrained_stats = {
        "holdout_excess_total": _excess_total(base_journal, split.holdout_dates),
        "full_excess_total": _excess_total(base_journal, None),
    }
    tuned_stats = {
        "holdout_excess_total": _excess_total(tuned_journal, split.holdout_dates),
        "full_excess_total": _excess_total(tuned_journal, None),
    }
    promoted = should_promote_experience_tune(pretrained_stats, tuned_stats)
    report.update(
        {
            "attempted": True,
            "promoted": promoted,
            "pretrained": pretrained_stats,
            "tuned": tuned_stats,
            "tuned_model_path": str(tuned_path),
            "holdout_days": len(split.holdout_dates),
        }
    )
    if promoted:
        tuned_champion = dict(champion)
        tuned_champion["model_path"] = str(tuned_path)
        tuned_champion["parent_model_path"] = champion["model_path"]
        tuned_champion["selected_by"] = "experience_tune_holdout_excess"
        tuned_champion["learner"] = "ppo"
        save_champion(cfg, tuned_champion)
        champion = tuned_champion
        log.info(
            "Experience tune promoted. Holdout excess %.2f%% -> %.2f%%",
            pretrained_stats["holdout_excess_total"] * 100,
            tuned_stats["holdout_excess_total"] * 100,
        )
    else:
        log.info(
            "Experience tune kept as a side model. Walk-forward copy remains the champion. "
            "Holdout excess %.2f%% vs tuned %.2f%%",
            pretrained_stats["holdout_excess_total"] * 100,
            tuned_stats["holdout_excess_total"] * 100,
        )
    _write_tune_report(cfg, report)
    return champion


def _write_tune_report(cfg: Config, report: dict) -> None:
    import json

    path = registry_dir(cfg) / "experience_tune.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
