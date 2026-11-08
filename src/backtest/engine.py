"""Historical replay: grade a frozen champion over the full data lake.

Supports portfolio mode (multi-name, close-to-close) and daytrade mode
(must-pick one name, open→close).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.lake import Lake
from src.env.daytrade_env import DaytradeEnv
from src.features.factory import build_daytrade_features, build_features, warmup_days
from src.metrics.scorecard import plain_english, scorecard
from src.registry.store import load_champion_model
from src.train.compete import write_diary_grade
from src.train.walkforward import _make_portfolio_env, _subset_daytrade_universe, rollout


log = logging.getLogger(__name__)


def backtest_dir(cfg: Config) -> Path:
    return cfg.artifacts_dir / "backtest"


def run_backtest(cfg: Config, cost_multiplier: float = 1.0, tag: str = "base") -> dict:
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
        open_ = None
        panel = build_features(close, volume, cfg.features, lake.benchmark)

    model, meta = load_champion_model(cfg)
    if meta["tickers"] != panel.tickers:
        raise RuntimeError(
            "The champion was trained on a different ticker set than the current "
            "data lake. Re-run download + train with the same config."
        )

    start = warmup_days(cfg.features)

    if cfg.mode == "daytrade":
        open_ = lake.open.reindex_like(close)
        end = len(close)
        env = DaytradeEnv(
            open_=open_,
            close=close,
            panel=panel,
            costs=cfg.costs,
            risk=cfg.risk,
            daytrade=cfg.daytrade,
            start=start,
            end=end,
            cost_multiplier=cost_multiplier,
            benchmark=lake.benchmark,
        )
        journal = rollout(model, env)
        weights = None
        # Same-day open→close benchmark proxy: SPY's own open→close each day.
        bench_ret = (close[lake.benchmark] / open_[lake.benchmark] - 1.0).iloc[start:end]
        bench_ret = bench_ret.dropna()
    else:
        end = len(close) - 1
        env = _make_portfolio_env(
            close,
            panel,
            cfg,
            start,
            end,
            cost_multiplier=cost_multiplier,
            volume=volume,
        )
        journal = rollout(model, env)
        weights = env.weight_history()
        bench_ret = close[lake.benchmark].iloc[start : end + 1].pct_change().dropna()

    turnover = journal["turnover"] if "turnover" in journal.columns else None
    agent_card = scorecard(journal["net_return"], turnover)
    bench_card = scorecard(bench_ret)

    out = backtest_dir(cfg) / tag
    out.mkdir(parents=True, exist_ok=True)
    journal.to_csv(out / "journal.csv")
    if weights is not None:
        weights.to_csv(out / "weights.csv")
    bench_equity = (1.0 + bench_ret).cumprod()
    bench_equity.to_csv(out / "benchmark_equity.csv")

    summary = {
        "tag": tag,
        "mode": cfg.mode,
        "cost_multiplier": cost_multiplier,
        "champion_fold": meta["fold"],
        "period": [
            str(close.index[start].date()),
            str(close.index[min(end, len(close)) - 1].date()),
        ],
        "agent": agent_card,
        "benchmark": bench_card,
        "agent_says": plain_english(agent_card, "the agent"),
        "benchmark_says": plain_english(
            bench_card,
            f"{'SPY open→close' if cfg.mode == 'daytrade' else 'buy-and-hold ' + lake.benchmark}",
        ),
    }
    if cfg.compete.enabled and "excess_return" in journal.columns:
        excess_card = scorecard(journal["excess_return"])
        tbill_card = scorecard(journal["rf"])
        summary["excess_vs_tbill"] = excess_card
        summary["tbill"] = tbill_card
        summary["tbill_says"] = (
            f"Cash in 13-week bills compounded to {tbill_card['total_return'] * 100:.1f}% "
            f"over {tbill_card['n_days']} trading days "
            f"(about {tbill_card['cagr'] * 100:.1f}% a year). "
            "A bill yield barely wobbles day to day, so its Sharpe ratio is not a useful comparison."
        )
        summary["excess_says"] = plain_english(excess_card, "the agent minus the Treasury hurdle")
        stitched = _stitch_portfolio_oos(cfg)
        if stitched:
            summary["walk_forward_oos"] = stitched
        diary = write_diary_grade(cfg, close.index)
        if diary:
            summary["saved_diary_vs_tbill"] = diary
    if cfg.mode == "daytrade" and "forced" in journal.columns:
        summary["forced_rate"] = float(journal["forced"].mean())
        summary["forced_says"] = (
            f"On {summary['forced_rate'] * 100:.0f}% of days the agent marked its pick as "
            f"'hand was forced' (wanted to sit out, but the rules require a trade)."
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "Backtest [%s/%s]: agent %.1f%% (Sharpe %.2f) vs bench %.1f%% (Sharpe %.2f)",
        cfg.mode,
        tag,
        agent_card["total_return"] * 100,
        agent_card["sharpe"],
        bench_card["total_return"] * 100,
        bench_card["sharpe"],
    )
    if cfg.mode == "daytrade" and tag == "base":
        _write_walk_forward_oos_report(cfg)
    return summary


def _stitch_portfolio_oos(cfg: Config) -> dict | None:
    """Stitch never-seen exam windows for the treasury copy."""
    from src.registry.store import fold_dir, load_experiments

    records = load_experiments(cfg)
    if not records:
        return None
    nets, excesses, rfs = [], [], []
    for record in records:
        path = fold_dir(cfg, record["fold"]) / "oos_journal.csv"
        if not path.exists():
            continue
        journal = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        nets.append(journal["net_return"])
        if "excess_return" in journal.columns:
            excesses.append(journal["excess_return"])
        if "rf" in journal.columns:
            rfs.append(journal["rf"])
    if not nets:
        return None

    def _cat(frames: list[pd.Series]) -> pd.Series:
        series = pd.concat(frames).sort_index()
        return series[~series.index.duplicated(keep="first")]

    agent = scorecard(_cat(nets))
    out = {
        "kind": "walk_forward_oos_only",
        "note": (
            "Only never-seen exam windows, stitched together. "
            "Prefer this over the full-history backtest."
        ),
        "n_folds": len(records),
        "agent": agent,
        "plain_english": plain_english(agent, "the copy on exam windows only"),
    }
    if excesses:
        excess = scorecard(_cat(excesses))
        out["excess_vs_tbill"] = excess
        out["excess_says"] = plain_english(excess, "exam-window excess over the Treasury hurdle")
    if rfs:
        out["tbill_total_return"] = scorecard(_cat(rfs))["total_return"]
    return out


def _write_walk_forward_oos_report(cfg: Config) -> dict | None:
    """Stitch exam-only journals — the honest grade, free of train-window contamination."""
    from src.registry.store import load_experiments, fold_dir

    records = load_experiments(cfg)
    if not records:
        return None
    frames = []
    for r in records:
        path = fold_dir(cfg, r["fold"]) / "oos_journal.csv"
        if not path.exists():
            continue
        j = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        frames.append(j["net_return"])
    if not frames:
        return None
    rets = pd.concat(frames).sort_index()
    rets = rets[~rets.index.duplicated(keep="first")]
    card = scorecard(rets)
    green = sum(
        1
        for r in records
        if r["out_of_sample"]["total_return"] > 0 and r["out_of_sample"]["sharpe"] > 0
    )
    report = {
        "kind": "walk_forward_oos_only",
        "note": (
            "Only never-seen exam windows, stitched together. Prefer this over the "
            "full-history backtest when judging daytrade skill."
        ),
        "green_folds": green,
        "total_folds": len(records),
        "green_fraction": green / len(records),
        "scorecard": card,
        "plain_english": plain_english(card, "walk-forward daytrade (exam windows only)"),
    }
    out = backtest_dir(cfg) / "walk_forward_oos"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    rets.to_csv(out / "returns.csv")
    log.info(
        "Walk-forward OOS-only: %.1f%% (Sharpe %.2f) | green folds %d/%d",
        card["total_return"] * 100,
        card["sharpe"],
        green,
        len(records),
    )
    return report
