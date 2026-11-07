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
from src.modes import is_concentrated, is_daytrade
from src.registry.store import load_champion_model
from src.train.walkforward import _make_portfolio_env, _subset_daytrade_universe, rollout


log = logging.getLogger(__name__)


def backtest_dir(cfg: Config) -> Path:
    return cfg.artifacts_dir / "backtest"


def run_backtest(cfg: Config, cost_multiplier: float = 1.0, tag: str = "base") -> dict:
    lake = Lake(cfg)
    close = lake.close
    volume = lake.volume

    if is_daytrade(cfg):
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

    if is_daytrade(cfg):
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
            benchmark=lake.benchmark,
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
            f"{'SPY open→close' if is_daytrade(cfg) else 'buy-and-hold ' + lake.benchmark}",
        ),
    }
    if is_daytrade(cfg) and "forced" in journal.columns:
        summary["forced_rate"] = float(journal["forced"].mean())
        summary["forced_says"] = (
            f"On {summary['forced_rate'] * 100:.0f}% of days the agent marked its pick as "
            f"'hand was forced' (wanted to sit out, but the rules require a trade)."
        )
    if "excess_return" in journal.columns:
        summary["mean_excess_return"] = float(journal["excess_return"].mean())
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
    if tag == "base" and (is_daytrade(cfg) or is_concentrated(cfg)):
        _write_walk_forward_oos_report(cfg)
    return summary


def _write_walk_forward_oos_report(cfg: Config) -> dict | None:
    """Stitch exam-only journals — the honest grade, free of train-window contamination."""
    from src.registry.store import load_experiments, fold_dir

    records = load_experiments(cfg)
    if not records:
        return None
    ret_frames = []
    excess_frames = []
    for r in records:
        path = fold_dir(cfg, r["fold"]) / "oos_journal.csv"
        if not path.exists():
            continue
        j = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        if "net_return" not in j.columns:
            continue
        ret_frames.append(j["net_return"])
        if "excess_return" in j.columns:
            excess_frames.append(j["excess_return"])
    if not ret_frames:
        return None
    rets = pd.concat(ret_frames).sort_index()
    rets = rets[~rets.index.duplicated(keep="first")]
    card = scorecard(rets)

    if is_concentrated(cfg):
        green = sum(1 for r in records if float(r.get("oos_excess_mean") or 0.0) > 0)
        mean_ex_meta = float(
            sum(float(r.get("oos_excess_mean") or 0.0) for r in records) / len(records)
        )
        label = "walk-forward concentrated (exam windows only)"
        note = (
            "Only never-seen exam windows, stitched together. Prefer this over the "
            "full-history backtest when judging concentrated skill — the long curve "
            "includes train-window contamination."
        )
    else:
        green = sum(
            1
            for r in records
            if r["out_of_sample"]["total_return"] > 0 and r["out_of_sample"]["sharpe"] > 0
        )
        mean_ex_meta = None
        label = "walk-forward daytrade (exam windows only)"
        note = (
            "Only never-seen exam windows, stitched together. Prefer this over the "
            "full-history backtest when judging daytrade skill."
        )

    report = {
        "kind": "walk_forward_oos_only",
        "mode": cfg.mode,
        "note": note,
        "green_folds": green,
        "total_folds": len(records),
        "green_fraction": green / len(records),
        "scorecard": card,
        "plain_english": plain_english(card, label),
    }
    if excess_frames:
        excess = pd.concat(excess_frames).sort_index()
        excess = excess[~excess.index.duplicated(keep="first")]
        report["mean_daily_excess"] = float(excess.mean())
        report["excess_scorecard"] = scorecard(excess)
        out_excess = backtest_dir(cfg) / "walk_forward_oos"
        out_excess.mkdir(parents=True, exist_ok=True)
        excess.to_csv(out_excess / "excess_returns.csv")
    if mean_ex_meta is not None:
        report["mean_oos_excess_across_folds"] = mean_ex_meta

    out = backtest_dir(cfg) / "walk_forward_oos"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    rets.to_csv(out / "returns.csv")
    log.info(
        "Walk-forward OOS-only [%s]: %.1f%% (Sharpe %.2f) | green folds %d/%d",
        cfg.mode,
        card["total_return"] * 100,
        card["sharpe"],
        green,
        len(records),
    )
    return report
