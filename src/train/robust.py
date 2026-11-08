"""A locked rule for keeping a model only when the edge repeats.

Each exam year is a model that was trained only on earlier data. The years
are stitched in order. The learned book is kept when that stitch beats SPY
after costs and at least half of the years finish ahead. Otherwise the
deployed book is SPY itself.

The sealed final year is not part of the stitch and is not scored. The
thresholds are fixed in this file. They are not revised after a run.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from src.config import Config
from src.metrics.scorecard import scorecard
from src.registry.store import fold_dir, registry_dir, save_champion

log = logging.getLogger(__name__)


def passes_repeated_edge(year_gaps: list[float], stitch_gap: float) -> bool:
    """True only when the whole stitch and at least half the years beat SPY."""
    if not year_gaps:
        return False
    ahead = sum(1 for gap in year_gaps if gap > 0.0)
    return stitch_gap > 0.0 and ahead * 2 >= len(year_gaps)


def _compound(series: pd.Series) -> float:
    return float((1.0 + series.astype(float)).prod() - 1.0)


def _load_journals(cfg: Config, records: list[dict]) -> pd.DataFrame:
    frames = []
    for record in sorted(records, key=lambda row: row["fold"]):
        path = fold_dir(cfg, int(record["fold"])) / "oos_journal.csv"
        frame = pd.read_csv(path)
        if "date" not in frame.columns:
            raise RuntimeError(f"Exam journal {path} has no date column.")
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")
        frames.append(frame)
    stitched = pd.concat(frames)
    stitched = stitched[~stitched.index.duplicated(keep="first")].sort_index()
    return stitched


def _year_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in sorted(records, key=lambda row: row["fold"]):
        if "spy_oos" not in record:
            raise RuntimeError("Repeated-edge check needs each exam's SPY return.")
        agent = float(record["out_of_sample"]["total_return"])
        spy = float(record["spy_oos"]["total_return"])
        rows.append(
            {
                "fold": record["fold"],
                "test_start": record["test_start"],
                "test_end": record["test_end"],
                "agent": agent,
                "spy": spy,
                "gap": agent - spy,
                "ahead": agent > spy,
            }
        )
    return rows


def apply_repeated_edge_gate(cfg: Config, close: pd.DataFrame, panel, records: list[dict]) -> dict:
    """Decide from the stitched exams. Does not score the sealed year."""
    from src.train.holdout import describe_seal, holdout_bounds

    years = _year_rows(records)
    journal = _load_journals(cfg, records)
    agent = _compound(journal["net_return"])
    spy = _compound(journal["benchmark_return"])
    bill = _compound(journal["rf"]) if "rf" in journal.columns else 0.0
    stitch_gap = agent - spy
    gaps = [row["gap"] for row in years]
    kept = passes_repeated_edge(gaps, stitch_gap)
    ahead = sum(1 for row in years if row["ahead"])

    report = {
        "rule": (
            "Keep the learned book only if the stitched unseen years beat SPY "
            "after the configured commission and slippage, and at least half of "
            "those years finish ahead. Otherwise hold SPY. The sealed final year "
            "is not scored."
        ),
        "n_years": len(years),
        "years_ahead": ahead,
        "years": years,
        "stitch": {
            "agent": agent,
            "spy": spy,
            "tbill": bill,
            "gap": stitch_gap,
            "n_days": int(len(journal)),
            "start": str(journal.index[0].date()),
            "end": str(journal.index[-1].date()),
            "avg_daily_turnover": float(journal["turnover"].mean()) if "turnover" in journal else None,
            "agent_card": scorecard(journal["net_return"], journal.get("turnover")),
            "spy_card": scorecard(journal["benchmark_return"]),
        },
        "kept_learned_book": kept,
        "deploy": "model" if kept else "hold_spy",
        "sealed_year_scored": False,
    }
    if cfg.compete.final_holdout_days > 0:
        report["sealed_year"] = describe_seal(close.index, cfg)

    out_dir = cfg.artifacts_dir / "stitch"
    out_dir.mkdir(parents=True, exist_ok=True)
    journal.to_csv(out_dir / "journal.csv")
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str))

    log.info(
        "Stitched unseen years: agent %.1f%% | SPY %.1f%% | bill %.1f%% | years ahead %d/%d",
        agent * 100,
        spy * 100,
        bill * 100,
        ahead,
        len(years),
    )
    log.info("Repeated-edge decision: %s", "keep the learned book" if kept else "hold SPY")

    champion = {
        "fold": None,
        "mode": cfg.mode,
        "tickers": panel.tickers,
        "feature_names": panel.names,
        "config_path": cfg.config_path,
        "objective": "repeated_edge_vs_spy",
        "focus": cfg.compete.focus,
        "yield_weight": cfg.compete.yield_weight,
        "rebalance_interval": cfg.compete.rebalance_interval,
        "spy_anchor": cfg.compete.spy_anchor,
        "n_tickers": len(panel.tickers),
        "stitch_report": str(out_dir / "summary.json"),
        "years_ahead": ahead,
        "total_folds": len(years),
        "stitch_agent": agent,
        "stitch_spy": spy,
        "stitch_gap": stitch_gap,
        "sealed_year_scored": False,
        "deploy": "hold_spy",
        "learner": "hold_spy",
        "model_path": None,
        "selected_by": "repeated_edge_gate_failed_hold_spy",
    }
    if kept:
        champion.update(_train_presample_model(cfg, close, panel))
    save_champion(cfg, champion)
    (registry_dir(cfg) / "champion_walkforward.json").write_text(
        json.dumps(champion, indent=2, default=str)
    )
    return champion


def _train_presample_model(cfg: Config, close: pd.DataFrame, panel) -> dict:
    """Fit one model on the years before the seal. Do not grade the seal."""
    from stable_baselines3.common.monitor import Monitor

    from src.train.holdout import holdout_bounds
    from src.train.walkforward import _make_portfolio_env, _new_ppo

    bounds = holdout_bounds(len(close), cfg)
    log.info(
        "Fitting the deployable model on %s through %s. The sealed year is not scored.",
        close.index[bounds["train_start"]].date(),
        close.index[bounds["train_end"] - 1].date(),
    )
    train_env = Monitor(
        _make_portfolio_env(close, panel, cfg, bounds["train_start"], bounds["train_end"])
    )
    model = _new_ppo(
        cfg,
        train_env,
        seed=cfg.training.seed + 10_000,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model.learn(total_timesteps=int(cfg.training.total_timesteps))
    out_dir = registry_dir(cfg) / "presample"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.zip"
    model.save(str(model_path))
    return {
        "deploy": "model",
        "learner": "ppo",
        "model_path": str(model_path),
        "selected_by": "repeated_edge_stitch_before_sealed_year",
        "train_start": str(close.index[bounds["train_start"]].date()),
        "train_end": str(close.index[bounds["train_end"] - 1].date()),
    }
