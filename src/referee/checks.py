"""The anti-cheat referee: automated traps for self-deception.

Every check answers one question in plain English:

1. lookahead_recompute — "If we delete the future and rebuild the features,
   do they change?" If yes, some feature was secretly reading ahead.
2. future_correlation — "Does any feature 'know' returns it shouldn't?"
   A legitimate clue has only a faint link to the next move; a near-perfect
   link means the feature contains tomorrow's answer.
3. embargo — "Is there a real gap between every practice window and its
   exam window?" No gap = answers bleeding onto the exam sheet.
4. cost_stress — "Does the profit survive if trading costs triple?" Edges
   that die at 3x costs were mostly an artifact of optimistic assumptions.
5. reality_gate — "Did the champion actually beat just buying the index on
   data it never saw?" If not, the complexity isn't earning its keep.

If ANY check fails, the profit numbers do not matter yet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.lake import Lake
from src.features.factory import FeaturePanel, build_features, warmup_days
from src.registry.store import load_experiments

log = logging.getLogger(__name__)


def referee_dir(cfg: Config) -> Path:
    return cfg.artifacts_dir / "referee"


# ---------------------------------------------------------------- check 1
def check_lookahead_recompute(cfg: Config, lake: Lake, n_dates: int | None = None) -> dict:
    """Rebuild features from future-deleted data and demand identical values."""
    close, volume = lake.close, lake.volume
    full = build_features(close, volume, cfg.features, lake.benchmark)
    warmup = warmup_days(cfg.features)

    rng = np.random.default_rng(0)
    n = n_dates or cfg.referee.leakage_sample_dates
    candidates = np.arange(warmup, len(close))
    sample = rng.choice(candidates, size=min(n, len(candidates)), replace=False)

    worst = 0.0
    for idx in sorted(sample):
        when = close.index[idx]
        view = lake.as_of(when)
        truncated = build_features(view.close, view.volume, cfg.features, lake.benchmark)
        diff = np.abs(truncated.values[-1] - full.values[idx])
        worst = max(worst, float(np.nanmax(diff)))

    passed = worst < 1e-6
    return {
        "name": "lookahead_recompute",
        "passed": bool(passed),
        "details": {"dates_checked": int(len(sample)), "worst_difference": worst},
        "plain_english": (
            f"Rebuilt the agent's clues on {len(sample)} random days with the future "
            f"deleted. Biggest change: {worst:.2e}. "
            + ("Identical — nothing was peeking ahead." if passed
               else "THEY CHANGED — some feature is reading the future. Fix before trusting any result.")
        ),
    }


# ---------------------------------------------------------------- check 2
def feature_future_correlations(close: pd.DataFrame, panel: FeaturePanel) -> dict[str, float]:
    """|corr| between each feature and returns the feature must not know.

    Under the lab's timing rules, the feature row for day T knows prices only
    through T-1. So it must have no strong link to either the T-1→T move
    (contemporaneous leak) or the T→T+1 move (future leak).
    """
    rets = close.pct_change()  # rets row t = move from t-1 to t
    forbidden = {
        "contemporaneous": rets,          # aligned: feature[t] vs move ending at t
        "future": rets.shift(-1),         # aligned: feature[t] vs move ending at t+1
    }
    out: dict[str, float] = {}
    for f, name in enumerate(panel.names):
        feat = pd.DataFrame(panel.values[:, :, f], index=panel.dates, columns=panel.tickers)
        worst = 0.0
        for kind, target in forbidden.items():
            x = feat.values.ravel()
            y = target.reindex(index=panel.dates, columns=panel.tickers).values.ravel()
            mask = np.isfinite(x) & np.isfinite(y) & (x != 0)
            if mask.sum() < 100:
                continue
            if x[mask].std() == 0 or y[mask].std() == 0:
                continue  # constant series (e.g. an always-on regime flag)
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            if np.isfinite(corr):
                worst = max(worst, abs(corr))
        out[name] = worst
    return out


def check_future_correlation(cfg: Config, lake: Lake) -> dict:
    panel = build_features(lake.close, lake.volume, cfg.features, lake.benchmark)
    corrs = feature_future_correlations(lake.close, panel)
    threshold = cfg.referee.cheat_corr_threshold
    offenders = {k: round(v, 3) for k, v in corrs.items() if v > threshold}
    passed = not offenders
    return {
        "name": "future_correlation",
        "passed": bool(passed),
        "details": {
            "threshold": threshold,
            "max_correlation": round(max(corrs.values()), 4) if corrs else 0.0,
            "offenders": offenders,
        },
        "plain_english": (
            "No feature has a suspiciously strong link to returns it shouldn't know."
            if passed
            else f"These features look like they contain the answer key: {offenders}. "
                 "That is cheating, not skill."
        ),
    }


# ---------------------------------------------------------------- check 3
def check_embargo(cfg: Config) -> dict:
    records = load_experiments(cfg)
    if not records:
        return {
            "name": "embargo",
            "passed": False,
            "details": {"error": "no training records found — run train first"},
            "plain_english": "No training has been logged yet, so the train/exam gap can't be verified.",
        }
    violations = []
    for r in records:
        gap = r["test_start_idx"] - r["train_end_idx"]
        if gap < cfg.training.embargo_days:
            violations.append({"fold": r["fold"], "gap_days": gap})
    passed = not violations
    return {
        "name": "embargo",
        "passed": bool(passed),
        "details": {
            "required_gap_days": cfg.training.embargo_days,
            "folds_checked": len(records),
            "violations": violations,
        },
        "plain_english": (
            f"All {len(records)} folds keep at least {cfg.training.embargo_days} market days "
            "between training data and exam data."
            if passed
            else f"Some folds have train/exam windows too close together: {violations}."
        ),
    }


# ---------------------------------------------------------------- check 4
def check_cost_stress(cfg: Config) -> dict:
    from src.backtest.engine import run_backtest

    base = run_backtest(cfg, cost_multiplier=1.0, tag="base")
    stressed = run_backtest(cfg, cost_multiplier=cfg.costs.stress_multiplier, tag="stress")

    base_ret = base["agent"]["total_return"]
    stress_ret = stressed["agent"]["total_return"]
    passed = stress_ret > 0 or (base_ret <= 0)
    return {
        "name": "cost_stress",
        "passed": bool(passed),
        "details": {
            "cost_multiplier": cfg.costs.stress_multiplier,
            "base_total_return": round(base_ret, 4),
            "stressed_total_return": round(stress_ret, 4),
        },
        "plain_english": (
            f"With trading costs multiplied by {cfg.costs.stress_multiplier:g}, total return goes "
            f"from {base_ret * 100:.1f}% to {stress_ret * 100:.1f}%. "
            + ("The edge survives rough conditions." if passed
               else "The 'profit' disappears once trading isn't nearly free — the edge is likely costs-fragile.")
        ),
    }


# ---------------------------------------------------------------- check 5
def check_reality_gate(cfg: Config) -> dict:
    records = load_experiments(cfg)
    if not records:
        return {
            "name": "reality_gate",
            "passed": False,
            "details": {"error": "no training records found — run train first"},
            "plain_english": "No exam results logged yet.",
        }
    rows = []
    wins = 0
    for r in records:
        agent_sh = r["out_of_sample"]["sharpe"]
        bench_sh = r["benchmark_oos"]["sharpe"]
        won = agent_sh >= bench_sh
        wins += int(won)
        rows.append(
            {
                "fold": r["fold"],
                "agent_oos_sharpe": round(agent_sh, 2),
                "benchmark_sharpe": round(bench_sh, 2),
                "beats_benchmark": won,
            }
        )
    frac = wins / len(records)
    passed = frac >= 0.5
    return {
        "name": "reality_gate",
        "passed": bool(passed),
        "details": {"folds": rows, "win_fraction": round(frac, 2)},
        "plain_english": (
            f"On exam (never-seen) data, the agent matched or beat buy-and-hold in "
            f"{wins} of {len(records)} periods. "
            + ("Good enough to keep researching." if passed
               else "It loses to simply buying the index most of the time — the current brain isn't earning its complexity.")
        ),
    }


# ---------------------------------------------------------------- runner
def run_referee(cfg: Config) -> dict:
    lake = Lake(cfg)
    checks = [
        check_lookahead_recompute(cfg, lake),
        check_future_correlation(cfg, lake),
        check_embargo(cfg),
        check_cost_stress(cfg),
        check_reality_gate(cfg),
    ]
    overall = all(c["passed"] for c in checks)
    report = {
        "overall_pass": overall,
        "verdict": (
            "PASS — no cheating detected and the results hold up under stress. "
            "Still simulated money, but honestly simulated."
            if overall
            else "FAIL — at least one check failed. Treat every profit number as untrustworthy until it's fixed."
        ),
        "checks": checks,
    }
    referee_dir(cfg).mkdir(parents=True, exist_ok=True)
    (referee_dir(cfg) / "report.json").write_text(json.dumps(report, indent=2))
    for c in checks:
        log.info("[%s] %s: %s", "PASS" if c["passed"] else "FAIL", c["name"], c["plain_english"])
    log.info("Referee overall: %s", "PASS" if overall else "FAIL")
    return report
