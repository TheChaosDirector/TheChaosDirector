"""Use saved paper decisions as a nudge, scored against the Treasury hurdle.

The live champion is not loaded here. Hints are date → weights for the copy's
own training env. A day counts only when most of that saved book's weight sits
inside the universe being trained, so a 12-name smoke run does not pretend the
270-name diary was its own portfolio.

The last slice of labeled days is a holdout: it is graded and never used as a
training reward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config
from src.experience.decisions import decisions_path
from src.metrics.scorecard import scorecard


def _load_summaries(cfg: Config) -> list[dict]:
    """Diary rows without the bulky observation vectors."""
    path = decisions_path(cfg)
    if not path.exists():
        return []
    keep = ("as_of", "outcome", "weights")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        rows.append({k: raw.get(k) for k in keep})
    return rows


def rf_on(daily_rf: pd.Series, day: pd.Timestamp) -> float | None:
    """Last published daily yield on or before ``day`` (no peeking ahead)."""
    if daily_rf.empty:
        return None
    series = daily_rf.sort_index()
    day = pd.Timestamp(day).tz_localize(None).normalize()
    pos = int(series.index.searchsorted(day, side="right")) - 1
    if pos < 0:
        return None
    quote_day = pd.Timestamp(series.index[pos])
    if (day - quote_day).days > 5:
        return None
    value = float(series.iloc[pos])
    if not np.isfinite(value):
        return None
    return value


def weights_on_universe(saved: dict, tickers: list[str]) -> tuple[np.ndarray, float]:
    """Project a saved weight map onto ``tickers``. Coverage is the share of weight kept."""
    vec = np.zeros(len(tickers), dtype=np.float64)
    index = {t: i for i, t in enumerate(tickers)}
    covered = 0.0
    total = 0.0
    for sym, weight in (saved or {}).items():
        w = float(weight)
        if w <= 0:
            continue
        total += w
        slot = index.get(sym)
        if slot is None:
            continue
        vec[slot] = w
        covered += w
    coverage = covered / total if total > 0 else 0.0
    return vec, coverage


def allocation_similarity(target: np.ndarray, saved: np.ndarray) -> float:
    """1 when the book matches the saved decision, including the cash residual."""
    cash_t = 1.0 - float(np.asarray(target).sum())
    cash_s = 1.0 - float(np.asarray(saved).sum())
    dist = 0.5 * (float(np.abs(target - saved).sum()) + abs(cash_t - cash_s))
    return float(np.clip(1.0 - dist, 0.0, 1.0))


def hint_reward(target: np.ndarray, saved: np.ndarray, beat: bool) -> float:
    """Positive when copying a decision that beat bills, negative when copying a loser."""
    similarity = allocation_similarity(target, saved)
    return similarity if beat else -similarity


@dataclass
class ExperienceSplit:
    train_hints: dict = field(default_factory=dict)
    holdout_dates: list = field(default_factory=list)
    report: dict = field(default_factory=dict)


def build_experience_split(
    cfg: Config,
    tickers: list[str],
    daily_rf: pd.Series,
) -> ExperienceSplit:
    """Split labeled diary days into train hints and a held-out tail."""
    usable = []
    for rec in _load_summaries(cfg):
        outcome = rec.get("outcome") or {}
        portfolio_return = outcome.get("portfolio_return")
        if portfolio_return is None:
            continue
        day = pd.Timestamp(rec["as_of"]).tz_localize(None).normalize()
        rf = rf_on(daily_rf, day)
        if rf is None:
            continue
        saved, coverage = weights_on_universe(rec.get("weights") or {}, tickers)
        usable.append(
            {
                "day": day,
                "weights": saved,
                "beat": float(portfolio_return) > rf,
                "coverage": coverage,
                "portfolio_return": float(portfolio_return),
                "tbill_return": rf,
            }
        )
    usable.sort(key=lambda row: row["day"])

    holdout_n = 0
    if len(usable) >= 4 and cfg.compete.holdout_frac > 0:
        holdout_n = max(1, int(round(len(usable) * cfg.compete.holdout_frac)))
    train_rows = usable[:-holdout_n] if holdout_n else usable
    hold_rows = usable[-holdout_n:] if holdout_n else []

    hints = {}
    skipped = 0
    for row in train_rows:
        if row["coverage"] < cfg.compete.min_coverage:
            skipped += 1
            continue
        hints[row["day"].date()] = {"weights": row["weights"], "beat": row["beat"]}

    report = {
        "labeled_with_yield": len(usable),
        "train_candidates": len(train_rows),
        "train_hints": len(hints),
        "skipped_low_coverage": skipped,
        "holdout_days": len(hold_rows),
        "min_coverage": cfg.compete.min_coverage,
        "note": (
            "A saved day becomes a training hint only when at least "
            f"{cfg.compete.min_coverage:.0%} of its weight is inside this universe. "
            "The held-out tail is never a training reward."
        ),
    }
    return ExperienceSplit(
        train_hints=hints,
        holdout_dates=[row["day"].date() for row in hold_rows],
        report=report,
    )


def grade_diary_against_tbill(cfg: Config, daily_rf: pd.Series) -> dict:
    """How the already-saved paper choices did versus the bill, day by day.

    Uses the diary's own next-day portfolio return (cash in that grade earns
    nothing, which matches how it was labeled). The bill is one trading day's
    yield. This does not retrain anything.
    """
    rows = []
    for rec in _load_summaries(cfg):
        outcome = rec.get("outcome") or {}
        portfolio_return = outcome.get("portfolio_return")
        if portfolio_return is None:
            continue
        day = pd.Timestamp(rec["as_of"]).tz_localize(None).normalize()
        rf = rf_on(daily_rf, day)
        if rf is None:
            continue
        rows.append(
            {
                "as_of": day,
                "portfolio_return": float(portfolio_return),
                "tbill_return": rf,
                "excess_return": float(portfolio_return) - rf,
            }
        )
    if not rows:
        return {
            "n_days": 0,
            "plain_english": "No labeled paper decisions overlapped the Treasury series.",
        }
    frame = pd.DataFrame(rows).drop_duplicates("as_of").set_index("as_of").sort_index()
    portfolio = scorecard(frame["portfolio_return"])
    tbill = scorecard(frame["tbill_return"])
    excess = scorecard(frame["excess_return"])
    beat_days = int((frame["excess_return"] > 0).sum())
    return {
        "n_days": int(len(frame)),
        "from": str(frame.index[0].date()),
        "to": str(frame.index[-1].date()),
        "portfolio": portfolio,
        "tbill": tbill,
        "excess": excess,
        "beat_days": beat_days,
        "plain_english": (
            f"Over {len(frame)} labeled paper days ({frame.index[0].date()} to "
            f"{frame.index[-1].date()}), the saved book made "
            f"{portfolio['total_return'] * 100:.2f}% and the 13-week bill hurdle "
            f"made {tbill['total_return'] * 100:.2f}%. "
            f"Excess versus the bill was {excess['total_return'] * 100:.2f}% "
            f"({beat_days} of {len(frame)} days ahead of the bill)."
        ),
    }


def should_promote_experience_tune(pretrained: dict, tuned: dict) -> bool:
    """Keep the walk-forward copy unless the tune wins the holdout without wrecking the full sample."""
    holdout_better = tuned["holdout_excess_total"] > pretrained["holdout_excess_total"]
    full_held_up = tuned["full_excess_total"] >= pretrained["full_excess_total"] - 0.02
    return bool(holdout_better and full_held_up)
