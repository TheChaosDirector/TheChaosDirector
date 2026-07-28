"""The paper-trading desk: a simulated live account with a hard blindfold.

Unlike the backtester (which replays a pre-built feature panel), the desk
advances a simulated calendar one market day at a time and, on every single
day, REBUILDS its features from a truncated copy of the lake that literally
ends that day. Anything after "today" does not exist in memory. That makes
lookahead impossible by construction, not just by discipline — this is the
closest thing to live trading without a brokerage.

The desk deliberately starts on the day after the final walk-forward exam
window, so it only ever trades data no fold was trained or graded on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.lake import Lake
from src.env.portfolio_env import action_to_weights
from src.features.factory import build_features
from src.metrics.scorecard import plain_english, scorecard
from src.registry.store import load_champion_model, load_experiments

log = logging.getLogger(__name__)


def paper_dir(cfg: Config) -> Path:
    return cfg.artifacts_dir / "paper"


def _state_path(cfg: Config) -> Path:
    return paper_dir(cfg) / "state.json"


def _journal_path(cfg: Config) -> Path:
    return paper_dir(cfg) / "journal.csv"


def _init_state(cfg: Config, lake: Lake) -> dict:
    experiments = load_experiments(cfg)
    if experiments:
        last_test_end = max(r["test_end_idx"] for r in experiments)
        start_idx = min(last_test_end, len(lake.close) - 1)
    else:
        start_idx = max(0, len(lake.close) - 60)
    start_date = lake.close.index[start_idx]
    return {
        "start_date": str(start_date.date()),
        "last_date": None,  # no day processed yet
        "cash_start": cfg.paper.starting_cash,
        "equity": cfg.paper.starting_cash,
        "weights": {},  # ticker -> portfolio fraction (rest is cash)
    }


def load_state(cfg: Config, lake: Lake) -> dict:
    path = _state_path(cfg)
    if path.exists():
        return json.loads(path.read_text())
    state = _init_state(cfg, lake)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return state


def _save_state(cfg: Config, state: dict) -> None:
    _state_path(cfg).write_text(json.dumps(state, indent=2))


def _append_journal(cfg: Config, row: dict) -> None:
    path = _journal_path(cfg)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def advance(cfg: Config, days: int = 1) -> dict:
    """Process the next `days` market days, one honest day at a time."""
    lake = Lake(cfg)
    model, meta = load_champion_model(cfg)
    state = load_state(cfg, lake)

    all_dates = lake.close.index
    if state["last_date"] is None:
        next_pos = all_dates.get_loc(pd.Timestamp(state["start_date"]))
    else:
        next_pos = all_dates.get_loc(pd.Timestamp(state["last_date"])) + 1

    tickers = meta["tickers"]
    processed = 0
    for pos in range(next_pos, min(next_pos + days, len(all_dates))):
        today = all_dates[pos]

        # The blindfold: rebuild the market as it existed at today's close.
        view = lake.as_of(today)
        panel = build_features(view.close, view.volume, cfg.features, lake.benchmark)
        obs_features = panel.values[-1].reshape(-1)

        prev_weights = np.array([state["weights"].get(t, 0.0) for t in tickers])

        # Step 1: realize profit/loss from the previous close to today's close.
        day_ret = 0.0
        if state["last_date"] is not None:
            prev_close = lake.close.loc[pd.Timestamp(state["last_date"]), tickers].values
            today_close = lake.close.loc[today, tickers].values
            valid_hold = np.isfinite(prev_close) & np.isfinite(today_close)
            rets = np.where(valid_hold, today_close / prev_close - 1.0, 0.0)
            day_ret = float((prev_weights * rets).sum())
            state["equity"] *= 1.0 + day_ret
            growth = 1.0 + day_ret
            if growth > 0:
                prev_weights = prev_weights * (1.0 + rets) / growth

        # Step 2: ask the frozen champion for today's allocation.
        drawdown_placeholder = 0.0
        obs = np.concatenate(
            [obs_features, prev_weights, [1.0 - prev_weights.sum()], [drawdown_placeholder]]
        ).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)

        today_close = lake.close.loc[today, tickers].values
        valid = np.isfinite(today_close).astype(np.float64)
        target = action_to_weights(
            action, valid, cfg.risk.max_weight_per_name, cfg.risk.max_gross_exposure
        )

        # Step 3: trade at today's close and pay for it.
        turnover = float(np.abs(target - prev_weights).sum())
        cost_frac = turnover * cfg.costs.total_bps / 1e4
        state["equity"] *= 1.0 - cost_frac

        state["weights"] = {
            t: float(w) for t, w in zip(tickers, target) if w > 1e-6
        }
        state["last_date"] = str(today.date())

        top = sorted(state["weights"].items(), key=lambda kv: -kv[1])[:5]
        _append_journal(
            cfg,
            {
                "date": str(today.date()),
                "equity": round(state["equity"], 2),
                "day_return": day_ret,
                "turnover": turnover,
                "cost_paid": round(cost_frac * state["equity"], 4),
                "cash_weight": round(1.0 - sum(state["weights"].values()), 4),
                "top_positions": "; ".join(f"{t}:{w:.1%}" for t, w in top),
            },
        )
        processed += 1
        log.info("Paper %s: equity $%.2f (day %+.2f%%)", today.date(), state["equity"], day_ret * 100)

    _save_state(cfg, state)

    report = summary(cfg)
    if processed == 0:
        log.info("Paper desk is caught up: no more market days in the lake.")
    return report


def summary(cfg: Config) -> dict:
    lake = Lake(cfg)
    state = load_state(cfg, lake)
    out = {
        "start_date": state["start_date"],
        "last_date": state["last_date"],
        "starting_cash": state["cash_start"],
        "equity": state["equity"],
        "pnl": state["equity"] - state["cash_start"],
        "positions": state["weights"],
    }
    jpath = _journal_path(cfg)
    if jpath.exists():
        journal = pd.read_csv(jpath)
        rets = pd.Series(journal["day_return"].values[1:])  # first row has no held period
        if len(rets):
            card = scorecard(rets)
            out["scorecard"] = card
            out["plain_english"] = plain_english(card, "the paper account")
    return out
