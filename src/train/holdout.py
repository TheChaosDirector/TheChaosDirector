"""A sealed final year for the dual-hurdle copy.

Walk-forward folds stop before this window, with the usual embargo gap
between them. One more model is fit on the years immediately before the
seal. That model is scored once, at the configured commission and slippage,
and again with costs tripled. Nothing in here is allowed to train again
after those numbers exist.
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import Config
from src.features.factory import warmup_days
from src.metrics.scorecard import scorecard
from src.registry.store import registry_dir, save_champion

TRADING_DAYS_PER_YEAR = 252


def fold_history_length(n_dates: int, cfg: Config) -> int:
    """How much history ``make_folds`` may see.

    The returned length still follows ``make_folds``'s own 'last bar is the
    next day's price' rule. The sealed tail and the embargo in front of it
    are not included.
    """
    if cfg.mode != "portfolio" or not cfg.compete.enabled or cfg.compete.final_holdout_days <= 0:
        return n_dates
    trimmed = n_dates - cfg.compete.final_holdout_days - cfg.training.embargo_days
    if trimmed <= warmup_days(cfg.features) + 50:
        raise RuntimeError(
            "The locked holdout leaves too little history to train. "
            "Shorten final_holdout_days or download a longer lake."
        )
    return trimmed


def holdout_bounds(n_dates: int, cfg: Config) -> dict[str, int]:
    """Index bounds. ``train_end`` and ``holdout_end`` are exclusive env ends."""
    holdout_days = int(cfg.compete.final_holdout_days)
    embargo = int(cfg.training.embargo_days)
    holdout_start = n_dates - 1 - holdout_days
    train_end = holdout_start - embargo
    train_len = int(cfg.training.train_years * TRADING_DAYS_PER_YEAR)
    train_start = train_end - train_len
    warm = warmup_days(cfg.features)
    if train_start < warm:
        raise RuntimeError(
            f"Need at least {warm + train_len + embargo + holdout_days} price rows "
            "for the sealed holdout protocol."
        )
    if not (warm <= train_start < train_end <= holdout_start < n_dates - 1):
        raise RuntimeError("Holdout bounds are not in chronological order.")
    return {
        "train_start": train_start,
        "train_end": train_end,
        "holdout_start": holdout_start,
        "holdout_end": n_dates - 1,
        "embargo_days": embargo,
        "holdout_days": holdout_days,
    }


def describe_seal(index: pd.DatetimeIndex, cfg: Config) -> str:
    bounds = holdout_bounds(len(index), cfg)
    return (
        "Sealed holdout (not used for training or model selection): "
        f"{index[bounds['holdout_start']].date()} through "
        f"{index[bounds['holdout_end'] - 1].date()}. "
        f"Embargo {bounds['embargo_days']} days. "
        f"Final model trains {index[bounds['train_start']].date()} through "
        f"{index[bounds['train_end'] - 1].date()}."
    )


def _block(journal: pd.DataFrame) -> dict:
    agent = scorecard(journal["net_return"], journal.get("turnover"))
    spy = scorecard(journal["benchmark_return"])
    bill = scorecard(journal["rf"])
    excess_bill = scorecard(journal["excess_return"])
    excess_spy = scorecard(journal["net_return"] - journal["benchmark_return"])
    beat_spy = agent["total_return"] > spy["total_return"]
    beat_bill = agent["total_return"] > bill["total_return"]
    return {
        "agent": agent,
        "spy": spy,
        "tbill": bill,
        "excess_vs_tbill": excess_bill,
        "excess_vs_spy": excess_spy,
        "beat_spy": beat_spy,
        "beat_tbill": beat_bill,
        "beat_both": beat_spy and beat_bill,
    }


def train_and_score_holdout(cfg: Config, close: pd.DataFrame, panel, volume=None) -> dict | None:
    """Fit the pre-holdout model and score the sealed window exactly once."""
    if cfg.compete.final_holdout_days <= 0:
        return None

    import logging

    from stable_baselines3.common.monitor import Monitor

    from src.train.walkforward import _make_portfolio_env, _new_ppo, rollout

    log = logging.getLogger(__name__)
    bounds = holdout_bounds(len(close), cfg)
    log.info("%s", describe_seal(close.index, cfg))
    log.info(
        "Costs charged inside every training and test day: %.1f bps commission + %.1f bps slippage "
        "on traded value. Stress test uses %.1fx those costs.",
        cfg.costs.commission_bps,
        cfg.costs.slippage_bps,
        cfg.costs.stress_multiplier,
    )

    train_env = Monitor(
        _make_portfolio_env(close, panel, cfg, bounds["train_start"], bounds["train_end"], volume=volume)
    )
    policy_kwargs = dict(net_arch=[128, 128] if cfg.compete.focus == "market" or cfg.compete.reward == "both" else [64, 64])
    model = _new_ppo(cfg, train_env, seed=cfg.training.seed + 10_000, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=int(cfg.training.total_timesteps))

    out_dir = registry_dir(cfg) / "holdout_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.zip"
    model.save(str(model_path))

    exam = _make_portfolio_env(
        close, panel, cfg, bounds["holdout_start"], bounds["holdout_end"], volume=volume
    )
    journal = rollout(model, exam)
    stressed_env = _make_portfolio_env(
        close,
        panel,
        cfg,
        bounds["holdout_start"],
        bounds["holdout_end"],
        cost_multiplier=cfg.costs.stress_multiplier,
        volume=volume,
    )
    stressed = rollout(model, stressed_env)

    base = _block(journal)
    stress = _block(stressed)
    report = {
        "protocol": (
            "One model trained only on the window before the embargo. "
            "The sealed year was not used to train, to pick this model, or to decide to train again."
            + (
                " Reward is after-cost wealth versus SPY, with the bill at "
                f"weight {cfg.compete.yield_weight:.2f}. This same calendar year was already "
                "scored once under the earlier dual-hurdle reward, so this grade is a follow-up."
                if cfg.compete.focus == "market"
                else ""
            )
        ),
        "focus": cfg.compete.focus,
        "yield_weight": cfg.compete.yield_weight,
        "spy_beat_weight": cfg.compete.spy_beat_weight,
        "train_start": str(close.index[bounds["train_start"]].date()),
        "train_end": str(close.index[bounds["train_end"] - 1].date()),
        "holdout_start": str(close.index[bounds["holdout_start"]].date()),
        "holdout_end": str(close.index[bounds["holdout_end"] - 1].date()),
        "embargo_days": bounds["embargo_days"],
        "commission_bps": cfg.costs.commission_bps,
        "slippage_bps": cfg.costs.slippage_bps,
        "stress_multiplier": cfg.costs.stress_multiplier,
        "timesteps": int(cfg.training.total_timesteps),
        "base": base,
        "stress": stress,
        "beat_both_after_costs": base["beat_both"],
        "beat_both_after_stress_costs": stress["beat_both"],
    }
    report_path = cfg.artifacts_dir / "holdout" / "summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    journal.to_csv(report_path.parent / "journal.csv")
    stressed.to_csv(report_path.parent / "journal_stress.csv")
    report_path.write_text(json.dumps(report, indent=2))

    log.info(
        "Locked holdout after costs: agent %.1f%% | SPY %.1f%% | bill %.1f%% | beat both: %s",
        base["agent"]["total_return"] * 100,
        base["spy"]["total_return"] * 100,
        base["tbill"]["total_return"] * 100,
        base["beat_both"],
    )
    log.info(
        "Same holdout at %.0fx costs: agent %.1f%% | beat both: %s",
        cfg.costs.stress_multiplier,
        stress["agent"]["total_return"] * 100,
        stress["beat_both"],
    )

    champion = {
        "fold": -1,
        "mode": cfg.mode,
        "model_path": str(model_path),
        "tickers": panel.tickers,
        "feature_names": panel.names,
        "selected_by": "most_recent_window_before_locked_holdout",
        "learner": "ppo",
        "objective": "beat_spy" if cfg.compete.focus == "market" else "beat_spy_and_tbill",
        "focus": cfg.compete.focus,
        "yield_weight": cfg.compete.yield_weight,
        "hurdle_symbol": cfg.compete.symbol,
        "config_path": cfg.config_path,
        "holdout_candidate": True,
        "holdout_report": str(report_path),
        "beat_both_after_costs": base["beat_both"],
        "beat_both_after_stress_costs": stress["beat_both"],
    }
    save_champion(cfg, champion)
    return champion
