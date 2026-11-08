"""Shared wiring for the Treasury-competing portfolio copy."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import Config
from src.data.yields import align_curve_lagged, align_tbill_daily
from src.train.experience_blend import grade_diary_against_tbill


def compete_env_kwargs(
    cfg: Config,
    index: pd.DatetimeIndex,
    experience: dict | None = None,
    close: pd.DataFrame | None = None,
) -> dict:
    """Env kwargs for a yield-aware gym. Empty when competition is off."""
    if not cfg.compete.enabled:
        return {}
    daily = align_tbill_daily(cfg, index)
    kwargs = {
        "cash_yield": np.asarray(daily, dtype=np.float64),
        "cash_earns": bool(cfg.compete.cash_earns_yield),
        "observe_yield": bool(cfg.compete.observe_yield),
        "reward_excess": cfg.compete.reward in ("excess", "both"),
        "experience": experience,
        "experience_scale": float(cfg.compete.experience_scale) if experience else 0.0,
        "yield_weight": float(cfg.compete.yield_weight),
        "focus": cfg.compete.focus,
    }
    if cfg.compete.observe_curve:
        kwargs["curve_obs"] = align_curve_lagged(cfg, index)
    if close is not None and cfg.universe.benchmark in close.columns:
        spy = close[cfg.universe.benchmark].astype(float)
        fwd = (spy.shift(-1) / spy - 1.0).to_numpy(dtype=np.float64)
        kwargs["benchmark_fwd"] = np.nan_to_num(fwd, nan=0.0, posinf=0.0, neginf=0.0)
        use_spy_weight = cfg.compete.reward == "both" or cfg.compete.focus == "market"
        kwargs["spy_beat_weight"] = float(cfg.compete.spy_beat_weight) if use_spy_weight else 0.0
    return kwargs


def write_diary_grade(cfg: Config, index: pd.DatetimeIndex) -> dict | None:
    """Score the saved paper diary against the bill. Does not touch model files."""
    if not cfg.compete.enabled:
        return None
    daily = align_tbill_daily(cfg, index)
    report = grade_diary_against_tbill(cfg, daily)
    out = cfg.artifacts_dir / "diary_vs_tbill.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report
