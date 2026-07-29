"""Mode helpers — keep daytrade special-cases from swallowing concentrated."""

from __future__ import annotations

from src.config import Config


def is_daytrade(cfg: Config) -> bool:
    return cfg.mode == "daytrade"


def is_allocation(cfg: Config) -> bool:
    """Portfolio-style daily allocation (includes concentrated)."""
    return cfg.mode in ("portfolio", "concentrated")


def is_concentrated(cfg: Config) -> bool:
    return cfg.mode == "concentrated"
