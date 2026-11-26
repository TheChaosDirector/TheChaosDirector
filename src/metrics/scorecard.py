"""Scorecard math shared by the trainer, backtester, paper desk and referee."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_from_returns(returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
    return starting_value * (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (TRADING_DAYS / n) - 1.0


def sharpe(returns: pd.Series) -> float:
    std = returns.std()
    if std is None or std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss, as a negative fraction (e.g. -0.23 = -23%)."""
    equity = equity_from_returns(returns)
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def scorecard(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    out = {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(returns),
        "n_days": int(len(returns)),
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
    }
    if turnover is not None and len(turnover):
        out["avg_daily_turnover"] = float(turnover.mean())
    return out


def plain_english(card: dict, label: str = "the strategy") -> str:
    """A one-paragraph, jargon-translated read of a scorecard."""
    tr = card["total_return"] * 100
    dd = card["max_drawdown"] * 100
    sh = card["sharpe"]
    return (
        f"Over {card['n_days']} trading days, {label} "
        f"{'made' if tr >= 0 else 'lost'} {abs(tr):.1f}% in total. "
        f"The worst losing stretch was {abs(dd):.1f}% from peak to bottom. "
        f"Return-per-unit-of-wobble (Sharpe) was {sh:.2f} — "
        f"{'suspiciously high, double-check for bugs' if sh > 3 else 'solid' if sh > 1 else 'modest' if sh > 0.3 else 'weak'}. "
        f"It finished up on {card['win_rate'] * 100:.0f}% of days."
    )
