"""Who may be held on a given day, using only information already known.

Membership used to be chosen once, from dollar volume over the whole file.
A name could enter the backtest because it became liquid years later, and a
name that was not listed yet still occupied a slot. Neither is allowed here.

Row t of the mask is decided from closes and volume through t-1 only.
Deleting every later row does not change it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def point_in_time_eligible(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    min_history_days: int,
    min_dollar_volume: float,
    max_names: int,
    benchmark: str,
) -> pd.DataFrame:
    """Boolean frame, same shape as ``close``. True means the name may be held.

    A name qualifies when, as of the prior close, it has ``min_history_days``
    closes and its trailing median dollar volume clears ``min_dollar_volume``.
    If more than ``max_names`` qualify, only the most liquid of them are kept.
    The benchmark is kept whenever it has a prior close, and it counts toward
    the cap.
    """
    if max_names < 1:
        raise ValueError("max_names must be at least 1")
    window = max(int(min_history_days), 1)
    close = close.sort_index()
    volume = volume.reindex(index=close.index, columns=close.columns)
    history = close.notna().cumsum().shift(1).fillna(0.0)
    past_dollar = (close * volume).shift(1)
    trail = past_dollar.rolling(window, min_periods=window).median()

    eligible = (history >= window) & (trail >= float(min_dollar_volume))
    score = trail.to_numpy(dtype=np.float64, copy=True)
    elig = eligible.to_numpy(dtype=bool, copy=True)
    if benchmark in close.columns:
        b = close.columns.get_loc(benchmark)
        live = history[benchmark].to_numpy(dtype=np.float64) >= 1.0
        elig[live, b] = True
        score[live, b] = np.inf

    score = np.where(elig, score, -np.inf)
    order = np.argsort(-score, axis=1, kind="mergesort")
    rank = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    rank[rows, order] = np.arange(order.shape[1])[None, :]
    keep = elig & (rank < int(max_names)) & (score > -np.inf)
    return pd.DataFrame(keep, index=close.index, columns=close.columns)
