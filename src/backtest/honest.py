"""Even-split replay that can hold companies after they have died.

Half the book is the benchmark. The other half is split evenly across the
names that, as of the prior close, were already trading and liquid. A name
that later goes to zero stays in that list until its prices stop, so the loss
is in the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.eligibility import point_in_time_eligible

# The delisted file starts in January 2000. A three-year wait would open the
# book after Enron had already fallen from about $80 to pennies, so that loss
# would never be taken. One quarter of prior closes is enough to know the name
# was trading. The dollar-volume median still uses only the past.
HISTORY_DAYS = 63
MIN_DOLLAR_VOLUME = 5_000_000.0
MAX_NAMES = 120
# 1 bp commission + 10 bp slippage, same locked cost as the other book.
COST_RATE = 11.0 / 10_000.0
REBALANCE_EVERY = 5


def replay_even_split(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    benchmark: str,
    min_history_days: int,
    min_dollar_volume: float,
    max_names: int,
    cost_rate: float,
    rebalance_every: int = 5,
    rf: pd.Series | None = None,
    dollar_volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per day the book could earn a return. Does not write files."""
    if benchmark not in close.columns:
        raise KeyError(f"{benchmark} is not in the price frame")
    close = close.sort_index()
    mask = point_in_time_eligible(
        close,
        volume,
        min_history_days=min_history_days,
        min_dollar_volume=min_dollar_volume,
        max_names=max_names,
        benchmark=benchmark,
        dollar_volume=dollar_volume,
    )
    prices = close.to_numpy(dtype=np.float64)
    eligible = mask.to_numpy(dtype=bool)
    bench = close.columns.get_loc(benchmark)
    rates = np.zeros(len(close), dtype=np.float64)
    if rf is not None:
        rates[:] = rf.reindex(close.index).fillna(0.0).to_numpy(dtype=np.float64)
    step = max(int(rebalance_every), 1)
    n = prices.shape[1]
    weights = np.zeros(n, dtype=np.float64)
    rows: list[dict] = []
    # The last row has no next price to earn.
    for t in range(len(close) - 1):
        today = prices[t]
        nxt = prices[t + 1]
        tradable = eligible[t] & np.isfinite(today) & np.isfinite(nxt) & (today > 0) & (nxt > 0)
        if t % step == 0:
            target = np.zeros(n, dtype=np.float64)
            count = int(tradable.sum())
            if count:
                target[tradable] = 0.5 / count
            if tradable[bench]:
                target[bench] += 0.5
        else:
            target = weights
        returns = np.zeros(n, dtype=np.float64)
        ok = tradable
        returns[ok] = nxt[ok] / today[ok] - 1.0
        turnover = float(np.abs(target - weights).sum())
        cost = turnover * cost_rate
        cash = 1.0 - float(target.sum())
        gross = float((target * returns).sum()) + cash * rates[t]
        net = gross - cost
        growth = 1.0 + net
        weights = (target * (1.0 + returns)) / growth if growth > 0 else target
        spy_today = today[bench]
        spy_next = nxt[bench]
        spy = (
            spy_next / spy_today - 1.0
            if np.isfinite(spy_today) and np.isfinite(spy_next) and spy_today > 0 and spy_next > 0
            else 0.0
        )
        rows.append(
            {
                "date": close.index[t],
                "net_return": net,
                "benchmark_return": float(spy),
                "n_names": int(tradable.sum()),
                "turnover": turnover,
            }
        )
    return pd.DataFrame(rows).set_index("date")
