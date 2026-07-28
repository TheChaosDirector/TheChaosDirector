"""Turn raw prices into the "clues" the agent sees — without leaking the future.

The one rule that matters: **the feature row for day T may only use data
through day T-1.** Every feature below is computed on raw history and then
shifted forward one day, so when the agent makes a decision on day T it is
standing where a real trader would stand on that morning.

The factory is a pure function of (close, volume) matrices, which lets the
referee re-run it on truncated data and verify byte-for-byte that no feature
ever changes when the future is deleted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import FeaturesCfg


@dataclass
class FeaturePanel:
    """Stacked features: values[t, n, f] for date t, ticker n, feature f."""

    dates: pd.DatetimeIndex
    tickers: list[str]
    names: list[str]
    values: np.ndarray  # shape (T, N, F), float32

    @property
    def n_features(self) -> int:
        return len(self.names)

    def frame(self, name: str) -> pd.DataFrame:
        f = self.names.index(name)
        return pd.DataFrame(self.values[:, :, f], index=self.dates, columns=self.tickers)

    def row(self, when: pd.Timestamp) -> np.ndarray:
        t = self.dates.get_loc(pd.Timestamp(when))
        return self.values[t]


def _zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = df.rolling(window).mean()
    std = df.rolling(window).std()
    return (df - mean) / std.replace(0.0, np.nan)


def build_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    fcfg: FeaturesCfg,
    benchmark: str,
    extra_unlagged: dict[str, pd.DataFrame] | None = None,
) -> FeaturePanel:
    """Build the leakage-safe feature panel.

    ``extra_unlagged`` exists only for the referee's own tests: it lets a test
    plant a deliberately cheating feature (NOT shifted) and prove it gets
    caught. Normal pipelines never pass it.
    """
    rets = close.pct_change()
    feats: dict[str, pd.DataFrame] = {}

    for w in fcfg.momentum_windows:
        feats[f"mom_{w}d"] = close.pct_change(w)

    feats[f"vol_{fcfg.vol_window}d"] = rets.rolling(fcfg.vol_window).std() * np.sqrt(252)
    feats[f"volume_z_{fcfg.volume_window}d"] = _zscore(volume, fcfg.volume_window)

    for w in fcfg.ma_windows:
        ma = close.rolling(w).mean()
        feats[f"ma_dist_{w}d"] = close / ma - 1.0

    # Cross-sectional rank: how this name's recent momentum compares to peers
    # on the same day (0 = weakest, 1 = strongest), centered at zero.
    mom_key = f"mom_{fcfg.momentum_windows[1] if len(fcfg.momentum_windows) > 1 else fcfg.momentum_windows[0]}d"
    feats["xsec_mom_rank"] = feats[mom_key].rank(axis=1, pct=True) - 0.5

    # Market regime, broadcast to every ticker: is the benchmark in an uptrend,
    # and how rough are markets overall right now?
    bench = close[benchmark]
    regime_up = (bench > bench.rolling(fcfg.regime_ma).mean()).astype(float)
    bench_vol = bench.pct_change().rolling(fcfg.vol_window).std() * np.sqrt(252)
    feats["regime_up"] = pd.DataFrame(
        np.tile(regime_up.values[:, None], (1, close.shape[1])),
        index=close.index,
        columns=close.columns,
    )
    feats["regime_vol"] = pd.DataFrame(
        np.tile(bench_vol.values[:, None], (1, close.shape[1])),
        index=close.index,
        columns=close.columns,
    )

    # THE anti-cheat step: shift everything so day T sees only data
    # through day T-1.
    feats = {k: v.shift(1) for k, v in feats.items()}

    if extra_unlagged:
        feats.update(extra_unlagged)

    names = list(feats.keys())
    stacked = np.stack(
        [feats[k].reindex(index=close.index, columns=close.columns).values for k in names],
        axis=-1,
    ).astype(np.float32)
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

    return FeaturePanel(
        dates=close.index,
        tickers=list(close.columns),
        names=names,
        values=stacked,
    )


def warmup_days(fcfg: FeaturesCfg) -> int:
    """How many leading days are unreliable while rolling windows fill up."""
    return max(
        max(fcfg.momentum_windows),
        fcfg.vol_window,
        fcfg.volume_window,
        max(fcfg.ma_windows),
        fcfg.regime_ma,
    ) + 1
