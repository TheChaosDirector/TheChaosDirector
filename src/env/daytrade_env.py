"""Day-trade gym: must pick ONE name, buy the open, sell the close.

Rules (plain English):

- Every morning the agent MUST choose exactly one stock/ETF. Sitting in cash
  is not allowed.
- It may emit a "reluctance" score. If that score is high, we log
  ``forced=True`` — meaning "I wanted to hold, but the rules made me go in."
  Reluctance never cancels the trade.
- Fill at today's open, exit at today's close. Costs are charged on both legs.
- Clues are the same lagged feature panel as the portfolio mode (prior-day
  information only).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.config import CostsCfg, DaytradeCfg, RiskCfg
from src.features.factory import FeaturePanel


def pick_one(
    action: np.ndarray,
    valid_mask: np.ndarray,
    forced_threshold: float,
) -> tuple[int, float, bool]:
    """Map raw action → (ticker_index, reluctance_01, forced_flag).

    The last action dimension is reluctance. The rest are scores over names;
    we take argmax among names that have a valid open and close today.
    """
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    scores = action[:-1]
    reluctance_raw = float(action[-1])
    # Squash to 0..1 so the threshold is human-readable.
    reluctance = 1.0 / (1.0 + np.exp(-reluctance_raw))

    masked = np.where(valid_mask > 0, scores, -np.inf)
    if not np.isfinite(masked).any():
        # Absolute last resort: pick the first name that has any finite score,
        # else index 0. The env guarantees at least the benchmark is valid.
        idx = int(np.argmax(valid_mask)) if valid_mask.sum() > 0 else 0
    else:
        idx = int(np.argmax(masked))

    forced = reluctance >= forced_threshold
    return idx, float(reluctance), bool(forced)


class DaytradeEnv(gym.Env):
    """Must-pick open→close day-trader over a contiguous window."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        open_: pd.DataFrame,
        close: pd.DataFrame,
        panel: FeaturePanel,
        costs: CostsCfg,
        risk: RiskCfg,
        daytrade: DaytradeCfg,
        start: int,
        end: int,
        cost_multiplier: float = 1.0,
    ):
        super().__init__()
        assert list(close.columns) == panel.tickers
        assert list(open_.columns) == panel.tickers
        assert 0 <= start < end <= len(close)

        self.open_df = open_
        self.close = close
        self.opens = open_.values.astype(np.float64)
        self.closes = close.values.astype(np.float64)
        self.panel = panel
        self.costs = costs
        self.risk = risk
        self.daytrade = daytrade
        self.start = start
        self.end = end
        # Buy + sell ⇒ two charges of the per-leg cost rate.
        self.cost_rate = costs.total_bps / 1e4 * cost_multiplier

        self.n_assets = len(panel.tickers)
        obs_dim = self.n_assets * panel.n_features + 2  # + equity_norm, drawdown
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (self.n_assets + 1,), np.float32)

        self._reset_state()

    def _reset_state(self) -> None:
        self.t = self.start
        self.equity = 1.0
        self.peak = 1.0
        self.drawdown = 0.0
        self.history: list[dict] = []

    def _valid_mask(self, t: int) -> np.ndarray:
        o = self.opens[t]
        c = self.closes[t]
        return (np.isfinite(o) & np.isfinite(c) & (o > 0) & (c > 0)).astype(np.float64)

    def _obs(self) -> np.ndarray:
        flat = self.panel.values[self.t].reshape(-1)
        state = np.array([self.equity - 1.0, self.drawdown], dtype=np.float64)
        return np.concatenate([flat, state]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        t = self.t
        valid = self._valid_mask(t)
        idx, reluctance, forced = pick_one(action, valid, self.daytrade.forced_threshold)

        o = float(self.opens[t, idx])
        c = float(self.closes[t, idx])
        gross_ret = c / o - 1.0
        cost = 2.0 * self.cost_rate  # open buy + close sell
        net_ret = gross_ret - cost

        self.equity *= 1.0 + net_ret
        self.peak = max(self.peak, self.equity)
        new_dd = 1.0 - self.equity / self.peak
        dd_increment = max(0.0, new_dd - self.drawdown)
        self.drawdown = new_dd

        reward = float(np.log(max(1.0 + net_ret, 1e-6))) - self.risk.drawdown_penalty * dd_increment

        self.history.append(
            {
                "date": self.close.index[t],
                "ticker": self.panel.tickers[idx],
                "open": o,
                "close": c,
                "gross_return": gross_ret,
                "net_return": net_ret,
                "cost": cost,
                "turnover": 2.0,  # full book in + full book out
                "equity": self.equity,
                "reluctance": reluctance,
                "forced": forced,
            }
        )

        self.t += 1
        terminated = self.t >= self.end
        if terminated:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_obs = self._obs()
        return next_obs, reward, terminated, False, {}

    def results(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history).set_index("date")
