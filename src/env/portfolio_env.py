"""The trading gym: a daily portfolio game the agent plays to learn.

Rules of the game, in plain English:

- Each morning the agent sees yesterday's-and-older market clues (the feature
  panel is already lagged) plus its own current positions.
- It answers one question: "what fraction of the portfolio goes into each
  name, and how much stays in cash?" (long-only, capped per name).
- Trades execute at today's close; the portfolio then earns tomorrow's move.
- Every trade costs money (commission + slippage on the value traded).
- The score (reward) is the day's log profit, minus a penalty whenever the
  portfolio digs itself into a *new* deepest drawdown — so the agent learns
  that steady gains beat wild swings.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.config import CostsCfg, RiskCfg
from src.features.factory import FeaturePanel


def action_to_weights(
    action: np.ndarray,
    valid_mask: np.ndarray,
    max_weight: float,
    max_gross: float,
) -> np.ndarray:
    """Map raw agent output (logits over names + cash) to legal weights.

    Softmax turns arbitrary numbers into a budget that sums to 1; the per-name
    cap and gross-exposure cap are then enforced, with any excess parked in
    cash. Names without a tradable price today are forced to zero.
    """
    logits = np.clip(np.asarray(action, dtype=np.float64), -10.0, 10.0)
    exp = np.exp(logits - logits.max())
    budget = exp / exp.sum()

    weights = budget[:-1] * valid_mask  # last slot is cash
    weights = np.minimum(weights, max_weight)
    gross = weights.sum()
    if gross > max_gross and gross > 0:
        weights *= max_gross / gross
    return weights


class PortfolioEnv(gym.Env):
    """Gymnasium environment over a contiguous window of market history."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        close: pd.DataFrame,
        panel: FeaturePanel,
        costs: CostsCfg,
        risk: RiskCfg,
        start: int,
        end: int,
        cost_multiplier: float = 1.0,
    ):
        super().__init__()
        assert list(close.columns) == panel.tickers
        assert 0 <= start < end <= len(close) - 1, "need t+1 prices for every step"

        self.close = close
        self.prices = close.values.astype(np.float64)
        self.panel = panel
        self.costs = costs
        self.risk = risk
        self.start = start
        self.end = end
        self.cost_rate = costs.total_bps / 1e4 * cost_multiplier

        self.n_assets = len(panel.tickers)
        obs_dim = self.n_assets * panel.n_features + self.n_assets + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (self.n_assets + 1,), np.float32)

        self._reset_state()

    def _reset_state(self) -> None:
        self.t = self.start
        self.weights = np.zeros(self.n_assets)  # 100% cash
        self.equity = 1.0
        self.peak = 1.0
        self.drawdown = 0.0
        self.history: list[dict] = []

    def _valid_mask(self, t: int) -> np.ndarray:
        today = self.prices[t]
        tomorrow = self.prices[t + 1]
        return (np.isfinite(today) & np.isfinite(tomorrow)).astype(np.float64)

    def _obs(self) -> np.ndarray:
        flat = self.panel.values[self.t].reshape(-1)
        state = np.concatenate(
            [
                self.weights,
                [1.0 - self.weights.sum()],  # cash
                [self.drawdown],
            ]
        )
        return np.concatenate([flat, state]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        t = self.t
        valid = self._valid_mask(t)
        target = action_to_weights(
            action, valid, self.risk.max_weight_per_name, self.risk.max_gross_exposure
        )

        turnover = float(np.abs(target - self.weights).sum())
        cost = turnover * self.cost_rate

        rets = np.where(valid > 0, self.prices[t + 1] / self.prices[t] - 1.0, 0.0)
        gross_ret = float((target * rets).sum())
        net_ret = gross_ret - cost

        self.equity *= 1.0 + net_ret
        self.peak = max(self.peak, self.equity)
        new_dd = 1.0 - self.equity / self.peak
        dd_increment = max(0.0, new_dd - self.drawdown)
        self.drawdown = new_dd

        reward = float(np.log(max(1.0 + net_ret, 1e-6))) - self.risk.drawdown_penalty * dd_increment

        # Positions drift with prices until the next rebalance.
        growth = 1.0 + net_ret
        self.weights = (target * (1.0 + rets)) / growth if growth > 0 else target

        self.history.append(
            {
                "date": self.close.index[t],
                "net_return": net_ret,
                "gross_return": gross_ret,
                "cost": cost,
                "turnover": turnover,
                "equity": self.equity,
                "weights": target.copy(),
            }
        )

        self.t += 1
        terminated = self.t >= self.end
        return self._obs() if not terminated else np.zeros_like(self._obs()), reward, terminated, False, {}

    def results(self) -> pd.DataFrame:
        """Trade-by-trade journal of the episode just played."""
        if not self.history:
            return pd.DataFrame()
        df = pd.DataFrame(
            [{k: v for k, v in h.items() if k != "weights"} for h in self.history]
        ).set_index("date")
        return df

    def weight_history(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(
            [h["weights"] for h in self.history],
            index=[h["date"] for h in self.history],
            columns=self.panel.tickers,
        )
