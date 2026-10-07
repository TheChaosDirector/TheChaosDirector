"""Day-trade gym: must pick ONE name, buy the open, sell the close.

Rules (plain English):

- Every morning the agent MUST choose exactly one stock/ETF. Sitting in cash
  is not allowed.
- It also chooses whether to mark the trade as ``forced`` ("I wanted to hold,
  but the rules made me go in"). That flag is logged only — it never cancels
  the trade.
- Fill at today's open, exit at today's close. Costs are charged on both legs.
- Clues are the lagged feature panel (prior-day information only).
- Reward is mainly *excess* open→close return vs the benchmark after costs,
  so the brain is graded on stock-picking skill, not just market weather.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.config import CostsCfg, DaytradeCfg, RiskCfg
from src.features.factory import FeaturePanel


def resolve_pick(
    action: np.ndarray | int,
    valid_mask: np.ndarray,
) -> tuple[int, bool]:
    """Map a MultiDiscrete / (ticker, forced) action onto a valid name.

    If the chosen ticker can't trade today, fall back to the first valid name
    (benchmark is always expected to be valid on market days).
    """
    action = np.asarray(action).reshape(-1)
    if action.size == 1:
        raw_idx = int(action[0])
        forced = False
    else:
        raw_idx = int(action[0])
        forced = bool(int(action[1]))

    n = len(valid_mask)
    raw_idx = int(np.clip(raw_idx, 0, n - 1))
    if valid_mask[raw_idx] > 0:
        return raw_idx, forced

    valid_idxs = np.flatnonzero(valid_mask > 0)
    if len(valid_idxs) == 0:
        return 0, forced
    return int(valid_idxs[0]), forced


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
        benchmark: str | None = None,
        randomize_episodes: bool = False,
        min_episode_days: int = 63,
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
        self.cost_rate = costs.total_bps / 1e4 * cost_multiplier
        self.randomize_episodes = randomize_episodes
        self.min_episode_days = min(min_episode_days, max(5, end - start))

        self.n_assets = len(panel.tickers)
        self.benchmark = benchmark or panel.tickers[0]
        if self.benchmark not in panel.tickers:
            raise ValueError(f"Benchmark {self.benchmark} not in panel tickers")
        self.bench_idx = panel.tickers.index(self.benchmark)

        obs_dim = self.n_assets * panel.n_features + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.MultiDiscrete([self.n_assets, 2])
        self.np_random = np.random.default_rng(0)

        self._reset_state()

    def _reset_state(self) -> None:
        span = self.end - self.start
        if self.randomize_episodes and span > self.min_episode_days:
            ep_len = int(self.np_random.integers(self.min_episode_days, span + 1))
            max_start = self.end - ep_len
            self.t = int(self.np_random.integers(self.start, max_start + 1))
            self.episode_end = self.t + ep_len
        else:
            self.t = self.start
            self.episode_end = self.end
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
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        t = self.t
        valid = self._valid_mask(t)
        idx, forced = resolve_pick(action, valid)

        o = float(self.opens[t, idx])
        c = float(self.closes[t, idx])
        gross_ret = c / o - 1.0
        cost = 2.0 * self.cost_rate
        net_ret = gross_ret - cost

        bo = float(self.opens[t, self.bench_idx])
        bc = float(self.closes[t, self.bench_idx])
        if np.isfinite(bo) and np.isfinite(bc) and bo > 0:
            bench_gross = bc / bo - 1.0
            bench_net = bench_gross - cost
        else:
            bench_gross = 0.0
            bench_net = -cost

        self.equity *= 1.0 + net_ret
        self.peak = max(self.peak, self.equity)
        new_dd = 1.0 - self.equity / self.peak
        dd_increment = max(0.0, new_dd - self.drawdown)
        self.drawdown = new_dd

        excess = net_ret - bench_net
        reward = (
            float(excess)
            + 0.5 * float(net_ret)
            - self.risk.drawdown_penalty * dd_increment
        )

        self.history.append(
            {
                "date": self.close.index[t],
                "ticker": self.panel.tickers[idx],
                "open": o,
                "close": c,
                "gross_return": gross_ret,
                "net_return": net_ret,
                "bench_net_return": bench_net,
                "excess_return": excess,
                "cost": cost,
                "turnover": 2.0,
                "equity": self.equity,
                "reluctance": 1.0 if forced else 0.0,
                "forced": forced,
            }
        )

        self.t += 1
        terminated = self.t >= self.episode_end
        if terminated:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_obs = self._obs()
        return next_obs, reward, terminated, False, {}

    def results(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history).set_index("date")
