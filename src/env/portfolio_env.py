"""The trading gym: a daily portfolio game the agent plays to learn.

Rules of the game, in plain English:

- Each morning the agent sees yesterday's-and-older market clues (the feature
  panel is already lagged) plus its own current positions.
- It answers one question: "what fraction of the portfolio goes into each
  name, and how much stays in cash?" (long-only, capped per name).
- Trades execute at today's close; the portfolio then earns tomorrow's move.
- Every trade costs money (commission + slippage on the value traded).
- Default score (reward) is the day's log profit, minus a penalty whenever the
  portfolio digs itself into a *new* deepest drawdown — so the agent learns
  that steady gains beat wild swings.
- Concentrated mode can also score excess vs a benchmark, keep only top-k names,
  and force a minimum invested fraction so it cannot hide in cash.
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
    max_names: int | None = None,
    min_gross: float | None = None,
) -> np.ndarray:
    """Map raw agent output (logits over names + cash) to legal weights.

    Softmax turns arbitrary numbers into a budget that sums to 1; the per-name
    cap and gross-exposure cap are then enforced, with any excess parked in
    cash. Names without a tradable price today are forced to zero.

    If ``max_names`` is set, only the top-k weights (among valid names) are kept;
    the rest are zeroed and caps are re-applied.

    If ``min_gross`` is set, scale (or force-pick) so invested weight is at least
    that fraction — cash cannot eat the book.
    """
    logits = np.clip(np.asarray(action, dtype=np.float64), -10.0, 10.0)
    exp = np.exp(logits - logits.max())
    budget = exp / exp.sum()

    weights = budget[:-1] * valid_mask  # last slot is cash
    name_logits = logits[:-1]

    if max_names is not None and max_names > 0 and len(weights) > max_names:
        # Rank only among currently valid names so dead tickers don't steal slots.
        scores = np.where(valid_mask > 0, weights, -np.inf)
        # If cash ate almost everything, fall back to raw name logits for ranking.
        if not np.isfinite(scores).any() or float(np.nanmax(scores)) <= 1e-12:
            scores = np.where(valid_mask > 0, name_logits, -np.inf)
        keep_idx = np.argpartition(scores, -max_names)[-max_names:]
        mask = np.zeros_like(weights)
        keep_idx = keep_idx[np.isfinite(scores[keep_idx])]
        mask[keep_idx] = 1.0
        weights = weights * mask

    weights = np.minimum(weights, max_weight)
    gross = float(weights.sum())
    if gross > max_gross and gross > 0:
        weights *= max_gross / gross
        gross = float(weights.sum())

    if min_gross is not None and min_gross > 0:
        target = min(float(min_gross), float(max_gross))
        weights = _enforce_min_gross(weights, valid_mask, name_logits, max_weight, target, max_names)

    return weights


def _enforce_min_gross(
    weights: np.ndarray,
    valid_mask: np.ndarray,
    name_logits: np.ndarray,
    max_weight: float,
    min_gross: float,
    max_names: int | None,
) -> np.ndarray:
    """Push invested weight up to min_gross among a concentrated sleeve."""
    w = weights.astype(np.float64).copy()
    gross = float(w.sum())
    if gross + 1e-12 >= min_gross:
        return w

    # If basically all cash, force-pick top names by logits and seed equal weights.
    if gross <= 1e-12:
        k = max_names if max_names and max_names > 0 else int(valid_mask.sum())
        k = max(1, min(k, int(valid_mask.sum()) or 1))
        scores = np.where(valid_mask > 0, name_logits, -np.inf)
        if not np.isfinite(scores).any():
            return w
        keep_idx = np.argpartition(scores, -k)[-k:]
        keep_idx = keep_idx[np.isfinite(scores[keep_idx])]
        if len(keep_idx) == 0:
            return w
        seed = min(min_gross / len(keep_idx), max_weight)
        w[:] = 0.0
        w[keep_idx] = seed
        gross = float(w.sum())

    if gross + 1e-12 >= min_gross:
        return np.minimum(w, max_weight)

    # Scale up held names, then iteratively clip at max_weight and refill room.
    held = w > 1e-12
    if not held.any():
        return w
    w[held] *= min_gross / gross
    for _ in range(8):
        w = np.minimum(w, max_weight)
        shortfall = min_gross - float(w.sum())
        if shortfall <= 1e-12:
            break
        room = np.where(held & (valid_mask > 0), max_weight - w, 0.0)
        room_sum = float(room.sum())
        if room_sum <= 1e-12:
            break
        w += room * (shortfall / room_sum)
    return np.minimum(w, max_weight)


def apply_hold_deadband(
    prev: np.ndarray,
    target: np.ndarray,
    deadband: float,
    *,
    max_weight: float,
    max_gross: float,
    min_gross: float | None,
    name_logits: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    max_names: int | None = None,
) -> np.ndarray:
    """Keep yesterday's weight when the proposed change is tiny (cuts fidgeting)."""
    if deadband <= 0:
        return target
    prev = np.asarray(prev, dtype=np.float64)
    out = np.asarray(target, dtype=np.float64).copy()
    if prev.shape != out.shape:
        return target
    small = np.abs(out - prev) < deadband
    out = np.where(small, prev, out)
    out = np.minimum(np.maximum(out, 0.0), max_weight)
    gross = float(out.sum())
    if gross > max_gross and gross > 0:
        out *= max_gross / gross
    if min_gross is not None and min_gross > 0:
        logits = name_logits if name_logits is not None else out
        mask = valid_mask if valid_mask is not None else np.ones_like(out)
        out = _enforce_min_gross(out, mask, logits, max_weight, min(min_gross, max_gross), max_names)
    return out


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
        *,
        benchmark_returns: np.ndarray | None = None,
        max_names: int | None = None,
        min_gross: float | None = None,
        excess_reward_weight: float = 0.0,
        absolute_reward_weight: float = 1.0,
        turnover_penalty: float = 0.0,
        hold_deadband: float = 0.0,
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
        self.max_names = max_names
        self.min_gross = min_gross
        self.excess_reward_weight = float(excess_reward_weight)
        self.absolute_reward_weight = float(absolute_reward_weight)
        self.turnover_penalty = float(turnover_penalty)
        self.hold_deadband = float(hold_deadband)

        # Per-bar close-to-close benchmark return aligned to env step index t
        # (reward at t uses return from t → t+1). Length must cover [start, end).
        if benchmark_returns is None:
            self.benchmark_returns = None
        else:
            br = np.asarray(benchmark_returns, dtype=np.float64)
            if len(br) != len(close):
                raise ValueError(
                    f"benchmark_returns length {len(br)} != close length {len(close)}"
                )
            self.benchmark_returns = br

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
            action,
            valid,
            self.risk.max_weight_per_name,
            self.risk.max_gross_exposure,
            max_names=self.max_names,
            min_gross=self.min_gross,
        )
        if self.hold_deadband > 0 and float(self.weights.sum()) > 1e-9:
            logits = np.asarray(action, dtype=np.float64)
            name_logits = logits[:-1] if len(logits) == self.n_assets + 1 else target
            target = apply_hold_deadband(
                self.weights,
                target,
                self.hold_deadband,
                max_weight=self.risk.max_weight_per_name,
                max_gross=self.risk.max_gross_exposure,
                min_gross=self.min_gross,
                name_logits=name_logits,
                valid_mask=valid,
                max_names=self.max_names,
            )

        turnover = float(np.abs(target - self.weights).sum())
        cost = turnover * self.cost_rate

        rets = np.where(valid > 0, self.prices[t + 1] / self.prices[t] - 1.0, 0.0)
        gross_ret = float((target * rets).sum())
        net_ret = gross_ret - cost

        if self.benchmark_returns is not None:
            bench_ret = float(self.benchmark_returns[t])
            if not np.isfinite(bench_ret):
                bench_ret = 0.0
        else:
            bench_ret = 0.0
        excess = net_ret - bench_ret

        self.equity *= 1.0 + net_ret
        self.peak = max(self.peak, self.equity)
        new_dd = 1.0 - self.equity / self.peak
        dd_increment = max(0.0, new_dd - self.drawdown)
        self.drawdown = new_dd

        abs_term = float(np.log(max(1.0 + net_ret, 1e-6)))
        if self.benchmark_returns is None or (
            self.excess_reward_weight == 0.0 and self.absolute_reward_weight == 1.0
        ):
            # Classic portfolio reward (unchanged default).
            reward = abs_term - self.risk.drawdown_penalty * dd_increment
        else:
            reward = (
                self.excess_reward_weight * excess
                + self.absolute_reward_weight * abs_term
                - self.risk.drawdown_penalty * dd_increment
            )
        if self.turnover_penalty > 0:
            reward -= self.turnover_penalty * turnover

        # Positions drift with prices until the next rebalance.
        growth = 1.0 + net_ret
        self.weights = (target * (1.0 + rets)) / growth if growth > 0 else target

        row = {
            "date": self.close.index[t],
            "net_return": net_ret,
            "gross_return": gross_ret,
            "cost": cost,
            "turnover": turnover,
            "equity": self.equity,
            "gross_exposure": float(target.sum()),
            "weights": target.copy(),
        }
        if self.benchmark_returns is not None:
            row["benchmark_return"] = bench_ret
            row["excess_return"] = excess
        self.history.append(row)

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
