"""Historical replay: grade a frozen champion over the full data lake.

This answers "if this exact brain had been running for the whole history,
what would the account look like?" — always side by side with the boring
alternative of buying and holding the benchmark.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.lake import Lake
from src.env.portfolio_env import PortfolioEnv
from src.features.factory import build_features, warmup_days
from src.metrics.scorecard import plain_english, scorecard
from src.registry.store import load_champion_model
from src.train.walkforward import rollout

log = logging.getLogger(__name__)


def backtest_dir(cfg: Config) -> Path:
    return cfg.artifacts_dir / "backtest"


def run_backtest(cfg: Config, cost_multiplier: float = 1.0, tag: str = "base") -> dict:
    lake = Lake(cfg)
    close = lake.close
    panel = build_features(close, lake.volume, cfg.features, lake.benchmark)

    model, meta = load_champion_model(cfg)
    if meta["tickers"] != panel.tickers:
        raise RuntimeError(
            "The champion was trained on a different ticker set than the current "
            "data lake. Re-run download + train with the same config."
        )

    start = warmup_days(cfg.features)
    end = len(close) - 1
    env = PortfolioEnv(
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        start=start,
        end=end,
        cost_multiplier=cost_multiplier,
    )
    journal = rollout(model, env)
    weights = env.weight_history()

    bench_ret = close[lake.benchmark].iloc[start : end + 1].pct_change().dropna()

    agent_card = scorecard(journal["net_return"], journal["turnover"])
    bench_card = scorecard(bench_ret)

    out = backtest_dir(cfg) / tag
    out.mkdir(parents=True, exist_ok=True)
    journal.to_csv(out / "journal.csv")
    weights.to_csv(out / "weights.csv")
    bench_equity = (1.0 + bench_ret).cumprod()
    bench_equity.to_csv(out / "benchmark_equity.csv")

    summary = {
        "tag": tag,
        "cost_multiplier": cost_multiplier,
        "champion_fold": meta["fold"],
        "period": [str(close.index[start].date()), str(close.index[end].date())],
        "agent": agent_card,
        "benchmark": bench_card,
        "agent_says": plain_english(agent_card, "the agent"),
        "benchmark_says": plain_english(bench_card, f"buy-and-hold {lake.benchmark}"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "Backtest [%s]: agent %.1f%% (Sharpe %.2f) vs %s %.1f%% (Sharpe %.2f)",
        tag,
        agent_card["total_return"] * 100,
        agent_card["sharpe"],
        lake.benchmark,
        bench_card["total_return"] * 100,
        bench_card["sharpe"],
    )
    return summary
