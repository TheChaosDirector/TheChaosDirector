"""Decision / experience log: save choices and grade them later."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.experience.decisions import (
    append_decision,
    build_decision_record,
    export_training_table,
    iter_decisions,
    label_outcomes,
    status,
)


def _write_tiny_lake(lake_dir: Path) -> None:
    lake_dir.mkdir(parents=True, exist_ok=True)
    prices = lake_dir / "prices"
    prices.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    for ticker, px0, step in [("AAA", 100.0, 10.0), ("SPY", 200.0, 2.0)]:
        px = [px0, px0 + step, px0 + 2 * step]
        df = pd.DataFrame(
            {
                "date": dates,
                "open": px,
                "high": px,
                "low": px,
                "close": px,
                "volume": [1_000_000] * 3,
            }
        )
        df.to_parquet(prices / f"{ticker}.parquet", index=False)
    (lake_dir / "manifest.json").write_text(
        json.dumps({"tickers": ["AAA", "SPY"], "benchmark": "SPY", "n_tickers": 2})
    )


def test_append_and_label_decision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"), mode="portfolio")
    _write_tiny_lake(Path(cfg.paths.lake_dir))

    tickers = ["AAA", "SPY"]
    weights = np.array([1.0, 0.0], dtype=np.float32)
    rec = build_decision_record(
        cfg=cfg,
        as_of="2024-01-02",
        tickers=tickers,
        obs=np.zeros(4, dtype=np.float32),
        action=np.array([1.0, 0.0], dtype=np.float32),
        weights=weights,
        targets={"AAA": 1.0},
        current_weights={},
        equity=10_000.0,
        cash=0.0,
        orders=[],
        executed=True,
        source="unit_test",
    )
    path = append_decision(cfg, rec)
    assert path.exists()
    assert status(cfg)["n_decisions"] == 1
    assert status(cfg)["n_labeled"] == 0

    # Re-append same day replaces, does not duplicate.
    append_decision(cfg, rec)
    assert status(cfg)["n_decisions"] == 1

    labeled = label_outcomes(cfg, horizon_days=1)
    assert labeled["labeled_now"] == 1
    rows = iter_decisions(cfg)
    assert abs(rows[0]["outcome"]["portfolio_return"] - 0.1) < 1e-9  # 100 -> 110
    assert abs(rows[0]["outcome"]["benchmark_return"] - 0.01) < 1e-9  # 200 -> 202

    out = export_training_table(cfg, labeled_only=True)
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert json.loads(path.read_text().splitlines()[0])["as_of"] == "2024-01-02"
