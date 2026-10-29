"""Save live/paper choices so they can become future training data.

Plain English:
- Every day the model picks a portfolio, we write that choice down.
- Later we can stamp how that choice actually did (the "outcome").
- Those stamped rows are homework the next training run can study.

Files (git-friendly, not under gitignored artifacts/):
  experience/<mode>/decisions.jsonl   one JSON object per decision day
  experience/<mode>/index.csv         quick human/table view
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.data.lake import Lake
from src.registry.store import load_champion_meta

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def experience_dir(cfg: Config) -> Path:
    return Path("experience") / cfg.mode


def decisions_path(cfg: Config) -> Path:
    return experience_dir(cfg) / "decisions.jsonl"


def index_path(cfg: Config) -> Path:
    return experience_dir(cfg) / "index.csv"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def append_decision(cfg: Config, record: dict[str, Any]) -> Path:
    """Append one decision day. Idempotent on (source, as_of, executed)."""
    path = decisions_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = dict(record)
    record.setdefault("schema_version", SCHEMA_VERSION)
    record.setdefault("mode", cfg.mode)
    record.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    record.setdefault("outcome", None)

    existing = list(iter_decisions(cfg))
    key = (record.get("source"), record.get("as_of"), bool(record.get("executed")))
    kept = [
        r
        for r in existing
        if (r.get("source"), r.get("as_of"), bool(r.get("executed"))) != key
    ]
    # Preserve prior outcome if we are re-saving the same day.
    for r in existing:
        if (r.get("source"), r.get("as_of"), bool(r.get("executed"))) == key and r.get("outcome"):
            record["outcome"] = r["outcome"]
            break
    kept.append(record)

    lines = [json.dumps(r, default=_json_default) for r in kept]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    _rewrite_index(cfg, kept)
    log.info("Saved decision for %s → %s", record.get("as_of"), path)
    return path


def iter_decisions(cfg: Config) -> list[dict[str, Any]]:
    path = decisions_path(cfg)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _rewrite_index(cfg: Config, records: list[dict[str, Any]] | None = None) -> None:
    records = records if records is not None else iter_decisions(cfg)
    rows = []
    for r in records:
        outcome = r.get("outcome") or {}
        top = r.get("top") or []
        rows.append(
            {
                "as_of": r.get("as_of"),
                "source": r.get("source"),
                "executed": r.get("executed"),
                "equity": r.get("equity"),
                "n_targets": r.get("n_targets"),
                "cash_weight": r.get("cash_weight"),
                "n_orders": r.get("n_orders"),
                "top": "; ".join(f"{t}:{w:.1%}" for t, w in top[:5]),
                "labeled": bool(outcome),
                "horizon_days": outcome.get("horizon_days"),
                "portfolio_return": outcome.get("portfolio_return"),
                "benchmark_return": outcome.get("benchmark_return"),
                "excess_return": outcome.get("excess_return"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and "as_of" in df.columns:
        df = df.sort_values("as_of")
    index_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(index_path(cfg), index=False)


def status(cfg: Config) -> dict[str, Any]:
    records = iter_decisions(cfg)
    labeled = sum(1 for r in records if r.get("outcome"))
    return {
        "mode": cfg.mode,
        "path": str(decisions_path(cfg)),
        "n_decisions": len(records),
        "n_labeled": labeled,
        "n_unlabeled": len(records) - labeled,
        "as_of_min": min((r.get("as_of") for r in records), default=None),
        "as_of_max": max((r.get("as_of") for r in records), default=None),
        "plain_english": (
            f"Saved {len(records)} choice-days "
            f"({labeled} already graded with real outcomes, "
            f"{len(records) - labeled} still waiting for later prices)."
        ),
    }


def label_outcomes(cfg: Config, *, horizon_days: int = 1) -> dict[str, Any]:
    """Stamp each unlabeled decision with what happened next in the price lake."""
    records = iter_decisions(cfg)
    if not records:
        return {"labeled_now": 0, "still_unlabeled": 0, "plain_english": "No decisions saved yet."}

    lake = Lake(cfg)
    close = lake.close.sort_index()
    bench = cfg.universe.benchmark
    if bench not in close.columns:
        raise RuntimeError(f"Benchmark {bench} missing from lake — run download first.")

    labeled_now = 0
    still = 0
    updated: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("outcome"):
            updated.append(rec)
            continue
        as_of = pd.Timestamp(rec["as_of"])
        if as_of not in close.index:
            # nearest prior session
            prior = close.index[close.index <= as_of]
            if len(prior) == 0:
                still += 1
                updated.append(rec)
                continue
            as_of = prior[-1]

        loc = close.index.get_loc(as_of)
        if isinstance(loc, slice):
            loc = loc.start
        end = loc + int(horizon_days)
        if end >= len(close.index):
            still += 1
            updated.append(rec)
            continue

        start_px = close.iloc[loc]
        end_px = close.iloc[end]
        weights = rec.get("weights") or {}
        port_ret = 0.0
        w_sum = 0.0
        for sym, w in weights.items():
            if sym not in start_px.index:
                continue
            s = float(start_px[sym])
            e = float(end_px[sym])
            if not np.isfinite(s) or not np.isfinite(e) or s <= 0:
                continue
            port_ret += float(w) * (e / s - 1.0)
            w_sum += float(w)
        # leftover treated as cash (0 return)
        b0 = float(start_px[bench])
        b1 = float(end_px[bench])
        bench_ret = (b1 / b0 - 1.0) if b0 > 0 and np.isfinite(b0) and np.isfinite(b1) else float("nan")

        rec = dict(rec)
        rec["outcome"] = {
            "horizon_days": int(horizon_days),
            "entry_date": str(pd.Timestamp(close.index[loc]).date()),
            "exit_date": str(pd.Timestamp(close.index[end]).date()),
            "portfolio_return": float(port_ret),
            "benchmark_return": float(bench_ret) if np.isfinite(bench_ret) else None,
            "excess_return": float(port_ret - bench_ret) if np.isfinite(bench_ret) else None,
            "weight_sum_priced": float(w_sum),
            "labeled_at": datetime.now(timezone.utc).isoformat(),
        }
        labeled_now += 1
        updated.append(rec)

    lines = [json.dumps(r, default=_json_default) for r in updated]
    _atomic_write_text(decisions_path(cfg), "\n".join(lines) + ("\n" if lines else ""))
    _rewrite_index(cfg, updated)
    return {
        "labeled_now": labeled_now,
        "still_unlabeled": still,
        "n_decisions": len(updated),
        "horizon_days": horizon_days,
        "plain_english": (
            f"Graded {labeled_now} choice-days with the next {horizon_days} market day return(s); "
            f"{still} still waiting for prices that haven't happened yet."
        ),
    }


def export_training_table(cfg: Config, *, labeled_only: bool = True) -> Path:
    """Flatten decisions into a parquet the next training recipe can read."""
    records = iter_decisions(cfg)
    rows = []
    for r in records:
        if labeled_only and not r.get("outcome"):
            continue
        outcome = r.get("outcome") or {}
        rows.append(
            {
                "as_of": r.get("as_of"),
                "source": r.get("source"),
                "executed": r.get("executed"),
                "obs": r.get("obs"),
                "action": r.get("action"),
                "weights": r.get("weights"),
                "tickers": r.get("tickers"),
                "portfolio_return": outcome.get("portfolio_return"),
                "benchmark_return": outcome.get("benchmark_return"),
                "excess_return": outcome.get("excess_return"),
                "horizon_days": outcome.get("horizon_days"),
                "champion_fold": (r.get("champion") or {}).get("fold"),
            }
        )
    out = experience_dir(cfg) / ("training_ready.parquet" if labeled_only else "all_decisions.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def build_decision_record(
    *,
    cfg: Config,
    as_of: str,
    tickers: list[str],
    obs: np.ndarray,
    action: np.ndarray,
    weights: np.ndarray,
    targets: dict[str, float],
    current_weights: dict[str, float],
    equity: float,
    cash: float,
    orders: list[dict],
    executed: bool,
    source: str = "alpaca_paper",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pack one trading day's choice into a training-friendly record."""
    try:
        champ = load_champion_meta(cfg)
        champion = {
            "fold": champ.get("fold"),
            "model_path": champ.get("model_path"),
            "learner": champ.get("learner"),
        }
    except FileNotFoundError:
        champion = None

    weight_map = {t: float(w) for t, w in zip(tickers, weights) if float(w) > 1e-12}
    top = sorted(targets.items(), key=lambda kv: -kv[1])[:10]
    rec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "mode": cfg.mode,
        "as_of": as_of,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "executed": bool(executed),
        "champion": champion,
        "tickers": list(tickers),
        "obs": np.asarray(obs, dtype=np.float32).tolist(),
        "action": np.asarray(action, dtype=np.float32).tolist(),
        "weights": weight_map,
        "weight_vector": np.asarray(weights, dtype=np.float32).tolist(),
        "targets": {k: float(v) for k, v in targets.items()},
        "current_weights": {k: float(v) for k, v in current_weights.items()},
        "top": top,
        "n_targets": len(targets),
        "cash_weight": float(max(0.0, 1.0 - sum(targets.values()))),
        "equity": float(equity),
        "cash": float(cash),
        "n_orders": len(orders),
        "orders": orders,
        "outcome": None,
    }
    if extra:
        rec.update(extra)
    return rec


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")
