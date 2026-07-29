"""Daily portfolio rebalance against Alpaca paper (practice money).

Flow (plain English):
1. Refresh the local price lake (optional).
2. Ask the frozen portfolio champion for today's target weights.
3. Look at what the Alpaca paper account currently holds.
4. Place market orders to move toward those targets.
5. Journal everything under artifacts/<mode>/alpaca_paper/.

Safety rails:
- Paper endpoint only (no live money URL).
- dry-run by default until you pass ``--execute``.
- Skips tiny notional changes under ``min_notional``.
- Can refuse to trade when the market is closed (unless ``--force``).
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.broker.alpaca import AlpacaPaperClient
from src.config import Config
from src.data.download import run_download
from src.data.lake import Lake
from src.env.portfolio_env import action_to_weights
from src.experience.decisions import append_decision, build_decision_record
from src.features.factory import build_features
from src.registry.store import load_champion_model

log = logging.getLogger(__name__)


def alpaca_dir(cfg: Config) -> Path:
    return cfg.artifacts_dir / "alpaca_paper"


def _ensure_portfolio_champion(cfg: Config) -> None:
    """Migrate pre-mode-split artifacts into artifacts/portfolio if needed."""
    dest = cfg.artifacts_dir / "registry" / "champion.json"
    if dest.exists():
        return
    legacy = Path("artifacts/registry/champion.json")
    if legacy.exists() and cfg.mode == "portfolio":
        log.info("Copying legacy portfolio champion into %s", cfg.artifacts_dir / "registry")
        shutil.copytree(Path("artifacts/registry"), cfg.artifacts_dir / "registry", dirs_exist_ok=True)
        if Path("artifacts/backtest").exists():
            shutil.copytree(Path("artifacts/backtest"), cfg.artifacts_dir / "backtest", dirs_exist_ok=True)


def to_alpaca_symbol(yahoo_symbol: str) -> str:
    """Yahoo uses BRK-B; Alpaca uses BRK.B."""
    return yahoo_symbol.replace("-", ".")


def to_yahoo_symbol(alpaca_symbol: str) -> str:
    return alpaca_symbol.replace(".", "-")


def _latest_targets(cfg: Config) -> tuple[dict[str, float], dict, dict]:
    """Ask the frozen champion for target weights using the latest lake bar.

    Returns targets, info, and a decision_ctx blob (obs/action/weights) for the experience log.
    """
    lake = Lake(cfg)
    model, meta = load_champion_model(cfg)
    tickers = list(meta["tickers"])

    # Preserve training ticker order/dimension so the policy obs size matches.
    close = lake.close.reindex(columns=tickers)
    volume = lake.volume.reindex(columns=tickers)
    panel = build_features(close, volume, cfg.features, lake.benchmark)

    t = len(panel.dates) - 1
    obs_features = panel.values[t].reshape(-1)
    prev = np.zeros(len(tickers))
    obs = np.concatenate([obs_features, prev, [1.0], [0.0]]).astype(np.float32)
    action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    last_prices = close.iloc[-1].values.astype(float)
    valid = np.isfinite(last_prices).astype(float)
    weights = action_to_weights(
        action, valid, cfg.risk.max_weight_per_name, cfg.risk.max_gross_exposure
    )
    targets = {tck: float(w) for tck, w in zip(tickers, weights) if w > 1e-6}
    info = {
        "as_of": str(panel.dates[t].date()),
        "n_targets": len(targets),
        "cash_weight": float(1.0 - sum(targets.values())),
        "top": sorted(targets.items(), key=lambda kv: -kv[1])[:10],
    }
    decision_ctx = {
        "tickers": tickers,
        "obs": obs,
        "action": action,
        "weights": np.asarray(weights, dtype=np.float32),
    }
    return targets, info, decision_ctx


def _current_weights(positions: dict[str, dict], equity: float) -> dict[str, float]:
    """Alpaca positions keyed by Yahoo-style symbols for diffing vs model targets."""
    if equity <= 0:
        return {}
    out = {}
    for sym, pos in positions.items():
        if pos.get("market_value", 0) <= 0:
            continue
        out[to_yahoo_symbol(sym)] = pos["market_value"] / equity
    return out


def plan_rebalance(
    targets: dict[str, float],
    current: dict[str, float],
    equity: float,
    min_notional: float,
) -> list[dict]:
    """Turn weight diffs into dollar buy/sell orders (Alpaca symbols)."""
    symbols = sorted(set(targets) | set(current))
    orders = []
    for sym in symbols:
        delta_w = targets.get(sym, 0.0) - current.get(sym, 0.0)
        notional = delta_w * equity
        if abs(notional) < min_notional:
            continue
        orders.append(
            {
                "symbol": to_alpaca_symbol(sym),
                "yahoo_symbol": sym,
                "side": "buy" if notional > 0 else "sell",
                "notional": abs(float(notional)),
                "target_weight": targets.get(sym, 0.0),
                "current_weight": current.get(sym, 0.0),
            }
        )
    orders.sort(key=lambda o: (0 if o["side"] == "sell" else 1, -o["notional"]))
    return orders


def status(cfg: Config) -> dict:
    _ensure_portfolio_champion(cfg)
    client = AlpacaPaperClient()
    acct = client.account()
    clock = client.clock()
    positions = client.positions()
    return {
        "paper": True,
        "account_status": acct.status,
        "equity": acct.equity,
        "cash": acct.cash,
        "buying_power": acct.buying_power,
        "market_open": bool(clock.get("is_open")),
        "next_open": clock.get("next_open"),
        "next_close": clock.get("next_close"),
        "positions": {
            s: {"qty": p["qty"], "market_value": p["market_value"]}
            for s, p in sorted(positions.items())
        },
        "n_positions": len(positions),
    }


def rebalance(
    cfg: Config,
    *,
    execute: bool = False,
    refresh_data: bool = True,
    force: bool = False,
    skip_if_closed: bool = False,
    min_notional: float | None = None,
) -> dict:
    """Compute (and optionally send) the daily Alpaca paper rebalance."""
    if cfg.mode != "portfolio":
        raise RuntimeError("Alpaca daily rebalance is wired for portfolio mode only right now.")

    _ensure_portfolio_champion(cfg)
    min_notional = float(min_notional if min_notional is not None else cfg.alpaca.min_notional)

    client = AlpacaPaperClient()
    clock = client.clock()
    market_open = bool(clock.get("is_open"))

    if not market_open and execute and not force:
        if skip_if_closed:
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "paper": True,
                "execute": False,
                "skipped": True,
                "reason": "market_closed",
                "market_open": False,
                "next_open": clock.get("next_open"),
                "next_close": clock.get("next_close"),
                "n_orders": 0,
                "orders": [],
                "submitted": [],
                "errors": [],
            }
            out = alpaca_dir(cfg)
            out.mkdir(parents=True, exist_ok=True)
            (out / "latest_plan.json").write_text(json.dumps(report, indent=2))
            log.info("Market closed — skipping rebalance (automation-friendly).")
            return report
        raise RuntimeError(
            "US market looks closed. Re-run with --force if you still want to send paper orders, "
            "use --skip-if-closed for cron/CI, or omit --execute for a dry-run plan."
        )

    if refresh_data:
        log.info("Refreshing price lake before rebalance...")
        run_download(cfg, synthetic=False)

    acct = client.account()
    positions = client.positions()
    targets, target_info, decision_ctx = _latest_targets(cfg)
    current = _current_weights(positions, acct.equity)
    orders = plan_rebalance(targets, current, acct.equity, min_notional)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper": True,
        "execute": bool(execute),
        "equity": acct.equity,
        "cash": acct.cash,
        "market_open": bool(clock.get("is_open")),
        "target_info": target_info,
        "n_orders": len(orders),
        "orders": orders,
        "submitted": [],
        "errors": [],
        "decision_log": None,
    }

    out = alpaca_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _save_experience(executed: bool) -> str:
        rec = build_decision_record(
            cfg=cfg,
            as_of=target_info["as_of"],
            tickers=decision_ctx["tickers"],
            obs=decision_ctx["obs"],
            action=decision_ctx["action"],
            weights=decision_ctx["weights"],
            targets=targets,
            current_weights=current,
            equity=acct.equity,
            cash=acct.cash,
            orders=orders,
            executed=executed,
            source="alpaca_paper",
            extra={"market_open": bool(clock.get("is_open"))},
        )
        path = append_decision(cfg, rec)
        report["decision_log"] = str(path)
        return str(path)

    def _write_plan() -> None:
        (out / f"plan_{stamp}.json").write_text(json.dumps(report, indent=2))
        (out / "latest_plan.json").write_text(json.dumps(report, indent=2))

    if not execute:
        _save_experience(executed=False)
        _write_plan()
        log.info(
            "Dry-run only: %d paper orders planned (pass --execute to send them). Choice saved to experience log.",
            len(orders),
        )
        return report

    client.cancel_open_orders()
    submitted = []
    errors = []
    for order in orders:
        try:
            resp = client.submit_notional_market_order(
                order["symbol"], order["side"], order["notional"]
            )
            submitted.append(
                {
                    "request": order,
                    "response_id": resp.get("id"),
                    "status": resp.get("status"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Order failed for %s", order["symbol"])
            errors.append({"request": order, "error": str(exc)})

    report["submitted"] = submitted
    report["errors"] = errors
    _save_experience(executed=True)
    _write_plan()
    (out / f"execution_{stamp}.json").write_text(json.dumps(report, indent=2))
    (out / "latest_execution.json").write_text(json.dumps(report, indent=2))

    journal = out / "journal.csv"
    row = {
        "timestamp": report["timestamp"],
        "equity": acct.equity,
        "n_orders": len(orders),
        "n_submitted": len(submitted),
        "n_errors": len(errors),
        "as_of": target_info["as_of"],
        "top": "; ".join(f"{t}:{w:.1%}" for t, w in target_info["top"][:5]),
    }
    pd.DataFrame([row]).to_csv(journal, mode="a", header=not journal.exists(), index=False)
    log.info(
        "Submitted %d/%d Alpaca paper orders (%d errors). Choice saved to experience log.",
        len(submitted),
        len(orders),
        len(errors),
    )
    return report
