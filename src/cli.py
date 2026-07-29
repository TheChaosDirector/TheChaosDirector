"""Command-line entrypoint for the lab.

Usage:
    python -m src.cli download ...
    python -m src.cli train|backtest|paper|referee ...
    python -m src.cli alpaca-status
    python -m src.cli alpaca-rebalance [--execute] [--force] [--no-refresh]
    python -m src.cli decisions-status|decisions-label|decisions-export
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import VALID_MODES, load_config


def _load_dotenv() -> None:
    """Load local .env if present (keys stay off git)."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except Exception:
        # Fallback: tiny parser so missing python-dotenv doesn't block.
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            import os

            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    _load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="src.cli", description="Autonomous Trading Agent Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str, with_mode: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", default="configs/default.yaml", help="YAML config to use")
        if with_mode:
            p.add_argument(
                "--mode",
                choices=VALID_MODES,
                default=None,
                help="portfolio (multi-name) or daytrade (must-pick open→close). Overrides YAML.",
            )
        return p

    p_dl = add("download", "Fill the data lake with daily prices", with_mode=False)
    p_dl.add_argument("--synthetic", action="store_true", help="Build a fake market (offline mode)")

    add("train", "Walk-forward self-training (parallel folds when n_workers > 1)")
    add("backtest", "Grade the champion over the full history")

    p_paper = add("paper", "Advance the simulated live paper-trading desk")
    p_paper.add_argument("--days", type=int, default=1, help="How many market days to advance")

    add("referee", "Run the anti-cheat report")

    add("alpaca-status", "Show Alpaca PAPER account equity/positions/clock", with_mode=True)

    p_live = add(
        "alpaca-rebalance",
        "Daily portfolio rebalance on Alpaca PAPER (dry-run unless --execute)",
        with_mode=True,
    )
    p_live.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit paper orders (default is dry-run plan only)",
    )
    p_live.add_argument(
        "--force",
        action="store_true",
        help="Allow submitting even if the US market clock says closed",
    )
    p_live.add_argument(
        "--skip-if-closed",
        action="store_true",
        help="For cron/CI: exit cleanly (no error) when the US market is closed",
    )
    p_live.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip refreshing the Yahoo price lake before targeting",
    )
    p_live.add_argument(
        "--min-notional",
        type=float,
        default=None,
        help="Ignore drifts smaller than this many dollars (default from config)",
    )

    add("decisions-status", "How many saved live/paper choices we have (training diary)")
    p_lab = add(
        "decisions-label",
        "Grade saved choices with what the market did next (fills training outcomes)",
    )
    p_lab.add_argument(
        "--horizon-days",
        type=int,
        default=1,
        help="How many market days after the choice to measure return (default 1)",
    )
    p_exp = add(
        "decisions-export",
        "Write a parquet table of saved choices for future training",
    )
    p_exp.add_argument(
        "--all",
        action="store_true",
        help="Include unlabeled choices too (default: labeled/graded only)",
    )

    args = parser.parse_args()
    mode = getattr(args, "mode", None)
    # Alpaca / decisions commands default to portfolio mode.
    if args.command.startswith("alpaca") and mode is None:
        mode = "portfolio"
    if args.command.startswith("decisions") and mode is None:
        mode = "portfolio"
    cfg = load_config(args.config, mode=mode)

    if args.command == "download":
        from src.data.download import run_download

        manifest = run_download(cfg, synthetic=args.synthetic)
        print(json.dumps(manifest, indent=2))
    elif args.command == "train":
        if cfg.mode == "daytrade" and cfg.daytrade.learner == "supervised":
            from src.train.daytrade_supervised import run_supervised_training

            champion = run_supervised_training(cfg)
        elif cfg.mode == "concentrated" and cfg.concentrated.learner == "supervised":
            from src.train.concentrated_supervised import run_supervised_training

            champion = run_supervised_training(cfg)
        else:
            from src.train.walkforward import run_training

            champion = run_training(cfg)
        print(json.dumps(champion.get("metrics", champion), indent=2, default=str))
    elif args.command == "backtest":
        from src.backtest.engine import run_backtest

        summary = run_backtest(cfg)
        print(json.dumps(summary, indent=2))
        print("\n" + summary["agent_says"])
        print(summary["benchmark_says"])
        if "forced_says" in summary:
            print(summary["forced_says"])
    elif args.command == "paper":
        from src.paper.desk import advance

        report = advance(cfg, days=args.days)
        print(json.dumps(report, indent=2))
        if "plain_english" in report:
            print("\n" + report["plain_english"])
    elif args.command == "referee":
        from src.referee.checks import run_referee

        report = run_referee(cfg)
        print(json.dumps(report, indent=2))
        print("\n" + report["verdict"])
    elif args.command == "alpaca-status":
        from src.live.rebalance import status

        report = status(cfg)
        print(json.dumps(report, indent=2))
    elif args.command == "alpaca-rebalance":
        from src.live.rebalance import rebalance

        report = rebalance(
            cfg,
            execute=bool(args.execute),
            refresh_data=not bool(args.no_refresh),
            force=bool(args.force),
            skip_if_closed=bool(args.skip_if_closed),
            min_notional=args.min_notional,
        )
        print(json.dumps(report, indent=2))
        if report.get("skipped"):
            print("\nSkipped: US market is closed (automation-friendly exit).")
        elif report.get("decision_log"):
            print(f"\nChoice saved for later training: {report['decision_log']}")
        if not args.execute and not report.get("skipped"):
            print(
                "\nDry-run complete. To send these orders to Alpaca PAPER, add --execute "
                "(and --force if the market is closed)."
            )
    elif args.command == "decisions-status":
        from src.experience.decisions import status as decisions_status

        report = decisions_status(cfg)
        print(json.dumps(report, indent=2))
        print("\n" + report["plain_english"])
    elif args.command == "decisions-label":
        from src.experience.decisions import label_outcomes

        report = label_outcomes(cfg, horizon_days=int(args.horizon_days))
        print(json.dumps(report, indent=2))
        print("\n" + report["plain_english"])
    elif args.command == "decisions-export":
        from src.experience.decisions import export_training_table

        path = export_training_table(cfg, labeled_only=not bool(args.all))
        print(json.dumps({"path": str(path)}, indent=2))
        print(f"\nTraining table written to {path}")


if __name__ == "__main__":
    main()
