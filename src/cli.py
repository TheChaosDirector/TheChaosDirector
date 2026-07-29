"""Command-line entrypoint for the lab.

Usage:
    python -m src.cli download [--config configs/smoke.yaml] [--synthetic]
    python -m src.cli train    [--config ...] [--mode portfolio|daytrade]
    python -m src.cli backtest [--config ...] [--mode ...]
    python -m src.cli paper    [--config ...] [--mode ...] [--days 5]
    python -m src.cli referee  [--config ...] [--mode ...]
"""

from __future__ import annotations

import argparse
import json
import logging

from src.config import VALID_MODES, load_config


def main() -> None:
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

    args = parser.parse_args()
    mode = getattr(args, "mode", None)
    cfg = load_config(args.config, mode=mode)

    if args.command == "download":
        from src.data.download import run_download

        manifest = run_download(cfg, synthetic=args.synthetic)
        print(json.dumps(manifest, indent=2))
    elif args.command == "train":
        if cfg.mode == "daytrade" and cfg.daytrade.learner == "supervised":
            from src.train.daytrade_supervised import run_supervised_training

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


if __name__ == "__main__":
    main()
