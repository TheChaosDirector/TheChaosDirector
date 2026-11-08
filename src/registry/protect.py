"""Keep the live S&P-relative champion out of the treasury-copy training path."""

from __future__ import annotations

from pathlib import Path

from src.config import Config

# Daily paper trading loads this folder (via the portfolio-champion release,
# which unpacks to the same layout). Treasury training must not write here.
LIVE_PORTFOLIO_DIR = Path("artifacts/portfolio")
LIVE_ARTIFACT_ROOT = Path("artifacts")


def assert_training_copy_is_isolated(cfg: Config) -> None:
    """Refuse to train the yield-competing copy on top of the live champion.

    Default portfolio training (``compete.enabled`` false) is unchanged. The
    yield configs point ``paths.artifacts_dir`` at ``artifacts/yield-compete``
    or ``artifacts/yield-compete-smoke``.
    """
    if not cfg.compete.enabled:
        return

    root = Path(cfg.paths.artifacts_dir).resolve()
    scoped = Path(cfg.artifacts_dir).resolve()
    live = LIVE_PORTFOLIO_DIR.resolve()
    shared = LIVE_ARTIFACT_ROOT.resolve()

    blocked = (
        root == live
        or root == shared
        or scoped == live
        or live in root.parents
        or live in scoped.parents
        or scoped in live.parents
    )
    if blocked:
        raise RuntimeError(
            "Refusing to train the Treasury-competing copy into the live champion "
            f"folder ({live}). Use configs/yield-compete.yaml so the existing "
            "portfolio model stays put. This command does not republish the "
            "portfolio-champion release."
        )
