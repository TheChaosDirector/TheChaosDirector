"""Champion model storage: which trained brain is currently 'the one'."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import Config


def registry_dir(cfg: Config) -> Path:
    return cfg.artifacts_dir / "registry"


def fold_dir(cfg: Config, fold: int) -> Path:
    return registry_dir(cfg) / f"fold_{fold:02d}"


def save_fold(cfg: Config, fold: int, model, metrics: dict) -> Path:
    d = fold_dir(cfg, fold)
    d.mkdir(parents=True, exist_ok=True)
    model.save(d / "model.zip")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return d


def save_champion(cfg: Config, meta: dict) -> None:
    registry_dir(cfg).mkdir(parents=True, exist_ok=True)
    (registry_dir(cfg) / "champion.json").write_text(json.dumps(meta, indent=2, default=str))


def load_champion_meta(cfg: Config) -> dict:
    path = registry_dir(cfg) / "champion.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No champion model at {path}. Run: python -m src.cli train"
        )
    return json.loads(path.read_text())


def load_champion_model(cfg: Config):
    meta = load_champion_meta(cfg)
    path = meta["model_path"]
    learner = meta.get("learner", "")
    mode = meta.get("mode") or cfg.mode

    if learner == "concentrated_ppo_ensemble" or (
        mode == "concentrated" and Path(path).name == "members.json"
    ):
        import json

        from stable_baselines3 import PPO

        from src.train.walkforward import AllocationPPOEnsemble

        members = json.loads(Path(path).read_text())
        models = [PPO.load(Path(path).parent / f"fold_{int(f):02d}.zip") for f in members]
        return AllocationPPOEnsemble(models), meta

    if mode == "concentrated" and (
        "concentrated" in learner or str(path).endswith(".joblib")
    ):
        if "ensemble" in learner or str(path).endswith("ensemble.joblib"):
            from src.train.concentrated_supervised import ConcentratedEnsembleAdapter

            return ConcentratedEnsembleAdapter.load(path), meta
        from src.train.concentrated_supervised import ConcentratedRankerAdapter

        return ConcentratedRankerAdapter.load(path), meta

    if "ensemble" in learner or str(path).endswith("ensemble.joblib"):
        from src.train.daytrade_supervised import DaytradeEnsembleAdapter

        return DaytradeEnsembleAdapter.load(path), meta
    if learner.startswith("supervised") or str(path).endswith(".joblib"):
        from src.train.daytrade_supervised import DaytradeRankerAdapter

        return DaytradeRankerAdapter.load(path), meta

    from stable_baselines3 import PPO

    return PPO.load(path), meta


def save_experiments(cfg: Config, records: list[dict]) -> None:
    registry_dir(cfg).mkdir(parents=True, exist_ok=True)
    (registry_dir(cfg) / "experiments.json").write_text(
        json.dumps(records, indent=2, default=str)
    )


def load_experiments(cfg: Config) -> list[dict]:
    path = registry_dir(cfg) / "experiments.json"
    return json.loads(path.read_text()) if path.exists() else []
