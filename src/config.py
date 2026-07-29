"""Load YAML config files into simple, typed objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PathsCfg:
    lake_dir: str = "data/lake"
    artifacts_dir: str = "artifacts"


@dataclass
class UniverseCfg:
    name: str = "default"
    benchmark: str = "SPY"
    min_history_days: int = 756
    max_tickers: int = 500


@dataclass
class DataCfg:
    start: str = "2010-01-01"
    end: str | None = None
    min_dollar_volume: float = 5_000_000


@dataclass
class FeaturesCfg:
    momentum_windows: list[int] = field(default_factory=lambda: [5, 21, 63, 126])
    vol_window: int = 21
    volume_window: int = 21
    ma_windows: list[int] = field(default_factory=lambda: [10, 50, 200])
    regime_ma: int = 200


@dataclass
class CostsCfg:
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    stress_multiplier: float = 3.0

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


@dataclass
class RiskCfg:
    max_weight_per_name: float = 0.10
    max_gross_exposure: float = 1.0
    drawdown_penalty: float = 0.10


@dataclass
class TrainingCfg:
    train_years: float = 3.0
    test_months: int = 6
    embargo_days: int = 5
    total_timesteps: int = 150_000
    seed: int = 42
    n_env_steps: int = 2048
    learning_rate: float = 3e-4
    n_workers: int = 1  # parallel walk-forward folds (1 = serial)


@dataclass
class DaytradeCfg:
    # Reluctance / forced threshold kept for backwards-compatible YAML;
    # forced is now an explicit discrete action bit.
    forced_threshold: float = 0.5
    # Cap how many names the daytrader sees (benchmark always kept). None = all.
    max_names: int | None = None


@dataclass
class PaperCfg:
    starting_cash: float = 100_000.0


@dataclass
class RefereeCfg:
    leakage_sample_dates: int = 20
    cheat_corr_threshold: float = 0.30


VALID_MODES = ("portfolio", "daytrade")


@dataclass
class Config:
    mode: str = "portfolio"
    paths: PathsCfg = field(default_factory=PathsCfg)
    universe: UniverseCfg = field(default_factory=UniverseCfg)
    data: DataCfg = field(default_factory=DataCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    costs: CostsCfg = field(default_factory=CostsCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    daytrade: DaytradeCfg = field(default_factory=DaytradeCfg)
    paper: PaperCfg = field(default_factory=PaperCfg)
    referee: RefereeCfg = field(default_factory=RefereeCfg)
    config_path: str = ""

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unknown mode '{self.mode}' (expected {VALID_MODES})")

    @property
    def lake_dir(self) -> Path:
        return Path(self.paths.lake_dir)

    @property
    def artifacts_dir(self) -> Path:
        # Mode-scoped so portfolio and daytrade champions never overwrite each other.
        return Path(self.paths.artifacts_dir) / self.mode


_SECTIONS = {
    "paths": PathsCfg,
    "universe": UniverseCfg,
    "data": DataCfg,
    "features": FeaturesCfg,
    "costs": CostsCfg,
    "risk": RiskCfg,
    "training": TrainingCfg,
    "daytrade": DaytradeCfg,
    "paper": PaperCfg,
    "referee": RefereeCfg,
}


def load_config(path: str | Path, mode: str | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    kwargs = {}
    for section, cls in _SECTIONS.items():
        kwargs[section] = cls(**(raw.get(section) or {}))
    resolved_mode = mode or raw.get("mode") or "portfolio"
    cfg = Config(**kwargs, mode=resolved_mode, config_path=str(path))
    return cfg
