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
    # forced is now an explicit discrete action bit (PPO) or derived (supervised).
    forced_threshold: float = 0.5
    # Cap how many names the daytrader sees (benchmark always kept). None = all.
    max_names: int | None = None
    # "supervised" = walk-forward gradient-boosted ranker (less overfit-prone)
    # "ppo" = reinforcement learning with randomized train episodes
    learner: str = "supervised"


@dataclass
class AlpacaCfg:
    # Practice-money daily rebalance settings. Live trading is intentionally unsupported here.
    min_notional: float = 25.0  # ignore weight drifts smaller than this many dollars
    refresh_data: bool = True


@dataclass
class CompeteCfg:
    """Train a separate copy whose hurdle is the short-term US Treasury yield.

    Off by default. Turning it on must use its own artifacts directory: the
    guard in ``src.registry.protect`` refuses to write into the live champion
    folder (``artifacts/portfolio``).
    """

    enabled: bool = False
    # ^IRX is the 13-week T-bill discount rate, quoted in percent (5.0 = 5%).
    symbol: str = "^IRX"
    # "excess" pays the agent only for beating the bill.
    # "both" also pays for beating the benchmark that day, after costs.
    # "absolute" keeps the old log-wealth reward while cash still earns the bill.
    reward: str = "excess"
    # Used only when reward == "both". Added to the log-excess reward as
    # weight * (net return − benchmark return). Small on purpose: a large
    # weight just teaches the model to hug the index and then lose to costs.
    spy_beat_weight: float = 0.25
    observe_yield: bool = True
    cash_earns_yield: bool = True
    # Nudge toward saved paper decisions that beat the bill, and away from ones
    # that lost to it. Only dates inside a fold's train window are used.
    replay_experience: bool = True
    experience_scale: float = 0.001
    min_coverage: float = 0.50
    holdout_frac: float = 0.25
    # Extra PPO steps on the saved decisions, starting from the walk-forward
    # copy. 0 skips that pass. The live champion is never the starting point.
    experience_timesteps: int = 0
    # Last N decision days are sealed. No fold trains on them, no fold is
    # picked because of them, and saved paper decisions are not replayed.
    # One model is then fit on the years just before that seal and scored once.
    final_holdout_days: int = 0
    # 1.0 subtracts the whole bill from the reward. Lower values keep the bill
    # as a light penalty so the book is not paid mainly for hugging cash.
    yield_weight: float = 1.0
    # "bill" ranks exams by excess over the T-bill.
    # "market" ranks them by after-cost excess over the benchmark, and the
    # reward becomes log wealth minus the benchmark, minus yield_weight times
    # the bill.
    focus: str = "bill"
    # Also show lagged 5-year and 10-year yields, and the 10y-minus-bill slope.
    # These are inputs only. They do not raise the bill's weight in the reward.
    observe_curve: bool = False
    # 1 means the agent may trade every day. 5 means it trades, then holds for
    # the next four sessions. Locked before a run; not tuned to a result.
    rebalance_interval: int = 1
    # Share of the book pinned to the benchmark before the policy's own bets.
    # 0 leaves the book free. 0.5 keeps half in SPY on every rebalance.
    spy_anchor: float = 0.0
    # When true, the learned book is kept only if the stitched unseen years
    # beat SPY after costs AND at least half of those years finish ahead.
    # The sealed final year is not scored and cannot change that decision.
    repeated_edge: bool = False

    def __post_init__(self) -> None:
        if self.reward not in ("excess", "absolute", "both"):
            raise ValueError(
                "compete.reward must be 'excess', 'absolute', or 'both', "
                f"got {self.reward!r}"
            )
        if self.focus not in ("bill", "market"):
            raise ValueError("compete.focus must be 'bill' or 'market', " f"got {self.focus!r}")
        if self.final_holdout_days < 0:
            raise ValueError("compete.final_holdout_days cannot be negative")
        if self.yield_weight < 0:
            raise ValueError("compete.yield_weight cannot be negative")
        if self.rebalance_interval < 1:
            raise ValueError("compete.rebalance_interval must be at least 1")
        if not 0.0 <= self.spy_anchor < 1.0:
            raise ValueError("compete.spy_anchor must be in [0, 1)")


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
    alpaca: AlpacaCfg = field(default_factory=AlpacaCfg)
    compete: CompeteCfg = field(default_factory=CompeteCfg)
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
    "alpaca": AlpacaCfg,
    "compete": CompeteCfg,
}


def load_config(path: str | Path, mode: str | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    kwargs = {}
    for section, cls in _SECTIONS.items():
        kwargs[section] = cls(**(raw.get(section) or {}))
    resolved_mode = mode or raw.get("mode") or "portfolio"
    cfg = Config(**kwargs, mode=resolved_mode, config_path=str(path))
    return cfg
