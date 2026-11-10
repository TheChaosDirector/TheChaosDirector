"""Walk-forward ranker for the delisted book.

A gradient-boosted tree predicts the next session's return minus SPY from the
lagged feature row. At the rebalance it turns those scores into portfolio
weights through the same cost, eligibility, and holding rules as the other
book. It does not train or overwrite the saved PPO.

The tree settings and the score scale below were fixed before the exams.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import Config
from src.data.eligibility import point_in_time_eligible
from src.env.portfolio_env import PortfolioEnv
from src.features.factory import build_features, warmup_days
from src.metrics.scorecard import scorecard
from src.registry.protect import assert_training_copy_is_isolated
from src.registry.store import fold_dir, registry_dir, save_champion
from src.train.holdout import describe_seal, fold_history_length, holdout_bounds
from src.train.robust import passes_repeated_edge
from src.train.walkforward import make_folds, rollout

log = logging.getLogger(__name__)

# Fixed before any exam. A score is turned into a cross-sectional z-score and
# multiplied by this. 1 keeps a typical winner under the 10% name cap, so the
# cap's leftover does not sit in cash. Cash's own logit stays at 0.
LOGIT_SCALE = 1.0
MAX_ITER = 60
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 200
LEARNING_RATE = 0.08


class _ZeroModel:
    def predict(self, rows):
        return np.zeros(len(np.asarray(rows)), dtype=np.float64)


class PortfolioRankerAdapter:
    """Scores each name, then hands the gym one logit per name plus cash."""

    def __init__(self, model, n_assets: int, n_features: int, logit_scale: float = LOGIT_SCALE):
        self.model = model
        self.n_assets = n_assets
        self.n_features = n_features
        self.logit_scale = logit_scale

    def predict(self, obs, deterministic: bool = True):
        del deterministic
        flat = np.asarray(obs, dtype=np.float64).reshape(-1)
        feats = flat[: self.n_assets * self.n_features].reshape(self.n_assets, self.n_features)
        scores = np.asarray(self.model.predict(feats), dtype=np.float64)
        center = float(scores.mean())
        spread = float(scores.std())
        if spread < 1e-8:
            logits = np.zeros(self.n_assets, dtype=np.float64)
        else:
            logits = (scores - center) / spread * self.logit_scale
        logits = np.clip(logits, -5.0, 5.0)
        action = np.concatenate([logits, [0.0]])
        return action.astype(np.float32), None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "n_assets": self.n_assets,
                "n_features": self.n_features,
                "logit_scale": self.logit_scale,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PortfolioRankerAdapter":
        blob = joblib.load(path)
        return cls(
            blob["model"],
            int(blob["n_assets"]),
            int(blob["n_features"]),
            float(blob.get("logit_scale", LOGIT_SCALE)),
        )


def _new_tree(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=MAX_ITER,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        l2_regularization=0.1,
        random_state=seed,
    )


def decision_rows(
    close: pd.DataFrame,
    panel,
    eligible: pd.DataFrame,
    start: int,
    end: int,
    benchmark: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature rows and next-session excess returns inside ``[start, end)``.

    The label on row t uses the close at t+1. The last row that is included
    has its next close still inside ``end`` (the index ``end`` itself is not
    read). Names that were not eligible that morning are left out.
    """
    bench = close.columns.get_loc(benchmark)
    prices = close.to_numpy(dtype=np.float64)
    mask = (
        eligible.reindex(index=close.index, columns=close.columns)
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    pieces_x: list[np.ndarray] = []
    pieces_y: list[np.ndarray] = []
    last = min(end, len(close)) - 1
    for t in range(start, last):
        today = prices[t]
        nxt = prices[t + 1]
        spy_today = today[bench]
        spy_next = nxt[bench]
        if not (
            np.isfinite(spy_today)
            and np.isfinite(spy_next)
            and spy_today > 0
            and spy_next > 0
        ):
            continue
        spy_ret = spy_next / spy_today - 1.0
        live = np.flatnonzero(
            mask[t] & np.isfinite(today) & np.isfinite(nxt) & (today > 0) & (nxt > 0)
        )
        if len(live) == 0:
            continue
        feats = panel.values[t, live, :]
        keep = np.any(feats != 0.0, axis=1)
        live = live[keep]
        feats = feats[keep]
        if len(live) == 0:
            continue
        excess = nxt[live] / today[live] - 1.0 - spy_ret
        pieces_x.append(np.asarray(feats, dtype=np.float32))
        pieces_y.append(excess.astype(np.float64))
    if not pieces_x:
        return np.zeros((0, panel.n_features), dtype=np.float32), np.zeros(0, dtype=np.float64)
    return np.vstack(pieces_x), np.concatenate(pieces_y)


def _fit(close, panel, eligible, start: int, end: int, benchmark: str, seed: int):
    features, target = decision_rows(close, panel, eligible, start, end, benchmark)
    model = _new_tree(seed)
    if len(target) < MIN_SAMPLES_LEAF * 2:
        log.warning("Only %s training rows in [%s, %s). The tree will predict zeros.", len(target), start, end)
        return PortfolioRankerAdapter(_ZeroModel(), panel.values.shape[1], panel.n_features), len(target)
    model.fit(features, target)
    return PortfolioRankerAdapter(model, panel.values.shape[1], panel.n_features), len(target)


def _env(close, panel, cfg: Config, start: int, end: int, eligible, cost_multiplier: float = 1.0):
    from src.train.compete import compete_env_kwargs

    return PortfolioEnv(
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        start=start,
        end=end,
        cost_multiplier=cost_multiplier,
        eligible=eligible,
        **compete_env_kwargs(cfg, close.index, close=close),
    )


def _load_delisted(cfg: Config):
    from src.data.delisted import ensure_price_file, load_price_frames, with_benchmark

    path = Path(cfg.data.price_file) if cfg.data.price_file else Path("data/external/sp500_delisted_prices.csv")
    ensure_price_file(path)
    spy_path = cfg.lake_dir / "prices" / f"{cfg.universe.benchmark}.parquet"
    spy = pd.read_parquet(spy_path)
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.set_index("date").sort_index()
    close, volume, dollar = load_price_frames(path)
    return with_benchmark(
        close,
        volume,
        dollar,
        spy["close"],
        spy["volume"],
        benchmark=cfg.universe.benchmark,
    )


def _panel(close, volume, cfg: Config):
    extra = None
    if cfg.features.include_macro:
        from src.data.macro import macro_known_before
        from src.data.yields import tbill_path

        path = tbill_path(cfg)
        if not path.exists():
            raise FileNotFoundError(
                f"No Treasury yield file at {path}. The ranker needs it for the real rate."
            )
        quoted = pd.read_parquet(path)
        quoted["date"] = pd.to_datetime(quoted["date"]).dt.tz_localize(None)
        bill = (
            quoted.drop_duplicates("date")
            .set_index("date")["annual_percent"]
            .astype(float)
            .sort_index()
        )
        macro = macro_known_before(close.index, bill_annual_percent=bill)
        extra = {name: macro[name] for name in macro.columns}
    return build_features(close, volume, cfg.features, cfg.universe.benchmark, extra_lagged=extra)


def run_ranker_training(cfg: Config) -> dict:
    """Fit one tree per exam year on the delisted point-in-time book.

    The learned book is kept only when the stitched unseen years beat SPY
    after costs and at least half of those years finish ahead. Otherwise the
    deployed book is SPY. The sealed final year is not part of that decision
    and is not scored.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert_training_copy_is_isolated(cfg)
    if not cfg.compete.repeated_edge:
        raise RuntimeError("The ranker config must leave repeated_edge on.")
    close, volume, dollar = _load_delisted(cfg)
    panel = _panel(close, volume, cfg)
    eligible = point_in_time_eligible(
        close,
        volume,
        min_history_days=cfg.universe.min_history_days,
        min_dollar_volume=cfg.data.min_dollar_volume,
        max_names=cfg.universe.max_tickers,
        benchmark=cfg.universe.benchmark,
        dollar_volume=dollar,
    )
    n_dates = fold_history_length(len(close), cfg)
    folds = make_folds(n_dates, cfg, warmup_days(cfg.features))
    if not folds:
        raise RuntimeError("No walk-forward folds. The price history is shorter than the windows.")
    log.info(
        "Ranker: %s names, %s features, %s folds, sealed tail left out (%s).",
        close.shape[1],
        panel.n_features,
        len(folds),
        describe_seal(close.index, cfg),
    )
    records = []
    for fold in folds:
        adapter, n_rows = _fit(
            close,
            panel,
            eligible,
            fold.train_start,
            fold.train_end,
            cfg.universe.benchmark,
            seed=cfg.training.seed + fold.index,
        )
        env = _env(close, panel, cfg, fold.test_start, fold.test_end, eligible)
        journal = rollout(adapter, env)
        agent = scorecard(journal["net_return"], journal.get("turnover"))
        spy = scorecard(journal["benchmark_return"])
        record = {
            "fold": fold.index,
            "train_start": str(close.index[fold.train_start].date()),
            "train_end": str(close.index[fold.train_end - 1].date()),
            "test_start": str(close.index[fold.test_start].date()),
            "test_end": str(close.index[fold.test_end - 1].date()),
            "train_rows": int(n_rows),
            "out_of_sample": agent,
            "spy_oos": spy,
        }
        records.append(record)
        out = fold_dir(cfg, fold.index)
        out.mkdir(parents=True, exist_ok=True)
        adapter.save(out / "model.joblib")
        (out / "learner.txt").write_text("ranker")
        (out / "metrics.json").write_text(json.dumps(record, indent=2, default=str))
        journal.to_csv(out / "oos_journal.csv")
        log.info(
            "Fold %s exam %s to %s: book %.1f%% | SPY %.1f%% | rows %s",
            fold.index,
            record["test_start"],
            record["test_end"],
            agent["total_return"] * 100,
            spy["total_return"] * 100,
            n_rows,
        )

    from src.train.robust import _compound, _load_journals, _year_rows

    years = _year_rows(records)
    journal = _load_journals(cfg, records)
    agent_total = _compound(journal["net_return"])
    spy_total = _compound(journal["benchmark_return"])
    gaps = [row["gap"] for row in years]
    kept = passes_repeated_edge(gaps, agent_total - spy_total)
    ahead = sum(1 for row in years if row["ahead"])
    report = {
        "rule": (
            "Keep the ranker only if the stitched unseen years beat SPY after "
            "commission and slippage, and at least half of those years finish "
            "ahead. Otherwise hold SPY. The sealed final year is not scored."
        ),
        "learner": "ranker",
        "n_years": len(years),
        "years_ahead": ahead,
        "years": years,
        "stitch": {
            "agent": agent_total,
            "spy": spy_total,
            "gap": agent_total - spy_total,
            "n_days": int(len(journal)),
            "start": str(journal.index[0].date()),
            "end": str(journal.index[-1].date()),
            "agent_card": scorecard(journal["net_return"], journal.get("turnover")),
            "spy_card": scorecard(journal["benchmark_return"]),
        },
        "kept_learned_book": kept,
        "deploy": "model" if kept else "hold_spy",
        "sealed_year_scored": False,
        "sealed_year": describe_seal(close.index, cfg),
        "features": panel.names,
    }
    out_dir = cfg.artifacts_dir / "stitch"
    out_dir.mkdir(parents=True, exist_ok=True)
    journal.to_csv(out_dir / "journal.csv")
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str))

    champion = {
        "fold": None,
        "mode": cfg.mode,
        "tickers": panel.tickers,
        "feature_names": panel.names,
        "config_path": cfg.config_path,
        "objective": "repeated_edge_vs_spy",
        "learner": "hold_spy",
        "model_path": None,
        "deploy": "hold_spy",
        "selected_by": "repeated_edge_gate_failed_hold_spy",
        "years_ahead": ahead,
        "total_folds": len(years),
        "stitch_agent": agent_total,
        "stitch_spy": spy_total,
        "stitch_gap": agent_total - spy_total,
        "sealed_year_scored": False,
        "metrics": report,
    }
    if kept:
        bounds = holdout_bounds(len(close), cfg)
        adapter, n_rows = _fit(
            close,
            panel,
            eligible,
            bounds["train_start"],
            bounds["train_end"],
            cfg.universe.benchmark,
            seed=cfg.training.seed + 10_000,
        )
        model_path = registry_dir(cfg) / "presample" / "model.joblib"
        adapter.save(model_path)
        (model_path.parent / "learner.txt").write_text("ranker")
        champion.update(
            {
                "learner": "ranker",
                "model_path": str(model_path),
                "deploy": "model",
                "selected_by": "repeated_edge_stitch_before_sealed_year",
                "train_rows": int(n_rows),
                "train_start": str(close.index[bounds["train_start"]].date()),
                "train_end": str(close.index[bounds["train_end"] - 1].date()),
            }
        )
    save_champion(cfg, champion)
    log.info(
        "Ranker stitch: book %.1f%% | SPY %.1f%% | years ahead %s/%s | deploy %s",
        agent_total * 100,
        spy_total * 100,
        ahead,
        len(years),
        champion["deploy"],
    )
    return champion
