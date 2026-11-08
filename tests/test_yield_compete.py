"""Treasury hurdle, diary nudge, and the live-champion guard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.data.yields import align_tbill_daily, annualized_observation, write_tbill_series
from src.env.portfolio_env import PortfolioEnv
from src.features.factory import build_features, warmup_days
from src.registry.protect import assert_training_copy_is_isolated
from src.train.experience_blend import (
    allocation_similarity,
    build_experience_split,
    grade_diary_against_tbill,
    hint_reward,
    rf_on,
    should_promote_experience_tune,
    weights_on_universe,
)
from tests.test_env import _make_env


def test_live_champion_folder_is_blocked():
    cfg = Config()
    cfg.compete.enabled = True
    try:
        assert_training_copy_is_isolated(cfg)
    except RuntimeError as exc:
        assert "live champion" in str(exc)
    else:
        raise AssertionError("training into artifacts/portfolio should be refused")


def test_yield_copy_folder_is_allowed():
    cfg = Config()
    cfg.compete.enabled = True
    cfg.paths.artifacts_dir = "artifacts/yield-compete"
    assert_training_copy_is_isolated(cfg)


def test_all_cash_earns_the_bill_and_excess_reward_is_flat(fake_market, cfg):
    close, volume = fake_market
    panel = build_features(close, volume, cfg.features, benchmark="SPY")
    start = warmup_days(cfg.features)
    daily = np.full(len(close), 0.05 / 252.0)
    env = PortfolioEnv(
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        start=start,
        end=len(close) - 1,
        cash_yield=daily,
        cash_earns=True,
        observe_yield=True,
        reward_excess=True,
    )
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.shape[0] == _make_env(fake_market, cfg).observation_space.shape[0] + 1
    assert abs(obs[-1] - annualized_observation(daily[start])) < 1e-6

    action = np.full(env.action_space.shape, -5.0, dtype=np.float32)
    action[-1] = 5.0
    done = False
    rewards = []
    while not done:
        _, reward, done, _, _ = env.step(action)
        rewards.append(reward)

    journal = env.results()
    assert journal["equity"].iloc[-1] > 1.01
    # A pure cash book matches the bill, so excess over the hurdle stays ~0.
    assert abs(journal["excess_return"]).max() < 1e-4
    assert abs(float(np.mean(rewards))) < 1e-4


def test_hint_reward_sign_and_coverage():
    target = np.array([0.5, 0.0])
    saved = np.array([0.5, 0.0])
    assert allocation_similarity(target, saved) == 1.0
    assert hint_reward(target, saved, beat=True) > 0
    assert hint_reward(target, saved, beat=False) < 0

    vec, coverage = weights_on_universe({"AAA": 0.2, "ZZZ": 0.8}, ["AAA", "BBB"])
    assert vec[0] == 0.2
    assert coverage == 0.2


def test_holdout_is_not_a_training_hint(monkeypatch, tmp_path):
    cfg = Config()
    cfg.compete.holdout_frac = 0.25
    cfg.compete.min_coverage = 0.5
    rows = []
    for i, day in enumerate(pd.bdate_range("2024-01-02", periods=8)):
        rows.append(
            {
                "as_of": str(day.date()),
                "weights": {"AAA": 0.6},
                "outcome": {"portfolio_return": 0.01 if i % 2 == 0 else -0.01},
            }
        )
    monkeypatch.setattr("src.train.experience_blend._load_summaries", lambda _cfg: rows)
    rf = pd.Series(0.0, index=pd.bdate_range("2024-01-02", periods=8))
    split = build_experience_split(cfg, ["AAA"], rf)
    assert split.report["holdout_days"] == 2
    assert len(split.train_hints) == 6
    assert all(day not in split.train_hints for day in split.holdout_dates)


def test_diary_grade_against_tbill(monkeypatch):
    cfg = Config()
    rows = [
        {
            "as_of": "2024-01-02",
            "weights": {"AAA": 1.0},
            "outcome": {"portfolio_return": 0.02},
        },
        {
            "as_of": "2024-01-03",
            "weights": {"AAA": 1.0},
            "outcome": {"portfolio_return": 0.0},
        },
    ]
    monkeypatch.setattr("src.train.experience_blend._load_summaries", lambda _cfg: rows)
    rf = pd.Series([0.001, 0.001], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    report = grade_diary_against_tbill(cfg, rf)
    assert report["n_days"] == 2
    assert report["beat_days"] == 1
    assert report["excess"]["total_return"] > 0


def test_promote_only_when_holdout_improves_and_full_sample_holds():
    base = {"holdout_excess_total": 0.01, "full_excess_total": 0.20}
    better = {"holdout_excess_total": 0.03, "full_excess_total": 0.19}
    wrecked = {"holdout_excess_total": 0.04, "full_excess_total": 0.10}
    assert should_promote_experience_tune(base, better)
    assert not should_promote_experience_tune(base, wrecked)


def test_align_tbill_does_not_peek_ahead(tmp_path):
    cfg = Config()
    cfg.paths.lake_dir = str(tmp_path)
    quotes = pd.Series([2.0, 4.0], index=pd.to_datetime(["2024-01-03", "2024-01-05"]))
    write_tbill_series(cfg, quotes, "test")
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    daily = align_tbill_daily(cfg, pd.DatetimeIndex(index))
    assert daily.iloc[0] == 0.0  # before the first quote
    assert abs(daily.iloc[1] - 2.0 / 100 / 252) < 1e-12
    assert abs(daily.iloc[2] - 2.0 / 100 / 252) < 1e-12  # forward fill, not the 4% from tomorrow
    assert abs(daily.iloc[3] - 4.0 / 100 / 252) < 1e-12


def test_both_reward_uses_the_outcome_not_the_observation(fake_market, cfg):
    close, volume = fake_market
    panel = build_features(close, volume, cfg.features, benchmark="SPY")
    start = warmup_days(cfg.features)
    daily = np.full(len(close), 0.02 / 252.0)
    fwd = np.zeros(len(close))
    env_a = PortfolioEnv(
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        start=start,
        end=start + 2,
        cash_yield=daily,
        cash_earns=True,
        observe_yield=True,
        reward_excess=True,
        benchmark_fwd=fwd,
        spy_beat_weight=0.25,
    )
    fwd_b = fwd.copy()
    fwd_b[start] = 0.05
    env_b = PortfolioEnv(
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        start=start,
        end=start + 2,
        cash_yield=daily,
        cash_earns=True,
        observe_yield=True,
        reward_excess=True,
        benchmark_fwd=fwd_b,
        spy_beat_weight=0.25,
    )
    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    np.testing.assert_array_equal(obs_a, obs_b)
    action = np.zeros(env_a.action_space.shape, dtype=np.float32)
    _, reward_a, _, _, _ = env_a.step(action)
    _, reward_b, _, _, _ = env_b.step(action)
    assert reward_b < reward_a


def test_locked_holdout_is_outside_every_fold():
    from src.train.holdout import fold_history_length, holdout_bounds
    from src.train.walkforward import make_folds

    cfg = Config()
    cfg.compete.enabled = True
    cfg.compete.final_holdout_days = 252
    cfg.training.train_years = 2.0
    cfg.training.test_months = 6
    cfg.training.embargo_days = 10
    n_dates = 4000
    folds = make_folds(fold_history_length(n_dates, cfg), cfg, warmup=201, need_next_day=True)
    bounds = holdout_bounds(n_dates, cfg)
    assert folds
    assert max(f.test_end for f in folds) <= bounds["train_end"]
    assert bounds["train_end"] <= bounds["holdout_start"] - bounds["embargo_days"]


def test_higher_costs_reduce_a_trading_episode(fake_market, cfg):
    close, volume = fake_market
    panel = build_features(close, volume, cfg.features, benchmark="SPY")
    start = warmup_days(cfg.features)
    end = len(close) - 1
    action = np.zeros(panel.tickers.__len__() + 1, dtype=np.float32)
    action[0] = 5.0

    def _run(multiplier: float) -> float:
        env = PortfolioEnv(
            close=close,
            panel=panel,
            costs=cfg.costs,
            risk=cfg.risk,
            start=start,
            end=end,
            cost_multiplier=multiplier,
        )
        env.reset()
        done = False
        while not done:
            _, _, done, _, _ = env.step(action)
        return float(env.results()["equity"].iloc[-1])

    assert _run(3.0) < _run(1.0)


def test_lighter_bill_weight_stops_paying_the_agent_to_match_cash(fake_market, cfg):
    close, volume = fake_market
    panel = build_features(close, volume, cfg.features, benchmark="SPY")
    start = warmup_days(cfg.features)
    daily = np.full(len(close), 0.05 / 252.0)
    env = PortfolioEnv(
        close=close,
        panel=panel,
        costs=cfg.costs,
        risk=cfg.risk,
        start=start,
        end=start + 15,
        cash_yield=daily,
        cash_earns=True,
        reward_excess=True,
        yield_weight=0.0,
    )
    action = np.full(env.action_space.shape, -5.0, dtype=np.float32)
    action[-1] = 5.0
    env.reset()
    done = False
    rewards = []
    while not done:
        _, reward, done, _, _ = env.step(action)
        rewards.append(reward)
    assert float(np.mean(rewards)) > 0.5 * np.log(1.0 + daily[start])


def test_market_focus_cares_more_about_spy_than_the_bill(fake_market, cfg):
    close, volume = fake_market
    panel = build_features(close, volume, cfg.features, benchmark="SPY")
    start = warmup_days(cfg.features)
    daily = np.full(len(close), 0.05 / 252.0)
    flat = np.zeros(len(close))
    up = flat.copy()
    up[start] = 0.02

    def _reward(fwd, weight: float) -> float:
        env = PortfolioEnv(
            close=close,
            panel=panel,
            costs=cfg.costs,
            risk=cfg.risk,
            start=start,
            end=start + 2,
            cash_yield=daily,
            cash_earns=True,
            reward_excess=True,
            benchmark_fwd=fwd,
            spy_beat_weight=1.0,
            yield_weight=weight,
            focus="market",
        )
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        return float(reward)

    spy_effect = abs(_reward(up, 0.2) - _reward(flat, 0.2))
    bill_effect = abs(_reward(flat, 1.0) - _reward(flat, 0.2))
    assert spy_effect > 10 * bill_effect


def test_curve_observation_excludes_same_day_and_future_quotes(tmp_path):
    from src.data.yields import align_curve_lagged, write_curve_series

    cfg = Config()
    cfg.paths.lake_dir = str(tmp_path)
    dates = pd.bdate_range("2020-01-01", periods=4)
    write_curve_series(
        cfg,
        pd.DataFrame(
            {
                "date": dates,
                "irx": [1.0, 1.0, 9.0, 1.0],
                "fvx": [2.0, 2.0, 9.0, 2.0],
                "tnx": [3.0, 3.0, 9.0, 3.0],
            }
        ),
        "test",
    )
    obs = align_curve_lagged(cfg, dates)
    assert obs.shape == (4, 3)
    assert np.all(obs[0] == 0.0)
    np.testing.assert_allclose(obs[1], [0.03, 0.02, 0.02])
    np.testing.assert_allclose(obs[2], [0.03, 0.02, 0.02])
    np.testing.assert_allclose(obs[3], [0.09, 0.09, 0.00])


def test_rf_lookup_refuses_a_stale_quote():
    rf = pd.Series([0.0001], index=pd.to_datetime(["2024-01-02"]))
    assert rf_on(rf, pd.Timestamp("2024-01-03")) == 0.0001
    assert rf_on(rf, pd.Timestamp("2024-02-01")) is None
