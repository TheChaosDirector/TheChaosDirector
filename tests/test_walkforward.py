"""Fold construction: no overlap, embargo respected, chronology preserved."""

from __future__ import annotations

from src.config import Config
from src.train.walkforward import make_folds


def test_folds_respect_embargo_and_chronology():
    cfg = Config()
    cfg.training.train_years = 1.0
    cfg.training.test_months = 3
    cfg.training.embargo_days = 5

    folds = make_folds(n_dates=1500, cfg=cfg, warmup=200, need_next_day=True)
    assert len(folds) >= 2

    for f in folds:
        assert f.train_start >= 200, "fold starts inside the feature warmup zone"
        assert f.test_start - f.train_end >= 5, "embargo gap violated"
        assert f.train_start < f.train_end < f.test_start < f.test_end
        assert f.test_end <= 1499, "portfolio fold needs a next-day price"

    for a, b in zip(folds, folds[1:]):
        assert b.train_start > a.train_start


def test_daytrade_folds_can_use_last_bar():
    cfg = Config(mode="daytrade")
    cfg.training.train_years = 1.0
    cfg.training.test_months = 3
    cfg.training.embargo_days = 5

    folds = make_folds(n_dates=1500, cfg=cfg, warmup=200, need_next_day=False)
    assert folds
    assert folds[-1].test_end <= 1500


def test_no_folds_when_history_too_short():
    cfg = Config()
    cfg.training.train_years = 3.0
    cfg.training.test_months = 6
    folds = make_folds(n_dates=300, cfg=cfg, warmup=200)
    assert folds == []
