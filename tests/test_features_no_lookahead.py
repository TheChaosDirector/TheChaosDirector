"""Prove the feature factory can't see the future: truncate history at a date
and demand the feature row for that date is bit-identical to the full build."""

from __future__ import annotations

import numpy as np

from src.features.factory import build_features, warmup_days


def test_truncated_features_match_full_build(fake_market, cfg):
    close, volume = fake_market
    full = build_features(close, volume, cfg.features, benchmark="SPY")
    warmup = warmup_days(cfg.features)

    for idx in [warmup + 3, len(close) // 2, len(close) - 10]:
        t_close = close.iloc[: idx + 1]
        t_volume = volume.iloc[: idx + 1]
        truncated = build_features(t_close, t_volume, cfg.features, benchmark="SPY")
        np.testing.assert_allclose(
            truncated.values[-1],
            full.values[idx],
            atol=1e-6,
            err_msg=f"Features at index {idx} change when the future is deleted",
        )


def test_features_are_lagged_one_day(fake_market, cfg):
    """Changing ONLY today's price must not change today's features."""
    close, volume = fake_market
    panel_a = build_features(close, volume, cfg.features, benchmark="SPY")

    poked = close.copy()
    poked.iloc[-1] = poked.iloc[-1] * 1.5  # violently move today's close
    panel_b = build_features(poked, volume, cfg.features, benchmark="SPY")

    np.testing.assert_allclose(
        panel_a.values[-1],
        panel_b.values[-1],
        atol=1e-9,
        err_msg="Today's features reacted to today's close — the lag is broken",
    )
