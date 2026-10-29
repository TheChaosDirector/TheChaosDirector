"""The most important test in the repo.

We plant a deliberately cheating feature — tomorrow's return, disguised as a
normal column — and assert the referee's correlation trap flags it while
waving the honest features through. If this test ever fails, the seatbelt is
broken and no profit number can be trusted.
"""

from __future__ import annotations

from src.features.factory import build_features
from src.referee.checks import feature_future_correlations

THRESHOLD = 0.30


def test_referee_flags_planted_future_return(fake_market, cfg):
    close, volume = fake_market

    cheat_future = close.pct_change().shift(-1)  # literally tomorrow's move
    panel = build_features(
        close, volume, cfg.features, benchmark="SPY",
        extra_unlagged={"totally_innocent_signal": cheat_future},
    )

    corrs = feature_future_correlations(close, panel)
    assert corrs["totally_innocent_signal"] > THRESHOLD, (
        "Referee failed to flag a feature that IS tomorrow's return"
    )


def test_referee_flags_unshifted_momentum(fake_market, cfg):
    """A subtler cheat: today's momentum without the one-day lag."""
    close, volume = fake_market

    cheat_contemporaneous = close.pct_change(5)  # includes today's close
    panel = build_features(
        close, volume, cfg.features, benchmark="SPY",
        extra_unlagged={"sneaky_mom": cheat_contemporaneous},
    )

    corrs = feature_future_correlations(close, panel)
    assert corrs["sneaky_mom"] > THRESHOLD, (
        "Referee failed to flag an unlagged momentum feature"
    )


def test_referee_passes_honest_features(fake_market, cfg):
    close, volume = fake_market
    panel = build_features(close, volume, cfg.features, benchmark="SPY")
    corrs = feature_future_correlations(close, panel)
    offenders = {k: v for k, v in corrs.items() if v > THRESHOLD}
    assert not offenders, f"Honest features wrongly flagged: {offenders}"
