"""Macro releases show up the session after they are public, and not before."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.macro import macro_known_before
from src.features.factory import build_features


def _monthly():
    dates = pd.date_range("1998-01-01", "2001-12-01", freq="MS")
    return pd.DataFrame(
        {
            "date": dates,
            "cpi": np.linspace(100.0, 110.0, len(dates)),
            "unrate_percent": np.linspace(4.0, 5.0, len(dates)),
        }
    )


def test_a_release_dated_today_is_visible_the_next_session():
    monthly = _monthly()
    index = pd.bdate_range("2000-02-07", "2000-02-10")
    known = macro_known_before(index, monthly)
    # January 2000 unemployment is public on February 8, so that session still
    # shows the prior release and the next session shows the new one.
    assert known.loc[pd.Timestamp("2000-02-08"), "unrate"] == known.loc[pd.Timestamp("2000-02-07"), "unrate"]
    assert known.loc[pd.Timestamp("2000-02-09"), "unrate"] != known.loc[pd.Timestamp("2000-02-08"), "unrate"]
    feb = pd.bdate_range("2000-02-15", "2000-02-17")
    cpi = macro_known_before(feb, monthly)
    assert float(cpi.loc[pd.Timestamp("2000-02-16"), "cpi_yoy"]) == float(cpi.loc[pd.Timestamp("2000-02-15"), "cpi_yoy"])
    assert float(cpi.loc[pd.Timestamp("2000-02-17"), "cpi_yoy"]) != float(cpi.loc[pd.Timestamp("2000-02-16"), "cpi_yoy"])


def test_a_future_month_does_not_change_an_earlier_session():
    monthly = _monthly()
    index = pd.bdate_range("2000-06-01", periods=5)
    bill = pd.Series(5.0, index=pd.bdate_range("2000-05-02", periods=40))
    full = macro_known_before(index, monthly, bill_annual_percent=bill)
    trimmed = macro_known_before(index, monthly.iloc[:-3].copy(), bill_annual_percent=bill)
    pd.testing.assert_series_equal(full["cpi_yoy"], trimmed["cpi_yoy"])
    pd.testing.assert_series_equal(full["unrate"], trimmed["unrate"])
    assert full["real_bill"].notna().all()


def test_a_coarser_bill_clock_still_lines_up():
    monthly = _monthly()
    index = pd.bdate_range("2000-06-01", periods=5)
    bill = pd.Series(5.0, index=pd.bdate_range("2000-05-02", periods=40).astype("datetime64[ms]"))
    known = macro_known_before(index, monthly, bill_annual_percent=bill)
    assert known["real_bill"].notna().all()
    assert abs(float(known["real_bill"].iloc[-1]) - (0.05 - float(known["cpi_yoy"].iloc[-1]))) < 1e-12


def test_macro_columns_are_not_shifted_a_second_time(fake_market, cfg):
    close, volume = fake_market
    lagged = pd.Series(0.0, index=close.index)
    lagged.iloc[-1] = 0.25
    panel = build_features(
        close, volume, cfg.features, benchmark="SPY", extra_lagged={"cpi_yoy": lagged}
    )
    assert panel.names[-1] == "cpi_yoy"
    assert abs(float(panel.values[-1, 0, -1]) - 0.25) < 1e-6
    plain = build_features(close, volume, cfg.features, benchmark="SPY")
    assert "cpi_yoy" not in plain.names
    assert plain.n_features == panel.n_features - 1
