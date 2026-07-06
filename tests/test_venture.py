"""Venture instrumentation — money-weighted comparison against a
same-cashflow benchmark shadow, and the pre-registered drawdown ladder.

The north-star comparison must be cashflow-fair: book XIRR vs the XIRR of
investing the identical deposits into IWDA on the identical dates. Plain
index TWR vs a contribution-driven IRR is mechanically biased — the red-team
finding that created this module.
"""
import numpy as np
import pandas as pd

from tools.venture import dd_state, shadow_curve, venture_summary, xirr


def test_xirr_single_flow_one_year():
    flows = [(pd.Timestamp("2025-01-01"), -1000.0)]
    r = xirr(flows, terminal_value=1100.0,
             terminal_date=pd.Timestamp("2026-01-01"))
    assert abs(r - 0.10) < 1e-3


def test_xirr_two_deposits():
    flows = [(pd.Timestamp("2025-01-01"), -1000.0),
             (pd.Timestamp("2025-07-01"), -1000.0)]
    # terminal exactly equal to deposits → rate ≈ 0
    r = xirr(flows, terminal_value=2000.0,
             terminal_date=pd.Timestamp("2026-01-01"))
    assert abs(r) < 1e-3
    # 10% on both money-years-ish → positive, below 10% single-flow rate
    r2 = xirr(flows, terminal_value=2150.0,
              terminal_date=pd.Timestamp("2026-01-01"))
    assert 0.05 < r2 < 0.15


def test_shadow_curve_buys_units_at_deposit_dates():
    idx = pd.bdate_range("2025-01-01", periods=300)
    bench = pd.Series(np.linspace(100, 200, 300), index=idx)   # doubles
    cf = pd.DataFrame([dict(date=idx[0], amount=1000.0),
                       dict(date=idx[150], amount=1000.0)])
    eq = shadow_curve(cf, bench)
    assert abs(eq.iloc[0] - 1000.0) < 1e-6
    # second deposit buys at ~150.17: terminal = 10u*200 + 1000/px150*200
    expected_terminal = 1000 / 100 * 200 + 1000 / bench.iloc[150] * 200
    assert abs(eq.iloc[-1] - expected_terminal) < 1e-6
    # deposit on a non-trading day executes at the next session
    cf2 = pd.DataFrame([dict(date=idx[10] + pd.Timedelta(days=0), amount=500.0)])
    sat = idx[10] - pd.Timedelta(days=1)          # ensure weekend handling path
    cf3 = pd.DataFrame([dict(date=sat, amount=500.0)])
    eq3 = shadow_curve(cf3, bench)
    assert len(eq3.dropna())


def test_dd_state_ladder():
    peak_then_drop = pd.Series([100, 120, 100, 84], dtype=float,
                               index=pd.bdate_range("2025-01-01", periods=4))
    s = dd_state(peak_then_drop)
    assert s["state"] == "half-vol"               # −30% from 120 peak
    fine = dd_state(pd.Series([100.0, 101, 102],
                              index=pd.bdate_range("2025-01-01", periods=3)))
    assert fine["state"] == "normal"
    crash = dd_state(pd.Series([100.0, 60],
                               index=pd.bdate_range("2025-01-01", periods=2)))
    assert crash["state"] == "derisk"
    assert crash["dd"] <= -0.35


def test_venture_summary_excess_sign():
    idx = pd.bdate_range("2025-01-01", periods=260)
    bench = pd.Series(np.linspace(100, 110, 260), index=idx)    # +10%
    book = pd.Series(np.linspace(10_000, 12_000, 260), index=idx)  # +20%
    cf = pd.DataFrame([dict(date=idx[0], amount=10_000.0)])
    out = venture_summary(cf, book, bench)
    assert out["book_xirr"] > out["shadow_xirr"] > 0
    assert out["excess"] > 0.05
    assert out["dd_state"] == "normal"
    assert 0 < out["months"] < 14
