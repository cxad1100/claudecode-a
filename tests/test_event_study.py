"""Canonical event-study machinery — market-model CARs, BMP t, calendar-time.

Pins: a planted post-event jump shows up as positive CAR with a significant
BMP t; the estimation window is strictly pre-event (a jump inside the event
window must NOT contaminate alpha/beta); the calendar-time portfolio holds a
name only AFTER publication (the PIT rule again, portfolio form).
"""
import numpy as np
import pandas as pd

from tools.event_study import calendar_time_portfolio, car_stats, market_model_ar


def _flat_market(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    mkt = pd.Series((1 + rng.normal(0.0002, 0.01, n)).cumprod() * 100, index=idx)
    return idx, mkt, rng


def test_planted_jump_positive_car_significant_bmp():
    idx, mkt, rng = _flat_market()
    mret = mkt.pct_change().fillna(0.0)
    prices = {}
    events = []
    for k in range(12):
        r = 0.8 * mret + rng.normal(0, 0.008, len(idx))
        e = 250 + 5 * k
        r.iloc[e] += 0.08                          # +8% on event day
        prices[f"S{k}"] = 100 * (1 + r).cumprod()
        events.append(dict(ticker=f"S{k}", event_date=idx[e]))
    pf = pd.DataFrame(prices)
    res = market_model_ar(pf, mkt, pd.DataFrame(events),
                          est_window=(-120, -21), evt_window=(-5, 20))
    assert res["n"] == 12
    stats = car_stats(res)
    assert stats["car"][0] > 0.06                  # day-0 CAR ≈ +8%
    # persists above the α/β-estimation noise floor (12 overlapping events
    # share one market draw → the CAR path drifts a few bp/day either way)
    assert stats["car"][20] > 0.04
    assert stats["bmp_t"] > 3.0
    assert abs(stats["car"][-5]) < 0.02            # nothing before the event


def test_estimation_window_strictly_pre_event():
    idx, mkt, rng = _flat_market()
    mret = mkt.pct_change().fillna(0.0)
    r = 0.5 * mret + rng.normal(0, 0.005, len(idx))
    e = 300
    r.iloc[e + 2] += 0.5                           # huge jump INSIDE evt window
    pf = pd.DataFrame({"X": 100 * (1 + r).cumprod()})
    ev = pd.DataFrame([dict(ticker="X", event_date=idx[e])])
    res = market_model_ar(pf, mkt, ev, est_window=(-120, -21),
                          evt_window=(-5, 20))
    b = res["betas"]["X"]
    assert 0.2 < b < 0.8                           # beta from clean window only


def test_calendar_time_holds_only_after_publication():
    idx, mkt, rng = _flat_market()
    pf = pd.DataFrame({"A": np.linspace(100, 120, len(idx)),
                       "B": np.linspace(100, 90, len(idx))}, index=idx)
    ev = pd.DataFrame([dict(ticker="A", published_at=idx[100]),
                       dict(ticker="B", published_at=idx[300])])
    port = calendar_time_portfolio(pf, ev, hold_days=21)
    assert port.loc[:idx[99]].dropna().eq(0).all() or \
           port.loc[:idx[99]].dropna().empty      # nothing held pre-publication
    assert port.loc[idx[102]:idx[118]].notna().all()
    # after A's window closes and before B publishes → flat again
    gap = port.loc[idx[130]:idx[295]]
    assert gap.dropna().eq(0).all() or gap.dropna().empty
