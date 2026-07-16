import numpy as np
import pandas as pd
import pytest

from tools.momentum import rebalance_dates
from tools.dipbuy import dip_returns, dip_scores, dip_scores_by_date, run_dipbuy


def _rw(seed, n=400, drift=0.0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))


# ── signal: trailing return + reversal score (pure, PIT) ──────────────────────

def test_dip_returns_is_trailing_window_return():
    idx = pd.bdate_range("2020-01-01", periods=30)
    px = pd.DataFrame({"A": np.linspace(100.0, 130.0, 30)}, index=idx)
    r = dip_returns(px, idx[-1], window=5)
    expected = px["A"].iloc[-1] / px["A"].iloc[-6] - 1.0     # last vs 5 bars prior
    assert abs(float(r["A"]) - expected) < 1e-12


def test_dip_score_is_negative_trailing_return():
    idx = pd.bdate_range("2020-01-01", periods=30)
    px = pd.DataFrame({"DROP": np.linspace(130.0, 100.0, 30),    # falling
                       "RISE": np.linspace(100.0, 130.0, 30)},   # rising
                      index=idx)
    s = dip_scores(px, idx[-1], window=5)
    assert s["DROP"] > 0 > s["RISE"]          # a faller scores positive, a riser negative


def test_dip_scores_rank_bigger_dip_higher():
    idx = pd.bdate_range("2020-01-01", periods=30)
    px = pd.DataFrame({"BIG":   list(np.full(29, 100.0)) + [80.0],    # -20% last bar
                       "SMALL": list(np.full(29, 100.0)) + [95.0],    # -5% last bar
                       "FLAT":  np.full(30, 100.0)}, index=idx)
    s = dip_scores(px, idx[-1], window=1).sort_values(ascending=False)
    assert list(s.index) == ["BIG", "SMALL", "FLAT"]     # deepest dip ranked first


def test_dip_scores_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=400)
    px = pd.DataFrame({"A": _rw(0), "B": _rw(1)}, index=idx)
    asof = idx[300]
    full = dip_scores(px, asof, window=21)                # later data present
    truncated = dip_scores(px.loc[:asof], asof, window=21)  # later data removed
    pd.testing.assert_series_equal(full.sort_index(), truncated.sort_index())


def test_dip_scores_too_little_history_is_empty():
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame({"A": np.linspace(100, 110, 5)}, index=idx)
    assert dip_scores(px, idx[-1], window=21).empty       # < window+1 rows → empty


def test_dip_scores_by_date_matches_run_momentum_contract():
    idx = pd.bdate_range("2020-01-01", periods=60)
    px = pd.DataFrame({"A": _rw(0, 60), "B": _rw(1, 60)}, index=idx)
    dates = [idx[30], idx[59]]
    sbd = dip_scores_by_date(px, dates, window=5)
    assert set(sbd) == set(dates)
    for d in dates:                                       # {"raw","voladj"} both present, identical
        assert set(sbd[d]) == {"raw", "voladj"}
        pd.testing.assert_series_equal(sbd[d]["raw"], sbd[d]["voladj"])


# ── engine: reversal book on the top-N cap screen ─────────────────────────────

def _flat_cap(px):
    """Equal PIT cap for every name at every rebalance → the top-N screen admits all."""
    dates = rebalance_dates(px.index)
    return pd.DataFrame({c: 1e12 for c in px.columns}, index=dates)


def _rand_universe(seed=1, n=400, names=("A", "B", "C", "D", "E")):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    px = pd.DataFrame({c: 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
                       for c in names}, index=idx)
    return px, idx


def test_run_dipbuy_holds_biggest_recent_loser():
    idx = pd.bdate_range("2020-01-01", periods=120)
    px = pd.DataFrame({"A": np.linspace(100, 160, 120),     # rising
                       "B": np.linspace(160, 100, 120),     # falling → deepest dip
                       "C": np.linspace(100, 130, 120)}, index=idx)
    slip = {c: 5 for c in px.columns}
    res = run_dipbuy(px, slip, _flat_cap(px), n=3, k=1, window=21, lookback=20,
                     cost_mults=(1.0,))
    picks = [h["picks"] for h in res["holdings_log"] if h["picks"]]
    assert picks and all(p == ["B"] for p in picks)          # the faller, every rebalance


def test_run_dipbuy_stays_fully_invested_when_screen_full():
    px, idx = _rand_universe()
    slip = {c: 5 for c in px.columns}
    res = run_dipbuy(px, slip, _flat_cap(px), n=5, k=3, window=21, lookback=30,
                     cost_mults=(1.0,))
    hp = [h["picks"] for h in res["holdings_log"]]
    assert hp and all(len(p) == 3 for p in hp)               # exactly k every rebalance


def test_run_dipbuy_walk_forward_holdings_stable_under_truncation():
    px, idx = _rand_universe()
    cap = _flat_cap(px)
    slip = {c: 5 for c in px.columns}
    kw = dict(n=5, k=2, window=21, lookback=30, cost_mults=(1.0,))
    full = run_dipbuy(px, slip, cap, **kw)
    cut = cap.index[len(cap.index) // 2]                     # a real month-end
    part = run_dipbuy(px.loc[:cut], slip, cap.loc[:cut], **kw)
    full_by = {h["date"]: h["picks"] for h in full["holdings_log"]}
    for h in part["holdings_log"]:                           # nothing in the past may change
        assert full_by.get(h["date"]) == h["picks"]


def test_run_dipbuy_defaults_to_t_plus_one_execution():
    px, idx = _rand_universe(seed=7)
    cap = _flat_cap(px)
    slip = {c: 5 for c in px.columns}
    kw = dict(n=5, k=3, window=21, lookback=30, cost_mults=(1.0,))
    default = run_dipbuy(px, slip, cap, **kw)["runs"][1.0]["equity"]
    lag1 = run_dipbuy(px, slip, cap, execute_lag=1, **kw)["runs"][1.0]["equity"]
    lag0 = run_dipbuy(px, slip, cap, execute_lag=0, **kw)["runs"][1.0]["equity"]
    pd.testing.assert_series_equal(default, lag1)            # default == honest t+1 fill
    assert not default.equals(lag0)                          # and t+1 genuinely differs from same-bar


def test_run_dipbuy_cost_monotonic_same_schedule():
    px, idx = _rand_universe()
    res = run_dipbuy(px, {c: 5 for c in px.columns}, _flat_cap(px),
                     n=5, k=3, window=21, lookback=30, cost_mults=(0.0, 1.0, 2.0))
    e0 = float(res["runs"][0.0]["equity"].iloc[-1])
    e1 = float(res["runs"][1.0]["equity"].iloc[-1])
    e2 = float(res["runs"][2.0]["equity"].iloc[-1])
    assert e0 >= e1 >= e2                                    # more cost never helps
