import numpy as np
import pandas as pd
import pytest

from tools.momentum import (rebalance_dates, momentum_scores, eligible, select_topk,
                            winsorize_prices, _exec_date)


def test_winsorize_caps_glitch_spikes_keeps_clean_column():
    idx = pd.bdate_range("2020-01-01", periods=12)
    clean = pd.Series(np.linspace(100, 112, 12), index=idx)
    glitch = clean.copy()
    glitch.iloc[6] = glitch.iloc[5] * 10.0                 # +900% split glitch
    w = winsorize_prices(pd.DataFrame({"CLEAN": clean, "GLITCH": glitch}), cap=0.5)
    assert np.allclose(w["CLEAN"].values, clean.values)    # clean column untouched
    assert w["GLITCH"].pct_change().dropna().max() <= 0.5 + 1e-9   # spike capped


def test_winsorize_preserves_dead_tail_nan():
    idx = pd.bdate_range("2020-01-01", periods=10)
    s = pd.Series(list(np.linspace(100, 50, 6)) + [np.nan] * 4, index=idx)
    w = winsorize_prices(pd.DataFrame({"D": s}), cap=0.5)
    assert w["D"].iloc[:6].notna().all() and w["D"].iloc[6:].isna().all()


def test_exec_date_lag_shifts_to_next_bar():
    idx = pd.bdate_range("2020-01-01", periods=5)
    assert _exec_date(idx, idx[0], 0) == idx[0]            # lag 0 = signal-day close
    assert _exec_date(idx, idx[0], 1) == idx[1]            # lag 1 = next bar
    assert _exec_date(idx, idx[2], 1) == idx[3]
    assert _exec_date(idx, idx[-1], 1) == idx[-1]          # clamps at the end


def test_eligible_drops_illiquid_and_short_history():
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({
        "LIQ":   np.linspace(100, 110, 300),         # ok
        "WIDE":  np.linspace(100, 110, 300),         # too-wide spread
        "SHORT": [np.nan] * 250 + list(np.linspace(100, 110, 50)),  # short history
    }, index=idx)
    slip = {"LIQ": 10, "WIDE": 80, "SHORT": 10}
    elig = eligible(px, idx[-1], slip, liq_max=30, min_obs=273)
    assert elig == {"LIQ"}


def test_eligible_drops_sub_min_price():
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({"OK": np.linspace(5, 6, 300),               # >= EUR1
                       "PENNY": np.linspace(0.004, 0.03, 300)}, idx)  # sub-EUR1 penny
    slip = {"OK": 10, "PENNY": 10}
    assert eligible(px, idx[-1], slip, liq_max=30, min_obs=273, min_price=1.0) == {"OK"}


def test_select_topk_ranks_and_respects_eligibility():
    scores = pd.Series({"A": 0.5, "B": 0.9, "C": 0.1, "D": 0.7})
    picks = select_topk(scores, {"A", "B", "C", "D"}, k=2)
    assert picks == ["B", "D"]                       # highest two, ordered
    picks2 = select_topk(scores, {"A", "C"}, k=5)    # eligibility limits pool
    assert picks2 == ["A", "C"]                       # k larger than pool is fine


def test_select_topk_sector_neutral_round_robin():
    scores = pd.Series({"A1": 0.9, "A2": 0.8, "A3": 0.7, "B1": 0.6, "C1": 0.5})
    sectors = {"A1": "Tech", "A2": "Tech", "A3": "Tech", "B1": "Energy", "C1": "Banks"}
    elig = set(scores.index)
    picks = select_topk(scores, elig, k=3, sectors=sectors)
    assert picks == ["A1", "B1", "C1"]               # best of each distinct sector first
    assert select_topk(scores, elig, k=3) == ["A1", "A2", "A3"]   # plain = global top-3


def test_trend_ok_above_below_200d_ma():
    from tools.momentum import trend_ok
    idx = pd.bdate_range("2019-01-01", periods=260)
    up = pd.Series(np.linspace(100.0, 200.0, 260), index=idx)     # last > 200d mean
    down = pd.Series(np.linspace(200.0, 100.0, 260), index=idx)   # last < 200d mean
    assert trend_ok(up, idx[-1], ma=200) is True
    assert trend_ok(down, idx[-1], ma=200) is False
    short = up.iloc[:50]
    assert trend_ok(short, short.index[-1], ma=200) is True       # too short → don't gate


def test_rebalance_dates_monthly_count():
    idx = pd.bdate_range("2022-01-03", periods=400)   # ~19 months
    dates = rebalance_dates(idx, "M")
    # one date per calendar month spanned, each a date present in the index
    months = len(set((d.year, d.month) for d in idx))
    assert len(dates) == months
    assert all(d in idx for d in dates)
    # each is the LAST index date in its month
    for d in dates:
        same_month = [x for x in idx if (x.year, x.month) == (d.year, d.month)]
        assert d == same_month[-1]


def test_rebalance_dates_quarterly():
    idx = pd.bdate_range("2020-01-01", periods=520)   # ~2 years
    q = rebalance_dates(idx, "Q")                       # upgrade E uses freq="Q"
    assert 6 <= len(q) <= 9                             # ~8 quarter-ends
    assert all(d in idx for d in q) and q == sorted(q)


def _rw(seed, n=400, drift=0.0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))


def test_momentum_scores_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=400)
    px = pd.DataFrame({"A": _rw(0), "B": _rw(1)}, index=idx)
    asof = idx[300]
    full = momentum_scores(px, asof)                 # later data present
    truncated = momentum_scores(px.loc[:asof], asof)  # later data removed
    pd.testing.assert_series_equal(full.sort_index(), truncated.sort_index())


def test_momentum_scores_is_12_1_return():
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({"UP": np.linspace(100.0, 200.0, 300),
                       "FLAT": np.full(300, 100.0)}, index=idx)
    s = momentum_scores(px, idx[-1], lookback=252, skip=21)
    assert s["UP"] > s["FLAT"]                       # trending beats flat
    assert abs(s["FLAT"]) < 1e-9                      # flat ~ zero momentum


def test_momentum_scores_insufficient_history_empty():
    idx = pd.bdate_range("2020-01-01", periods=100)
    px = pd.DataFrame({"A": _rw(2, 100)}, index=idx)
    assert momentum_scores(px, idx[-1], lookback=252, skip=21).empty


def test_vol_adjust_prefers_lower_vol_same_trend():
    idx = pd.bdate_range("2020-01-01", periods=300)
    t = np.arange(300)
    smooth = np.linspace(100.0, 150.0, 300)                        # same up-trend
    jagged = np.linspace(100.0, 150.0, 300) + 8.0 * np.sin(t * 1.3)  # + oscillation = higher vol
    px = pd.DataFrame({"SMOOTH": smooth, "JAGGED": jagged}, index=idx)
    adj = momentum_scores(px, idx[-1], lookback=252, skip=21, vol_adjust=True)
    assert adj["SMOOTH"] > adj["JAGGED"]                           # vol-adjust rewards the steady climber


from tools.momentum import run_momentum


def _multi(seed=1, n=400, ncols=20):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = {f"T{i}": 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
            for i in range(ncols)}
    return pd.DataFrame(cols, idx)


def test_run_momentum_cost_monotonic_same_schedule():
    px = _multi()
    slip = {t: 10 for t in px.columns}
    r = run_momentum(px, slip, k=5, cost_mults=(0.0, 1.0, 2.0))
    n0 = r["runs"][0.0]["stats"]["net_return"]
    n1 = r["runs"][1.0]["stats"]["net_return"]
    n2 = r["runs"][2.0]["stats"]["net_return"]
    assert n0 >= n1 >= n2                              # more cost -> lower net
    assert len(r["holdings_log"]) >= 2                 # schedule is cost-independent
    assert all(len(h["picks"]) <= 5 for h in r["holdings_log"])


def test_run_momentum_equity_starts_at_capital():
    px = _multi()
    slip = {t: 10 for t in px.columns}
    r = run_momentum(px, slip, k=5, capital=10_000.0, cost_mults=(1.0,))
    eq = r["runs"][1.0]["equity"]
    assert abs(eq.iloc[0] - 10_000.0) < 1e-6
    assert "stats" in r["runs"][1.0] and "sharpe" in r["runs"][1.0]["stats"]
    assert eq.index[0] == r["holdings_log"][0]["date"]   # anchored at first rebalance


def test_run_momentum_no_history_returns_empty_schedule():
    idx = pd.bdate_range("2020-01-01", periods=100)
    px = pd.DataFrame({"A": np.linspace(100, 110, 100)}, index=idx)
    r = run_momentum(px, {"A": 10}, k=5, cost_mults=(1.0,))
    assert r["holdings_log"] == []


def test_run_momentum_start_clips_first_rebalance_without_lookahead():
    px = _multi()                                    # daily 2019-01 .. ~2020-07
    slip = {t: 10 for t in px.columns}
    full = run_momentum(px, slip, k=5, cost_mults=(1.0,))
    clipped = run_momentum(px, slip, k=5, cost_mults=(1.0,), start="2020-03-01")

    assert full["holdings_log"][0]["date"] < pd.Timestamp("2020-03-01")
    assert clipped["holdings_log"]                                    # non-empty
    assert clipped["holdings_log"][0]["date"] >= pd.Timestamp("2020-03-01")
    # walk-forward: clipping the start changes nothing about the picks on shared dates
    full_by_date = {h["date"]: h["picks"] for h in full["holdings_log"]}
    for h in clipped["holdings_log"]:
        assert h["picks"] == full_by_date[h["date"]]


def test_run_momentum_graveyard_liquidates_dead_holding():
    from tools.universe_pit import PITUniverse
    idx = pd.bdate_range("2019-01-01", periods=400)
    winner = 100.0 * np.cumprod(1 + np.full(400, 0.001))
    bomb = np.concatenate([np.linspace(100, 300, 300), np.linspace(300, 30, 60),
                           [np.nan] * 40])
    px = pd.DataFrame({"WIN": winner, "BOMB": bomb,
                       "C": np.linspace(100, 90, 400)}, index=idx)
    slip = {t: 10 for t in px.columns}
    pit = PITUniverse(px, delisting={"BOMB": idx[359]})           # dies at bar 359
    r = run_momentum(px, slip, k=2, lookback=200, skip=10, cost_mults=(0.0,), pit=pit)
    for h in r["holdings_log"]:
        if h["date"] > idx[359]:
            assert "BOMB" not in h["picks"]                       # dead → never picked after death
    assert r["runs"][0.0]["equity"].notna().all()                # no NaN from the dead leg
    h0 = next(h for h in r["holdings_log"] if h["picks"])
    assert "ret" in h0 and "dead" in h0                          # timeline data attached
    assert all(t in h0["ret"] for t in h0["picks"])              # every pick has a period return


def test_run_momentum_elig_cache_matches_uncached():
    from tools.momentum import precompute_eligibility
    px = _multi()
    slip = {t: 10 for t in px.columns}
    dates = [d for d in rebalance_dates(px.index, "M") if len(px.loc[:d]) >= 201]
    cache = precompute_eligibility(px, slip, dates, min_obs=210)
    a = run_momentum(px, slip, k=5, lookback=200, skip=10, cost_mults=(1.0,))
    b = run_momentum(px, slip, k=5, lookback=200, skip=10, cost_mults=(1.0,),
                     elig_by_date=cache)
    assert [h["picks"] for h in a["holdings_log"]] == [h["picks"] for h in b["holdings_log"]]


def test_run_momentum_score_cache_matches_uncached():
    from tools.momentum import precompute_scores
    px = _multi()
    slip = {t: 10 for t in px.columns}
    dates = [d for d in rebalance_dates(px.index, "M") if len(px.loc[:d]) >= 201]
    sc = precompute_scores(px, dates, 200, 10)
    a = run_momentum(px, slip, k=5, lookback=200, skip=10, cost_mults=(1.0,), vol_adjust=True)
    b = run_momentum(px, slip, k=5, lookback=200, skip=10, cost_mults=(1.0,),
                     vol_adjust=True, score_by_date=sc)
    assert [h["picks"] for h in a["holdings_log"]] == [h["picks"] for h in b["holdings_log"]]


from tools.momentum import benchmark_curves, equal_weight_curve


def test_benchmark_curve_buy_hold_normalized():
    idx = pd.bdate_range("2021-01-01", periods=100)
    bench = pd.DataFrame({"MSCI World": np.linspace(100, 120, 100)}, index=idx)
    window = idx[10:90]
    curves = benchmark_curves(bench, window, capital=10_000.0)
    c = curves["MSCI World"]
    assert abs(c.iloc[0] - 10_000.0) < 1e-6           # starts at capital
    assert c.iloc[-1] > c.iloc[0]                      # rising benchmark rises
    assert list(c.index) == list(window)


def test_equal_weight_curve_starts_at_capital():
    idx = pd.bdate_range("2021-01-01", periods=120)
    px = pd.DataFrame({"A": np.linspace(100, 130, 120),
                       "B": np.linspace(100, 110, 120)}, index=idx)
    window = idx[5:115]
    c = equal_weight_curve(px, ["A", "B"], window, capital=10_000.0)
    assert abs(c.iloc[0] - 10_000.0) < 1e-6
    assert c.iloc[-1] > c.iloc[0]


def test_equal_weight_curve_empty_tickers_returns_empty():
    idx = pd.bdate_range("2021-01-01", periods=10)
    px = pd.DataFrame({"A": np.linspace(100, 110, 10)}, index=idx)
    assert equal_weight_curve(px, [], idx, capital=10_000.0).empty


import re

import build_momentum_report as bmr


def _fake_gather():
    idx = pd.bdate_range("2019-01-01", periods=400)
    rng = np.random.default_rng(3)
    px = pd.DataFrame({f"T{i}": 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 400)))
                       for i in range(20)}, index=idx)
    slip = {t: 10 for t in px.columns}
    from tools.momentum import run_momentum
    res = run_momentum(px, slip, k=5, cost_mults=(0.0, 1.0, 2.0), capital=10_000.0)
    return dict(res=res, prices=px, benchmarks=pd.DataFrame(index=idx), capital=10_000.0,
                meta={t: dict(name=t, local_id="000", country="X", sector="Y") for t in px.columns})


def test_public_report_has_no_euro_amounts():
    d = _fake_gather()
    html = bmr.build(d, public=True)
    euros = re.findall(r"€[0-9][0-9.,]*", html)
    assert all(e == "€1" for e in euros), euros


def test_private_report_builds_nonempty():
    d = _fake_gather()
    html = bmr.build(d, public=False)
    assert "<html" in html.lower() and "momentum" in html.lower()
    assert "Sharpe" in html


def test_grid_sections_local_only():
    assert hasattr(bmr, "sec_grid") and hasattr(bmr, "sec_survivorship")
    d = _fake_gather()
    d["grid"] = None                                   # gather may skip the grid
    html_pub = bmr.build(d, public=True)
    assert "64-permutation" not in html_pub            # grid is private-only


def test_pnl_color_buckets():
    assert bmr._pnl_color(0.30, False) == "#0a6b00"    # a lot up → dark green
    assert bmr._pnl_color(0.05, False) == "#46c84e"    # up → green
    assert bmr._pnl_color(-0.05, False) == "#ef4444"   # down → red
    assert bmr._pnl_color(-0.40, False) == "#7a0000"   # a lot down → dark red
    assert bmr._pnl_color(0.30, True) == "#000000"     # defaulted overrides → black


def test_eligible_turnover_gate_is_point_in_time():
    idx = pd.bdate_range("2024-01-02", periods=420)
    prices = pd.DataFrame({"X": 10.0, "Y": 10.0, "Z": 10.0}, index=idx)
    slip = {"X": 10, "Y": 10, "Z": 10}
    months = pd.date_range("2024-01-31", periods=12, freq="ME")
    turn = pd.DataFrame({"X": [200_000.0] * 12,
                         "Y": [50_000.0] * 6 + [200_000.0] * 6}, index=months)
    early, late = months[4], months[11]
    e_early = eligible(prices, early, slip, min_obs=10, turnover=turn)
    e_late = eligible(prices, late, slip, min_obs=10, turnover=turn)
    assert "Y" not in e_early          # trailing 6m median 50k < 100k floor THEN
    assert "Y" in e_late               # liquidity recovered -> eligible NOW
    assert "X" in e_early and "Z" in e_early   # Z uncovered -> static-gate fallback
    assert eligible(prices, early, slip, min_obs=10) == {"X", "Y", "Z"}  # None -> old behavior
