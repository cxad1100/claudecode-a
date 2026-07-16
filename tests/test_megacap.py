import numpy as np
import pandas as pd
from tools.megacap import cap_panel, yoy_growth_panel


def test_cap_panel_pit_no_future_leak():
    idx = pd.bdate_range("2022-01-03", periods=260)
    prices = pd.DataFrame({"A": np.linspace(10, 20, 260)}, index=idx)
    # shares become AVAILABLE only at 2022-06-01; a later bump at 2022-12-01
    shares = {"A": pd.Series({pd.Timestamp("2022-06-01"): 100.0,
                              pd.Timestamp("2022-12-01"): 200.0})}
    dates = [pd.Timestamp("2022-03-31"), pd.Timestamp("2022-07-29"),
             pd.Timestamp("2022-12-30")]
    cap = cap_panel(shares, prices, dates)
    assert np.isnan(cap.loc[dates[0], "A"])                    # before any shares avail
    px_jul = prices["A"].loc[:dates[1]].iloc[-1]
    assert cap.loc[dates[1], "A"] == 100.0 * px_jul            # uses 100, not future 200
    px_dec = prices["A"].loc[:dates[2]].iloc[-1]
    assert cap.loc[dates[2], "A"] == 200.0 * px_dec            # bump now available


def test_yoy_growth_pit_and_value():
    idx = pd.bdate_range("2022-01-03", periods=400)
    prices = pd.DataFrame({"A": np.ones(400)}, index=idx)      # price irrelevant here
    rev = {"A": pd.DataFrame(
        {"revenue": [500.0, 550.0], "avail": [pd.Timestamp("2022-04-28"),
                                              pd.Timestamp("2023-04-28")]},
        index=[pd.Timestamp("2022-03-31"), pd.Timestamp("2023-03-31")])}
    before = pd.Timestamp("2023-04-01")                        # 2023 Q1 not yet filed
    after = pd.Timestamp("2023-05-01")                         # 2023 Q1 filed
    g = yoy_growth_panel(rev, [before, after])
    assert np.isnan(g.loc[before, "A"])                        # no prior-year pair available yet
    assert abs(g.loc[after, "A"] - (550.0 / 500.0 - 1.0)) < 1e-9


from tools.megacap import megacap_screen, cap_scores_by_date, growth_scores_by_date


def _cap_frame():
    dates = [pd.Timestamp("2022-01-31"), pd.Timestamp("2022-02-28")]
    return pd.DataFrame({"BIG": [100.0, 100.0], "MID": [50.0, 50.0],
                         "SMALL": [10.0, np.nan]}, index=dates), dates


def test_megacap_screen_topn_excludes_nan():
    cap, dates = _cap_frame()
    scr = megacap_screen(cap, dates, n=2)
    assert scr[dates[0]] == {"BIG", "MID"}
    assert scr[dates[1]] == {"BIG", "MID"}          # SMALL is NaN in Feb -> excluded anyway


def test_megacap_screen_n1():
    cap, dates = _cap_frame()
    assert megacap_screen(cap, dates, n=1)[dates[0]] == {"BIG"}


def test_scores_by_date_shape():
    cap, dates = _cap_frame()
    sc = cap_scores_by_date(cap, dates)
    assert set(sc[dates[0]]) == {"raw", "voladj"}
    assert sc[dates[0]]["raw"].equals(sc[dates[0]]["voladj"])
    assert sc[dates[0]]["raw"]["BIG"] == 100.0


from tools.megacap import run_arms, ARMS
from tools.momentum import rebalance_dates


def _synth_market():
    # ~14 months daily so >=1 monthly rebalance clears lookback; two names.
    idx = pd.bdate_range("2021-01-01", periods=300)
    # BIG rises fastest (best momentum); SMALL flat.
    prices = pd.DataFrame({"BIG": np.linspace(10, 40, 300),
                           "SMALL": np.linspace(10, 11, 300)}, index=idx)
    dates = rebalance_dates(prices.index)
    cap = pd.DataFrame({"BIG": 5.0, "SMALL": 100.0}, index=dates)   # SMALL bigger cap
    yoy = pd.DataFrame({"BIG": 0.9, "SMALL": 0.1}, index=dates)     # BIG grows faster
    slip = {"BIG": 10, "SMALL": 10}
    return prices, cap, yoy, slip


def test_run_arms_returns_three_runs():
    prices, cap, yoy, slip = _synth_market()
    res = run_arms(prices, slip, cap, yoy, n=2, k=1, lookback=200, skip=21)
    assert set(res) == set(ARMS)
    assert all("runs" in res[a] for a in ARMS)


def test_arms_pick_by_their_own_score():
    prices, cap, yoy, slip = _synth_market()
    res = run_arms(prices, slip, cap, yoy, n=2, k=1, lookback=200, skip=21)
    size_picks = [p for h in res["size"]["holdings_log"] for p in h["picks"]]
    grow_picks = [p for h in res["growth"]["holdings_log"] for p in h["picks"]]
    mom_picks = [p for h in res["momentum"]["holdings_log"] for p in h["picks"]]
    assert set(size_picks) == {"SMALL"}      # SMALL has the larger cap
    assert set(grow_picks) == {"BIG"}        # BIG has the faster revenue growth
    assert set(mom_picks) == {"BIG"}         # BIG has the stronger 12-1 momentum


# ── value / GARP arms (price-to-sales; US filers only for currency safety) ─────

from tools.megacap import (ttm_revenue_panel, ps_panel, value_scores_by_date,
                           garp_scores_by_date, run_value_arms)


def _quarterly_rev(vals, start="2021-03-31", avail_lag_days=30):
    idx = pd.date_range(start, periods=len(vals), freq="QE")
    return pd.DataFrame({"revenue": [float(v) for v in vals],
                         "avail": idx + pd.Timedelta(days=avail_lag_days)}, index=idx)


def test_ttm_revenue_sums_last_four_quarters_pit():
    rev = {"A": _quarterly_rev([100, 110, 120, 130, 140])}       # 5 quarters
    d = rev["A"].index[-1] + pd.Timedelta(days=40)               # all 5 filed
    ttm = ttm_revenue_panel(rev, [d])
    assert abs(ttm.loc[d, "A"] - (110 + 120 + 130 + 140)) < 1e-9  # trailing 4, not 5


def _no_ttm(ttm, d, t):
    """No TTM = the name is absent (omit-when-empty, like cap_panel) or NaN."""
    return t not in ttm.columns or np.isnan(ttm.loc[d, t])


def test_ttm_revenue_incomplete_year_is_nan():
    rev = {"A": _quarterly_rev([100, 110, 120])}                 # only 3 quarters
    d = rev["A"].index[-1] + pd.Timedelta(days=40)
    assert _no_ttm(ttm_revenue_panel(rev, [d]), d, "A")          # <4Q → no TTM


def test_ttm_revenue_no_lookahead():
    rev = {"A": _quarterly_rev([100, 110, 120, 130])}
    before_last = rev["A"]["avail"].iloc[-1] - pd.Timedelta(days=1)  # 4th Q not yet filed
    assert _no_ttm(ttm_revenue_panel(rev, [before_last]), before_last, "A")


def test_ttm_revenue_annual_fallback():
    idx = pd.date_range("2020-12-31", periods=3, freq="YE")      # annual cadence (20-F)
    rev = {"A": pd.DataFrame({"revenue": [400.0, 450.0, 500.0],
                              "avail": idx + pd.Timedelta(days=60)}, index=idx)}
    d = idx[-1] + pd.Timedelta(days=70)
    assert ttm_revenue_panel(rev, [d]).loc[d, "A"] == 500.0      # last annual = TTM


def test_ps_panel_is_cap_over_ttm_and_drops_foreign():
    dates = pd.DatetimeIndex(["2023-03-31", "2023-06-30"])
    cap = pd.DataFrame({"AAPL": [3000., 3200.], "SAP.DE": [2000., 2100.]}, index=dates)
    ttm = pd.DataFrame({"AAPL": [100., 100.], "SAP.DE": [50., 50.]}, index=dates)
    ps = ps_panel(cap, ttm)
    assert "SAP.DE" not in ps.columns                            # foreign filer → currency-unsafe
    assert abs(ps.loc[dates[0], "AAPL"] - 30.0) < 1e-9


def test_ps_panel_nan_when_rev_nonpositive():
    dates = pd.DatetimeIndex(["2023-03-31"])
    cap = pd.DataFrame({"AAPL": [3000.]}, index=dates)
    ttm = pd.DataFrame({"AAPL": [0.0]}, index=dates)
    ps = ps_panel(cap, ttm)
    assert "AAPL" not in ps.columns or np.isnan(ps.loc[dates[0], "AAPL"])


def test_value_scores_rank_cheapest_highest():
    dates = pd.DatetimeIndex(["2023-03-31"])
    ps = pd.DataFrame({"CHEAP": [5.0], "MID": [15.0], "RICH": [40.0]}, index=dates)
    sc = value_scores_by_date(ps, dates)[dates[0]]["raw"].sort_values(ascending=False)
    assert list(sc.index) == ["CHEAP", "MID", "RICH"]


def test_garp_scores_prefer_cheap_and_growing():
    dates = pd.DatetimeIndex(["2023-03-31"])
    ps = pd.DataFrame({"CHEAPGROW": [10.], "RICHGROW": [40.], "CHEAPFLAT": [10.]}, index=dates)
    yoy = pd.DataFrame({"CHEAPGROW": [0.5], "RICHGROW": [0.5], "CHEAPFLAT": [0.05]}, index=dates)
    sc = garp_scores_by_date(ps, yoy, dates)[dates[0]]["raw"].sort_values(ascending=False)
    assert sc.index[0] == "CHEAPGROW"                            # cheap AND growing = best
    assert sc["CHEAPGROW"] > sc["RICHGROW"]                      # cheaper wins at equal growth
    assert sc["CHEAPGROW"] > sc["CHEAPFLAT"]                     # faster growth wins at equal price


def test_run_value_arms_pick_by_price_to_sales():
    idx = pd.bdate_range("2021-01-01", periods=300)
    prices = pd.DataFrame({"CHEAP": np.linspace(10, 12, 300),
                           "RICH": np.linspace(10, 12, 300)}, index=idx)
    dates = rebalance_dates(prices.index)
    cap = pd.DataFrame({"CHEAP": 100.0, "RICH": 100.0}, index=dates)   # equal cap → both in screen
    rev = {"CHEAP": _quarterly_rev([50] * 10, start="2019-03-31"),     # high rev → low P/S
           "RICH":  _quarterly_rev([5] * 10, start="2019-03-31")}      # low rev  → high P/S
    yoy = pd.DataFrame({"CHEAP": 0.1, "RICH": 0.1}, index=dates)
    res = run_value_arms(prices, {"CHEAP": 10, "RICH": 10}, cap, rev, yoy,
                         n=2, k=1, lookback=200)
    assert set(res) == {"value", "garp"}
    val_picks = {p for h in res["value"]["holdings_log"] for p in h["picks"]}
    assert val_picks == {"CHEAP"}                               # value buys the cheaper P/S name
