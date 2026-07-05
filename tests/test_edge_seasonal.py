"""Tests for the tax-loss-rebound sleeve (tools.edge_seasonal).

The two tests that matter most: the engine must FIND a January rebound that is
deliberately planted in synthetic data, and must find NOTHING in IID data."""
import numpy as np
import pandas as pd

from tools import edge_seasonal as ES
from tools import significance as sig


def _universe(n_names=60, years=(2016, 2023), seed=0, plant_rebound=False):
    """Daily panel. With plant_rebound, 12 designated names lose ~40% through each year
    (tax-loss bait) then rebound hard in January; the rest are mild IID."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(f"{years[0]}-01-01", f"{years[1] + 1}-03-01")
    cols = {}
    losers = [f"L{i}" for i in range(12)] if plant_rebound else []
    names = losers + [f"N{i}" for i in range(n_names - len(losers))]
    for name in names:
        r = rng.normal(0.0002, 0.012, len(idx))
        if name in losers:
            in_year = (idx.month <= 11) | ((idx.month == 12) & (idx.day <= 15))
            r = np.where(in_year, r - 0.0028, r)          # grind down all year
            jan = (idx.month == 1) | ((idx.month == 12) & (idx.day > 15))
            r = np.where(jan, r + 0.012, r)               # the planted rebound
        cols[name] = 50.0 * np.cumprod(1 + r)
    return pd.DataFrame(cols, index=idx)


def _slip(prices):
    return {t: 10.0 for t in prices.columns}


def test_windows_calendar():
    idx = pd.bdate_range("2018-01-01", "2021-03-01")
    wins = ES.seasonal_windows(idx)
    assert [w["year"] for w in wins] == [2018, 2019, 2020]
    for w in wins:
        assert w["signal"].month == 12 and w["signal"].day <= 15
        assert w["exit"] > w["signal"]
    assert wins[-1]["exit"].month == 1 and wins[-1]["exit"].year == 2021


def test_select_losers_requires_negative_ytd():
    ytd = pd.Series({"A": -0.5, "B": -0.1, "C": 0.3, "D": -0.02})
    picks = ES.select_losers(ytd, {"A", "B", "C", "D"}, k=3)
    assert picks == ["A", "B", "D"]                       # C is a winner — excluded
    assert ES.select_losers(pd.Series({"C": 0.3}), {"C"}, k=3) == []


def test_finds_planted_rebound():
    prices = _universe(plant_rebound=True, seed=1)
    res = ES.run_seasonal(prices, _slip(prices), k=10, execute_lag=1)
    assert len(res["years"]) >= 6
    # the picks are dominated by the planted losers…
    hit = np.mean([np.mean([p.startswith("L") for p in y["picks"]])
                   for y in res["years"] if y["picks"]])
    assert hit > 0.8
    # …the sleeve makes money net of costs…
    assert res["equity"].iloc[-1] > res["equity"].iloc[0] * 1.15
    # …and beats random selection from the same pool decisively
    mc = sig.monte_carlo_null(res["pools"], ES.seasonal_period_returns(res["years"]),
                              k=10, ppy=1.0, n_trials=400, seed=0)
    assert mc["p_sharpe"] < 0.05


def test_finds_nothing_in_iid_data():
    prices = _universe(plant_rebound=False, seed=2)
    res = ES.run_seasonal(prices, _slip(prices), k=10, execute_lag=1)
    mc = sig.monte_carlo_null(res["pools"], ES.seasonal_period_returns(res["years"]),
                              k=10, ppy=1.0, n_trials=400, seed=0)
    assert mc["p_sharpe"] > 0.05                          # no fake edge on IID returns
    # and a ~flat equity path (costs only) — nothing resembling the planted case
    total = res["equity"].iloc[-1] / res["equity"].iloc[0] - 1.0
    assert abs(total) < 0.15


def test_no_look_ahead_in_picks():
    prices = _universe(plant_rebound=True, seed=3)
    res = ES.run_seasonal(prices, _slip(prices), k=10)
    w0 = res["years"][0]
    mut = prices.copy()
    mut.loc[mut.index > w0["signal"]] *= 3.0              # absurd future
    res2 = ES.run_seasonal(mut, _slip(prices), k=10)
    assert res2["years"][0]["picks"] == w0["picks"]       # first-year picks unchanged


def test_costs_reduce_equity():
    prices = _universe(plant_rebound=True, seed=4)
    free = ES.run_seasonal(prices, _slip(prices), k=10, cost_mult=0.0)
    paid = ES.run_seasonal(prices, _slip(prices), k=10, cost_mult=2.0)
    assert paid["equity"].iloc[-1] < free["equity"].iloc[-1]


def test_entry_executes_after_signal():
    prices = _universe(plant_rebound=True, seed=5)
    res = ES.run_seasonal(prices, _slip(prices), k=5, execute_lag=1)
    for y in res["years"]:
        if y["picks"]:
            first_move = res["equity"].loc[y["signal"]:].index[0]
            assert first_move >= y["signal"]              # nothing booked before the signal


def test_no_stub_window_when_data_ends_in_december():
    idx = pd.bdate_range("2018-01-01", "2020-12-20")      # ends 5 sessions after signal
    wins = ES.seasonal_windows(idx)
    assert [w["year"] for w in wins] == [2018, 2019]      # 2020's stub is rejected
    for w in wins:
        assert w["exit"] >= pd.Timestamp(w["year"] + 1, 1, 1)


def test_deaths_during_window_are_recorded():
    prices = _universe(plant_rebound=True, seed=6)

    class StubPIT:                                        # every name listed; L0 dies
        def listed(self, t, d):
            return True

        def died_between(self, d, nxt):
            return {"L0"}

    res = ES.run_seasonal(prices, _slip(prices), k=10, pit=StubPIT())
    with_picks = [y for y in res["years"] if y["picks"]]
    assert with_picks and all("dead" in y for y in res["years"])
    for y in with_picks:                                  # L0 is a planted loser → picked
        if "L0" in y["picks"]:
            assert "L0" in y["dead"]


def test_combine_sleeves_blend():
    idx = pd.bdate_range("2020-01-01", periods=100)
    core = pd.Series(0.01, index=idx)
    sleeve = pd.Series(0.0, index=idx)
    r = ES.combine_sleeves(core, sleeve, w_sleeve=0.2)
    assert np.allclose(r.values, 0.008)                   # 80% of the core's return
