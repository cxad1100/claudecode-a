"""Unit tests for the pairs-trading universe and engine math."""

import numpy as np
import pandas as pd
import pytest

import tools.pairs_universe as pu
from tools.pairs_universe import UNIVERSE, candidate_pairs, _CURATED
from tools.pairs_engine import engle_granger
from tools.pairs_engine import half_life


def test_universe_entries_complete():
    # Live UNIVERSE may come from data/universe_meta.csv (extra "name" key,
    # liquidity-tiered slippage incl. 25 bps). Assert the engine-required keys.
    for tk, meta in UNIVERSE.items():
        assert {"sector", "currency", "slippage_bps"} <= set(meta), tk
        assert isinstance(meta["currency"], str) and meta["currency"]
        assert isinstance(meta["slippage_bps"], int) and meta["slippage_bps"] > 0


def test_curated_fallback_well_formed():
    for tk, meta in _CURATED.items():
        assert set(meta) == {"sector", "currency", "slippage_bps"}, tk
        assert meta["currency"] in ("USD", "EUR")
        assert meta["slippage_bps"] in (5, 10, 15)


def test_candidate_pairs_same_sector_and_currency():
    # Source-agnostic invariant: every emitted pair shares sector AND currency,
    # and none crosses either. Checked against the live UNIVERSE.
    pairs = candidate_pairs()
    for a, b in pairs:
        assert UNIVERSE[a]["sector"] == UNIVERSE[b]["sector"]
        assert UNIVERSE[a]["currency"] == UNIVERSE[b]["currency"]


def test_candidate_pairs_curated_counts(monkeypatch):
    # Pin the pairing logic against the deterministic curated set.
    monkeypatch.setattr(pu, "UNIVERSE", _CURATED)
    pairs = pu.candidate_pairs()
    assert ("BAC", "JPM") in pairs            # same sector, same currency
    assert ("NVDA", "SIE.DE") not in pairs    # cross-sector
    assert ("JPM", "DBK.DE") not in pairs     # cross-currency
    assert 30 <= len(pairs) <= 80             # ~52 with the curated universe


def test_candidate_pairs_modes(monkeypatch):
    fake = {
        "AAA.DE": dict(sector="Banks", currency="EUR", slippage_bps=5, country="Germany"),
        "BBB.DE": dict(sector="Banks", currency="EUR", slippage_bps=6, country="Germany"),
        "CCC.MI": dict(sector="Banks", currency="EUR", slippage_bps=7, country="Italy"),
        "DDD.DE": dict(sector="Autos", currency="EUR", slippage_bps=8, country="Germany"),
    }
    monkeypatch.setattr(pu, "UNIVERSE", fake)
    sect = set(pu.candidate_pairs(mode="sector"))
    cs = set(pu.candidate_pairs(mode="country_sector"))
    # cross-country same-sector pair: in sector mode, not in country_sector mode
    assert ("AAA.DE", "CCC.MI") in sect
    assert ("AAA.DE", "CCC.MI") not in cs
    # same-country same-sector pair: in both
    assert ("AAA.DE", "BBB.DE") in sect and ("AAA.DE", "BBB.DE") in cs
    # cross-sector never paired
    for a, b in sect | cs:
        assert fake[a]["sector"] == fake[b]["sector"]


def test_candidate_pairs_per_group_cap(monkeypatch):
    fake = {f"T{i}.DE": dict(sector="Banks", currency="EUR", slippage_bps=i + 1,
                             country="Germany") for i in range(10)}
    monkeypatch.setattr(pu, "UNIVERSE", fake)
    pairs = pu.candidate_pairs(mode="sector", max_per_group=4)
    assert len(pairs) == 6                    # top-4 by slippage → C(4,2)=6


def make_cointegrated(n=750, seed=42, beta=0.8, alpha=0.5, phi=0.7, noise=0.02):
    """Price pair sharing a random-walk driver:
    log x = random walk; log y = alpha + beta*log x + AR(1) noise."""
    rng = np.random.default_rng(seed)
    lx = np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = phi * eps[i - 1] + rng.normal(0, noise)
    ly = alpha + beta * lx + eps
    idx = pd.bdate_range("2022-01-03", periods=n)
    return (pd.Series(np.exp(ly), idx, name="YYY"),
            pd.Series(np.exp(lx), idx, name="XXX"))


def make_independent(n=750, seed=7):
    """Two unrelated random walks — correlated by luck at most, never cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    a = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)), idx, name="AAA")
    b = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)), idx, name="BBB")
    return a, b


def test_detects_cointegrated_pair():
    y, x = make_cointegrated()
    r = engle_granger(y, x)
    assert r["pvalue"] < 0.05
    assert r["beta"] > 0
    assert {r["y"], r["x"]} == {"YYY", "XXX"}
    assert set(r) >= {"y", "x", "alpha", "beta", "pvalue", "spread"}


def test_rejects_independent_walks():
    a, b = make_independent()
    r = engle_granger(a, b)
    assert r["pvalue"] >= 0.05


def test_spread_is_ols_residual():
    y, x = make_cointegrated()
    r = engle_granger(y, x)
    # residual of OLS with constant has (near-)zero mean by construction
    assert abs(r["spread"].mean()) < 1e-10
    assert len(r["spread"]) == len(y)


def test_half_life_known_ar1():
    # AR(1) with phi=0.9: Δs = (phi-1)·s + ε, so HL ≈ ln2/0.1 ≈ 6.9 days
    rng = np.random.default_rng(3)
    s = np.zeros(5000)
    for i in range(1, 5000):
        s[i] = 0.9 * s[i - 1] + rng.normal(0, 1)
    hl = half_life(pd.Series(s, pd.bdate_range("2010-01-04", periods=5000)))
    assert 5.5 < hl < 8.5


def test_half_life_random_walk_is_huge():
    rng = np.random.default_rng(4)
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 3000)),
                   pd.bdate_range("2012-01-02", periods=3000))
    assert half_life(rw) > 60


from tools.pairs_engine import select_pairs, walkforward_windows


def test_walkforward_windows_roll_without_overlap():
    idx = pd.bdate_range("2022-01-03", periods=400)
    wins = walkforward_windows(idx, formation_days=252, trading_days=63)
    assert len(wins) == 3
    for f, t in wins:
        assert len(f) == 252
        assert f[-1] < t[0]                     # formation strictly before trading
    # consecutive trading windows must not overlap
    for (_, t1), (_, t2) in zip(wins, wins[1:]):
        assert t1[-1] < t2[0]


def test_select_pairs_picks_cointegrated_only():
    y, x = make_cointegrated()
    a, b = make_independent()
    prices = pd.concat([y, x, a, b], axis=1)
    res = select_pairs(prices, [("YYY", "XXX"), ("AAA", "BBB")])
    assert res["n_tested"] == 2
    assert len(res["selected"]) == 1
    sel = res["selected"][0]
    assert {sel["y"], sel["x"]} == {"YYY", "XXX"}
    # frozen params present; raw window data must NOT leak out
    assert set(sel) >= {"alpha", "beta", "pvalue", "half_life", "mu", "sigma"}
    assert "spread" not in sel
    assert sel["sigma"] > 0


def test_select_pairs_respects_min_obs():
    y, x = make_cointegrated(n=50)
    prices = pd.concat([y, x], axis=1)
    res = select_pairs(prices, [("YYY", "XXX")], min_obs=100)
    assert res["n_tested"] == 0
    assert res["selected"] == []


from tools.pairs_engine import generate_signals, pair_zscore


def _z(vals):
    return pd.Series(vals, pd.bdate_range("2024-01-01", periods=len(vals)), dtype=float)


def test_pair_zscore_uses_frozen_params():
    idx = pd.bdate_range("2024-01-01", periods=3)
    py = pd.Series([np.e] * 3, idx)            # log = 1
    px = pd.Series([1.0] * 3, idx)             # log = 0
    pair = dict(alpha=0.0, beta=1.0, mu=0.0, sigma=0.5)
    z = pair_zscore(py, px, pair)
    assert np.allclose(z.values, 2.0)          # (1 - 0)/0.5


def test_signals_long_entry_and_exit():
    sig = generate_signals(_z([0, -2.5, -1, 0.1, 0]))
    assert sig.tolist() == [0, 1, 1, 0, 0]


def test_signals_short_entry_and_exit():
    sig = generate_signals(_z([0, 2.5, 1, -0.1]))
    assert sig.tolist() == [0, -1, -1, 0]


def test_signals_stop_loss_bans_reentry():
    sig = generate_signals(_z([0, 2.5, 3.6, 2.5, 2.5]))
    assert sig.tolist() == [0, -1, 0, 0, 0]    # stopped at 3.6, no re-entry at 2.5


def test_signals_no_lookahead():
    rng = np.random.default_rng(11)
    z = _z(np.cumsum(rng.normal(0, 0.6, 200)))
    full = generate_signals(z)
    for k in (10, 50, 150):
        assert generate_signals(z.iloc[:k]).tolist() == full.iloc[:k].tolist()


# ── half_life must be total: the live .F universe feeds it degenerate spreads ──

def test_half_life_constant_spread_is_non_reverting_not_a_crash():
    """Two near-perfectly collinear legs (cross-listings of the same issuer, which the
    .F universe is full of) produce a numerically constant spread. statsmodels'
    add_constant then skips its own constant column and params.iloc[1] used to raise
    IndexError — killing the whole walk-forward scan ~18 minutes in."""
    flat = pd.Series([3.0] * 300, index=pd.bdate_range("2020-01-01", periods=300))
    assert half_life(flat) == float("inf")


def test_half_life_degenerate_inputs_return_inf():
    idx = pd.bdate_range("2020-01-01", periods=300)
    assert half_life(pd.Series(dtype=float)) == float("inf")          # empty
    assert half_life(pd.Series([1.0, 2.0], index=idx[:2])) == float("inf")   # too short
    nan = pd.Series([float("nan")] * 300, index=idx)
    assert half_life(nan) == float("inf")
    inf = pd.Series([float("inf")] * 300, index=idx)
    assert half_life(inf) == float("inf")


def test_degenerate_pair_is_filtered_not_fatal():
    """select_pairs must survive a collinear pair and simply not select it — the
    hl_min<=hl<=hl_max rule rejects inf through the normal path, no special case."""
    from tools.pairs_engine import select_pairs
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    px = pd.DataFrame({"A": base, "B": base * 2.0}, index=idx)   # perfectly collinear
    out = select_pairs(px, [("A", "B")], p_max=0.05, top_n=5)
    assert out["selected"] == []
    for pr in out["selected"]:
        assert np.isfinite(pr["half_life"])


def test_half_life_still_measures_a_real_ar1():
    """The fix must not change a pair that previously fitted."""
    rng = np.random.default_rng(1)
    n, rho = 5000, -0.05
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i - 1] + rho * s[i - 1] + rng.normal(0, 0.1)
    hl = half_life(pd.Series(s, pd.bdate_range("2010-01-04", periods=n)))
    assert 5 < hl < 30 and np.isfinite(hl)
