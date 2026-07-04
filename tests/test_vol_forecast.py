"""Sanity + no-look-ahead checks for tools.vol_forecast."""
import numpy as np
import pandas as pd
import pytest

from tools import quant_grade as Q
from tools import vol_forecast as VF


def _returns(n=1500, seed=0, mu=0.0003, sig=0.011):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(mu, sig, n), index=idx)


def _garch_returns(n=4000, alpha=0.08, beta=0.90, s2_unc=0.012 ** 2, seed=7):
    """Simulate a zero-mean GARCH(1,1) path with known params."""
    rng = np.random.default_rng(seed)
    omega = s2_unc * (1 - alpha - beta)
    r = np.empty(n)
    s2 = s2_unc
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return pd.Series(r, index=pd.bdate_range("2010-01-01", periods=n))


def _clustered_returns(n=3000, seed=11):
    """Vol-clustered (regime-switching sigma) series — vol is predictable here."""
    rng = np.random.default_rng(seed)
    sig = np.where((np.arange(n) // 250) % 2 == 0, 0.006, 0.025)
    return pd.Series(rng.normal(0, 1, n) * sig, index=pd.bdate_range("2012-01-01", periods=n))


# ── no look-ahead: mutating the future must not change past forecasts ──

@pytest.mark.parametrize("method,kw", [
    ("rolling", {}),
    ("ewma", {}),
    ("garch", dict(min_train=500, refit_every=63)),
    ("har", dict(min_train=500, refit_every=63)),
])
def test_no_look_ahead(method, kw):
    r = _garch_returns(n=1200)
    t0 = 1000
    r_mut = r.copy()
    r_mut.iloc[t0 + 1:] = 0.30 * np.sign(r_mut.iloc[t0 + 1:] + 1e-9)   # absurd future
    f_base = VF.forecast_vol(r, method=method, **kw)
    f_mut = VF.forecast_vol(r_mut, method=method, **kw)
    pd.testing.assert_series_equal(f_base.iloc[:t0], f_mut.iloc[:t0])


def test_ewma_matches_hand_recursion():
    r = _returns(n=70, seed=2)
    lam = 0.94
    f = VF.ewma_vol(r, lam=lam, seed_window=63)
    x = r.to_numpy()
    s2 = np.var(x[:63], ddof=1)
    for t in range(63, 66):
        s2 = lam * s2 + (1 - lam) * x[t] ** 2
    assert np.isclose(f.iloc[65], np.sqrt(252 * s2))
    assert f.iloc[:62].isna().all()                       # nothing before the seed


def test_garch_fit_recovers_params():
    r = _garch_returns(n=4000, alpha=0.08, beta=0.90)
    fit = VF.garch11_fit(r)
    assert fit["converged"]
    assert abs(fit["alpha"] - 0.08) < 0.05
    assert abs(fit["persistence"] - 0.98) < 0.03


def test_garch_constant_series_survives():
    r = pd.Series(0.0, index=pd.bdate_range("2015-01-01", periods=900))
    fit = VF.garch11_fit(r)                               # degenerate → fallback params
    assert 0 < fit["alpha"] < 1 and 0 < fit["beta"] < 1
    f = VF.garch11_vol(r, min_train=800)
    tail = f.dropna()
    assert (tail >= 0).all() and np.isfinite(tail).all()


def test_har_positive_and_predictive_on_clustered_data():
    r = _clustered_returns()
    f = VF.har_rv_vol(r, min_train=756)
    tail = f.dropna()
    assert len(tail) > 500 and (tail > 0).all() and np.isfinite(tail).all()
    r2 = VF.oos_r2(f, VF.forward_realized_vol(r, horizon=21), horizon=21)
    assert r2 > 0                                          # vol IS predictable here


def test_parkinson_scale():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2018-01-01", periods=400)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 400)), index=idx)
    high, low = close * 1.01, close * 0.99
    pk = VF.parkinson_vol(high, low, lookback=21).dropna()
    assert (pk > 0).all() and pk.std() < 0.01              # constant range → ~constant vol


def test_dispatcher_rejects_unknown():
    with pytest.raises(ValueError):
        VF.forecast_vol(_returns(), method="nope")


# ── evaluation math ──

def test_qlike_minimised_at_truth():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2019-01-01", periods=800)
    true_var = pd.Series(rng.uniform(0.5, 2.0, 800) * 1e-4, index=idx)
    realized = true_var * rng.chisquare(1, 800)            # E[realized] = true_var
    good = VF.qlike(true_var, realized)
    assert good < VF.qlike(true_var * 0.5, realized)
    assert good < VF.qlike(true_var * 2.0, realized)


def test_mincer_zarnowitz_perfect_forecast():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2019-01-01", periods=500)
    f = pd.Series(rng.uniform(0.1, 0.4, 500), index=idx)
    mz = VF.mincer_zarnowitz(f, f)
    assert abs(mz["beta"] - 1.0) < 1e-9 and abs(mz["alpha"]) < 1e-9 and mz["r2"] > 0.999


def test_mean_null_near_zero_on_iid():
    out = VF.mean_null(_returns(n=2000, seed=9))
    for v in out.values():                                 # returns are NOT predictable
        assert v == v and v < 0.02


# ── vol_managed accounting ──

def test_vol_managed_caps_and_floors():
    r = _returns(n=800, sig=0.004)                         # calm → exposure pins at cap
    f = VF.rolling_vol(r, 63)
    m = VF.vol_managed(r, f, target_vol=0.15, cost_bps=0.0)
    exp = m["exposure"]
    assert (exp <= 1.0 + 1e-12).all() and (exp >= 0).all()
    assert exp.iloc[100:].min() > 0.99                     # calm tape → fully invested
    huge = f * 0 + 10.0                                    # absurd forecast → ~0 exposure
    m2 = VF.vol_managed(r, huge, target_vol=0.15, cost_bps=0.0)
    assert m2["exposure"].max() <= 0.02


def test_vol_managed_costs_and_band():
    r = _clustered_returns(n=1500)
    f = VF.ewma_vol(r)
    free = VF.vol_managed(r, f, band=0.0, cost_bps=0.0, fee_eur=0.0)
    paid = VF.vol_managed(r, f, band=0.0, cost_bps=25.0, fee_eur=1.0)
    assert paid["equity"].iloc[-1] < free["equity"].iloc[-1]    # costs bite
    tight = VF.vol_managed(r, f, band=0.02, cost_bps=5.0)
    wide = VF.vol_managed(r, f, band=0.25, cost_bps=5.0)
    assert wide["n_trades_per_year"] < tight["n_trades_per_year"]


def test_vol_managed_parity_with_vol_target():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=900)
    eq = pd.Series(1000 * np.cumprod(1 + rng.normal(0.0006, 0.02, 900)), index=idx)
    vt = Q.vol_target(eq, target_vol=0.15, lookback=63)
    r = eq.pct_change().dropna()
    vm = VF.vol_managed(r, VF.rolling_vol(r, 63), target_vol=0.15,
                        band=0.0, cost_bps=0.0, fee_eur=0.0)
    a = vt["equity"] / vt["equity"].iloc[0]
    b = vm["equity"] / vm["equity"].iloc[0]
    j = pd.concat([a, b], axis=1).dropna()
    assert np.allclose(j.iloc[:, 0], j.iloc[:, 1], rtol=1e-10)


def test_vol_managed_gate_zeroes_exposure():
    r = _returns(n=600, sig=0.02)
    f = VF.rolling_vol(r, 63)
    gate = pd.Series(0.0, index=r.index)                   # trend filter fully off
    m = VF.vol_managed(r, f, gate=gate, cost_bps=0.0)
    assert m["exposure"].abs().max() == 0.0
    assert np.allclose(m["equity"].to_numpy(), 1.0)
