"""Tests for the adaptive/learned vol forecasters (tools.vol_ml)."""
import numpy as np
import pandas as pd
import pytest

from tools import vol_forecast as VF
from tools import vol_ml as ML


def _ewma_returns(n=3000, lam=0.90, s0=0.012, seed=5):
    """Simulate returns whose conditional vol follows an EWMA recursion with known λ."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = s0 ** 2
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = lam * s2 + (1 - lam) * r[t] ** 2
    return pd.Series(r, index=pd.bdate_range("2012-01-01", periods=n))


def _clustered(n=2500, seed=11):
    rng = np.random.default_rng(seed)
    sig = np.where((np.arange(n) // 250) % 2 == 0, 0.006, 0.025)
    return pd.Series(rng.normal(0, 1, n) * sig, index=pd.bdate_range("2013-01-01", periods=n))


# ── no look-ahead: the property every forecaster must satisfy ──

@pytest.mark.parametrize("method,kw", [
    ("adaptive_ewma", dict(min_train=500, refit_every=126)),
    ("ridge", dict(min_train=500, refit_every=63)),
    ("ensemble", {}),
])
def test_no_look_ahead(method, kw):
    r = _ewma_returns(n=1200)
    t0 = 1000
    mut = r.copy()
    mut.iloc[t0 + 1:] = 0.30 * np.sign(mut.iloc[t0 + 1:] + 1e-9)
    if method == "ensemble":                              # cheap components for speed
        comp = lambda s: dict(rolling=VF.rolling_vol(s), ewma=VF.ewma_vol(s))
        f1 = ML.ensemble_vol(r, components=comp(r))
        f2 = ML.ensemble_vol(mut, components=comp(mut))
    else:
        f1 = VF.forecast_vol(r, method=method, **kw)
        f2 = VF.forecast_vol(mut, method=method, **kw)
    pd.testing.assert_series_equal(f1.iloc[:t0], f2.iloc[:t0])


# ── adaptive λ ──

def test_lambda_recovery():
    fit = ML.fit_ewma_lambda(_ewma_returns(n=3000, lam=0.90).to_numpy())
    assert fit["converged"] and abs(fit["lam"] - 0.90) < 0.04


def test_adaptive_ewma_tracks_fixed_when_lambda_is_094():
    r = _ewma_returns(n=2200, lam=0.94, seed=8)
    ad = ML.adaptive_ewma_vol(r, min_train=756)
    fx = VF.ewma_vol(r)
    j = pd.concat([ad, fx], axis=1).dropna()
    corr = j.iloc[:, 0].corr(j.iloc[:, 1])
    assert corr > 0.98                                    # fitted λ lands near the truth


# ── learned ridge ──

def test_ridge_positive_finite_and_predictive():
    r = _clustered()
    f = ML.ridge_vol(r, min_train=756)
    tail = f.dropna()
    assert len(tail) > 500 and (tail > 0).all() and np.isfinite(tail).all()
    r2 = VF.oos_r2(f, VF.forward_realized_vol(r, 21), horizon=21)
    assert r2 > 0


def test_ridge_alpha_selection_runs_all_grid():
    # degenerate-but-valid: tiny sample under min threshold → all-NaN, no crash
    r = _clustered(n=500)
    assert ML.ridge_vol(r, min_train=756).isna().all()


# ── online ensemble ──

def test_ensemble_shifts_weight_to_the_better_model():
    n, seed = 2000, 3
    rng = np.random.default_rng(seed)
    sig = np.where((np.arange(n) // 250) % 2 == 0, 0.006, 0.025)
    r = pd.Series(rng.normal(0, 1, n) * sig, index=pd.bdate_range("2013-01-01", periods=n))
    # "good" = the true conditional vol for t+1 (oracle); "bad" = constant 500% vol
    good = pd.Series(np.append(sig[1:], sig[-1]) * np.sqrt(252), index=r.index)
    bad = pd.Series(5.0, index=r.index)
    fc, w = ML.ensemble_vol(r, components=dict(good=good, bad=bad), return_weights=True)
    wlate = w.dropna().iloc[-250:]
    assert wlate["good"].mean() > 0.95                    # trust reallocated to the winner
    j = pd.concat([fc, good], axis=1).dropna().iloc[-250:]
    assert np.allclose(j.iloc[:, 0], j.iloc[:, 1], rtol=0.15)


def test_ensemble_beats_worst_component_on_qlike():
    r = _clustered(n=2000, seed=9)
    comp = dict(rolling=VF.rolling_vol(r), ewma=VF.ewma_vol(r))
    fc = ML.ensemble_vol(r, components=comp)
    fwd = VF.forward_realized_vol(r, 1)
    ql = {m: VF.qlike(comp[m].loc[fc.dropna().index] ** 2,
                      fwd.loc[fc.dropna().index] ** 2) for m in comp}
    ql_ens = VF.qlike(fc.dropna() ** 2, fwd.loc[fc.dropna().index] ** 2)
    assert ql_ens <= max(ql.values()) + 1e-9              # never worse than the worst


def test_dispatcher_knows_ml_methods():
    r = _clustered(n=1000)
    f = VF.forecast_vol(r, method="ensemble",
                        components=dict(rolling=VF.rolling_vol(r), ewma=VF.ewma_vol(r)))
    assert isinstance(f, pd.Series)
    with pytest.raises(ValueError, match="ensemble"):
        VF.forecast_vol(r, method="lstm")                 # helpful error names the menu
