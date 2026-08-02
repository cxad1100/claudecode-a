"""Phase-transition indicators — breadth, susceptibility, absorption ratio.

All causal (trailing windows only) and observational: the culture pin at the
bottom asserts the selection engine never imports this module. The throttle
mirrors quant_grade.vol_target's contract exactly (shift(1) sizing,
|Δexposure|·bps cost) so the two overlays compare fairly.
"""
import inspect

import numpy as np
import pandas as pd

from tools.phase_transition import (absorption_ratio, ar_throttle, breadth,
                                    susceptibility)


def _iid_panel(n_days=300, n=20, seed=0, scale=0.02):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    return pd.DataFrame((1 + rng.normal(0, scale, (n_days, n))).cumprod(axis=0) * 100,
                        index=idx, columns=[f"S{k}" for k in range(n)])


def test_breadth_bounded_and_min_names():
    p = _iid_panel()
    m = breadth(p, min_names=5)
    assert m.dropna().between(-1, 1).all()
    thin = breadth(p.iloc[:, :3], min_names=5)      # too few names → NaN
    assert thin.isna().all()


def test_susceptibility_spikes_on_synchronization():
    rng = np.random.default_rng(1)
    n_days, n = 300, 20
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    r = rng.normal(0, 0.01, (n_days, n))
    common = rng.normal(0, 0.03, 60)
    r[-60:] = common[:, None] + rng.normal(0, 0.001, (60, n))   # sync episode
    p = pd.DataFrame((1 + r).cumprod(axis=0) * 100, index=idx)
    chi = susceptibility(breadth(p, min_names=5), window=30)
    assert chi.iloc[-1] > 4 * chi.iloc[180]


def test_absorption_rank1_vs_iid():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=300)
    f = rng.normal(0, 0.02, 300)
    rank1 = pd.DataFrame({f"S{k}": (1 + pd.Series(f + rng.normal(0, 1e-4, 300),
                                                  index=idx)).cumprod() * 100
                          for k in range(20)})
    ar1 = absorption_ratio(rank1, window=200, k_frac=0.2, step=50)
    assert ar1["ar"].dropna().iloc[-1] > 0.95
    ar0 = absorption_ratio(_iid_panel(), window=200, k_frac=0.2, step=50)
    v = ar0["ar"].dropna().iloc[-1]
    assert 0.15 < v < 0.5
    assert "avg_corr" in ar0.columns


def test_absorption_truncate_future_invariance():
    p = _iid_panel()
    full = absorption_ratio(p, window=200, step=50)
    d = full["ar"].dropna().index[1]
    trunc = absorption_ratio(p.loc[:d], window=200, step=50)
    assert np.isclose(full.loc[d, "ar"], trunc.loc[d, "ar"], equal_nan=True)


def test_ar_throttle_uses_only_past_ar():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=600)
    eq = pd.Series((1 + rng.normal(0.0005, 0.01, 600)).cumprod() * 10_000,
                   index=idx)
    ar = pd.Series(0.2 + rng.normal(0, 0.02, 600), index=idx)
    jump = idx[400]
    ar.loc[jump:] = 0.9 + rng.normal(0, 0.02, 200)   # regime jumps AT `jump`
    out = ar_throttle(eq, ar, min_hist=100, floor=0.3)
    expo = out["exposure"]
    # sizing reacts one bar AFTER the jump, never on the jump bar itself:
    # the jump-day exposure was set from yesterday's (calm) AR
    assert expo.loc[jump] > 0.8
    assert expo.loc[idx[401]] < 0.5
    assert 0.3 <= expo.dropna().min() <= 1.0
    assert "equity" in out and out["equity"].iloc[0] > 0


def test_phase_not_imported_by_selection():
    import tools.momentum as momentum
    import tools.momentum_grid as grid
    for mod in (momentum, grid):
        src = inspect.getsource(mod)
        assert "phase_transition" not in src
        assert "leadlag" not in src
