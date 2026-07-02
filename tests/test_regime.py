"""Sanity checks for the observational regime diagnostics (tools.regime).

These power a *read-only* lens on the strategy page — an HMM regime label and the
200d-MA trend state, shown in parallel. They never feed selection or sizing, so the
key invariants are (1) the HMM is walk-forward / no-look-ahead and (2) the module is
not imported by the selection code.
"""
import inspect

import numpy as np
import pandas as pd

from tools import regime


def test_trend_state_tracks_200d_ma():
    idx = pd.bdate_range("2019-01-01", periods=300)
    up = pd.Series(np.linspace(100.0, 200.0, 300), index=idx)      # last > 200d mean
    down = pd.Series(np.linspace(200.0, 100.0, 300), index=idx)    # last < 200d mean
    assert bool(regime.trend_state(up, window=200).iloc[-1]) is True
    assert bool(regime.trend_state(down, window=200).iloc[-1]) is False


def test_hmm_regime_flags_high_vol_regime():
    # alternating calm/storm blocks so the model sees both states; once it has, it
    # flags risk-off in the high-vol blocks far more than the calm ones
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1200)
    segs, storm = [], []
    for (mu, sig), hi in zip([(0.0005, 0.004), (-0.001, 0.035)] * 2,
                             [False, True, False, True]):
        segs.append(rng.normal(mu, sig, 300))
        storm += [hi] * 300
    rets = pd.Series(np.concatenate(segs), index=idx)
    is_storm = pd.Series(storm, index=idx)
    out = regime.hmm_regime(rets, list(idx[300::150]))
    assert set(out.columns) >= {"prob_risk_off", "risk_off"}
    ro = out["risk_off"].astype(float)
    st = is_storm.reindex(ro.index).astype(bool)
    assert ro[st].mean() > ro[~st].mean() + 0.3                    # storm flagged far more


def test_hmm_regime_no_lookahead():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    rets = pd.Series(np.concatenate([rng.normal(0, 0.005, 500),
                                     rng.normal(0, 0.030, 500)]), index=idx)
    refits = list(idx[200::150])
    full = regime.hmm_regime(rets, refits)
    asof = idx[700]
    sub = [d for d in refits if d <= asof]
    trunc = regime.hmm_regime(rets.loc[:asof], sub)
    last = max(sub)                                                # strictly frozen region
    common = full.index.intersection(trunc.index)
    common = common[common <= last]
    assert len(common) > 0
    pd.testing.assert_series_equal(full.loc[common, "risk_off"],
                                   trunc.loc[common, "risk_off"])


def test_regime_not_imported_by_selection():
    # observational-only: the picks/sizing code must never depend on the diagnostics
    import tools.momentum as m
    import tools.momentum_grid as g
    for mod in (m, g):
        src = inspect.getsource(mod)
        assert "tools.regime" not in src and "import regime" not in src
