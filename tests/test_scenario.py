"""Pure checks for tools.scenario — regime-conditioned block bootstrap, no network.

The scenario layer resamples the strategy's OWN daily returns in blocks, tilting the
sampling toward risk-off (bear) or risk-on (bull) regime blocks. It is observational:
numbers in → distribution out, it never touches selection or sizing.
"""
import numpy as np
import pandas as pd

from tools.scenario import regime_scenarios


def _series(n=800, off_frac=0.30, off_mu=-0.001, on_mu=0.001, vol=0.01, seed=0):
    """Daily returns where risk-off days drift down and risk-on days drift up, plus the
    aligned risk-off mask — so bear (more risk-off blocks) must end lower than bull."""
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(seed)
    ro = pd.Series(rng.random(n) < off_frac, index=idx)
    mu = np.where(ro.values, off_mu, on_mu)
    r = pd.Series(rng.normal(mu, vol), index=idx)
    return r, ro


def test_shapes_and_percentile_ordering():
    r, ro = _series()
    out = regime_scenarios(r, ro, horizon=252, block=21, n_sims=400, seed=0)
    assert out["horizon"] == 252
    for name in ("bear", "base", "bull"):
        sc = out["scenarios"][name]
        assert len(sc["p50"]) == 253                     # horizon + 1 (starts at wealth 1.0)
        assert sc["p50"][0] == 1.0
        assert sc["term_p5"] <= sc["term_p50"] <= sc["term_p95"]   # ordered within scenario


def test_bear_below_base_below_bull():
    r, ro = _series()
    out = regime_scenarios(r, ro, horizon=252, block=21, n_sims=600, seed=1)
    bear = out["scenarios"]["bear"]["term_p50"]
    base = out["scenarios"]["base"]["term_p50"]
    bull = out["scenarios"]["bull"]["term_p50"]
    assert bear < base < bull                            # oversampling risk-off drags terminal down


def test_deterministic_by_seed():
    r, ro = _series()
    a = regime_scenarios(r, ro, horizon=126, block=21, n_sims=300, seed=7)
    b = regime_scenarios(r, ro, horizon=126, block=21, n_sims=300, seed=7)
    assert a["scenarios"]["base"]["term_p50"] == b["scenarios"]["base"]["term_p50"]
    assert np.allclose(a["scenarios"]["bear"]["p95"], b["scenarios"]["bear"]["p95"])


def test_no_riskoff_collapses_scenarios_together():
    # all risk-on → nothing to up-weight → bear == base == bull (common random numbers)
    r, ro = _series(off_frac=0.0)
    out = regime_scenarios(r, ro, horizon=126, block=21, n_sims=300, seed=0)
    bear = out["scenarios"]["bear"]["term_p50"]
    base = out["scenarios"]["base"]["term_p50"]
    bull = out["scenarios"]["bull"]["term_p50"]
    assert bear == base == bull
    assert out["frac_off"] == 0.0


def test_too_short_returns_none():
    r, ro = _series(n=20)
    assert regime_scenarios(r, ro, horizon=252, block=21) is None   # graceful, not a crash
