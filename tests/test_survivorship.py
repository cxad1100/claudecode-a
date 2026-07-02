"""Synthetic delisting injection (tools.survivorship) — on-population survivorship test.

The bolt-on real graveyard overlaps the live `.DE`/`.F` universe only ~2%, so it can't
honestly test the *holding* leak (does the strategy hold a name into its death?). Instead
kill LIVE names at a realistic hazard with a terminal crash, then NaN (delisted) — 100%
representative by construction. Pins the injector: terminal loss then untradeable, seeded
+ reproducible, hazard-monotone, min-life respected, zero-hazard a no-op."""
import numpy as np
import pandas as pd

from tools import survivorship as sv


def _prices(n=400, k=12, seed=0):
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))) for i in range(k)},
        index=idx)


def test_inject_applies_terminal_crash_then_delists():
    px = _prices()
    out, deaths = sv.inject_delistings(px, hazard_annual=0.5, loss_lo=0.3, loss_hi=0.9, seed=1)
    assert deaths                                                  # high hazard ⇒ some die
    for t, (d, loss) in deaths.items():
        pos = out.index.get_loc(d)
        prev = px[t].iloc[pos - 1]
        assert abs(out[t].iloc[pos] - prev * (1 - loss)) < 1e-6    # terminal crash on death day
        assert out[t].iloc[pos + 1:].isna().all()                 # untradeable after death
        assert 0.3 <= loss <= 0.9                                  # loss drawn in band


def test_inject_zero_hazard_is_noop():
    px = _prices()
    out, deaths = sv.inject_delistings(px, hazard_annual=0.0, seed=3)
    assert deaths == {}
    pd.testing.assert_frame_equal(out, px)                         # nothing touched


def test_inject_reproducible_by_seed():
    px = _prices()
    a, da = sv.inject_delistings(px, hazard_annual=0.2, seed=7)
    b, db = sv.inject_delistings(px, hazard_annual=0.2, seed=7)
    assert da == db and a.equals(b)                               # same seed ⇒ identical


def test_inject_more_hazard_more_deaths():
    px = _prices(k=40)
    _, low = sv.inject_delistings(px, hazard_annual=0.02, seed=0)
    _, high = sv.inject_delistings(px, hazard_annual=0.6, seed=0)
    assert len(high) > len(low)                                   # hazard drives death count


def test_inject_respects_min_life():
    px = _prices()
    _, deaths = sv.inject_delistings(px, hazard_annual=0.9, seed=2, min_life=200)
    for _t, (d, _loss) in deaths.items():
        assert px.index.get_loc(d) >= 200                        # no death before min_life bars


def test_summarize_bounds_drag_around_base():
    base = 1.0                                                   # +100% clean
    rets = [0.98, 1.0, 1.02, 0.99, 1.01]
    s = sv.summarize(base, rets, sim_hits=[1, 0, 2, 1, 0], sim_deaths=[10, 12, 9, 11, 8])
    assert s["sims"] == 5
    assert abs(s["mean_return"] - np.mean(rets)) < 1e-9
    assert abs(s["delta_mean"] - (np.mean(rets) - base)) < 1e-9
    assert s["delta_lo"] <= s["delta_mean"] <= s["delta_hi"]     # band brackets the mean
    assert abs(s["hits_mean"] - 0.8) < 1e-9
    assert abs(s["deaths_mean"] - 10.0) < 1e-9
    assert abs(s["avoidance_rate"] - (1 - 0.8 / 10.0)) < 1e-9    # sold 92% before death


def test_summarize_avoidance_handles_zero_deaths():
    s = sv.summarize(1.0, [1.0], sim_hits=[0], sim_deaths=[0])
    assert s["avoidance_rate"] == 1.0                            # no deaths ⇒ nothing to hold
