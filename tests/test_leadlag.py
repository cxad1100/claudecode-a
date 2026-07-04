"""Lead-lag network signal — the invariants that keep it honest.

The known failure mode of daily lead-lag on a Frankfurt cross-listing feed is
nonsynchronous trading (Lo & MacKinlay 1990): a stale line echoes yesterday's
move of its liquid home listing and looks 'predicted'. These tests pin the
defenses (activity gate, cross-sectional demeaning, shuffle-null threshold)
and the no-lookahead / determinism contracts every score precompute must obey.
"""
import numpy as np
import pandas as pd
import pytest

from tools.leadlag import (lagged_corr, shuffle_threshold, leadlag_edges,
                           network_universe, leadlag_scores)


def _panel(n_days=400, seed=0, n_noise=18):
    """Synthetic panel: B follows A with lag 1 (planted lead), rest noise.

    n_noise keeps N large enough that cross-sectional demeaning's mechanical
    −1/(N−1) correlation floor sits below the shuffle threshold — at tiny N
    every pair inherits a spurious link from the demeaning itself."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    a = rng.normal(0, 0.02, n_days)
    b = 0.8 * np.roll(a, 1) + rng.normal(0, 0.005, n_days)
    b[0] = 0.0
    cols = {"A": a, "B": b}
    for k in range(n_noise):
        cols[f"N{k:02d}"] = rng.normal(0, 0.02, n_days)
    rets = pd.DataFrame(cols, index=idx)
    return (1 + rets).cumprod() * 100.0


def _returns(prices):
    return np.log(prices).diff()


def test_lagged_corr_detects_planted_lead():
    r = _returns(_panel())
    c = lagged_corr(r, lag=1)
    assert c.loc["A", "B"] > 0.5          # A(t-1) → B(t): strong
    assert abs(c.loc["B", "A"]) < 0.2     # reverse direction: nothing


def test_lagged_corr_orientation_is_leader_row_follower_col():
    r = _returns(_panel())
    c = lagged_corr(r, lag=1)
    assert c.loc["A", "B"] > c.loc["B", "A"]


def test_shuffle_threshold_deterministic_seeded():
    r = _returns(_panel())
    t1 = shuffle_threshold(r, lag=1, seed=7)
    t2 = shuffle_threshold(r, lag=1, seed=7)
    t3 = shuffle_threshold(r, lag=1, seed=8)
    assert t1 == t2
    assert t1 != t3
    assert 0.0 < t1 < 1.0


def test_leadlag_edges_keeps_planted_drops_noise():
    r = _returns(_panel())
    e = leadlag_edges(r, seed=0)
    kept = set(zip(e["leader"], e["follower"]))
    assert ("A", "B") in kept
    # noise pairs shouldn't flood in: planted edge + at most a few false ones
    # (expected false ≈ (1−edge_q)·N·(N−1) ≈ 0.4 here)
    assert len(e) <= 4
    assert e.attrs["rho_star"] > 0.0
    assert e.attrs["n_kept"] == len(e)


def test_stale_series_excluded_from_network():
    prices = _panel()
    # G repeats each price 5 bars (stale ffill) — classic dead .F line
    g = prices["A"].iloc[::5].reindex(prices.index).ffill()
    prices = prices.assign(G=g)
    elig = set(prices.columns)
    uni = network_universe(prices, None, prices.index[-1], elig,
                           min_active=0.6, window=252)
    assert "G" not in uni
    assert "A" in uni


def test_demeaning_kills_common_factor_lag():
    # every name = same autocorrelated factor + tiny noise → after
    # cross-sectional demeaning no pair should clear the shuffle threshold
    rng = np.random.default_rng(3)
    n = 400
    f = np.zeros(n)
    for t in range(1, n):                          # AR(1) common factor
        f[t] = 0.4 * f[t - 1] + rng.normal(0, 0.02)
    idx = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({f"N{k:02d}": (1 + pd.Series(f + rng.normal(0, 1e-4, n),
                                                       index=idx)).cumprod() * 100
                           for k in range(20)})
    e = leadlag_edges(_returns(prices), seed=0)
    assert len(e) == 0


def test_scores_cover_all_dates_with_both_variants():
    prices = _panel()
    dates = list(prices.index[[260, 300, 340, 380]])
    elig = {d: set(prices.columns) for d in dates}
    out = leadlag_scores(prices, dates, elig, None, seed=0)
    assert set(out.keys()) == set(dates)
    for d in dates:
        assert set(out[d].keys()) == {"raw", "voladj"}
        assert isinstance(out[d]["raw"], pd.Series)


def test_scores_truncate_future_invariance():
    prices = _panel()
    d = prices.index[300]
    elig = {d: set(prices.columns)}
    full = leadlag_scores(prices, [d], elig, None, seed=0)[d]["raw"]
    trunc = leadlag_scores(prices.loc[:d], [d], elig, None, seed=0)[d]["raw"]
    pd.testing.assert_series_equal(full.sort_index(), trunc.sort_index())


def test_follower_scored_by_leader_recent_return():
    prices = _panel()
    d = prices.index[390]
    elig = {d: set(prices.columns)}
    out = leadlag_scores(prices, [d], elig, None, recent=21, seed=0)[d]["raw"]
    # B has in-edge from A → score defined and signed like A's recent return
    a_recent = prices["A"].iloc[-21:].iloc[-1] / prices["A"].iloc[-22] - 1
    assert not np.isnan(out["B"])
    assert np.sign(out["B"]) == np.sign(a_recent)
    # most noise names have no significant in-edges → absent from the Series
    # (no momentum fallback); a couple of false edges at edge_q=0.999 are fine
    n_scored = sum(f"N{k:02d}" in out.index for k in range(18))
    assert n_scored <= 2
