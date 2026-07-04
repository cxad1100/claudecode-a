"""Correlation-structure toolkit — RMT cleaning + statistical clustering.

Pins: Ledoit-Wolf shrinkage intensity behaves (∈[0,1], grows as T shrinks),
Marchenko-Pastur clipping is trace-preserving and keeps planted structure,
clustering recovers planted blocks, and the per-date cluster precompute is
truncate-future invariant (a full-sample clustering would be look-ahead).
"""
import numpy as np
import pandas as pd

from tools.corr_struct import (cluster_labels, cluster_maps_by_date,
                               corr_distance, lw_shrink, mp_clip, mst_edges)


def _two_block_panel(n_days=500, seed=0, k=8, rho=0.7):
    """Two planted blocks of k names each: strong within-block correlation."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    f1, f2 = rng.normal(0, 0.02, (2, n_days))
    cols = {}
    w = np.sqrt(rho)
    for i in range(k):
        cols[f"A{i}"] = w * f1 + np.sqrt(1 - rho) * rng.normal(0, 0.02, n_days)
        cols[f"B{i}"] = w * f2 + np.sqrt(1 - rho) * rng.normal(0, 0.02, n_days)
    rets = pd.DataFrame(cols, index=idx)
    return (1 + rets).cumprod() * 100


def test_lw_shrink_intensity_bounds_and_growth():
    prices = _two_block_panel()
    r = prices.pct_change().dropna()
    _, d_long = lw_shrink(r)
    _, d_short = lw_shrink(r.tail(40))
    assert 0.0 <= d_long <= 1.0 and 0.0 <= d_short <= 1.0
    assert d_short > d_long          # less data → shrink harder


def test_mp_clip_preserves_trace_and_planted_structure():
    prices = _two_block_panel()
    r = prices.pct_change().dropna()
    corr = r.corr().to_numpy()
    cleaned, info = mp_clip(corr, T=len(r))
    assert np.isclose(np.trace(cleaned), np.trace(corr))
    assert np.allclose(np.diag(cleaned), 1.0, atol=1e-8)
    assert info["n_signal_eigs"] >= 2          # two planted factors survive
    ev_raw = np.sort(np.linalg.eigvalsh(corr))
    ev_cln = np.sort(np.linalg.eigvalsh(cleaned))
    assert np.isclose(ev_cln[-1], ev_raw[-1], rtol=0.05)   # market eig kept


def test_corr_distance_metric_range():
    c = np.array([[1.0, 0.5], [0.5, 1.0]])
    d = corr_distance(c)
    assert d[0, 0] == 0.0
    assert np.isclose(d[0, 1], np.sqrt(2 * 0.5))
    assert corr_distance(np.array([[1.0, -1.0], [-1.0, 1.0]]))[0, 1] == 2.0


def test_cluster_labels_recover_planted_blocks():
    prices = _two_block_panel()
    corr = prices.pct_change().dropna().corr()
    labels = cluster_labels(corr, n_clusters=2)
    a = {labels[f"A{i}"] for i in range(8)}
    b = {labels[f"B{i}"] for i in range(8)}
    assert len(a) == 1 and len(b) == 1 and a != b


def test_mst_edges_span_all_names():
    prices = _two_block_panel()
    corr = prices.pct_change().dropna().corr()
    edges = mst_edges(corr)
    assert len(edges) == len(corr) - 1          # spanning tree
    names = set()
    for u, v, w in edges:
        names |= {u, v}
        assert w >= 0
    assert names == set(corr.columns)


def test_cluster_maps_truncate_future_invariance():
    prices = _two_block_panel()
    dates = list(prices.index[[300, 400]])
    elig = {d: set(prices.columns) for d in dates}
    full = cluster_maps_by_date(prices, dates, elig, window=252, n_clusters=2)
    d = dates[0]
    trunc = cluster_maps_by_date(prices.loc[:d], [d], elig, window=252,
                                 n_clusters=2)
    assert full[d] == trunc[d]
    assert set(full.keys()) == set(dates)
