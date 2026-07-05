"""Correlation-structure toolkit — RMT cleaning + statistical clustering.

Pure functions. The point of the module is honest correlation estimation on
short windows (T barely above N): Ledoit-Wolf shrinkage implemented from the
2004 closed form (no sklearn dependency — deriving the estimator is part of
the curriculum), Marchenko-Pastur eigenvalue clipping (Laloux et al. 1999 —
the one econophysics tool with a real validation record), Mantegna's
correlation distance, hierarchical clustering (average linkage default; MST
kept for visualisation only — it is the skeleton of single linkage, which
chains), and a per-rebalance PIT cluster precompute for cluster-neutral
selection (a full-sample clustering would be look-ahead).
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform

__all__ = ["lw_shrink", "mp_clip", "corr_distance", "cluster_labels",
           "mst_edges", "cluster_maps_by_date", "rand_index"]


def lw_shrink(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward the
    scaled identity μI. Returns (Σ_shrunk, δ). δ ∈ [0,1] grows as T shrinks —
    with little data the estimator trusts the structureless prior more."""
    x = returns.to_numpy(dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    mu = np.trace(s) / n
    d2 = float(np.linalg.norm(s - mu * np.eye(n), "fro") ** 2)
    b2_sum = 0.0
    for row in x:
        outer = np.outer(row, row)
        b2_sum += float(np.linalg.norm(outer - s, "fro") ** 2)
    b2 = min(b2_sum / t ** 2, d2)
    delta = (b2 / d2) if d2 > 0 else 1.0
    return delta * mu * np.eye(n) + (1.0 - delta) * s, float(delta)


def mp_clip(corr: np.ndarray, T: int) -> tuple[np.ndarray, dict]:
    """Marchenko-Pastur eigenvalue clipping of a correlation matrix: eigen-
    values below λ+ = (1+√(N/T))² are indistinguishable from noise and get
    replaced by their average (trace-preserving); the signal eigenvalues —
    market mode included — are kept. Diagonal renormalised back to 1."""
    n = corr.shape[0]
    lam_plus = (1.0 + np.sqrt(n / T)) ** 2
    ev, vec = np.linalg.eigh(corr)                      # ascending
    noise = ev < lam_plus
    ev_c = ev.copy()
    if noise.any():
        ev_c[noise] = ev[noise].mean()
    cleaned = (vec * ev_c) @ vec.T
    d = np.sqrt(np.clip(np.diag(cleaned), 1e-12, None))
    cleaned = cleaned / np.outer(d, d)
    np.fill_diagonal(cleaned, 1.0)
    info = dict(n_signal_eigs=int((~noise).sum()), lambda_plus=float(lam_plus),
                var_explained_market=float(ev[-1] / n))
    return cleaned, info


def corr_distance(corr: np.ndarray) -> np.ndarray:
    """Mantegna (1999) metric: d_ij = √(2(1−ρ_ij)) ∈ [0, 2]."""
    return np.sqrt(np.clip(2.0 * (1.0 - np.asarray(corr, dtype=float)), 0.0, None))


def cluster_labels(corr: pd.DataFrame, *, method: str = "average",
                   n_clusters: int | None = None,
                   dist_threshold: float | None = None) -> dict[str, int]:
    """Hierarchical clustering on the correlation-distance matrix. Cut either
    at a cluster count or a distance threshold. Average linkage default —
    single linkage (≡ MST) chains, ward is the tighter alternative."""
    names = list(corr.columns)
    d = corr_distance(corr.to_numpy())
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method=method)
    if n_clusters is not None:
        lab = fcluster(z, t=min(n_clusters, len(names)), criterion="maxclust")
    elif dist_threshold is not None:
        lab = fcluster(z, t=dist_threshold, criterion="distance")
    else:
        raise ValueError("pass n_clusters or dist_threshold")
    return dict(zip(names, (int(v) for v in lab)))


def mst_edges(corr: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Minimum spanning tree of the correlation-distance graph — N−1 edges.
    Visualisation only: edge-level decisions are unstable at T≈N."""
    names = list(corr.columns)
    d = corr_distance(corr.to_numpy())
    np.fill_diagonal(d, 0.0)
    # csgraph treats 0 as 'no edge'; lift weights by epsilon to keep ρ≈1 pairs
    mst = minimum_spanning_tree(csr_matrix(d + 1e-9)).tocoo()
    return [(names[i], names[j], float(max(w - 1e-9, 0.0)))
            for i, j, w in zip(mst.row, mst.col, mst.data)]


def rand_index(labels_a: dict, labels_b: dict) -> float:
    """Rand index between two labelings over their shared keys: fraction of
    pairs on which the two clusterings agree (same-cluster vs different).
    1.0 = identical partitions. 0 pairs → 0.0."""
    keys = sorted(set(labels_a) & set(labels_b))
    n = len(keys)
    if n < 2:
        return 0.0
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_a = labels_a[keys[i]] == labels_a[keys[j]]
            same_b = labels_b[keys[i]] == labels_b[keys[j]]
            agree += int(same_a == same_b)
            total += 1
    return agree / total


def cluster_maps_by_date(prices: pd.DataFrame, dates, elig_by_date: dict, *,
                         window: int = 252, method: str = "average",
                         n_clusters: int = 20, min_obs: int = 126) -> dict:
    """{rebalance_date: {ticker: cluster_id}} from TRAILING-window returns of
    the eligible names only — the PIT precompute cluster-neutral selection
    consumes (full-sample clusters would leak the future into the groups)."""
    out = {}
    for d in dates:
        cols = [t for t in prices.columns if t in elig_by_date.get(d, set())]
        w = prices.loc[:d, cols].tail(window + 1)
        r = w.pct_change()
        keep = list(r.columns[r.notna().sum() >= min_obs])
        if len(keep) < 3:
            out[d] = {}
            continue
        corr = r[keep].corr()
        out[d] = cluster_labels(corr, method=method, n_clusters=n_clusters)
    return out
