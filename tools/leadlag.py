"""Lead-lag network signal — lagged cross-correlation graph → neighbor momentum.

Pure functions (no I/O). The estimator is deliberately defensive because the
input is a Frankfurt cross-listing feed where nonsynchronous trading (Lo &
MacKinlay 1990) manufactures fake lead-lag out of stale prices:

- `network_universe` drops lines whose returns are mostly zeros (stale/ffilled)
  and caps the graph at the most-liquid names by PIT monthly turnover;
- callers demean returns cross-sectionally before correlating (an autocorrelated
  common factor otherwise makes everything appear to lead everything);
- `leadlag_edges` keeps only entries beyond a circular-shift shuffle null
  (per-column shifts preserve each series' own autocorrelation, destroy
  cross-links — Curme et al. 2015), soft-thresholded so weights are continuous.

The z-scored matrix-product correlation is exact on complete panels and a
pairwise-overlap approximation on gappy ones; the shuffle null runs through the
same estimator, so the threshold is calibrated to the estimator's own noise.
"""
import numpy as np
import pandas as pd

__all__ = ["lagged_corr", "shuffle_threshold", "leadlag_edges",
           "network_universe", "leadlag_scores",
           "size_leadlag_baseline_scores", "rank_ic"]


def _zscore(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Column z-scores with NaN→0 and the not-NaN mask (for overlap counts).
    Zero-variance columns z to 0 — a flat series can neither lead nor follow."""
    x = returns.to_numpy(dtype=float)
    mask = np.isfinite(x)
    x = np.where(mask, x, np.nan)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0.0)] = np.inf
    z = (x - mu) / sd
    return np.where(mask, z, 0.0), mask


def _corr_product(z_lead: np.ndarray, m_lead: np.ndarray, z_foll: np.ndarray,
                  m_foll: np.ndarray, lag: int, min_overlap: int) -> np.ndarray:
    """corr(lead(t−lag), foll(t)) for all pairs in one matrix product; entries
    with pairwise overlap < min_overlap are NaN."""
    zx, zy = z_lead[:-lag], z_foll[lag:]
    n_pair = (m_lead[:-lag].astype(float).T @ m_foll[lag:].astype(float))
    denom = np.maximum(n_pair - 1.0, 1.0)
    c = (zx.T @ zy) / denom
    c = np.clip(c, -1.0, 1.0)
    c[n_pair < min_overlap] = np.nan
    return c


def lagged_corr(returns: pd.DataFrame, lag: int = 1,
                min_overlap: int = 126) -> pd.DataFrame:
    """Lagged correlation matrix: entry [j, i] = corr(r_j(t−lag), r_i(t)) —
    row = leader, column = follower."""
    z, m = _zscore(returns)
    c = _corr_product(z, m, z, m, lag, min_overlap)
    return pd.DataFrame(c, index=returns.columns, columns=returns.columns)


def shuffle_threshold(returns: pd.DataFrame, lag: int, *, n_shuffle: int = 3,
                      edge_q: float = 0.999, min_shift: int = 21,
                      seed: int = 0) -> float:
    """|ρ| threshold from a circular-shift null: shift each LEADER column by an
    independent random offset ≥ min_shift (own autocorrelation preserved,
    cross-links destroyed), pool all off-diagonal |entries| across shuffles,
    take the edge_q quantile. Seeded → deterministic."""
    rng = np.random.default_rng(seed)
    z, m = _zscore(returns)
    t, n = z.shape
    hi = max(t - min_shift, min_shift + 1)
    pool = []
    for _ in range(n_shuffle):
        shifts = rng.integers(min_shift, hi, size=n)
        zs = np.empty_like(z)
        ms = np.empty_like(m)
        for j in range(n):
            zs[:, j] = np.roll(z[:, j], shifts[j])
            ms[:, j] = np.roll(m[:, j], shifts[j])
        c = _corr_product(zs, ms, z, m, lag, min_overlap=1)
        off = c[~np.eye(n, dtype=bool)]
        pool.append(np.abs(off[np.isfinite(off)]))
    pool = np.concatenate(pool) if pool else np.array([1.0])
    return float(np.quantile(pool, edge_q)) if len(pool) else 1.0


def leadlag_edges(returns: pd.DataFrame, *, lags: tuple = (1,),
                  min_overlap: int = 126, n_shuffle: int = 3,
                  edge_q: float = 0.999, seed: int = 0) -> pd.DataFrame:
    """Significant directed edges after cross-sectional demeaning.

    Long frame (leader, follower, lag, rho, weight) with soft-threshold weights
    w = sign(ρ)·max(|ρ|−ρ*, 0)·(1/lag); self-pairs excluded. attrs carry the
    diagnostics an honest page must print: rho_star per lag, kept-edge count,
    the count expected by chance, and mean |self-lag| (residual staleness)."""
    r = returns.sub(returns.mean(axis=1), axis=0)          # kill common factor
    names = list(r.columns)
    n = len(names)
    rows, rho_stars, self_lags = [], {}, []
    for lag in lags:
        c = lagged_corr(r, lag, min_overlap).to_numpy()
        rho = shuffle_threshold(r, lag, n_shuffle=n_shuffle, edge_q=edge_q,
                                seed=seed + lag)
        rho_stars[lag] = rho
        diag = np.abs(np.diag(c))
        self_lags.append(np.nanmean(diag) if np.isfinite(diag).any() else np.nan)
        w = np.sign(c) * np.maximum(np.abs(c) - rho, 0.0) / lag
        w[~np.isfinite(w)] = 0.0
        np.fill_diagonal(w, 0.0)
        lead_ix, foll_ix = np.nonzero(w)
        for a, b in zip(lead_ix, foll_ix):
            rows.append(dict(leader=names[a], follower=names[b], lag=lag,
                             rho=float(c[a, b]), weight=float(w[a, b])))
    out = pd.DataFrame(rows, columns=["leader", "follower", "lag", "rho", "weight"])
    out.attrs["rho_star"] = rho_stars[lags[0]] if len(lags) == 1 else rho_stars
    out.attrs["n_kept"] = len(out)
    out.attrs["n_expected_false"] = (1.0 - edge_q) * n * (n - 1) * len(lags)
    out.attrs["mean_self_lag"] = float(np.nanmean(self_lags)) if self_lags else float("nan")
    return out


def network_universe(prices: pd.DataFrame, turnover: pd.DataFrame | None,
                     asof, elig: set, *, top_n: int = 300,
                     min_active: float = 0.6, window: int = 252,
                     min_obs: int = 126) -> list[str]:
    """Names allowed into the graph at `asof`: eligible ∩ actively-printing
    (≥ min_obs returns in the window AND ≥ min_active of them nonzero — a
    stale/ffilled line is mostly zero returns), capped at the top_n by trailing
    6-month median EUR turnover when the PIT turnover frame is available."""
    cols = [t for t in prices.columns if t in elig]
    w = prices.loc[:asof, cols].tail(window + 1)
    r = w.pct_change()
    obs = r.notna().sum()
    nonzero = (r.ne(0) & r.notna()).sum()
    frac = nonzero / obs.where(obs > 0)
    keep = list(obs.index[(obs >= min_obs) & (frac >= min_active)])
    if turnover is not None and len(keep) > top_n:
        tw = turnover.loc[:asof].tail(6)
        med = tw.reindex(columns=keep).median()
        keep = list(med.sort_values(ascending=False).head(top_n).index)
    return sorted(keep)[:top_n] if turnover is None else sorted(keep)


def _date_rng_seed(seed: int, d: pd.Timestamp) -> int:
    """Per-date deterministic seed — identical whether the caller passed the
    full frame or one truncated at d (truncate-future invariance)."""
    return int(seed) * 100_000_000 + int(pd.Timestamp(d).strftime("%Y%m%d"))


def leadlag_scores(prices: pd.DataFrame, dates, elig_by_date: dict,
                   turnover: pd.DataFrame | None, *, lookback: int = 252,
                   lags: tuple = (1,), recent: int = 21, top_n: int = 300,
                   min_overlap: int = 126, min_active: float = 0.6,
                   n_shuffle: int = 3, edge_q: float = 0.999,
                   seed: int = 0, placebo_seed: int | None = None,
                   diag: dict | None = None) -> dict:
    """run_momentum-ready score precompute: {date: {"raw": Series, "voladj":
    Series}} covering EVERY date in `dates` (a missing date would silently fall
    back to price momentum inside run_momentum — never allowed).

    s_i(d) = Σ_j w_{j→i}·R_j^recent / Σ_j |w_{j→i}| over significant in-edges;
    names with no in-edges are absent from the Series (no momentum fallback).

    `placebo_seed` permutes leader identities per date AFTER the graph is
    built (same edges, same degree distribution, scrambled attribution): a
    pipeline that 'works' under the placebo is leaking, not predicting.
    A caller-supplied `diag` dict is filled per date with the honesty
    numbers the page must print (edges kept vs expected-by-chance,
    threshold, residual self-lag staleness).
    """
    out = {}
    for d in dates:
        hist = prices.loc[:d]
        uni = network_universe(hist, turnover, d, elig_by_date.get(d, set()),
                               top_n=top_n, min_active=min_active,
                               window=lookback)
        raw = pd.Series(dtype=float)
        voladj = pd.Series(dtype=float)
        if len(uni) >= 3:
            w = hist[uni].tail(lookback + 1)
            rets = np.log(w).diff()
            edges = leadlag_edges(rets, lags=lags, min_overlap=min_overlap,
                                  n_shuffle=n_shuffle, edge_q=edge_q,
                                  seed=_date_rng_seed(seed, d))
            if diag is not None:
                diag[d] = dict(n_uni=len(uni), n_kept=edges.attrs["n_kept"],
                               n_expected_false=edges.attrs["n_expected_false"],
                               rho_star=edges.attrs["rho_star"],
                               mean_self_lag=edges.attrs["mean_self_lag"])
            if placebo_seed is not None and len(edges):
                prng = np.random.default_rng([placebo_seed,
                                              _date_rng_seed(seed, d)])
                perm = dict(zip(uni, prng.permutation(uni)))
                edges = edges.assign(leader=edges["leader"].map(perm))
            if len(edges):
                recent_ret = w.iloc[-1] / w.iloc[-(recent + 1)] - 1.0
                num = edges.assign(x=lambda e: e["weight"]
                                   * e["leader"].map(recent_ret).values)
                s = (num.groupby("follower")["x"].sum()
                     / edges.assign(a=lambda e: e["weight"].abs())
                       .groupby("follower")["a"].sum())
                raw = s.dropna()
                vol = rets[raw.index].std() * np.sqrt(252.0)
                voladj = (raw / vol).replace([np.inf, -np.inf], np.nan).dropna()
        out[d] = {"raw": raw, "voladj": voladj}
    return out


def size_leadlag_baseline_scores(prices: pd.DataFrame, dates,
                                 elig_by_date: dict,
                                 turnover: pd.DataFrame | None, *,
                                 leader_frac: float = 0.2, recent: int = 21,
                                 lookback: int = 252,
                                 min_overlap: int = 60) -> dict:
    """The KNOWN lead-lag channel as a baseline: big (top-turnover) names lead
    small ones (Lo & MacKinlay). Each name is scored by the turnover-weighted
    leader basket's recent return × its own trailing lagged correlation to the
    basket. If the network signal can't beat this, it found nothing new.

    Requires the PIT turnover frame (that's what defines 'big'); without it
    every date returns empty Series."""
    out = {}
    for d in dates:
        raw = pd.Series(dtype=float)
        voladj = pd.Series(dtype=float)
        elig = sorted(elig_by_date.get(d, set()))
        if turnover is not None and len(elig) >= 3:
            hist = prices.loc[:d]
            w = hist.reindex(columns=elig).tail(lookback + 1)
            med = turnover.loc[:d].tail(6).reindex(columns=elig).median()
            med = med.dropna()
            n_lead = max(1, int(np.ceil(leader_frac * len(elig))))
            leaders = med.sort_values(ascending=False).head(n_lead)
            if len(leaders) and leaders.sum() > 0:
                rets = np.log(w).diff()
                bw = leaders / leaders.sum()
                basket = (rets[leaders.index] * bw).sum(axis=1)
                # corr(basket(t−1), r_i(t)) — the basket leads
                zb = (basket - basket.mean()) / (basket.std(ddof=1) or np.inf)
                c = {}
                for t in elig:
                    x, y = zb.iloc[:-1].to_numpy(), rets[t].iloc[1:].to_numpy()
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() < min_overlap or np.nanstd(y[ok]) == 0:
                        continue
                    yy = (y[ok] - y[ok].mean()) / y[ok].std(ddof=1)
                    c[t] = float(np.dot(x[ok], yy) / max(ok.sum() - 1, 1))
                bl = w[leaders.index].iloc[-1] / w[leaders.index].iloc[-(recent + 1)] - 1.0
                basket_recent = float((bl * bw).sum())
                raw = pd.Series({t: v * basket_recent for t, v in c.items()}).dropna()
                vol = rets[raw.index].std() * np.sqrt(252.0)
                voladj = (raw / vol).replace([np.inf, -np.inf], np.nan).dropna()
        out[d] = {"raw": raw, "voladj": voladj}
    return out


def rank_ic(score_by_date: dict, prices: pd.DataFrame, dates,
            elig_by_date: dict, execute_lag: int = 1) -> pd.DataFrame:
    """Per-rebalance Spearman rank IC of the 'raw' scores against realized
    next-period returns, executed with the same t+`execute_lag` convention as
    the engine. Rows: date → (ic, n). Interpret with a Newey–West t on the
    mean; the decay across holding horizons is the honest way to read a
    daily-horizon signal forced through a monthly/quarterly harness."""
    from scipy.stats import spearmanr

    from tools.momentum import _exec_date

    rows = []
    for i in range(len(dates) - 1):
        d, nxt = dates[i], dates[i + 1]
        s = score_by_date.get(d, {}).get("raw", pd.Series(dtype=float)).dropna()
        names = [t for t in s.index
                 if t in elig_by_date.get(d, set()) and t in prices.columns]
        if len(names) < 3:
            continue
        e0 = _exec_date(prices.index, d, execute_lag)
        e1 = _exec_date(prices.index, nxt, execute_lag)
        realized = prices.loc[e1, names] / prices.loc[e0, names] - 1.0
        ok = realized.notna()
        if ok.sum() < 3:
            continue
        ic = spearmanr(s[names][ok.values], realized[ok])[0]
        rows.append(dict(date=d, ic=float(ic), n=int(ok.sum())))
    return pd.DataFrame(rows).set_index("date") if rows else \
        pd.DataFrame(columns=["ic", "n"])
