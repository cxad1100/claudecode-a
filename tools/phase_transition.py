"""Phase-transition indicators — breadth, susceptibility, absorption ratio.

Pure functions, ALL causal (trailing windows only) and OBSERVATIONAL: like
tools/regime.py, nothing here may feed selection or sizing in the production
strategy (pinned by test_phase_not_imported_by_selection). The physics
framing — market as spin system, breadth M(t) as magnetization, Var(M) as
susceptibility χ, eigenvalue concentration as the order parameter — maps onto
known finance quantities (breadth, correlation spikes, Kritzman's absorption
ratio). The honest read: these are COINCIDENT stress gauges, not crash
predictors; the throttle must beat plain vol targeting out of sample before
it earns promotion, and expanding trailing quantiles keep the mapping free of
the full-sample-quantile look-ahead common in the literature.
"""
import numpy as np
import pandas as pd

from tools.quant_grade import perf_metrics

__all__ = ["breadth", "susceptibility", "absorption_ratio", "ar_throttle"]


def breadth(prices: pd.DataFrame, *, min_names: int = 50) -> pd.Series:
    """M(t) = cross-sectional mean of sign(daily return) ∈ [−1, 1] — the
    market's 'magnetization'. Days with fewer than min_names prints are NaN
    (a thin cross-section makes the mean meaningless)."""
    r = prices.pct_change()
    sgn = np.sign(r)
    n = sgn.notna().sum(axis=1)
    m = sgn.mean(axis=1, skipna=True)
    m[n < min_names] = np.nan
    return m


def susceptibility(m: pd.Series, window: int = 63) -> pd.Series:
    """χ(t) = trailing variance of the magnetization. Near a synchronization
    episode (all spins aligning) M swings hard and χ spikes — the finite-size
    analogue of diverging susceptibility at a phase transition."""
    return m.rolling(window, min_periods=max(10, window // 3)).var(ddof=1)


def absorption_ratio(prices: pd.DataFrame, *, window: int = 252,
                     k_frac: float = 0.2, step: int = 21, top_n: int = 300,
                     turnover: pd.DataFrame | None = None,
                     min_obs: int = 126) -> pd.DataFrame:
    """Kritzman et al. (2011): AR = share of total variance absorbed by the
    top ⌈k_frac·N⌉ eigenvalues of the TRAILING-window correlation matrix,
    recomputed every `step` bars and forward-filled between. Columns: ar,
    avg_corr (off-diagonal mean), d_ar (15d mean vs 1y mean, in 1y sds — the
    standardized shift Kritzman trades on). Rank-1 comovement → AR ≈ 1;
    independent names → AR ≈ k_frac. When a PIT `turnover` frame is given the
    cross-section is capped at the top_n most-liquid names per step."""
    idx = prices.index
    rows = {}
    for pos in range(window, len(idx), step):
        d = idx[pos]
        w = prices.iloc[pos - window:pos + 1]
        r = w.pct_change()
        keep = list(r.columns[r.notna().sum() >= min_obs])
        if turnover is not None and len(keep) > top_n:
            med = turnover.loc[:d].tail(6).reindex(columns=keep).median()
            keep = list(med.sort_values(ascending=False).head(top_n).index)
        if len(keep) < 10:
            continue
        corr = r[keep].corr().to_numpy()
        ev = np.linalg.eigvalsh(corr)                     # ascending
        k = max(1, int(np.ceil(k_frac * len(keep))))
        ar = float(ev[-k:].sum() / np.trace(corr))
        off = corr[~np.eye(len(keep), dtype=bool)]
        rows[d] = dict(ar=ar, avg_corr=float(np.nanmean(off)))
    out = pd.DataFrame.from_dict(rows, orient="index").reindex(idx).ffill()
    if "ar" in out.columns:
        mu_1y = out["ar"].rolling(252, min_periods=63).mean()
        sd_1y = out["ar"].rolling(252, min_periods=63).std(ddof=1)
        out["d_ar"] = (out["ar"].rolling(15, min_periods=5).mean() - mu_1y) / sd_1y
    return out


def ar_throttle(equity: pd.Series, ar: pd.Series, *, lo_q: float = 0.5,
                hi_q: float = 0.9, floor: float = 0.3,
                turn_cost_bps: float = 25.0, min_hist: int = 252) -> dict:
    """Exposure dial on the absorption ratio, mirroring vol_target's contract
    exactly (shift(1) sizing, |Δexposure|·bps resize cost) so the two overlays
    compare fairly. Exposure = 1 at/below the EXPANDING-TRAILING lo_q quantile
    of AR, sliding linearly to `floor` at/above the hi_q quantile — trailing
    quantiles because full-sample quantiles are the classic look-ahead in
    absorption-ratio papers. First min_hist days run fully invested."""
    r = equity.pct_change().dropna()
    a = ar.reindex(r.index).ffill()
    qlo = a.expanding(min_periods=min_hist).quantile(lo_q)
    qhi = a.expanding(min_periods=min_hist).quantile(hi_q)
    span = (qhi - qlo).replace(0.0, np.nan)
    x = ((a - qlo) / span).clip(0.0, 1.0)
    target = (1.0 - (1.0 - floor) * x).fillna(1.0)
    w = target.shift(1).fillna(1.0)                    # yesterday's sizing
    cost = w.diff().abs().fillna(0.0) * (turn_cost_bps / 1e4)
    scaled = (r * w - cost).dropna()
    eq = (1 + scaled).cumprod()
    eq = eq / eq.iloc[0] * float(equity.dropna().iloc[0])
    out = perf_metrics(eq) if len(eq) > 63 else {}
    out.update(equity=eq, exposure=w.reindex(scaled.index),
               avg_exposure=float(w.reindex(scaled.index).mean()),
               lo_q=lo_q, hi_q=hi_q, floor=floor)
    return out
