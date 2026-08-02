"""Pairs-engine math: cointegration testing, pair selection, z-score signals.

Pure functions — no I/O, no network. The walk-forward contract:
every parameter (alpha, beta, mu, sigma) is estimated on a formation
window and FROZEN before being applied to a later trading window.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def engle_granger(a: pd.Series, b: pd.Series) -> dict:
    """Engle-Granger two-step on log prices, both orientations; keep the
    lower p-value.

    Step 1: OLS log(y) = alpha + beta*log(x); residual = spread.
    Step 2: statsmodels `coint` — ADF on the residuals with the CORRECT
    Engle-Granger critical values (plain `adfuller` on estimated residuals
    uses the wrong distribution).
    """
    best = None
    for y, x in ((a, b), (b, a)):
        ly, lx = np.log(y), np.log(x)
        ols = sm.OLS(ly, sm.add_constant(lx)).fit()
        alpha, beta = float(ols.params.iloc[0]), float(ols.params.iloc[1])
        pval = float(coint(ly, lx)[1])
        spread = ly - (alpha + beta * lx)
        if best is None or pval < best["pvalue"]:
            best = dict(y=str(y.name), x=str(x.name), alpha=alpha, beta=beta,
                        pvalue=pval, spread=spread)
    return best


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life in trading days from an AR(1) fit:
    Δs_t = c + ρ·s_{t-1} + ε  →  HL = −ln2/ρ. Non-reverting (ρ ≥ 0) → inf.

    Total by design: a degenerate spread returns inf rather than raising. Two legs that
    are near-perfectly collinear (the live .F universe is full of them — cross-listings
    of the same issuer) produce a numerically CONSTANT spread. statsmodels'
    add_constant then skips adding its own constant, the design matrix has one column,
    and reading params.iloc[1] raised IndexError — which killed the entire walk-forward
    scan partway through, ~18 minutes in.

    inf is the honest answer, not merely a convenient one: a constant spread has no mean
    reversion to measure. It also composes with the caller's existing
    `hl_min <= hl <= hl_max` filter, so such a pair is rejected by the normal rule rather
    than by a special case. Pairs that previously fitted are unaffected."""
    ds = spread.diff().dropna()
    lag = spread.shift(1).dropna()
    if len(lag) < 3 or not np.isfinite(lag.to_numpy()).all():
        return float("inf")
    if float(lag.std()) == 0.0:            # constant spread: nothing to regress on
        return float("inf")
    try:
        params = sm.OLS(ds, sm.add_constant(lag)).fit().params
    except Exception:
        return float("inf")                # singular / pathological window
    if len(params) < 2:                    # add_constant skipped a constant column
        return float("inf")
    rho = float(params.iloc[1])
    if not np.isfinite(rho):
        return float("inf")
    return float("inf") if rho >= 0 else float(-np.log(2.0) / rho)


def walkforward_windows(index: pd.DatetimeIndex, formation_days: int = 252,
                        trading_days: int = 63,
                        ) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Formation/trading window pairs, rolled forward by trading_days.
    Trading windows never overlap; a partial last window needs ≥ 5 days."""
    out, start = [], 0
    while start + formation_days + 5 <= len(index):
        f = index[start:start + formation_days]
        t = index[start + formation_days:start + formation_days + trading_days]
        out.append((f, t))
        start += trading_days
    return out


def select_pairs(prices: pd.DataFrame, candidates: list[tuple[str, str]],
                 p_max: float = 0.05, hl_min: float = 2.0, hl_max: float = 60.0,
                 top_n: int = 10, min_obs: int = 100) -> dict:
    """Test every candidate on the formation window; return the top pairs.

    Filters: cointegration p < p_max, positive hedge ratio, half-life in
    [hl_min, hl_max] trading days. Capped at top_n by p-value to bound
    multiple-testing damage; n_tested is reported so the report can show
    the multiple-comparisons caveat honestly.
    """
    found, tested = [], 0
    for a, b in candidates:
        # log-prices need strictly positive, finite values. Some .F lines carry
        # NaN gaps or stray 0/negative prints that would feed inf into the OLS.
        df = prices[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        df = df[(df > 0).all(axis=1)]
        if len(df) < min_obs:
            continue
        # A flat/illiquid .F line (never trades) has zero variance → the OLS
        # residual is constant and coint's ADF rejects it. Skip degenerate series.
        if df[a].std() == 0 or df[b].std() == 0:
            continue
        tested += 1
        try:
            r = engle_granger(df[a], df[b])
        except Exception:
            continue                # singular OLS / collinear / pathological window
        if r["pvalue"] >= p_max or r["beta"] <= 0:
            continue
        hl = half_life(r["spread"])
        if not (hl_min <= hl <= hl_max):
            continue
        r["half_life"] = hl
        r["mu"] = float(r["spread"].mean())
        r["sigma"] = float(r["spread"].std())
        del r["spread"]            # frozen params only; no window data leaks out
        found.append(r)
    found.sort(key=lambda r: r["pvalue"])
    return {"selected": found[:top_n], "n_tested": tested}


def pair_zscore(py: pd.Series, px: pd.Series, pair: dict) -> pd.Series:
    """z-score of the spread using FROZEN formation params (alpha, beta, mu, sigma)."""
    spread = np.log(py) - (pair["alpha"] + pair["beta"] * np.log(px))
    return (spread - pair["mu"]) / pair["sigma"]


def generate_signals(z: pd.Series, entry: float = 2.0, exit_band: float = 0.0,
                     stop: float = 3.5) -> pd.Series:
    """Desired spread position per close: +1 long spread, -1 short, 0 flat.

    Sequential state machine — the position at t depends only on z up to t
    (no look-ahead). Long spread = spread cheap (z <= -entry): long Y, short X.
    Exit when z reverts through the exit band; stop when |z| >= stop —
    cointegration treated as broken, NO re-entry for the rest of the window.
    """
    pos, banned, out = 0, False, []
    for zt in z.to_numpy():
        if np.isnan(zt):
            out.append(pos)
            continue
        if pos == 0 and not banned:
            if zt <= -entry:
                pos = 1
            elif zt >= entry:
                pos = -1
        elif pos == 1:
            if zt <= -stop:
                pos, banned = 0, True
            elif zt >= -exit_band:
                pos = 0
        elif pos == -1:
            if zt >= stop:
                pos, banned = 0, True
            elif zt <= exit_band:
                pos = 0
        out.append(pos)
    return pd.Series(out, index=z.index, dtype=float)
