"""Point-in-time mega-cap screen + selection-arm inputs (pure functions, no I/O).

A market-cap size screen restricts the universe to the largest names at each
rebalance; three arms then rank inside that pool by cap, YoY revenue growth, or
12-1 momentum. All panels are strictly point-in-time — every value at date `d`
uses only information available on or before `d`.
"""
import numpy as np
import pandas as pd


def cap_panel(shares_hist: dict, prices: pd.DataFrame, dates) -> pd.DataFrame:
    """Monthly PIT market cap: last available shares (avail index <= d) times last
    available price (index <= d), per ticker per rebalance date. NaN when either is
    missing at `d`."""
    dates = list(dates)
    out = {}
    for t, sh in shares_hist.items():
        if t not in prices.columns or sh is None or len(sh) == 0:
            continue
        pser = prices[t]
        col = {}
        for d in dates:
            s = sh.loc[:d]
            p = pser.loc[:d].dropna()
            if len(s) and len(p):
                col[d] = float(s.iloc[-1]) * float(p.iloc[-1])
        if col:
            out[t] = pd.Series(col)
    return pd.DataFrame(out).reindex(dates)


def yoy_growth_panel(rev_hist: dict, dates, *, tol_days: int = 45) -> pd.DataFrame:
    """Trailing YoY quarterly revenue growth, PIT. For each date and ticker: among
    rows with `avail` <= d, take the latest period-end `p`; find the period ~365d
    before `p` (within `tol_days`); growth = rev(p)/rev(p-1yr) - 1. NaN otherwise."""
    dates = list(dates)
    out = {}
    for t, df in rev_hist.items():
        if df is None or len(df) == 0:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        col = {}
        for d in dates:
            av = df[df["avail"] <= pd.Timestamp(d)]
            if av.empty:
                continue
            p = av.index.max()
            target = p - pd.Timedelta(days=365)
            diffs = (av.index.to_series() - target).abs()
            prior = diffs[diffs <= pd.Timedelta(days=tol_days)]
            if prior.empty:
                continue
            pp = prior.idxmin()
            rev_now, rev_prev = float(av.loc[p, "revenue"]), float(av.loc[pp, "revenue"])
            if rev_prev > 0:
                col[d] = rev_now / rev_prev - 1.0
        if col:
            out[t] = pd.Series(col)
    return pd.DataFrame(out).reindex(dates)


def megacap_screen(cap: pd.DataFrame, dates, n: int) -> dict:
    """{date: set of the top-`n` tickers by PIT cap}. Names with NaN cap at a date
    are not rankable and are excluded (never crash)."""
    out = {}
    for d in dates:
        if d not in cap.index:
            out[d] = set()
            continue
        row = cap.loc[d].dropna().sort_values(ascending=False)
        out[d] = set(row.head(n).index)
    return out


def _scores_by_date(panel: pd.DataFrame, dates) -> dict:
    """{date: {'raw': Series, 'voladj': Series}} from a per-date score panel — both
    keys identical, matching the `score_by_date` contract `run_momentum` consumes."""
    out = {}
    for d in dates:
        s = panel.loc[d].dropna() if d in panel.index else pd.Series(dtype=float)
        out[d] = {"raw": s, "voladj": s}
    return out


def cap_scores_by_date(cap: pd.DataFrame, dates) -> dict:
    return _scores_by_date(cap, dates)


def growth_scores_by_date(yoy: pd.DataFrame, dates) -> dict:
    return _scores_by_date(yoy, dates)


from tools.momentum import run_momentum, rebalance_dates, precompute_scores

ARMS = ("size", "growth", "momentum")


def build_screen_and_scores(prices: pd.DataFrame, cap: pd.DataFrame,
                            yoy: pd.DataFrame, *, n: int):
    """Shared top-n cap eligibility + a `score_by_date` dict per arm. Panels must be
    indexed on `rebalance_dates(prices.index)` so the keys line up with the engine's
    internal rebalance loop."""
    dates = rebalance_dates(prices.index)
    elig = megacap_screen(cap, dates, n)
    scores = {"size": cap_scores_by_date(cap, dates),
              "growth": growth_scores_by_date(yoy, dates),
              "momentum": precompute_scores(prices, dates)}
    return elig, scores


def run_arms(prices: pd.DataFrame, slippage_bps: dict, cap: pd.DataFrame,
             yoy: pd.DataFrame, *, n: int, k: int = 10, **kw) -> dict:
    """Run size/growth/momentum on the shared top-n cap screen. Each arm injects the
    same `elig_by_date` (top-n cap) and its own `score_by_date`; the engine is not
    forked. Extra kwargs (lookback, skip, start, cost_mults, ...) pass through."""
    elig, scores = build_screen_and_scores(prices, cap, yoy, n=n)
    return {arm: run_momentum(prices, slippage_bps, k=k,
                              elig_by_date=elig, score_by_date=scores[arm], **kw)
            for arm in ARMS}
