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
