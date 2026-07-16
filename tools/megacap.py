"""Point-in-time mega-cap screen + selection-arm inputs (pure functions, no I/O).

A market-cap size screen restricts the universe to the largest names at each
rebalance; three arms then rank inside that pool by cap, YoY revenue growth, or
12-1 momentum. All panels are strictly point-in-time — every value at date `d`
uses only information available on or before `d`.
"""
import json
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "megacap_fundamentals.json"

# Tickers whose EDGAR share count is a known false-positive name-match → absurd cap
# (e.g. Southern Sun, a ~€1.5bn SA leisure firm, matched to a large US "Southern…"
# filer and shown at ~€1,000bn). Excluded from the cap screen. Grows as glitches surface.
CAP_GLITCH_BLOCKLIST = {"SSU.JO"}


def prune_shares(shares: dict, meta_df) -> dict:
    """Clean the shares dict feeding cap_panel: drop known glitch tickers, and collapse
    same-company duplicates (a name reachable via several listings — ASML + ASML.AS,
    ADR + local — would otherwise occupy multiple top-N slots and double its weight).
    Dedup keeps the listing with the longest share history (best coverage)."""
    from tools.edgar import norm_name
    name_of = dict(zip(meta_df["ticker"], meta_df["name"])) if meta_df is not None else {}
    kept, best_for_company = {}, {}
    # longest-history-first so the first seen per company is the one we keep
    for t in sorted(shares, key=lambda t: -len(shares[t]) if shares[t] is not None else 0):
        if t in CAP_GLITCH_BLOCKLIST:
            continue
        company = norm_name(name_of.get(t, t)) or t
        if company in best_for_company:
            continue                                # a longer-history listing already kept
        best_for_company[company] = t
        kept[t] = shares[t]
    return kept


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


def ttm_revenue_panel(rev_hist: dict, dates, *, min_quarters: int = 4) -> pd.DataFrame:
    """Trailing-12-month revenue, PIT. For each date and ticker: among rows with `avail`
    <= d, quarterly names (median period gap < 180d) sum their last four quarters (NaN
    until `min_quarters` have filed — an incomplete year is not a TTM); annual names
    (20-F filers) take their latest annual figure, which already spans a year. Native
    reporting currency per name — only ever consumed within a single-currency cohort
    (see `ps_panel`)."""
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
            gaps = av.index.to_series().diff().dt.days.dropna()
            quarterly = len(gaps) and float(gaps.median()) < 180.0
            if quarterly:
                if len(av) >= min_quarters:
                    col[d] = float(av["revenue"].iloc[-min_quarters:].sum())
            else:
                col[d] = float(av["revenue"].iloc[-1])
        if col:
            out[t] = pd.Series(col)
    return pd.DataFrame(out).reindex(dates)


def ps_panel(cap: pd.DataFrame, ttm_rev: pd.DataFrame) -> pd.DataFrame:
    """Price-to-sales = market cap / trailing-12m revenue, PIT, US filers ONLY.

    Cross-name P/S ranking is valid only when every name reports revenue in the SAME
    currency: cap is one currency for all, so a mixed USD/JPY/EUR revenue set makes a
    JPY filer look ~150x 'cheaper' purely on FX. EDGAR keeps each name's native currency
    (the cache does not even store the unit), so we restrict to US filers — the dotless
    tickers; every foreign 20-F filer carries an exchange suffix (`SAP.DE`, `7203.T`).
    NaN where revenue is missing or non-positive."""
    us = [t for t in ttm_rev.columns if "." not in t and t in cap.columns]
    out = {}
    for t in us:
        r = ttm_rev[t].where(ttm_rev[t] > 0)
        ps = (cap[t] / r).replace([np.inf, -np.inf], np.nan)
        if ps.notna().any():
            out[t] = ps
    return pd.DataFrame(out).reindex(cap.index)


def value_scores_by_date(ps: pd.DataFrame, dates) -> dict:
    """Value score = -P/S: the cheapest-on-sales name ranks highest."""
    return _scores_by_date(-ps, dates)


def garp_scores_by_date(ps: pd.DataFrame, yoy: pd.DataFrame, dates) -> dict:
    """GARP score = YoY revenue growth / P/S — 'growth at a reasonable price': high for a
    fast grower trading cheaply on sales, negative for a shrinking name (value-traps and
    de-growers sort to the bottom). Avoids the PEG divide-by-zero on flat growth. Columns
    collapse to the P/S cohort (US filers)."""
    return _scores_by_date(yoy / ps, dates)


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


def run_value_arms(prices: pd.DataFrame, slippage_bps: dict, cap: pd.DataFrame,
                   rev_hist: dict, yoy: pd.DataFrame, *, n: int, k: int = 10, **kw) -> dict:
    """Value + GARP arms on the shared top-n cap screen (US filers only — see `ps_panel`).
    Each injects the same `elig_by_date` (top-n cap) and its own P/S-based score; the
    engine is not forked. Returns {"value": run, "garp": run}."""
    dates = rebalance_dates(prices.index)
    elig = megacap_screen(cap, dates, n)
    ps = ps_panel(cap, ttm_revenue_panel(rev_hist, dates))
    scores = {"value": value_scores_by_date(ps, dates),
              "garp": garp_scores_by_date(ps, yoy, dates)}
    return {kind: run_momentum(prices, slippage_bps, k=k,
                               elig_by_date=elig, score_by_date=scores[kind], **kw)
            for kind in ("value", "garp")}


def candidate_pool(meta_df, prices_cols, *, max_names: int = 400,
                   liq_max: int = 30, turnover=None) -> list:
    """Liquid candidate names for the fundamentals fetch: present in the price
    panel, slippage <= `liq_max`, then the `max_names` largest by trailing median
    EUR turnover when a turnover panel is given — the real size proxy. The
    slippage sort is only the fallback: the TR universe meta carries ONE flat
    slippage value for every row, so spread order there is file order, and the
    'pool' it produced contained zero actual mega-caps. The obscure tail can never
    be top-N, so capping is free."""
    df = meta_df[meta_df["ticker"].isin(set(prices_cols))].copy()
    df = df[df["slippage_bps"] <= liq_max]
    if turnover is not None and len(getattr(turnover, "columns", ())):
        med = turnover.tail(126).median()
        df = df.assign(_turn=df["ticker"].map(med).fillna(0.0))
        df = df.sort_values("_turn", ascending=False)
    else:
        df = df.sort_values("slippage_bps")
    return list(df["ticker"].head(max_names))


def fetch_pool(tickers, *, key=None, get_fn=None, sleep: float = 0.3,
               lag_days: int = 75):
    """Fetch EODHD fundamentals for each ticker (native symbol), parse PIT shares +
    revenue. Returns (shares{t:Series}, rev{t:DataFrame}, cover{t:bool}). Paced by
    `sleep` between live calls; `get_fn` injected in tests skips the sleep."""
    from tools import eodhd
    key = key or eodhd.api_key()
    gf = get_fn or eodhd._http_get
    shares, rev, cover = {}, {}, {}
    for t in tickers:
        fund = eodhd.fetch_fundamentals(t, key=key, get_fn=gf)
        sh = eodhd.parse_shares_history(fund, lag_days=lag_days) if fund else pd.Series(dtype=float)
        rv = (eodhd.parse_revenue_history(fund, lag_days=lag_days)
              if fund else pd.DataFrame(columns=["revenue", "avail"]))
        cover[t] = bool(len(sh))
        if len(sh):
            shares[t] = sh
        if len(rv):
            rev[t] = rv
        if get_fn is None:
            time.sleep(sleep)
    return shares, rev, cover


def coverage_report(cover: dict) -> dict:
    n, hit = len(cover), sum(bool(v) for v in cover.values())
    return {"candidates": n, "covered": hit,
            "pct": round(100 * hit / n, 1) if n else 0.0}


def save_cache(shares: dict, rev: dict, path=CACHE) -> None:
    """Persist parsed panels so the report never re-hits the API. Series/DataFrames
    are round-tripped via ISO-dated JSON."""
    blob = {
        "shares": {t: {d.isoformat(): float(v) for d, v in s.items()}
                   for t, s in shares.items()},
        "rev": {t: {"period": [i.isoformat() for i in df.index],
                    "revenue": [float(x) for x in df["revenue"]],
                    "avail": [a.isoformat() for a in df["avail"]]}
                for t, df in rev.items()},
    }
    pathlib.Path(path).write_text(json.dumps(blob))


def load_cache(path=CACHE):
    blob = json.loads(pathlib.Path(path).read_text())
    shares = {t: pd.Series({pd.Timestamp(d): v for d, v in s.items()}).sort_index()
              for t, s in blob["shares"].items()}
    rev = {t: pd.DataFrame({"revenue": d["revenue"],
                            "avail": [pd.Timestamp(a) for a in d["avail"]]},
                           index=[pd.Timestamp(p) for p in d["period"]])
           for t, d in blob["rev"].items()}
    return shares, rev
