"""Price Trade Republic's REAL tradeable universe via yfinance.

`tools.tr_tradeable --enumerate` writes `tr_universe.csv` — every stock TR offers, by ISIN
(the authoritative list: includes Milan/Tokyo names EODHD can't serve). Here we turn that
into a priced universe:

  1. resolve each ISIN -> its home yfinance ticker (yf.Search; ENI -> ENI.MI, Toyota -> 7203.T)
  2. fetch 2017→ daily history, convert to EUR (pence/agorot handled), and apply the same
     liquidity criteria as build_market (≥€100k/day median turnover, ≥€1, glitch-clean)
  3. keep the existing EODHD-priced DEAD names as the survivorship graveyard (yfinance has no
     reliable delisted history), merged in unchanged

Output replaces data/universe/universe_{prices,meta}.csv, so the strategy/momentum reports
run on exactly what you can trade. Resolution + fetch are cached/resumable.

  .venv/bin/python -m tools.build_tr_universe resolve   # ISIN -> ticker (cached)
  .venv/bin/python -m tools.build_tr_universe fetch      # ticker -> prices -> universe_*
"""
import concurrent.futures as cf
import json
import sys
import time

import pandas as pd
import yfinance as yf

from tools.build_market import apply_criteria, CC_NAME, ROOT
from tools.synthetic_proxy import to_eur
from tools.universe_reconcile import reconcile

TR_UNI = ROOT / "data" / "universe" / "tr_universe.csv"
TKMAP = ROOT / "data" / "universe" / "tr_ticker_map.json"      # {isin: yfinance ticker} cache
OUT_PRICES = ROOT / "data" / "universe" / "universe_prices.csv"
OUT_META = ROOT / "data" / "universe" / "universe_meta.csv"
OUT_TURN = ROOT / "data" / "universe" / "universe_turnover.csv"  # month-end median daily EUR turnover

# yfinance suffix -> (currency, prices_in_minor_units e.g. GBp pence / ILA agorot).
SUFFIX = {
    "MI": ("EUR", 0), "PA": ("EUR", 0), "AS": ("EUR", 0), "MC": ("EUR", 0), "BR": ("EUR", 0),
    "LS": ("EUR", 0), "VI": ("EUR", 0), "DE": ("EUR", 0), "F": ("EUR", 0), "HE": ("EUR", 0),
    "IR": ("EUR", 0), "BE": ("EUR", 0), "MU": ("EUR", 0), "SG": ("EUR", 0), "DU": ("EUR", 0),
    "L": ("GBP", 1), "SW": ("CHF", 0), "ST": ("SEK", 0), "OL": ("NOK", 0), "CO": ("DKK", 0),
    "TO": ("CAD", 0), "V": ("CAD", 0), "HK": ("HKD", 0), "AX": ("AUD", 0), "T": ("JPY", 0),
    "WA": ("PLN", 0), "TA": ("ILS", 1), "KS": ("KRW", 0), "KQ": ("KRW", 0), "TW": ("TWD", 0),
    "TWO": ("TWD", 0), "VX": ("CHF", 0), "SI": ("SGD", 0),
}


def _ccy_pence(ticker: str):
    suf = ticker.rsplit(".", 1)[1] if "." in ticker else ""
    return SUFFIX.get(suf, ("USD", 0))


def resolve():
    """ISIN -> primary yfinance ticker via Yahoo search; cached + paced + resumable."""
    cache = json.loads(TKMAP.read_text()) if TKMAP.exists() else {}
    uni = pd.read_csv(TR_UNI)
    todo = [z for z in uni["isin"].dropna().unique() if z not in cache]
    print(f"resolving {len(todo)} ISINs ({len(cache)} cached)…", flush=True)
    for i, iz in enumerate(todo, 1):
        try:
            q = yf.Search(iz, max_results=1).quotes
            cache[iz] = q[0]["symbol"] if q else ""
        except Exception:
            cache[iz] = cache.get(iz, "")           # leave blank → retried next run
        if i % 50 == 0:
            TKMAP.write_text(json.dumps(cache))
            print(f"  {i}/{len(todo)} resolved", flush=True)
        time.sleep(0.25)                            # pace Yahoo search
    TKMAP.write_text(json.dumps(cache))
    hit = sum(1 for v in cache.values() if v)
    print(f"RESOLVE DONE: {hit}/{len(cache)} ISINs → tickers -> {TKMAP}", flush=True)


def _download_chunk(tickers, tries=4):
    """Batched yfinance download for a list of tickers (ONE request per chunk — Yahoo throttles
    per-ticker loops hard). Returns {ticker: (close, volume)}; retries the chunk on failure."""
    out = {}
    for attempt in range(tries):
        try:
            raw = yf.download(tickers, start="2017-01-01", auto_adjust=True, progress=False,
                              group_by="ticker", threads=True)
            break
        except Exception:
            if attempt < tries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return out
    if raw is None or raw.empty:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            sub = raw[t] if (multi and t in raw.columns.get_level_values(0)) else (raw if not multi else None)
            if sub is None or "Close" not in sub:
                continue
            idx = sub.index.tz_localize(None) if sub.index.tz is not None else sub.index
            close = pd.Series(sub["Close"].values, index=idx).dropna()
            vol = pd.Series(sub["Volume"].values, index=idx)
            if len(close):
                out[t] = (close, vol)
        except Exception:
            continue
    return out


def _fx_series(ccy: str):
    """EUR→ccy daily series (ccy per 1 EUR). EUR → None."""
    if ccy == "EUR":
        return None
    try:
        h = yf.Ticker(f"EUR{ccy}=X").history(start="2017-01-01")
        if h.empty:
            return None
        idx = h.index.tz_localize(None) if h.index.tz is not None else h.index
        return pd.Series(h["Close"].values, index=idx).dropna()
    except Exception:
        return None


def fetch(workers=12):
    cache = json.loads(TKMAP.read_text())
    # Load the WHOLE previous universe (live + dead): reconcile() below keeps the ledger —
    # dead names pass through, and previously-live names absent from this fetch are
    # classified (delisted/removed/demoted/carried) instead of silently dropped.
    prev_meta, prev_prices = None, None
    if OUT_META.exists() and OUT_PRICES.exists():
        prev_meta = pd.read_csv(OUT_META)
        prev_prices = pd.read_csv(OUT_PRICES, index_col=0, parse_dates=True)
        print(f"ledger: reconciling against {len(prev_meta)} existing names", flush=True)

    uni = pd.read_csv(TR_UNI)
    by_tk = {}                                      # ticker -> (isin, name, country); dedup
    for r in uni.itertuples(index=False):
        tk = cache.get(r.isin, "")
        if tk and tk not in by_tk:
            by_tk[tk] = (r.isin, str(r.name), str(r.country))
    tickers = list(by_tk)
    ccys = {_ccy_pence(t)[0] for t in tickers}
    fxs = {c: _fx_series(c) for c in ccys}
    print(f"fetching {len(tickers)} TR names in chunks…", flush=True)

    rows, meta, t0, chunk = {}, [], time.time(), 75
    fetched_ok, turn_cols = set(), {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        got = _download_chunk(batch)
        fetched_ok.update(got)                      # data came back (criteria may still fail)
        for tk, (close, vol) in got.items():
            ccy, pence = _ccy_pence(tk)
            if pence:
                close = close / 100.0
            eur = close if fxs.get(ccy) is None else to_eur(close, fxs[ccy])
            # monthly median daily EUR turnover — the PIT liquidity gate's source; kept
            # even for names that fail criteria this round (history matters later).
            turn_cols[tk] = (eur * vol.reindex(eur.index)).resample("ME").median()
            crit = apply_criteria(eur, vol, active=True)
            if crit is None:
                continue
            isin, name, country = by_tk[tk]
            rows[tk] = eur.dropna()
            meta.append(dict(ticker=tk, name=name, sector="Unknown",
                             country=CC_NAME.get(country, country), currency="EUR",
                             slippage_bps=25, local_id="", isin=isin, home=tk, **crit))
        print(f"  {min(i + chunk, len(tickers))}/{len(tickers)} kept={len(rows)} "
              f"{time.time()-t0:.0f}s", flush=True)
        time.sleep(0.5)                             # gentle pacing between chunks

    new_meta = pd.DataFrame(meta)
    new_prices = pd.DataFrame(rows).sort_index()
    if prev_meta is not None:
        merged_meta, merged_prices, rep = reconcile(
            prev_meta, prev_prices, new_meta, new_prices,
            tr_isins=set(uni["isin"].dropna().astype(str)),
            fetched_ok=fetched_ok, asof=pd.Timestamp.today().normalize())
        print("ledger: " + ", ".join(f"{k}={len(v)}" for k, v in rep.items()), flush=True)
        for k in ("delisted", "removed", "demoted"):
            if rep[k]:
                print(f"  {k}: {', '.join(rep[k][:20])}", flush=True)
    else:
        merged_meta, merged_prices = new_meta, new_prices
    turn = pd.DataFrame(turn_cols).sort_index()
    if OUT_TURN.exists():                           # keep turnover history for exited names
        prev_turn = pd.read_csv(OUT_TURN, index_col=0, parse_dates=True)
        turn = pd.concat([turn, prev_turn[[c for c in prev_turn.columns
                                           if c not in turn.columns]]], axis=1).sort_index()
    turn.to_csv(OUT_TURN)
    merged_prices.to_csv(OUT_PRICES)
    merged_meta.to_csv(OUT_META, index=False)
    n_exit = int(merged_meta["delisting_date"].notna().sum())
    print(f"TR UNIVERSE BUILT: {len(merged_meta) - n_exit} live + {n_exit} exited/dead "
          f"= {len(merged_prices.columns)} names -> {OUT_META}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "resolve"
    {"resolve": resolve, "fetch": fetch}[cmd]()
