"""Global mega-cap giants that EDGAR can't cover (non-SEC filers) → PIT-approx
market-cap inputs for the mega-cap screen, so the top-N pool isn't US-only.

Each name is pulled from a single yfinance listing where price AND share count sit
on the SAME basis (USD ADR where one exists → one FX pair), converted to EUR. Shares
come from yfinance `get_shares_full` (filing-stamped, so roughly point-in-time; not
the exact filed-date PIT that EDGAR gives, but share counts for giants barely move —
fine for a cap RANKING). Cached to data/megacap_global.json; the screen reads the
cache, never the network.

Deliberately ONLY the names EDGAR misses — ASML/SAP/Toyota/Novartis/Novo already
have exact EDGAR shares and must not be double-counted here.
"""
import json
import pathlib
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "megacap_global.json"

# display name → yfinance ticker (USD ADR ⇒ uniform USD→EUR FX). ONLY the world's
# biggest non-US names that EDGAR misses AND that aren't already in the universe under
# another listing — verified (AstraZeneca/Shell/Unilever/Total/HSBC/Toyota/SAP/ASML are
# already covered via .L/.PA/.T tickers, so they are deliberately absent here to avoid
# double-counting the same company).
# NB: Samsung is intentionally absent — its only USD ADR (SSNLF) is a stale pink-sheet
# line with garbage prices (negative/spurious); its real KRW listing (005930.KS) needs
# a KRW FX leg not wired here. Better 7 clean giants than 1 poisoned series.
GIANTS = {
    "TSMC": "TSM", "Tencent": "TCEHY", "Nestlé": "NSRGY", "LVMH": "LVMUY",
    "Roche": "RHHBY", "Alibaba": "BABA", "Sony": "SONY",
}
DEFAULT_SLIP_BPS = 15          # ADRs of the world's largest names are liquid


def _eurusd(get_hist) -> pd.Series:
    """USD per 1 EUR, daily."""
    fx = get_hist("EURUSD=X")
    return fx[~fx.index.duplicated()].sort_index()


def fetch_global(*, sleep: float = 0.3, get_hist=None, get_shares=None,
                 tickers=None) -> dict:
    """Pull EUR price + share history for each giant → {ticker: {prices, shares}}.
    Network injected for tests. Skips names with no usable data (never raises)."""
    import yfinance as yf

    def _hist(tk):
        df = yf.download(tk, start="2017-01-01", auto_adjust=True, progress=False)
        c = df["Close"]
        s = c.iloc[:, 0] if hasattr(c, "columns") else c
        return s.dropna()

    def _shares(tk):
        s = yf.Ticker(tk).get_shares_full(start="2017-01-01")
        return (s[~s.index.duplicated()].sort_index()
                if s is not None and len(s) else pd.Series(dtype=float))

    gh, gs = get_hist or _hist, get_shares or _shares
    names = tickers or GIANTS
    eurusd = _eurusd(gh)
    out = {}
    for i, tk in enumerate(names.values() if isinstance(names, dict) else names):
        try:
            px_usd = gh(tk)
            sh = gs(tk)
            if len(px_usd) < 50 or sh.empty:
                continue
            fx = eurusd.reindex(px_usd.index).ffill().bfill()
            px_eur = (px_usd / fx).dropna()                 # USD → EUR
            out[tk] = {"prices": {d.isoformat(): float(v) for d, v in px_eur.items()},
                       "shares": {d.isoformat(): float(v) for d, v in sh.items()}}
        except Exception:
            continue
        if get_hist is None:
            time.sleep(sleep)
    return out


def save_global(blob: dict, path=CACHE) -> None:
    pathlib.Path(path).write_text(json.dumps(blob))


def load_global(prices_index, path=CACHE):
    """Cache → (price_df on `prices_index`, shares_dict, slip_dict). Empty when the
    cache is absent, so the screen degrades to the EDGAR-only universe."""
    p = pathlib.Path(path)
    if not p.exists():
        return pd.DataFrame(index=prices_index), {}, {}
    blob = json.loads(p.read_text())
    price_cols, shares, slip = {}, {}, {}
    for tk, d in blob.items():
        pser = _series(d["prices"])
        price_cols[tk] = pser.reindex(prices_index).ffill()
        shares[tk] = _series(d["shares"])
        slip[tk] = DEFAULT_SLIP_BPS
    return pd.DataFrame(price_cols, index=prices_index), shares, slip


def _series(d: dict) -> pd.Series:
    """ISO-keyed dict → tz-naive sorted Series. yfinance share stamps are tz-aware
    (and mixing them into a dict makes an object Index, not a DatetimeIndex), so
    coerce everything through utc=True then strip tz — else the PIT `sh.loc[:d]`
    slice compares tz-aware to the tz-naive panel and raises."""
    if not d:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(list(d.keys()), utc=True).tz_localize(None)
    return pd.Series(list(d.values()), index=idx).sort_index()
