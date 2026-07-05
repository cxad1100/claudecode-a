"""On-disk data buffer for the live dashboard.

Caches the expensive yfinance inputs (5y price history, market caps) with a TTL
and keeps a last-good copy of live prices, so a page refresh is cheap and a
failed fetch degrades to the last-good value (with a staleness flag) instead of
silently substituting cost basis.

Buffer lives under local/buffer/ (gitignored). Pure helpers: a fetch function is
injectable so the cache logic is testable without the network.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from tools import optimizer as OPT
from tools import portfolio_tools as PT

BUFFER_DIR = Path(__file__).resolve().parent.parent / "local" / "buffer"


def _fresh(path: Path, ttl_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600


def _dir(buffer_dir: Path | None) -> Path:
    d = buffer_dir or BUFFER_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_price_history(tickers, period="5y", ttl_hours=12, force=False,
                         buffer_dir: Path | None = None, _fetch=None) -> pd.DataFrame:
    """5y price history with a TTL pickle cache. Reuse within TTL unless `force`."""
    _fetch = _fetch or OPT.fetch_price_history
    d = _dir(buffer_dir)
    h = hashlib.md5(f"{'_'.join(sorted(tickers))}|{period}".encode()).hexdigest()[:12]
    path = d / f"hist_{h}.pkl"
    if not force and _fresh(path, ttl_hours):
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
    df = _fetch(tickers, period=period)
    try:
        df.to_pickle(path)
    except Exception:
        pass
    return df


def _fetch_ohlc_yf(ticker: str, period: str = "max") -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    return raw[["Close", "High", "Low"]].dropna(how="all")


def cached_ohlc(ticker: str, period: str = "max", ttl_hours: float = 12, force=False,
                buffer_dir: Path | None = None, _fetch=None) -> pd.DataFrame:
    """OHLC (Close/High/Low) with a TTL pickle cache — cached_price_history stores Close
    only. A failed fetch degrades to the last-good pickle (stale beats dead) and only
    raises when nothing was ever cached."""
    _fetch = _fetch or _fetch_ohlc_yf
    d = _dir(buffer_dir)
    path = d / f"ohlc_{ticker.replace('^', 'i').replace('.', '_')}.pkl"
    if not force and _fresh(path, ttl_hours):
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
    try:
        df = _fetch(ticker, period)
    except Exception:
        if path.exists():
            return pd.read_pickle(path)                   # last-good fallback
        raise
    try:
        df.to_pickle(path)
    except Exception:
        pass
    return df


def cached_market_caps(tickers, ttl_hours=24, force=False,
                       buffer_dir: Path | None = None, _fetch=None) -> dict[str, float]:
    """Market caps with a long TTL JSON cache. A failed/partial fetch keeps the
    last-good value per ticker (new values win on merge)."""
    _fetch = _fetch or OPT.fetch_market_caps
    d = _dir(buffer_dir)
    path = d / "market_caps.json"
    cached: dict[str, float] = {}
    if path.exists():
        try:
            cached = json.loads(path.read_text())
        except Exception:
            cached = {}
    if not force and _fresh(path, ttl_hours) and set(tickers) <= set(cached):
        return {t: cached[t] for t in tickers if t in cached}
    fresh = _fetch(tickers)                       # dict, possibly partial/empty
    merged = {**cached, **fresh}                  # new wins, keep last-good for missing
    path.write_text(json.dumps(merged))
    return {t: merged[t] for t in tickers if t in merged}


def cached_current_prices(holdings, force=False,
                          buffer_dir: Path | None = None, _fetch=None):
    """Live EUR prices, buffered.

    Non-force: serve last-good from the buffer (no network) when it covers every
    holding; otherwise (cold/missing) fetch anyway. Force: fetch live and, for any
    ticker whose fetch returns None, fall back to the last-good buffered price
    (never average cost).

    Returns (prices, stale, as_of):
      prices  {ticker: eur_price | None}
      stale   {ticker: iso_ts} for tickers served from an older buffer after a
              failed live fetch (only populated on force)
      as_of   iso timestamp the prices are "as of" (or None if nothing buffered)
    """
    _fetch = _fetch or PT.fetch_current_prices
    d = _dir(buffer_dir)
    path = d / "live_prices.json"
    buf: dict[str, dict] = {}
    if path.exists():
        try:
            buf = json.loads(path.read_text())     # {ticker: {"price": float, "ts": iso}}
        except Exception:
            buf = {}

    covered = all(t in buf for t in holdings)
    if not force and buf and covered:
        prices = {t: buf[t]["price"] for t in holdings}
        as_of = max((buf[t]["ts"] for t in holdings), default=None)
        return prices, {}, as_of

    fresh = _fetch(holdings)                        # {ticker: price | None}
    now = datetime.now().isoformat(timespec="seconds")
    prices, stale = {}, {}
    for t in holdings:
        p = fresh.get(t)
        if p is not None:
            prices[t] = p
            buf[t] = {"price": p, "ts": now}
        elif t in buf:
            prices[t] = buf[t]["price"]            # last-good, NOT average cost
            stale[t] = buf[t]["ts"]
        else:
            prices[t] = None                       # never seen — summary cost-bases it
    path.write_text(json.dumps(buf))
    return prices, stale, now
