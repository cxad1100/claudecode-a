"""Offline: enrich data/universe/universe_meta.csv with real GICS sectors via yfinance.

The TR-native momentum universe ships sector='Unknown' for every name (the enumerate
step carries no sector), which makes sector-neutral selection (upgrade B) a silent
no-op. This fetches yfinance `.info['sector']` per LIVE ticker (the home listing is
already the ticker, e.g. NVDA/NASDAQ, P911.DE, 8411.T — far better sector coverage than
the .F lines), caches to data/universe/sector_map.json (resumable), and writes the
sector column back into the meta CSV.

  python -m tools.enrich_sectors                 # all live names
  python -m tools.enrich_sectors --floor 1e6     # only names >= €/day turnover floor

Idempotent: a re-run only fetches tickers absent from the cache, so an interrupted run
resumes where it left off. Dead (delisted) names are skipped — momentum ranks them last
and holds ~0 into death, so their sector never matters."""
import argparse
import json
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data/universe/universe_meta.csv"
CACHE = ROOT / "data/universe/sector_map.json"
JUNK = {"Unknown", "", None}


def merge_sectors(meta: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Apply the {ticker: sector|None} cache onto the meta's sector column. A real
    fetched sector overwrites; a None (tried, none found) or an absent ticker keeps the
    existing value. Pure — returns a new frame."""
    out = meta.copy()
    fetched = out["ticker"].astype(str).map(lambda t: cache.get(t))
    out["sector"] = fetched.where(fetched.notna(), out["sector"])
    return out


def _fetch_sector(ticker: str, retries: int = 4, base: float = 3.0) -> str | None:
    """yfinance sector with backoff. Yahoo rate-limits bursts (YFRateLimitError); on
    that we sleep base·2^attempt + jitter and retry, so the run self-heals as the
    throttle resets. A small leading jitter desynchronises worker threads. None means
    a genuine miss (this yfinance raises on rate-limit rather than returning empty)."""
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(0.0, 0.4))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sec = yf.Ticker(ticker).info.get("sector")
            return str(sec) if sec else None
        except Exception as e:
            if "RateLimit" in type(e).__name__ and attempt < retries - 1:
                time.sleep(base * (2 ** attempt) + random.uniform(0.0, 1.0))
                continue
            return None
    return None


def _load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True))


def targets(meta: pd.DataFrame, floor: float) -> list[str]:
    """Live names (not delisted) at/above the turnover floor — the realistic pool the
    strategy can actually pick."""
    live = meta[meta["delisting_date"].isna()]
    live = live[live["med_turnover"].fillna(0) >= floor]
    return sorted(set(live["ticker"].astype(str)))


def enrich(floor: float = 0.0, workers: int = 6, retry_misses: bool = False,
           batch: int = 0, cooldown_s: float = 0.0) -> dict:
    """Fetch sectors for the targeted names and write them back. `batch`>0 submits in
    chunks of that size with a `cooldown_s` pause between chunks — this PACES the load so
    Yahoo's rate bucket refills instead of throttling a single long burst (the burst gets
    ~700 names then YFRateLimitError-poisons the rest). `retry_misses` drops cached Nones
    (rate-limit victims) so they're refetched."""
    meta = pd.read_csv(META)
    cache = _load_cache()
    if retry_misses:
        cache = {k: v for k, v in cache.items() if v}
    todo = [t for t in targets(meta, floor) if t not in cache]
    chunk = batch if batch > 0 else len(todo) or 1
    print(f"{len(cache)} cached · {len(todo)} to fetch "
          f"(floor={floor:.0f}, workers={workers}, batch={chunk}, cooldown={cooldown_s:.0f}s)")
    done = 0
    for i in range(0, len(todo), chunk):
        names = todo[i:i + chunk]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fetch_sector, t): t for t in names}
            for fut in as_completed(futs):
                cache[futs[fut]] = fut.result()
                done += 1
        _save_cache(cache)
        got = sum(1 for v in cache.values() if v)
        print(f"  batch {i // chunk + 1}: {done}/{len(todo)} fetched · {got} real total", flush=True)
        if cooldown_s and i + chunk < len(todo):
            time.sleep(cooldown_s)
    out = merge_sectors(meta, cache)
    n_real = (~out["sector"].isin(JUNK) & out["sector"].notna()).sum()
    out.to_csv(META, index=False)
    print(f"wrote {META}: {n_real}/{len(out)} names now carry a real sector")
    return cache


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retry-misses", action="store_true",
                    help="drop cached Nones (rate-limit victims) and refetch gently")
    ap.add_argument("--batch", type=int, default=0, help="chunk size; pause between chunks")
    ap.add_argument("--cooldown", type=float, default=0.0, help="seconds to pause between chunks")
    args = ap.parse_args()
    enrich(floor=args.floor, workers=args.workers, retry_misses=args.retry_misses,
           batch=args.batch, cooldown_s=args.cooldown)
