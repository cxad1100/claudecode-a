"""German net-short-position register (Bundesanzeiger NLP) → PIT event store.

  python -m tools.short_register fetch          # current positions, append
  python -m tools.short_register fetch --full   # full history bootstrap

Public register of net short positions ≥ 0.5% (EU SSR Art. 6), the German
short-interest dataset (Jank, Roling, Smajlbegovic 2021 JFE study exactly
this data). Endpoint (verified 2026-07-04): GET the register page to open a
wicket session, POST the filter form with isHistorical=true for history,
then GET the session's CSV resource link:

  base:  https://www.bundesanzeiger.de/pub/de/nlp?<page>
  form:  action= …nlp?<p>-1.-nlp~filter~form~panel-form   (fields incl. isHistorical)
  csv:   href=  …csv~form~panel-form-csv~resource~link

CSV: BOM, quoted, columns Positionsinhaber,Emittent,ISIN,Position,Datum —
decimal-comma percentage, ISO dates. Publication is statutorily the next
business day after the position date (SSR Art. 9), so `published_at` is
imputed position_date+1bd and flagged `published_imputed` — PIT joins must
use published_at strictly-before, and any signal must survive a +1bd
sensitivity lag (see plan risk register).

Pure core (parse/normalize/archive) is unit-tested on a real-response
fixture; fetch is a thin paced IO wrapper with raw responses cached under
local/econo_cache/short_register/ (parser bugs replay offline). The store
data/universe/short_positions.csv is append-only: corrections append new
rows (sha1 row key), never overwrite — the reconcile()/snapshots.csv ledger
philosophy, row-keyed.
"""
import csv
import hashlib
import io
import pathlib
import re
import sys
import time
import urllib.request
from datetime import datetime
from http.cookiejar import CookieJar

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "universe" / "short_positions.csv"
CACHE = ROOT / "local" / "econo_cache" / "short_register"
BASE = "https://www.bundesanzeiger.de/pub/de/nlp"
UA = "Mozilla/5.0 (research; investment-monitor) python-urllib"

_COLS = ["key", "holder", "issuer", "isin", "pct", "position_date",
         "published_at", "published_imputed", "fetched_at"]


def pit_columns() -> list[str]:
    return list(_COLS)


# ── pure core ─────────────────────────────────────────────────────────────────

def parse(text: str) -> list[dict]:
    """Register CSV → rows. Handles the UTF-8 BOM, quoted commas and the
    German decimal comma in the position percentage."""
    text = text.lstrip("﻿")
    rows = []
    for rec in csv.DictReader(io.StringIO(text)):
        try:
            pct = float(str(rec["Position"]).replace(".", "").replace(",", "."))
        except (TypeError, ValueError):
            continue
        isin = str(rec.get("ISIN", "")).strip().upper()
        date = str(rec.get("Datum", "")).strip()
        if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{10}", isin) or not date:
            continue
        rows.append(dict(holder=str(rec["Positionsinhaber"]).strip(),
                         issuer=str(rec["Emittent"]).strip(),
                         isin=isin, pct=pct, position_date=date))
    return rows


def _row_key(r: dict) -> str:
    raw = f"{r['holder']}|{r['isin']}|{r['position_date']}|{r['pct']:.4f}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def normalize(rows: list[dict], fetched_at: str) -> pd.DataFrame:
    """Rows → canonical frame. published_at = position_date + 1 business day
    (SSR statutory next-day publication), always flagged imputed — the
    register does not expose the true publication timestamp."""
    out = []
    for r in rows:
        pub = (pd.Timestamp(r["position_date"]) + pd.offsets.BDay(1)).date()
        out.append(dict(key=_row_key(r), holder=r["holder"], issuer=r["issuer"],
                        isin=r["isin"], pct=float(r["pct"]),
                        position_date=r["position_date"],
                        published_at=pub.isoformat(), published_imputed=True,
                        fetched_at=fetched_at))
    return pd.DataFrame(out, columns=_COLS)


def archive(new: pd.DataFrame, store: pd.DataFrame | None) -> pd.DataFrame:
    """Append-only, idempotent by row key: re-fetching the same positions is
    a no-op; a changed position (new date or pct) appends a new row. Nothing
    is ever mutated or dropped — superseded rows are the history."""
    if store is None or not len(store):
        return new.drop_duplicates(subset="key").reset_index(drop=True)
    fresh = new[~new["key"].isin(set(store["key"]))]
    return pd.concat([store, fresh.drop_duplicates(subset="key")],
                     ignore_index=True)


# ── thin IO + CLI ─────────────────────────────────────────────────────────────

def _http_session():
    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", UA)]
    return opener


def _get(opener, url: str) -> str:
    with opener.open(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch(full: bool = False, get_fn=None) -> pathlib.Path:
    """Open a session, (for --full) toggle historised rows via the filter
    form, download the CSV resource, archive into the store."""
    ts = datetime.now().isoformat(timespec="seconds")
    if get_fn is None:
        op = _http_session()
        page = _get(op, f"{BASE}?0")
        if full:
            m = re.search(r'action="([^"]*nlp~filter~form~panel-form)"', page)
            if not m:
                raise RuntimeError("filter form not found — page layout changed")
            form_url = m.group(1).replace("&amp;", "&")
            data = ("fulltext=&positionsinhaber=&ermittent=&isin=&positionVon="
                    "&positionBis=&datumVon=&datumBis=&isHistorical=true")
            req = urllib.request.Request(form_url, data=data.encode(),
                                         headers={"User-Agent": UA})
            with op.open(req, timeout=60) as r:
                page = r.read().decode("utf-8", errors="replace")
            time.sleep(1.0)
        m = re.search(r'href="([^"]*csv~form~panel-form-csv~resource~link)"', page)
        if not m:
            raise RuntimeError("CSV link not found — page layout changed")
        text = _get(op, m.group(1).replace("&amp;", "&"))
    else:                                          # injected transport (tests)
        text = get_fn(full)
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "full" if full else "current"
    (CACHE / f"{ts[:10]}_{tag}.csv").write_text(text, encoding="utf-8")
    df = normalize(parse(text), fetched_at=ts)
    store = pd.read_csv(STORE, dtype={"published_imputed": bool}) \
        if STORE.exists() else None
    merged = archive(df, store)
    n_new = len(merged) - (len(store) if store is not None else 0)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(STORE, index=False)
    print(f"short_register: {len(df)} fetched, {n_new} new rows → {STORE}",
          flush=True)
    return STORE


if __name__ == "__main__":
    full = "--full" in sys.argv
    if len(sys.argv) < 2 or sys.argv[1] != "fetch":
        print("usage: python -m tools.short_register fetch [--full]")
        raise SystemExit(2)
    fetch(full=full)
