"""BaFin Directors' Dealings register (MAR Art. 19) → PIT event store.

  python -m tools.bafin_dd fetch --since 2026-06 --until 2026-07 [--limit N]

Endpoints (verified 2026-07-04, struts app at
portal.mvp.bafin.de/database/DealingsInfo/):

  search:  POST sucheForm.do  (zeitraumVon/zeitraumBis dd.mm.yyyy, zeitraum=0)
           results paginate via d-5010980-p=N (20 rows/page); rows carry
           ergebnisListe.do?cmd=loadEmittentenAction&meldepflichtigerId=<id>
  person:  GET that link → table with the person's COMPLETE dealing history:
           Emittent | BaFin-ID | ISIN | Meldepflichtiger | Position/Status |
           Art des Instruments | Art des Geschäfts | Datum des Geschäfts |
           Ort des Geschäfts | Datum der Aktivierung
           (Aktivierung = real publication timestamp → published_imputed=False)
  amounts: price/volume live one more hop down (transaktionListe.do modal);
           deliberately deferred — occurrence + side carries the documented
           insider signal (Lakonishok & Lee 2001), an optional --with-amounts
           can add the third hop later.

Crawl economics: person pages return full histories, so the register is
covered by crawling UNIQUE persons (~requests ≈ unique meldepflichtigerIds,
not dealings).

RETENTION (measured 2026-07-04): the search only returns the trailing ~12
months — earlier windows come back empty (and out-of-range dates on the
export path silently fall back to the unfiltered view, so a "114 pages"
result is the cap artifact, not data). Person pages DO reach further back,
so the store holds multi-year histories for persons active in the trailing
year. Coverage bias to document wherever this feeds analysis: going back in
time, only still-active insiders are covered; fresh-window signals (90d
tilt) are unbiased, deep backtests are not. Re-running fetch monthly keeps
discovery complete going forward. Fetch is windowed (--since/--until months), paced (1s +
backoff), and resumable: raw pages cached under local/econo_cache/bafin_dd/,
person pages skipped when cached same-day, and the archive is idempotent by
row key. Store data/universe/insider_dealings.csv is append-only — the
snapshots.csv/reconcile ledger philosophy, row-keyed.
"""
import hashlib
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from http.cookiejar import CookieJar

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "universe" / "insider_dealings.csv"
CACHE = ROOT / "local" / "econo_cache" / "bafin_dd"
BASE = "https://portal.mvp.bafin.de/database/DealingsInfo/"
UA = "Mozilla/5.0 (research; investment-monitor) python-urllib"

_COLS = ["key", "meldung_id", "isin", "issuer", "bafin_id", "person",
         "person_role", "instrument", "side", "event_date", "published_at",
         "published_imputed", "venue", "price", "volume_eur", "fetched_at"]

_SIDE = {"Kauf": "buy", "Verkauf": "sell"}


def _iso(d: str) -> str:
    """'30.06.2026[ 14:12:08]' → '2026-06-30' ('' when unparseable)."""
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", d or "")
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else ""


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", "", html)).strip()


# ── pure parsers (fixture-tested) ─────────────────────────────────────────────

def parse_results(html: str) -> dict:
    """Search-results page → {'rows': [...], 'max_page': int}. Rows carry the
    meldepflichtiger_id needed for the person-page hop."""
    rows = []
    for tr in re.findall(r"<tr[^>]*>.*?</tr>", html, re.S):
        m = re.search(r"meldepflichtigerId=(\d+)", tr)
        if not m:
            continue
        tds = [_strip(td) for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(tds) < 5:
            continue
        rows.append(dict(meldepflichtiger_id=m.group(1), nachname=tds[0],
                         vorname=tds[1], titel=tds[2], position=tds[3],
                         list_date=_iso(tds[4])))
    pages = [int(p) for p in re.findall(r"d-5010980-p=(\d+)", html)]
    return dict(rows=rows, max_page=max(pages) if pages else 1)


def parse_person(html: str) -> list[dict]:
    """Person-detail page → complete dealing records (one per table row),
    including the meldung_id from the transaction link and the REAL
    publication timestamp (Datum der Aktivierung)."""
    out = []
    for tr in re.findall(r"<tr[^>]*>.*?</tr>", html, re.S):
        m = re.search(r"meldungId=(\d+)", tr)
        if not m:
            continue
        tds = [_strip(td) for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(tds) < 10:
            continue
        out.append(dict(meldung_id=m.group(1), issuer=tds[0], bafin_id=tds[1],
                        isin=tds[2].upper(), person=tds[3], person_role=tds[4],
                        instrument=tds[5], side=tds[6],
                        event_date=_iso(tds[7]), venue=tds[8],
                        published_at=_iso(tds[9])))
    return out


def normalize(rows: list[dict], fetched_at: str) -> pd.DataFrame:
    """Records → canonical frame. Sides map Kauf/Verkauf → buy/sell (other
    instrument-specific labels kept verbatim); published_imputed is False —
    this register exposes the real activation timestamp."""
    out = []
    for r in rows:
        if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{10}", r.get("isin", "")):
            continue
        if not r.get("event_date") or not r.get("published_at"):
            continue
        key = hashlib.sha1(
            f"{r['meldung_id']}|{r['isin']}|{r['event_date']}|{r['side']}"
            .encode()).hexdigest()[:16]
        out.append(dict(key=key, meldung_id=r["meldung_id"], isin=r["isin"],
                        issuer=r["issuer"], bafin_id=r["bafin_id"],
                        person=r["person"], person_role=r["person_role"],
                        instrument=r["instrument"],
                        side=_SIDE.get(r["side"], r["side"]),
                        event_date=r["event_date"],
                        published_at=r["published_at"],
                        published_imputed=False, venue=r["venue"],
                        price="", volume_eur="", fetched_at=fetched_at))
    return pd.DataFrame(out, columns=_COLS)


def archive(new: pd.DataFrame, store: pd.DataFrame | None) -> pd.DataFrame:
    """Append-only, idempotent by key; corrections append, never overwrite."""
    if store is None or not len(store):
        return new.drop_duplicates(subset="key").reset_index(drop=True)
    fresh = new[~new["key"].isin(set(store["key"]))]
    return pd.concat([store, fresh.drop_duplicates(subset="key")],
                     ignore_index=True)


# ── thin IO + CLI ─────────────────────────────────────────────────────────────

def _opener():
    op = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar()))
    op.addheaders = [("User-Agent", UA)]
    return op


def _get(op, url: str, tries: int = 3) -> str:
    for k in range(tries):
        try:
            with op.open(url, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2.0 * 2 ** k)
    return ""


#: displaytag caps any query at 114 pages (2,272 rows) — a window hitting the
#: cap must be split (weekly) or rows silently vanish. zeitraum=3 activates
#: the custom date range; 0 would be Gesamtzeitraum and IGNORE von/bis.
_PAGE_CAP = 114


def _search_url(von: str, bis: str, page: int = 1) -> str:
    q = dict(zeitraum="3", zeitraumVon=von, zeitraumBis=bis, emittentName="",
             emittentIsin="", meldepflichtigerName="", sucheButton="Suche")
    if page > 1:
        q["d-5010980-p"] = str(page)
    return BASE + "sucheForm.do?" + urllib.parse.urlencode(q)


def fetch(since: str, until: str, limit: int | None = None,
          pace: float = 1.0) -> pathlib.Path:
    """Crawl [since, until) month windows: discover unique persons from the
    paginated search, pull each person page once (same-day cache), archive
    every dealing row. Resumable; safe to re-run."""
    ts = datetime.now().isoformat(timespec="seconds")
    day = ts[:10]
    CACHE.mkdir(parents=True, exist_ok=True)
    op = _opener()
    _get(op, BASE + "?locale=de_DE")                        # open session
    def _discover(von_ts: pd.Timestamp, bis_ts: pd.Timestamp) -> None:
        """Page one window into person_ids; split when the row cap bites."""
        von = von_ts.strftime("%d.%m.%Y")
        bis = bis_ts.strftime("%d.%m.%Y")
        page = 1
        while True:
            html = _get(op, _search_url(von, bis, page))
            time.sleep(pace)
            res = parse_results(html)
            if page == 1 and res["max_page"] >= _PAGE_CAP \
                    and (bis_ts - von_ts).days > 7:
                mid = von_ts + (bis_ts - von_ts) / 2
                _discover(von_ts, mid)
                _discover(mid, bis_ts)
                return
            for r in res["rows"]:
                person_ids.setdefault(r["meldepflichtiger_id"], r["nachname"])
            if page >= res["max_page"] or not res["rows"]:
                return
            page += 1

    months = pd.period_range(since, until, freq="M")
    person_ids: dict[str, str] = {}
    for p in months:
        _discover(p.start_time, p.end_time + pd.Timedelta(days=1))
        print(f"bafin_dd: {p} → {len(person_ids)} unique persons so far",
              flush=True)
    ids = list(person_ids)
    if limit:
        ids = ids[:limit]
    rows: list[dict] = []
    for i, pid in enumerate(ids):
        cache_f = CACHE / f"person_{pid}_{day}.html"
        if cache_f.exists():
            html = cache_f.read_text(encoding="utf-8")
        else:
            html = _get(op, BASE + "ergebnisListe.do?cmd=loadEmittentenAction"
                        f"&meldepflichtigerId={pid}")
            cache_f.write_text(html, encoding="utf-8")
            time.sleep(pace)
        rows.extend(parse_person(html))
        if (i + 1) % 25 == 0:
            print(f"bafin_dd: {i + 1}/{len(ids)} persons, "
                  f"{len(rows)} dealing rows", flush=True)
    df = normalize(rows, fetched_at=ts)
    store = pd.read_csv(STORE, dtype=str) if STORE.exists() else None
    merged = archive(df, store)
    n_new = len(merged) - (len(store) if store is not None else 0)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(STORE, index=False)
    print(f"bafin_dd: {len(df)} parsed, {n_new} new rows → {STORE}", flush=True)
    return STORE


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] != "fetch":
        print("usage: python -m tools.bafin_dd fetch --since YYYY-MM "
              "--until YYYY-MM [--limit N]")
        raise SystemExit(2)

    def _opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    fetch(since=_opt("--since", "2026-06"), until=_opt("--until", "2026-07"),
          limit=int(_opt("--limit", 0)) or None)
