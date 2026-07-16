"""SEC EDGAR companyfacts → PIT shares-outstanding + revenue histories.

Free, official, no key — the fundamentals source for the mega-cap screen after
EODHD's fundamentals API turned out to be a separate paid product. Better PIT than
the EODHD path too: every value becomes available at its exact `filed` date instead
of a 75-day-lag heuristic.

Coverage = SEC filers only: US natives plus foreign private issuers with a US
listing (20-F: SAP, ASML, Toyota, Sony, ...). Names that never file with the SEC
(most .T / .HK / EU-domestic-only lines) stay uncovered — the mega-cap page's
coverage line quantifies the gap, and the screen ranks only covered names.

Split handling: universe prices are yfinance auto-adjusted, but filings report
as-reported share counts, so `parse_shares_history` re-bases the count history onto
the latest split basis by detecting jump ratios (>=1.4x or <=0.7x between
consecutive filings — buybacks/issuance never move that fast, splits always do).
Dividend adjustment still leaves old caps understated by a few percent; tolerable
for a rank screen, not for absolute caps.

Etiquette per SEC guidance: descriptive User-Agent with a contact address, paced
well under their ~10 req/s ceiling.
"""
import json
import re
import time
import urllib.request

import pandas as pd

USER_AGENT = "investment-monitor zxmc1100@gmail.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Share-count tag chain. The weighted-average family first: it is total across
# share classes (companyfacts drops class dimensions, so per-class instant tags
# like GOOGL's are ambiguous or absent), and it spans the us-gaap -> ifrs-full
# taxonomy switch some 20-F filers made (Toyota). Instant tags are the fallback.
_SHARE_TAGS_WA = (("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
                  ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
                  ("ifrs-full", "WeightedAverageShares"),
                  ("ifrs-full", "AdjustedWeightedAverageShares"))
_SHARE_TAGS_INSTANT = (("dei", "EntityCommonStockSharesOutstanding"),
                       ("us-gaap", "CommonStockSharesOutstanding"),
                       ("ifrs-full", "NumberOfSharesOutstanding"),
                       ("ifrs-full", "NumberOfSharesIssued"))

# Revenue tag family across taxonomy eras (SalesRevenueNet pre-2018, RFC 2018+,
# ifrs-full for 20-F filers). Entries are unioned, then de-duplicated per period.
_REV_TAGS = (("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
             ("us-gaap", "Revenues"),
             ("us-gaap", "SalesRevenueNet"),
             ("us-gaap", "SalesRevenueGoodsNet"),
             ("ifrs-full", "Revenue"),
             ("ifrs-full", "RevenueFromContractsWithCustomers"))

_Q_DUR = (60, 120)            # days: a quarterly period
_A_DUR = (320, 400)           # days: an annual period
_MIN_QUARTERS = 8             # fewer quarterly rows than this -> use annual cadence
_SPLIT_UP, _SPLIT_DOWN = 1.4, 0.7   # consecutive-count jump ratios that mean "split"

# Legal-form tokens that differ between the broker list and SEC titles.
_NAME_STOP = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED",
              "PLC", "SE", "AG", "NV", "SA", "KGAA", "ADR", "THE", "CLASS"}


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")


# ── ticker / name → CIK ──────────────────────────────────────────────────────────

def fetch_company_tickers(get_fn=_http_get) -> dict:
    """SEC's full filer list {ticker, title, cik_str} — one fetch per run."""
    return json.loads(get_fn(TICKERS_URL))


def norm_name(s: str) -> str:
    """Uppercase, strip punctuation and legal-form tokens — the join key between
    broker display names ('Toyota Motor') and SEC titles ('TOYOTA MOTOR CORP/')."""
    s = re.sub(r"\(.*?\)", " ", str(s).upper())
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in _NAME_STOP]
    return " ".join(toks)


def cik_index(raw: dict) -> tuple[dict, dict]:
    """(by_ticker, by_name) lookups. First occurrence wins on collision — the SEC
    file lists the primary/largest registrant class first."""
    by_ticker, by_name = {}, {}
    for v in raw.values():
        t, cik, title = str(v["ticker"]).upper(), int(v["cik_str"]), v["title"]
        by_ticker.setdefault(t, cik)
        nn = norm_name(title)
        if nn:
            by_name.setdefault(nn, cik)
    return by_ticker, by_name


def match_cik(ticker: str, name: str, by_ticker: dict, by_name: dict) -> int | None:
    """CIK for a universe row, conservatively.

    Bare / `.US` tickers are trusted directly (they ARE US symbols). Exchange-
    suffixed tickers (`.T`, `.DE`, ...) never match by symbol — `DAI.DE` vs some
    unrelated US 'DAI' — they must match by normalized name (exact, or unique
    prefix either way). No match -> None -> the name is simply uncovered."""
    t = str(ticker).upper()
    if "." not in t or t.endswith(".US"):
        cik = by_ticker.get(t.split(".")[0])
        if cik is not None:
            return cik
    nn = norm_name(name)
    if not nn:
        return None
    if nn in by_name:
        return by_name[nn]
    pref = [c for n, c in by_name.items()
            if n.startswith(nn + " ") or nn.startswith(n + " ")]
    return pref[0] if len(pref) == 1 else None


# ── companyfacts fetch + parse ────────────────────────────────────────────────────

def fetch_companyfacts(cik: int, *, get_fn=_http_get, retries: int = 2,
                       retry_delay: float = 1.0):
    """Full companyfacts blob for a CIK → dict, or None on failure. Same retry
    shape as tools.eodhd so one blip never aborts a paced batch."""
    url = FACTS_URL.format(cik=int(cik))
    for attempt in range(retries + 1):
        try:
            data = json.loads(get_fn(url))
            return data if isinstance(data, dict) and data.get("facts") else None
        except Exception:
            if attempt >= retries:
                return None
            time.sleep(retry_delay * (attempt + 1))
    return None


def _entries(fund: dict, tags) -> list:
    """Union of share-unit fact entries across a tag chain."""
    out = []
    facts = (fund or {}).get("facts", {})
    for tax, tag in tags:
        units = facts.get(tax, {}).get(tag, {}).get("units", {})
        out.extend(units.get("shares", []))
    return out


def _split_normalize(ser: pd.Series) -> pd.Series:
    """Re-base an as-reported share-count series onto its latest split basis:
    walk backwards, multiplying every earlier value by each detected jump ratio."""
    if len(ser) < 2:
        return ser
    vals = ser.to_list()
    factor = 1.0
    out = [vals[-1]]
    for i in range(len(vals) - 2, -1, -1):
        r = vals[i + 1] / vals[i] if vals[i] else 1.0
        if r >= _SPLIT_UP or r <= _SPLIT_DOWN:
            factor *= r
        out.append(vals[i] * factor)
    return pd.Series(out[::-1], index=ser.index)


def _filing_series(rows) -> pd.Series:
    """entries → Series filed→count: one value per filing date, keeping the latest
    period-end (the current period, not a comparative restatement of an old year)."""
    recs = {}
    for e in rows:
        v, end, filed = e.get("val"), e.get("end"), e.get("filed")
        if not v or not end or not filed:
            continue
        f = pd.Timestamp(filed)
        if f not in recs or pd.Timestamp(end) > recs[f][0]:
            recs[f] = (pd.Timestamp(end), float(v))
    if not recs:
        return pd.Series(dtype=float)
    return pd.Series({f: v for f, (_, v) in recs.items()}, dtype=float).sort_index()


def parse_shares_history(fund: dict) -> pd.Series:
    """Share count per filing → Series indexed by availability (= `filed`),
    normalized to the latest basis. Weighted-average tags rule wherever they exist
    (total across classes); instant/cover-page tags fill the years BEFORE the WA
    span (some filers, e.g. Alphabet, only tagged WA totals recently). The
    normalizer reconciles the splice: a basis jump at the boundary — a split, or
    an instant tag that covered one share class — is detected as a ratio jump and
    the earlier history is re-based onto the trusted latest (WA) basis."""
    wa = _filing_series(_entries(fund, _SHARE_TAGS_WA))
    inst = _filing_series(_entries(fund, _SHARE_TAGS_INSTANT))
    if wa.empty:
        ser = inst
    elif not inst.empty and inst.index[0] < wa.index[0]:
        ser = pd.concat([inst[inst.index < wa.index[0]], wa]).sort_index()
    else:
        ser = wa
    if ser.empty:
        return ser
    return _split_normalize(ser)


def parse_revenue_history(fund: dict) -> pd.DataFrame:
    """Revenue rows → DataFrame(index=period end, revenue, avail=filed), one
    cadence per name: quarterly when >= _MIN_QUARTERS quarterly periods exist,
    else annual (20-F filers report annually). Single (majority) currency unit —
    growth is a within-name ratio, so mixing USD and JPY rows would corrupt it.
    Per period the EARLIEST filing wins (that is when the number became known)."""
    empty = pd.DataFrame(columns=["revenue", "avail"])
    facts = (fund or {}).get("facts", {})
    rows = []
    for tax, tag in _REV_TAGS:
        for unit, es in facts.get(tax, {}).get(tag, {}).get("units", {}).items():
            for e in es:
                if e.get("val") is None or not e.get("start") or not e.get("end") \
                        or not e.get("filed"):
                    continue
                rows.append((unit, pd.Timestamp(e["start"]), pd.Timestamp(e["end"]),
                             pd.Timestamp(e["filed"]), float(e["val"])))
    if not rows:
        return empty
    units = pd.Series([r[0] for r in rows])
    major = units.value_counts().idxmax()
    picked = {}
    for unit, start, end, filed, val in rows:
        if unit != major:
            continue
        d = (end - start).days
        if _Q_DUR[0] <= d <= _Q_DUR[1]:
            cad = "Q"
        elif _A_DUR[0] <= d <= _A_DUR[1]:
            cad = "A"
        else:
            continue                       # H1 / 9M cumulatives: not a YoY cadence
        key = (cad, end)
        if key not in picked or filed < picked[key][0]:
            picked[key] = (filed, val)
    q = {end: fv for (cad, end), fv in picked.items() if cad == "Q"}
    a = {end: fv for (cad, end), fv in picked.items() if cad == "A"}
    use = q if len(q) >= _MIN_QUARTERS else a
    if not use:
        return empty
    idx = sorted(use)
    return pd.DataFrame({"revenue": [use[e][1] for e in idx],
                         "avail": [use[e][0] for e in idx]}, index=idx)
