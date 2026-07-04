"""Short-selling register scraper — pure parse/normalize/archive contracts.

Fixture is a head of the real Bundesanzeiger NLP full-history CSV (BOM,
quoted fields, decimal-comma percentages, ISO dates). The fetch CLI is a thin
IO wrapper and is not network-tested; parsers and the append-only PIT archive
are the tested core (universe_snapshot.archive philosophy, row-keyed).
"""
from pathlib import Path

import pandas as pd

from tools.short_register import archive, normalize, parse, pit_columns

FIX = Path(__file__).parent / "fixtures" / "short_register_sample.csv"


def test_parse_handles_bom_quotes_decimal_comma():
    rows = parse(FIX.read_text(encoding="utf-8"))
    assert len(rows) == 5
    r = rows[0]
    assert r["holder"] == "VARENNE CAPITAL PARTNERS"
    assert r["isin"] == "DE0005158703"
    assert r["pct"] == 0.21                       # 0,21 → float
    assert r["position_date"] == "2026-07-02"
    # comma inside quoted holder name survives
    assert any(x["holder"] == "AQR Capital Management, LLC" for x in rows)


def test_normalize_imputes_next_bday_publication_flagged():
    rows = parse(FIX.read_text(encoding="utf-8"))
    df = normalize(rows, fetched_at="2026-07-04T12:00:00")
    assert list(df.columns) == pit_columns()
    r = df.iloc[0]
    # 2026-07-02 = Thursday → statutory publication next business day = Friday
    assert r["position_date"] == "2026-07-02"
    assert r["published_at"] == "2026-07-03"
    assert bool(r["published_imputed"]) is True
    assert r["fetched_at"] == "2026-07-04T12:00:00"


def test_archive_idempotent_and_appends_changes():
    rows = parse(FIX.read_text(encoding="utf-8"))
    df = normalize(rows, fetched_at="2026-07-04T12:00:00")
    store = archive(df, None)
    again = archive(df, store)
    assert len(again) == len(store) == 5          # re-run is a no-op
    # same rows re-fetched later (different fetched_at, same key) → no-op
    bigger = archive(normalize(parse(FIX.read_text(encoding="utf-8")),
                               fetched_at="x"), store)
    assert len(bigger) == 5
    # a position CHANGE (same holder/isin, new date+pct) goes through
    # normalize like every real fetch → new key → appends, never overwrites
    upd = normalize([dict(holder="VARENNE CAPITAL PARTNERS",
                          issuer="Bechtle Aktiengesellschaft",
                          isin="DE0005158703", pct=0.35,
                          position_date="2026-07-03")], fetched_at="y")
    grown = archive(upd, store)
    assert len(grown) == 6                        # correction appended
    assert (grown["pct"] == 0.35).any()
    assert (grown["pct"] == 0.21).any()           # old row kept (history)
