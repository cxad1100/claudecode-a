"""BaFin Directors' Dealings scraper — pure parser/archive contracts.

Fixtures are trimmed real responses from portal.mvp.bafin.de (2026-07-04).
The register's person-detail table carries the full dealing record including
`Datum der Aktivierung` — a REAL publication timestamp, so unlike the short
register nothing is imputed here. Fetch is a thin paced 2-hop crawler
(search pages → person pages) and is not network-tested.
"""
from pathlib import Path

from tools.bafin_dd import archive, normalize, parse_person, parse_results

FIX = Path(__file__).parent / "fixtures"


def test_parse_results_extracts_person_ids_and_pagination():
    out = parse_results((FIX / "bafin_dd_results.html").read_text())
    ids = {r["meldepflichtiger_id"] for r in out["rows"]}
    assert "34423" in ids and "34408" in ids
    r = next(x for x in out["rows"] if x["meldepflichtiger_id"] == "34423")
    assert r["nachname"] == "Herzberg"
    assert out["max_page"] >= 2                   # pagination detected


def test_parse_person_full_record_with_real_publication():
    rows = parse_person((FIX / "bafin_dd_person.html").read_text())
    assert len(rows) == 1
    r = rows[0]
    assert r["issuer"] == "Vidac Pharma Holding PLC"
    assert r["bafin_id"] == "50089006"
    assert r["isin"] == "GB00BM9XQ619"
    assert r["person"] == "Herzberg, Dr. Max"
    assert r["person_role"] == "Vorstand"
    assert r["instrument"] == "Aktie"
    assert r["side"] == "Verkauf"
    assert r["event_date"] == "2026-06-30"        # 30.06.2026 → ISO
    assert r["venue"] == "Tradegate"
    assert r["published_at"] == "2026-07-03"      # Datum der Aktivierung
    assert r["meldung_id"] == "34416"


def test_normalize_maps_sides_and_flags_real_publication():
    rows = parse_person((FIX / "bafin_dd_person.html").read_text())
    df = normalize(rows, fetched_at="2026-07-04T12:00:00")
    r = df.iloc[0]
    assert r["side"] == "sell"                    # Verkauf → sell
    assert bool(r["published_imputed"]) is False  # real timestamp
    assert r["key"]


def test_archive_idempotent_by_key():
    rows = parse_person((FIX / "bafin_dd_person.html").read_text())
    df = normalize(rows, fetched_at="x")
    store = archive(df, None)
    assert len(archive(df, store)) == len(store) == 1
