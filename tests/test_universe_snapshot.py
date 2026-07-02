"""Point-in-time universe snapshots (tools.universe_snapshot).

Tier-1 survivorship fix: freeze 'what TR offered' on a date, append-only, so future
backtests can ask membership_asof(t) instead of using today's survivors. Pins the two
pure pieces — the dated append (idempotent) and the as-of reader (latest snapshot ≤ t)."""
import pandas as pd

from tools import universe_snapshot as us


def _members(*isins):
    return pd.DataFrame({"isin": list(isins),
                         "name": [f"N{i}" for i in isins],
                         "country": ["US"] * len(isins)})


def test_archive_stamps_snapshot_date_onto_fresh_store():
    out = us.archive(_members("A", "B"), "2026-06-29", store=None)
    assert set(out["isin"]) == {"A", "B"}
    assert (out["snapshot_date"] == "2026-06-29").all()           # every row dated
    assert list(out.columns) == ["snapshot_date", "isin", "name", "country"]


def test_archive_same_date_replaces_not_duplicates():
    s1 = us.archive(_members("A", "B"), "2026-06-29", store=None)
    s2 = us.archive(_members("A", "C", "D"), "2026-06-29", store=s1)   # re-run same day
    rows = s2[s2["snapshot_date"] == "2026-06-29"]
    assert set(rows["isin"]) == {"A", "C", "D"}                   # second run wins
    assert len(rows) == 3                                         # no stale A/B duplicate


def test_archive_preserves_earlier_dates():
    s1 = us.archive(_members("A", "B"), "2026-05-01", store=None)
    s2 = us.archive(_members("A"), "2026-06-01", store=s1)        # B delisted between snaps
    assert set(s2["snapshot_date"]) == {"2026-05-01", "2026-06-01"}
    assert len(s2[s2["snapshot_date"] == "2026-05-01"]) == 2      # history intact


def test_membership_asof_returns_latest_on_or_before():
    s = us.archive(_members("A", "B"), "2026-05-01", store=None)
    s = us.archive(_members("A", "C"), "2026-06-01", store=s)     # B gone, C new
    assert us.membership_asof(s, "2026-06-15") == {"A", "C"}      # newest ≤ t
    assert us.membership_asof(s, "2026-05-15") == {"A", "B"}      # the may snapshot
    assert us.membership_asof(s, "2026-05-01") == {"A", "B"}      # boundary inclusive
    assert us.membership_asof(s, "2026-04-01") == set()           # before any snapshot


def test_membership_asof_empty_store_is_empty():
    assert us.membership_asof(us.empty_store(), "2026-06-29") == set()
