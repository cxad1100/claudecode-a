"""The cross-page strategy cache: it must round-trip a curve, refuse to pretend a
degenerate curve is a curve, and degrade quietly rather than break a page build."""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from tools import strategy_cache as sc


def _eq(n=10, start="2024-01-01"):
    return pd.Series(range(100, 100 + n), index=pd.bdate_range(start, periods=n),
                     dtype=float)


def test_roundtrip_carries_curve_source_and_meta(tmp_path):
    sc.save("pairs", _eq(), source="build_pairs_report", cache_dir=tmp_path,
            name="Pairs stat-arb", family="pairs", status="research",
            cost_model="slip + €1")
    got = sc.load("pairs", cache_dir=tmp_path)
    assert got["id"] == "pairs" and got["source"] == "build_pairs_report"
    assert len(got["equity"]) == 10
    assert got["meta"]["family"] == "pairs" and got["meta"]["cost_model"] == "slip + €1"
    assert isinstance(got["built_at"], datetime)


def test_degenerate_curve_is_stored_as_no_curve(tmp_path):
    """A one-point series is not a curve — storing it would make the page draw a dot and
    compute meaningless window metrics from it."""
    one = pd.Series([1.0], index=pd.bdate_range("2024-01-01", periods=1))
    sc.save("thin", one, source="x", cache_dir=tmp_path)
    assert sc.load("thin", cache_dir=tmp_path)["equity"] is None


def test_all_nan_curve_is_stored_as_no_curve(tmp_path):
    nan = pd.Series([float("nan")] * 5, index=pd.bdate_range("2024-01-01", periods=5))
    sc.save("nan", nan, source="x", cache_dir=tmp_path)
    assert sc.load("nan", cache_dir=tmp_path)["equity"] is None


def test_curveless_artifact_still_carries_its_verdict(tmp_path):
    """A strategy with genuinely no daily curve still gets an artifact — its metrics and
    verdict beat no artifact at all."""
    sc.save("nocurve", None, source="lab", cache_dir=tmp_path,
            name="Some lab cell", verdict="killed on the matched null")
    got = sc.load("nocurve", cache_dir=tmp_path)
    assert got["equity"] is None and got["meta"]["verdict"].startswith("killed")


def test_missing_and_corrupt_entries_degrade_to_none(tmp_path):
    assert sc.load("absent", cache_dir=tmp_path) is None
    sc.save("good", _eq(), source="x", cache_dir=tmp_path)
    (tmp_path / "broken.pkl").write_bytes(b"not a pickle at all")
    assert sc.load("broken", cache_dir=tmp_path) is None
    # a corrupt neighbour must not take the readable ones down with it
    assert sorted(sc.load_all(cache_dir=tmp_path)) == ["good"]


def test_wrong_schema_is_ignored(tmp_path):
    import pickle
    with open(tmp_path / "old.pkl", "wb") as fh:
        pickle.dump({"schema": 999, "id": "old", "equity": _eq()}, fh)
    assert sc.load("old", cache_dir=tmp_path) is None
    assert sc.load_all(cache_dir=tmp_path) == {}


def test_save_rejects_a_non_series(tmp_path):
    with pytest.raises(TypeError):
        sc.save("bad", [1, 2, 3], source="x", cache_dir=tmp_path)


def test_age_label_reports_staleness(tmp_path):
    sc.save("p", _eq(), source="x", cache_dir=tmp_path)
    got = sc.load("p", cache_dir=tmp_path)
    now = got["built_at"]
    assert "min ago" in sc.age_label(got, now=now)
    assert sc.age_label(got, now=now + timedelta(hours=5)) == "lab run 5h ago"
    assert sc.age_label(got, now=now + timedelta(days=9)).startswith("lab run ")
    assert round(sc.age_hours(got, now=now + timedelta(hours=3)), 3) == 3.0


def test_id_with_path_separators_cannot_escape_the_cache_dir(tmp_path):
    sc.save("../../evil", _eq(), source="x", cache_dir=tmp_path)
    assert not (tmp_path.parent.parent / "evil.pkl").exists()
    assert sc.load("../../evil", cache_dir=tmp_path) is not None


def test_clear_removes_entries(tmp_path):
    sc.save("a", _eq(), source="x", cache_dir=tmp_path)
    sc.save("b", _eq(), source="x", cache_dir=tmp_path)
    assert sc.clear("a", cache_dir=tmp_path) == 1
    assert sorted(sc.load_all(cache_dir=tmp_path)) == ["b"]
    assert sc.clear(cache_dir=tmp_path) == 1
