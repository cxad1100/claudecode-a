import json
import pandas as pd
from tools.megacap import (candidate_pool, fetch_pool, coverage_report,
                           save_cache, load_cache)

META = pd.DataFrame({"ticker": ["A.F", "B.F", "C.F", "D.F"],
                     "slippage_bps": [5, 40, 12, 8]})
FUND = {"outstandingShares": {"quarterly": {
            "0": {"dateFormatted": "2023-03-31", "shares": 1_000_000}}},
        "Financials": {"Income_Statement": {"quarterly": {
            "2023-03-31": {"date": "2023-03-31", "totalRevenue": "500000000",
                           "filing_date": "2023-04-28"}}}}}


def test_candidate_pool_liquid_sorted_capped():
    pool = candidate_pool(META, {"A.F", "B.F", "C.F", "D.F"}, max_names=2, liq_max=30)
    assert pool == ["A.F", "D.F"]                      # tightest spread first, B.F dropped (>30)


def test_fetch_pool_injected_coverage():
    def gf(url):
        return json.dumps(FUND) if "A.F" in url else "{}"
    shares, rev, cover = fetch_pool(["A.F", "B.F"], key="x", get_fn=gf)
    assert cover == {"A.F": True, "B.F": False}
    assert "A.F" in shares and "A.F" in rev and "B.F" not in shares


def test_coverage_report_pct():
    assert coverage_report({"A.F": True, "B.F": False})["pct"] == 50.0


def test_cache_roundtrip(tmp_path):
    def gf(url):
        return json.dumps(FUND)
    shares, rev, _ = fetch_pool(["A.F"], key="x", get_fn=gf)
    p = tmp_path / "fund.json"
    save_cache(shares, rev, p)
    s2, r2 = load_cache(p)
    assert s2["A.F"].equals(shares["A.F"])
    assert r2["A.F"]["revenue"].equals(rev["A.F"]["revenue"])
