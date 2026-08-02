import json
import pandas as pd
from tools.megacap import (candidate_pool, fetch_pool, coverage_report,
                           save_cache, load_cache, prune_shares)


def test_prune_shares_drops_glitch_and_dedups_same_company():
    idx = pd.date_range("2020-01-01", periods=5)
    shares = {"SSU.JO": pd.Series(1.0, index=idx),            # blocklisted glitch
              "ASML": pd.Series(1.0, index=idx),              # longer history → kept
              "ASML.AS": pd.Series(1.0, index=idx[:2])}       # same company, shorter → dropped
    meta = pd.DataFrame({"ticker": ["SSU.JO", "ASML", "ASML.AS"],
                         "name": ["Southern Sun", "ASML (ADR)", "ASML"]})
    out = prune_shares(shares, meta)
    assert "SSU.JO" not in out                                # glitch removed
    assert "ASML" in out and "ASML.AS" not in out            # same company collapsed to one

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


def test_candidate_pool_ranks_by_turnover_when_given():
    """Flat slippage columns made spread order meaningless — with a turnover panel
    the pool is the largest names by trailing median EUR turnover."""
    idx = pd.bdate_range("2024-01-01", periods=200)
    turn = pd.DataFrame({"A.F": 1e5, "C.F": 9e6, "D.F": 5e6}, index=idx)
    pool = candidate_pool(META, {"A.F", "B.F", "C.F", "D.F"}, max_names=2,
                          liq_max=30, turnover=turn)
    assert pool == ["C.F", "D.F"]                      # biggest turnover, A.F outranked
    # uncovered names rank at zero turnover but stay eligible below the cap
    pool3 = candidate_pool(META, {"A.F", "B.F", "C.F", "D.F"}, max_names=3,
                           liq_max=30, turnover=turn)
    assert pool3 == ["C.F", "D.F", "A.F"]


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
