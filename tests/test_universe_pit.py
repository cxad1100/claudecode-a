import numpy as np
import pandas as pd

from tools.universe_pit import PITUniverse


def _frame():
    idx = pd.bdate_range("2020-01-01", periods=400)              # ~2020-01 .. 2021-07
    live = pd.Series(np.linspace(10, 20, 400), index=idx)        # survivor
    late = pd.Series([np.nan] * 100 + list(np.linspace(5, 8, 300)), index=idx)  # late IPO
    dead = pd.Series(list(np.linspace(30, 2, 200)) + [np.nan] * 200, index=idx) # dies mid-frame
    return pd.DataFrame({"LIVE": live, "LATE": late, "DEAD": dead}, index=idx)


def test_first_trade_date_is_first_non_nan():
    pit = PITUniverse(_frame())
    idx = pd.bdate_range("2020-01-01", periods=400)
    assert pit.first_trade_date("LIVE") == idx[0]
    assert pit.first_trade_date("LATE") == idx[100]


def test_delisting_date_only_for_dead():
    idx = pd.bdate_range("2020-01-01", periods=400)
    pit = PITUniverse(_frame(), delisting={"DEAD": idx[199]})
    assert pit.delisting_date("DEAD") == idx[199]
    assert pit.delisting_date("LIVE") is None


def test_listed_respects_first_trade_and_delisting():
    idx = pd.bdate_range("2020-01-01", periods=400)
    pit = PITUniverse(_frame(), delisting={"DEAD": idx[199]})
    assert pit.listed("LATE", idx[50]) is False        # before its first trade
    assert pit.listed("LATE", idx[150]) is True
    assert pit.listed("DEAD", idx[150]) is True         # alive before delisting
    assert pit.listed("DEAD", idx[250]) is False        # after delisting


def test_tradeable_excludes_dead_and_short_history():
    idx = pd.bdate_range("2020-01-01", periods=400)
    pit = PITUniverse(_frame(), delisting={"DEAD": idx[199]})
    elig = pit.tradeable(idx[300], min_history_days=120)
    assert "LIVE" in elig and "LATE" in elig            # both alive w/ >120 days
    assert "DEAD" not in elig                            # dead by idx[300]


def test_died_between_window():
    idx = pd.bdate_range("2020-01-01", periods=400)
    pit = PITUniverse(_frame(), delisting={"DEAD": idx[199]})
    assert pit.died_between(idx[150], idx[250]) == {"DEAD"}
    assert pit.died_between(idx[250], idx[300]) == set()


def test_last_price_at_or_before_asof():
    idx = pd.bdate_range("2020-01-01", periods=400)
    pit = PITUniverse(_frame(), delisting={"DEAD": idx[199]})
    assert pit.last_price("DEAD") == 2.0                       # final bar before death
    assert pit.last_price("LIVE", asof=idx[0]) == 10.0          # clipped to asof


# ── snapshot membership gate + deaths/exits split ────────────────────────────

def _mprices(*tickers, start="2026-01-02", n=200):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({t: 1.0 for t in tickers}, index=idx)


def _store(entries):
    return pd.DataFrame([dict(snapshot_date=d, isin=z, name="", country="")
                         for d, z in entries])


def test_membership_no_knowledge_before_first_snapshot():
    pit = PITUniverse(_mprices("AAA", "BBB"), {},
                      membership=(_store([("2026-06-01", "IA")]),
                                  {"AAA": "IA", "BBB": "IB"}))
    assert pit.listed("BBB", "2026-05-15")            # pre-snapshot: unrestricted
    assert pit.listed("AAA", "2026-06-15")
    assert not pit.listed("BBB", "2026-06-15")        # not offered at the snapshot


def test_membership_latest_snapshot_wins():
    store = _store([("2026-03-02", "IA"), ("2026-03-02", "IB"), ("2026-06-01", "IA")])
    pit = PITUniverse(_mprices("AAA", "BBB"), {},
                      membership=(store, {"AAA": "IA", "BBB": "IB"}))
    assert pit.listed("BBB", "2026-04-01")            # in the March snapshot
    assert not pit.listed("BBB", "2026-06-02")        # dropped by June's


def test_membership_unknown_isin_and_empty_store_allowed():
    pit = PITUniverse(_mprices("CCC"), {},
                      membership=(_store([("2026-03-02", "IA")]), {}))
    assert pit.listed("CCC", "2026-04-01")            # no ISIN → no membership claim
    pit2 = PITUniverse(_mprices("CCC"), {}, membership=(_store([]), {"CCC": "IC"}))
    assert pit2.listed("CCC", "2026-04-01")           # empty store → unrestricted


def test_deaths_subset_feeds_died_between_exits_still_gate():
    exits = {"AAA": pd.Timestamp("2026-03-02"), "BBB": pd.Timestamp("2026-03-02")}
    pit = PITUniverse(_mprices("AAA", "BBB"), exits, deaths={"AAA": exits["AAA"]})
    assert pit.died_between("2026-02-01", "2026-04-01") == {"AAA"}   # demoted ≠ died
    assert not pit.listed("BBB", "2026-03-03")                       # but exit gates
