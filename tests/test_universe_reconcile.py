"""Ledger semantics: no previously-live name is ever silently dropped by a refresh."""
import pandas as pd

from tools.universe_reconcile import reconcile

ASOF = pd.Timestamp("2026-07-01")
COLS = ["ticker", "name", "isin", "delisting_date", "exit_reason", "med_turnover"]


def _meta(rows):
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLS]


def _px(last, n=60):
    return pd.Series(2.0, index=pd.bdate_range(end=last, periods=n))


def _run(prev_rows, prev_px, new_rows, new_px, tr_isins, fetched_ok):
    return reconcile(_meta(prev_rows), pd.DataFrame(prev_px),
                     _meta(new_rows), pd.DataFrame(new_px),
                     tr_isins=tr_isins, fetched_ok=fetched_ok, asof=ASOF)


def test_classification_delisted_removed_demoted_carried():
    prev = [dict(ticker=t, isin=f"I{t}") for t in ("A", "B", "C", "D", "K")]
    ppx = {"A": _px("2026-05-01"), "B": _px("2026-06-30"), "C": _px("2026-06-30"),
           "D": _px("2026-06-30"), "K": _px("2026-06-30")}
    new = [dict(ticker="K", isin="IK", med_turnover=9.9)]
    m, p, rep = _run(prev, ppx, new, {"K": _px("2026-06-30")},
                     tr_isins={"IA", "IC", "ID", "IK"}, fetched_ok={"A", "C", "K"})
    r = m.set_index("ticker")
    assert rep["delisted"] == ["A"] and r.loc["A", "exit_reason"] == "delisted"
    assert str(r.loc["A", "delisting_date"]) == "2026-05-01"      # last print
    assert rep["removed"] == ["B"] and r.loc["B", "exit_reason"] == "removed"
    assert rep["demoted"] == ["C"] and r.loc["C", "exit_reason"] == "demoted"
    assert rep["carried"] == ["D"] and pd.isna(r.loc["D", "delisting_date"])
    assert set(p.columns) == {"A", "B", "C", "D", "K"}            # nobody vanished
    assert r.loc["K", "med_turnover"] == 9.9                      # fresh row wins


def test_stale_new_live_reclassified_dead():
    new = [dict(ticker="E", isin="IE")]
    m, p, rep = _run([], {}, new, {"E": _px("2026-05-20")}, {"IE"}, {"E"})
    r = m.set_index("ticker")
    assert rep["stale_new"] == ["E"] and r.loc["E", "exit_reason"] == "delisted"
    assert str(r.loc["E", "delisting_date"]) == "2026-05-20"


def test_dead_passthrough_and_legacy_backfill():
    prev = [dict(ticker="W", isin="IW", delisting_date="2020-06-26")]   # no exit_reason
    m, p, rep = _run(prev, {"W": _px("2020-06-26")}, [], {}, set(), set())
    r = m.set_index("ticker")
    assert r.loc["W", "exit_reason"] == "delisted"                # legacy = real death
    assert "W" in p.columns


def test_resurrection_clears_exit():
    prev = [dict(ticker="F", isin="IF", delisting_date="2026-03-31", exit_reason="demoted")]
    new = [dict(ticker="F", isin="IF")]
    m, p, rep = _run(prev, {"F": _px("2026-03-31")}, new, {"F": _px("2026-06-30")},
                     {"IF"}, {"F"})
    r = m.set_index("ticker")
    assert rep["resurrected"] == ["F"] and pd.isna(r.loc["F", "delisting_date"])
    assert len(m) == 1


def test_empty_prev_passthrough():
    new = [dict(ticker="G", isin="IG")]
    m, p, rep = _run([], {}, new, {"G": _px("2026-06-30")}, {"IG"}, {"G"})
    assert list(m["ticker"]) == ["G"] and all(not v for v in rep.values())
