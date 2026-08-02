import json

import pandas as pd

from tools.edgar import (cik_index, match_cik, norm_name, fetch_companyfacts,
                         parse_shares_history, parse_revenue_history)

RAW = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
       "1": {"cik_str": 1094517, "ticker": "TM", "title": "TOYOTA MOTOR CORP/"},
       "2": {"cik_str": 1000184, "ticker": "SAP", "title": "SAP SE"},
       "3": {"cik_str": 99999, "ticker": "DAI", "title": "Delaware Dynamics Inc"}}


def _idx():
    return cik_index(RAW)


def test_match_bare_us_ticker_direct():
    bt, bn = _idx()
    assert match_cik("AAPL", "Apple", bt, bn) == 320193
    assert match_cik("AAPL.US", "Apple", bt, bn) == 320193


def test_match_suffixed_ticker_never_by_symbol():
    """DAI.DE must NOT hit the unrelated US 'DAI' — suffixed tickers go by name."""
    bt, bn = _idx()
    assert match_cik("DAI.DE", "Mercedes-Benz Group", bt, bn) is None


def test_match_foreign_by_normalized_name():
    bt, bn = _idx()
    assert match_cik("7203.T", "Toyota Motor", bt, bn) == 1094517
    assert match_cik("SAP.DE", "SAP SE", bt, bn) == 1000184


def test_norm_name_strips_legal_forms():
    assert norm_name("TOYOTA MOTOR CORP/") == norm_name("Toyota Motor")
    assert norm_name("Meta Platforms (A)") == norm_name("Meta Platforms Inc")


def _fund(entries_by_tag):
    facts = {}
    for (tax, tag, unit), es in entries_by_tag.items():
        units = facts.setdefault(tax, {}).setdefault(tag, {"units": {}})["units"]
        units[unit] = es
    return {"facts": facts}


def test_fetch_companyfacts_injected_and_none_on_error():
    ok = json.dumps({"facts": {"dei": {}}})
    assert fetch_companyfacts(1, get_fn=lambda u: ok) == {"facts": {"dei": {}}}
    assert fetch_companyfacts(1, get_fn=lambda u: "Forbidden",
                              retry_delay=0.0) is None


def test_parse_shares_prefers_weighted_average_and_unions_taxonomies():
    """us-gaap era + ifrs era union into one series (the Toyota switch)."""
    fund = _fund({
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"):
            [{"val": 1.0e9, "end": "2019-12-31", "filed": "2020-02-01"}],
        ("ifrs-full", "WeightedAverageShares", "shares"):
            [{"val": 1.02e9, "end": "2021-12-31", "filed": "2022-02-01"}],
        ("dei", "EntityCommonStockSharesOutstanding", "shares"):
            [{"val": 5.0e8, "end": "2019-12-31", "filed": "2020-02-01"}],
    })
    s = parse_shares_history(fund)
    assert list(s.index) == [pd.Timestamp("2020-02-01"), pd.Timestamp("2022-02-01")]
    assert s.iloc[0] == 1.0e9                    # WA wins over the dei fallback


def test_parse_shares_split_normalized_to_latest_basis():
    """A 4:1 split (count jumps 4x) re-bases pre-split history; buyback drift
    (-2%) does not trigger."""
    fund = _fund({
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"): [
            {"val": 1.00e9, "end": "2019-12-31", "filed": "2020-02-01"},
            {"val": 0.98e9, "end": "2020-06-30", "filed": "2020-08-01"},
            {"val": 3.92e9, "end": "2020-12-31", "filed": "2021-02-01"},
        ]})
    s = parse_shares_history(fund)
    assert s.iloc[-1] == 3.92e9
    assert abs(s.iloc[0] - 4.0e9) < 1e6          # 1.0e9 x 4
    assert abs(s.iloc[1] - 3.92e9) < 1e6         # 0.98e9 x 4


def test_parse_shares_instant_prefix_rebased_onto_wa_basis():
    """Alphabet-shaped: WA totals only tagged recently; older instant rows covered
    one class (half the count). The prefix is kept and re-based onto the WA basis
    by the jump normalizer; instant rows overlapping the WA span are ignored."""
    fund = _fund({
        ("us-gaap", "CommonStockSharesOutstanding", "shares"): [
            {"val": 5.0e9, "end": "2019-12-31", "filed": "2020-02-01"},
            {"val": 5.1e9, "end": "2020-12-31", "filed": "2021-02-01"},
            {"val": 5.2e9, "end": "2024-12-31", "filed": "2025-02-01"},
        ],
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"): [
            {"val": 10.2e9, "end": "2024-12-31", "filed": "2025-02-01"},
            {"val": 10.0e9, "end": "2025-12-31", "filed": "2026-02-01"},
        ]})
    s = parse_shares_history(fund)
    assert len(s) == 4                                        # 2 prefix + 2 WA
    assert s.iloc[-1] == 10.0e9                               # trusted latest basis
    assert abs(s.iloc[0] - 10.0e9) < 0.3e9                    # 5.0e9 x2 rebase


def test_parse_shares_one_value_per_filing_latest_end():
    """A 10-K restating older years: the current period-end wins for that filed."""
    fund = _fund({
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"): [
            {"val": 9.0e8, "end": "2019-12-31", "filed": "2021-02-01"},
            {"val": 1.0e9, "end": "2020-12-31", "filed": "2021-02-01"},
        ]})
    s = parse_shares_history(fund)
    assert len(s) == 1 and s.iloc[0] == 1.0e9


def test_parse_revenue_quarterly_preferred_earliest_filed_wins():
    q = [{"val": 100.0 + i, "start": f"20{18 + i // 4}-{1 + 3 * (i % 4):02d}-01",
          "end": f"20{18 + i // 4}-{3 + 3 * (i % 4):02d}-28",
          "filed": f"20{18 + i // 4}-{4 + 3 * (i % 4) if i % 4 < 3 else 12:02d}-15"}
         for i in range(8)]
    # a later comparative refiling of the first quarter must NOT move its avail
    q.append(dict(q[0], filed="2020-04-15"))
    ann = [{"val": 460.0, "start": "2018-01-01", "end": "2018-12-28",
            "filed": "2019-02-15"}]
    fund = _fund({("us-gaap", "Revenues", "USD"): q + ann})
    df = parse_revenue_history(fund)
    assert len(df) == 8                                       # quarterly cadence, no annual mix
    assert df.loc[pd.Timestamp("2018-03-28"), "avail"] == pd.Timestamp("2018-04-15")


def test_parse_revenue_annual_fallback_and_majority_unit():
    """A 20-F filer: annual-only rows, and a stray USD row in a EUR series is
    dropped rather than corrupting growth."""
    ann = [{"val": 25.0e9 * (1.1 ** i), "start": f"20{17 + i}-01-01",
            "end": f"20{17 + i}-12-31", "filed": f"20{18 + i}-02-25"}
           for i in range(4)]
    fund = _fund({("ifrs-full", "Revenue", "EUR"): ann,
                  ("ifrs-full", "Revenue", "USD"):
                      [{"val": 30.0e9, "start": "2017-01-01", "end": "2017-12-31",
                        "filed": "2018-02-25"}]})
    df = parse_revenue_history(fund)
    assert len(df) == 4
    assert abs(df["revenue"].iloc[0] - 25.0e9) < 1.0          # EUR kept, USD dropped
    growth = df["revenue"].iloc[1] / df["revenue"].iloc[0] - 1.0
    assert abs(growth - 0.1) < 1e-9
