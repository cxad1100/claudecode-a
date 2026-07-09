import json
import pandas as pd
from tools.eodhd import (fetch_fundamentals, parse_shares_history,
                         parse_revenue_history)

FUND = {
    "SharesStats": {"SharesOutstanding": 30_000_000},
    "outstandingShares": {"quarterly": {
        "0": {"dateFormatted": "2023-03-31", "shares": 30_000_000},
        "1": {"dateFormatted": "2022-03-31", "shares": 28_000_000},
    }},
    "Financials": {"Income_Statement": {"quarterly": {
        "2023-03-31": {"date": "2023-03-31", "totalRevenue": "550000000",
                       "filing_date": "2023-04-28"},
        "2022-03-31": {"date": "2022-03-31", "totalRevenue": "500000000"},
    }}},
}


def test_fetch_fundamentals_injected():
    got = fetch_fundamentals("MEDP.F", key="x", get_fn=lambda url: json.dumps(FUND))
    assert got["SharesStats"]["SharesOutstanding"] == 30_000_000


def test_fetch_fundamentals_empty_is_none():
    assert fetch_fundamentals("X.F", key="x", get_fn=lambda url: "{}") is None
    assert fetch_fundamentals("X.F", key="x", get_fn=lambda url: "null") is None


def test_parse_shares_history_lagged():
    s = parse_shares_history(FUND, lag_days=75)
    assert s.loc[pd.Timestamp("2023-03-31") + pd.Timedelta(days=75)] == 30_000_000
    assert list(s.sort_index().values) == [28_000_000, 30_000_000]


def test_parse_revenue_uses_filing_date_else_lag():
    df = parse_revenue_history(FUND, lag_days=75)
    assert df.loc[pd.Timestamp("2023-03-31"), "avail"] == pd.Timestamp("2023-04-28")
    assert df.loc[pd.Timestamp("2022-03-31"), "avail"] == \
        pd.Timestamp("2022-03-31") + pd.Timedelta(days=75)
    assert df.loc[pd.Timestamp("2023-03-31"), "revenue"] == 550_000_000.0
