"""Per-asset EUR value curves emitted by build_roi_timeseries.

No network: yfinance is monkeypatched with flat synthetic prices, so every
position value is exactly shares x a known constant.
"""

import pandas as pd
import pytest

import tools.portfolio_analytics as pa

PRICES = {"AAA.F": 100.0, "BBB.F": 50.0}


def _fake_download(tickers, start=None, auto_adjust=True, progress=False, **kw):
    """Stand in for yf.download: flat prices for AAA.F/BBB.F, nothing else.

    Returns the MultiIndex ('Close', ticker) column layout real yfinance uses
    for multi-ticker downloads. Unknown tickers (the benchmarks, EURUSD=X)
    produce an empty frame, so no benchmark series are built.
    """
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    known = [t for t in tickers if t in PRICES]
    if not known:
        return pd.DataFrame()
    idx = pd.bdate_range(start="2026-01-01", end=pd.Timestamp.today().normalize())
    cols = pd.MultiIndex.from_product([["Close"], known])
    data = {("Close", t): [PRICES[t]] * len(idx) for t in known}
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(pa.yf, "download", _fake_download)


def _txn(date, ticker, action, shares, pps):
    return {"date": date, "ticker": ticker, "action": action,
            "shares": shares, "price": shares * pps, "pps": pps}


# Two tickers in every fixture: yfinance's single-ticker download has a
# different column shape, and the production path is always multi-ticker.
BUYS = [
    _txn("2026-02-02", "AAA.F", "buy", 10.0, 100.0),   # 1000 EUR
    _txn("2026-03-02", "BBB.F", "buy", 20.0, 50.0),    # 1000 EUR
]


def test_asset_line_starts_on_its_buy_date():
    _roi, _bms, av = pa.build_roi_timeseries(BUYS)
    bbb = av["BBB.F"]
    assert bbb.loc[:"2026-02-27"].dropna().empty       # nothing before the buy
    assert bbb.loc["2026-03-02"] == pytest.approx(1000.0)
    assert av["AAA.F"].loc["2026-02-02"] == pytest.approx(1000.0)


def test_sold_asset_line_ends_at_the_sell():
    txns = BUYS + [_txn("2026-04-01", "BBB.F", "sell", 20.0, 50.0)]
    _roi, _bms, av = pa.build_roi_timeseries(txns)
    assert av["BBB.F"].loc["2026-03-02"] == pytest.approx(1000.0)
    assert av["BBB.F"].loc["2026-04-01":].dropna().empty
    # proceeds move into the cash trace, which is NaN until the first sell
    cash = av["__cash__"]
    assert cash.loc[:"2026-03-31"].dropna().empty
    assert cash.loc["2026-04-01"] == pytest.approx(1000.0)


def test_no_cash_trace_when_there_are_no_sells():
    _roi, _bms, av = pa.build_roi_timeseries(BUYS)
    assert "__cash__" not in av


def test_total_reconciles_to_assets_plus_cash():
    txns = BUYS + [_txn("2026-04-01", "BBB.F", "sell", 20.0, 50.0)]
    _roi, _bms, av = pa.build_roi_timeseries(txns)
    total = av["__total__"]
    per_asset = pd.DataFrame({k: v for k, v in av.items() if not k.startswith("__")})
    recon = per_asset.sum(axis=1, skipna=True) + av["__cash__"].fillna(0.0)
    pd.testing.assert_series_equal(recon, total, check_names=False, atol=1e-6)


def test_weekend_buy_lands_on_next_business_day():
    # 2026-02-01 is a Sunday (Tradegate trades are dated like this).
    txns = [_txn("2026-02-01", "AAA.F", "buy", 10.0, 100.0),
            _txn("2026-03-02", "BBB.F", "buy", 20.0, 50.0)]
    _roi, _bms, av = pa.build_roi_timeseries(txns)
    aaa = av["AAA.F"].dropna()
    assert aaa.index[0] == pd.Timestamp("2026-02-02")   # the Monday
    assert aaa.iloc[0] == pytest.approx(1000.0)


def test_roi_series_is_unchanged_by_the_new_return_value():
    roi, bms, av = pa.build_roi_timeseries(BUYS)
    assert isinstance(roi, pd.Series) and not roi.empty
    assert bms == {}                       # fake download yields no benchmarks
    assert set(av) >= {"AAA.F", "BBB.F", "__total__"}
    # flat prices => ROI is exactly 0% throughout
    assert roi.abs().max() == pytest.approx(0.0)
