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


def test_section_is_empty_in_the_public_build():
    import build_report as B
    _roi, _bms, av = pa.build_roi_timeseries(BUYS)
    assert B.sec_asset_value({"asset_values": av}, public=True) == ""


def test_section_is_empty_without_curve_data():
    import build_report as B
    assert B.sec_asset_value({"asset_values": {}}, public=False) == ""
    assert B.sec_asset_value({}, public=False) == ""


def test_section_renders_a_trace_per_asset_plus_total():
    import build_report as B
    txns = BUYS + [_txn("2026-04-01", "BBB.F", "sell", 20.0, 50.0)]
    _roi, _bms, av = pa.build_roi_timeseries(txns)
    html = B.sec_asset_value({"asset_values": av}, public=False)
    assert "<h2>" in html
    assert "Total portfolio" in html
    assert "Cash from sells" in html
    assert "AAA.F" in html and "BBB.F" in html


def test_overlay_renders_one_mode_per_figure():
    """% and € are separate figures behind separate tabs, not one figure with a toggle:
    log would break on the % view's negative returns, and cash has no return at all,
    so it belongs only to the € figure.
    """
    import build_report as B
    txns = BUYS + [_txn("2026-04-01", "BBB.F", "sell", 20.0, 50.0)]
    _roi, _bms, av = pa.build_roi_timeseries(txns)

    pct = B._asset_value_figure(av, "pct")
    assert pct.layout.yaxis.ticksuffix == "%" and pct.layout.yaxis.type != "log"
    assert "Cash from sells" not in [t.name for t in pct.data]
    assert pct.layout.legend.x > 1.0               # legend sits outside the plot area

    eur = B._asset_value_figure(av, "eur")
    assert eur.layout.yaxis.tickprefix == "€" and eur.layout.yaxis.type == "log"
    assert "Cash from sells" in [t.name for t in eur.data]
    # same positions in both, so the tabs can't disagree about what you hold
    assert ([t.name for t in pct.data if t.name != "Cash from sells"]
            == [t.name for t in eur.data if t.name != "Cash from sells"])


def test_tab_strip_labels_and_hides_all_but_the_first_pane():
    import build_report as B
    html = B._asset_tabs([("A", "<i>one</i>"), ("B", "<i>two</i>"), ("C", "<i>three</i>")])
    assert html.count("<button class='apv-tab") == 3
    assert "<button class='apv-tab on'" in html     # first tab starts selected
    assert ">A</button>" in html and ">C</button>" in html
    assert html.count("<div class='apv-pane' hidden>") == 2
    assert "Plotly.Plots.resize" in html           # a hidden pane measures zero width


def test_selector_shows_one_position_at_a_time_over_a_persistent_ghost():
    """The big view is a dropdown over a single trace. Pin that exactly one position
    starts visible and that every dropdown entry keeps the ghost on — a visible-array
    off by one would either blank the chart or stack all 14 lines back on top of
    each other, which is the tangle these views exist to escape.
    """
    import build_report as B
    _roi, _bms, av = pa.build_roi_timeseries(BUYS)
    fig = B._asset_selector_figure(av)
    n = len(BUYS)
    assert len(fig.data) == n + 1                          # ghost + one per position
    assert [t.visible for t in fig.data].count(False) == n - 1
    picker = fig.layout.updatemenus[0]
    assert picker.type == "dropdown" and len(picker.buttons) == n
    for b in picker.buttons:
        vis = b.args[0]["visible"]
        assert vis[0] is True and sum(bool(v) for v in vis) == 2   # ghost + exactly one


def test_selector_is_skipped_when_there_is_nothing_to_show():
    import build_report as B
    assert B._asset_selector_figure({}) is None
    assert B._asset_selector_figure({"__roi__": {}}) is None


def test_facets_give_each_position_its_own_panel_with_a_ghost_benchmark():
    """14 positions is double the ~7 hues a categorical palette can carry, so the
    overlay tangles no matter how it's coloured. The facet grid is the fix: pin that
    every position gets a panel and that the portfolio ghost is repeated in each one,
    since that ghost is what makes "did this beat my average" a local comparison.
    """
    import build_report as B
    _roi, _bms, av = pa.build_roi_timeseries(BUYS)
    fig = B._asset_facets_figure(av, cols=4)
    assert len(fig.data) == 2 * len(BUYS)          # one ghost + one line per position
    assert all(t.showlegend is False for t in fig.data)   # identity comes from titles
    titles = [a.text for a in fig.layout.annotations]
    assert any("AAA" in t for t in titles) and any("BBB" in t for t in titles)
    # panels share one y-axis, or a flat position would look as dramatic as a doubling
    assert fig.layout.yaxis2.matches == "y" or fig.layout.yaxis.ticksuffix == "%"


def test_facets_are_skipped_when_there_is_nothing_to_facet():
    import build_report as B
    assert B._asset_facets_figure({}) is None
    assert B._asset_facets_figure({"__roi__": {}}) is None


def test_per_asset_roi_uses_the_portfolio_formula_not_a_rebased_curve():
    """Topping up a position must not read as a gain. AAA is bought twice at the same
    price with a flat price series, so its EUR line doubles while its return stays 0%.
    A naive value/first-value rebase would report +100%.
    """
    txns = [_txn("2026-02-02", "AAA.F", "buy", 10.0, 100.0),
            _txn("2026-03-02", "AAA.F", "buy", 10.0, 100.0),
            _txn("2026-03-02", "BBB.F", "buy", 20.0, 50.0)]
    _roi, _bms, av = pa.build_roi_timeseries(txns)
    assert av["AAA.F"].loc["2026-02-02"] == pytest.approx(1000.0)
    assert av["AAA.F"].loc["2026-03-02"] == pytest.approx(2000.0)   # EUR doubles
    aaa_roi = av["__roi__"]["AAA.F"]
    assert aaa_roi.loc["2026-02-02"] == pytest.approx(0.0)
    assert aaa_roi.loc["2026-03-02"] == pytest.approx(0.0)          # return does not


def test_roi_series_share_the_euro_series_gaps():
    """The % view swaps y-arrays on the same traces, so the two must align exactly —
    same index, same NaN gaps, or a sold position's line would outlive its own data.
    """
    txns = BUYS + [_txn("2026-04-01", "BBB.F", "sell", 20.0, 50.0)]
    _roi, _bms, av = pa.build_roi_timeseries(txns)
    for tk in ("AAA.F", "BBB.F"):
        assert av["__roi__"][tk].index.equals(av[tk].index)
        assert av["__roi__"][tk].isna().equals(av[tk].isna())
    assert av["__roi__"]["__total__"].index.equals(av["__total__"].index)


def test_return_arity_is_pinned():
    """build_strategy_report.py unpacks this into 3 names inside a broad
    `except Exception`, so an arity mismatch there would silently drop the
    /strategy portfolio comparison instead of raising. Pin the contract here,
    on the cheapest possible input, so a regression fails loudly in this
    fast/no-network test instead of silently on that page.
    """
    roi, bms, av = pa.build_roi_timeseries([])   # no buys -> returns before any download
    assert isinstance(roi, pd.Series) and roi.empty
    assert bms == {}
    assert av == {}
