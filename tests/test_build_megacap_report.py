import numpy as np
import pandas as pd
import build_megacap_report as MC
from tools.momentum import rebalance_dates


def _data():
    idx = pd.bdate_range("2021-01-01", periods=320)
    prices = pd.DataFrame({"BIG": np.linspace(10, 40, 320),
                           "SMALL": np.linspace(10, 12, 320)}, index=idx)
    dates = rebalance_dates(prices.index)
    cap = pd.DataFrame({"BIG": 5.0, "SMALL": 100.0}, index=dates)
    yoy = pd.DataFrame({"BIG": 0.9, "SMALL": 0.1}, index=dates)
    slip = {"BIG": 10, "SMALL": 10}
    return dict(prices=prices, cap=cap, yoy=yoy, slip=slip,
                benchmarks=pd.DataFrame(index=idx), capital=10_000.0,
                coverage={"candidates": 2, "covered": 2, "pct": 100.0})


def test_build_html_has_arms_and_grid():
    html = MC.build_html(_data())
    assert "size" in html.lower() and "growth" in html.lower() and "momentum" in html.lower()
    assert "N=25" in html or "N = 25" in html
    assert "100.0%" in html or "coverage" in html.lower()   # coverage surfaced


def test_topn_cutoff_is_the_nth_largest():
    idx = pd.bdate_range("2021-01-01", periods=3)
    cap = pd.DataFrame({"A": [100, 100, 100], "B": [50, 50, 50],
                        "C": [10, 10, 10]}, index=idx)
    assert MC._topn_cutoff(cap, 2).iloc[0] == 50.0        # 2nd largest
    assert MC._topn_cutoff(cap, 1).iloc[0] == 100.0


def test_sec_holdings_renders_latest_book_and_history():
    d = _data()
    res = MC.run_arms(d["prices"], d["slip"], d["cap"], d["yoy"],
                      n=MC.HEADLINE_N, k=MC.K, capital=d["capital"])
    h = MC.sec_holdings(res, "size", d["prices"], {"BIG": "Big Co"}, giants={"BIG"})
    assert "latest book" in h
    assert "period return" in h and "contribution" in h
    assert "Full holdings history" in h
    assert "BIG" in h and "non-US" in h                       # giant tag shown


def test_build_html_includes_holdings_section():
    html = MC.build_html(_data())
    assert "What the book actually held" in html
    assert "Full holdings history" in html


def test_survivorship_check_quantifies_cutoff_and_verdict():
    """The panel proves death-survivorship ≈ 0 for the cap screen by showing the
    entry cutoff, and points the real correction at the momentum family."""
    idx = pd.bdate_range("2021-01-01", periods=3)
    # 60 names so a top-50 cutoff exists; caps in the tens-of-billions range
    cap = pd.DataFrame({f"N{i}": [1e9 * (60 - i)] * 3 for i in range(60)}, index=idx)
    h = MC.sec_survivorship_check(cap)
    assert "Survivorship" in h and "top-25" in h and "€" in h and "bn" in h
    assert "momentum family" in h                         # points to where it DOES matter
    assert "equal-weight" in h                            # names the real residual
    assert MC.sec_survivorship_check(pd.DataFrame()) == ""   # graceful when empty
