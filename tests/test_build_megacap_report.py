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
