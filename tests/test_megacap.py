import numpy as np
import pandas as pd
from tools.megacap import cap_panel, yoy_growth_panel


def test_cap_panel_pit_no_future_leak():
    idx = pd.bdate_range("2022-01-03", periods=260)
    prices = pd.DataFrame({"A": np.linspace(10, 20, 260)}, index=idx)
    # shares become AVAILABLE only at 2022-06-01; a later bump at 2022-12-01
    shares = {"A": pd.Series({pd.Timestamp("2022-06-01"): 100.0,
                              pd.Timestamp("2022-12-01"): 200.0})}
    dates = [pd.Timestamp("2022-03-31"), pd.Timestamp("2022-07-29"),
             pd.Timestamp("2022-12-30")]
    cap = cap_panel(shares, prices, dates)
    assert np.isnan(cap.loc[dates[0], "A"])                    # before any shares avail
    px_jul = prices["A"].loc[:dates[1]].iloc[-1]
    assert cap.loc[dates[1], "A"] == 100.0 * px_jul            # uses 100, not future 200
    px_dec = prices["A"].loc[:dates[2]].iloc[-1]
    assert cap.loc[dates[2], "A"] == 200.0 * px_dec            # bump now available


def test_yoy_growth_pit_and_value():
    idx = pd.bdate_range("2022-01-03", periods=400)
    prices = pd.DataFrame({"A": np.ones(400)}, index=idx)      # price irrelevant here
    rev = {"A": pd.DataFrame(
        {"revenue": [500.0, 550.0], "avail": [pd.Timestamp("2022-04-28"),
                                              pd.Timestamp("2023-04-28")]},
        index=[pd.Timestamp("2022-03-31"), pd.Timestamp("2023-03-31")])}
    before = pd.Timestamp("2023-04-01")                        # 2023 Q1 not yet filed
    after = pd.Timestamp("2023-05-01")                         # 2023 Q1 filed
    g = yoy_growth_panel(rev, [before, after])
    assert np.isnan(g.loc[before, "A"])                        # no prior-year pair available yet
    assert abs(g.loc[after, "A"] - (550.0 / 500.0 - 1.0)) < 1e-9


from tools.megacap import megacap_screen, cap_scores_by_date, growth_scores_by_date


def _cap_frame():
    dates = [pd.Timestamp("2022-01-31"), pd.Timestamp("2022-02-28")]
    return pd.DataFrame({"BIG": [100.0, 100.0], "MID": [50.0, 50.0],
                         "SMALL": [10.0, np.nan]}, index=dates), dates


def test_megacap_screen_topn_excludes_nan():
    cap, dates = _cap_frame()
    scr = megacap_screen(cap, dates, n=2)
    assert scr[dates[0]] == {"BIG", "MID"}
    assert scr[dates[1]] == {"BIG", "MID"}          # SMALL is NaN in Feb -> excluded anyway


def test_megacap_screen_n1():
    cap, dates = _cap_frame()
    assert megacap_screen(cap, dates, n=1)[dates[0]] == {"BIG"}


def test_scores_by_date_shape():
    cap, dates = _cap_frame()
    sc = cap_scores_by_date(cap, dates)
    assert set(sc[dates[0]]) == {"raw", "voladj"}
    assert sc[dates[0]]["raw"].equals(sc[dates[0]]["voladj"])
    assert sc[dates[0]]["raw"]["BIG"] == 100.0
