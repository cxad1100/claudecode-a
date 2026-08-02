import json

import pandas as pd

from tools.megacap_global import fetch_global, save_global, load_global, _series


def test_fetch_global_converts_usd_to_eur():
    idx = pd.bdate_range("2017-01-02", periods=100)

    def gh(tk):
        if tk == "EURUSD=X":
            return pd.Series(1.10, index=idx)          # USD per 1 EUR
        return pd.Series(110.0, index=idx)             # USD price

    def gs(tk):
        return pd.Series(1e9, index=idx[:5])

    blob = fetch_global(get_hist=gh, get_shares=gs, tickers={"X": "X"})
    assert "X" in blob
    v = next(iter(blob["X"]["prices"].values()))
    assert abs(v - 100.0) < 1e-6                        # 110 USD / 1.10 = 100 EUR


def test_series_coerces_tz_aware_to_naive():
    s = _series({"2017-01-02T00:00:00+00:00": 1e9, "2017-04-01T00:00:00+00:00": 1.1e9})
    assert s.index.tz is None
    assert list(s.values) == [1e9, 1.1e9]


def test_load_global_reindexes_and_strips_tz(tmp_path):
    blob = {"X": {"prices": {"2017-01-02T00:00:00": 100.0, "2017-01-03T00:00:00": 101.0},
                  "shares": {"2017-01-02T00:00:00+00:00": 1e9}}}   # tz-aware shares
    p = tmp_path / "g.json"
    p.write_text(json.dumps(blob))
    idx = pd.bdate_range("2017-01-02", periods=5)
    gpx, gsh, gslip = load_global(idx, path=p)
    assert "X" in gpx.columns and list(gpx.index) == list(idx)
    assert gsh["X"].index.tz is None                   # the tz bug that crashed cap_panel
    assert gslip["X"] > 0


def test_load_global_absent_is_empty(tmp_path):
    gpx, gsh, gslip = load_global(pd.bdate_range("2020-01-01", periods=3),
                                  path=tmp_path / "missing.json")
    assert gpx.empty and gsh == {} and gslip == {}
