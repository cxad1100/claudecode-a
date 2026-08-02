"""PIT event→signal layer: strict publication-time joins, insider tilt,
short-pressure (crowded + covering), small-cap interaction.

The one rule that matters: nothing published AT or AFTER the rebalance
instant may influence it — pit_slice is strictly `published_at < asof`,
and every score builder routes through it.
"""
import pandas as pd

from tools.event_signal import (insider_score_by_date, interact_small,
                                pit_slice, short_pressure_by_date)


def _dd(rows):
    return pd.DataFrame(rows)


def test_pit_slice_strictly_before():
    ev = _dd([dict(isin="DE0001", published_at="2026-06-01"),
              dict(isin="DE0002", published_at="2026-06-15"),
              dict(isin="DE0003", published_at="2026-03-01")])
    out = pit_slice(ev, pd.Timestamp("2026-06-15"))
    assert set(out["isin"]) == {"DE0001", "DE0003"}   # same-day excluded
    out2 = pit_slice(ev, pd.Timestamp("2026-06-16"), lookback_days=30)
    assert set(out2["isin"]) == {"DE0001", "DE0002"}  # window trims March


def test_insider_score_sign_decay_and_coverage():
    im = {"DE0001": "AAA", "DE0002": "BBB", "XX9999": "ZZZ"}
    ev = _dd([
        dict(isin="DE0001", side="buy", published_at="2026-06-20"),
        dict(isin="DE0002", side="buy", published_at="2026-04-05"),
        dict(isin="DE0001", side="sell", published_at="2026-01-10"),
        dict(isin="NOPE12", side="buy", published_at="2026-06-20"),
    ])
    dates = [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-02-01")]
    out = insider_score_by_date(ev, dates, im, window_days=180, halflife=30)
    assert set(out.keys()) == set(dates)
    d = dates[0]
    assert set(out[d].keys()) == {"raw", "voladj"}
    s = out[d]["raw"]
    assert s["AAA"] > 0                       # fresh buy dominates old sell
    assert 0 < s["BBB"] < s["AAA"]            # older buy decayed smaller
    assert "NOPE12" not in s.index and "ZZZ" not in s.index
    # at Feb 1 the June events don't exist yet
    s2 = out[dates[1]]["raw"]
    assert s2["AAA"] < 0                      # only the Jan sell is visible


def test_short_pressure_crowded_covering_and_pit():
    im = {"DE0001": "AAA"}
    sh = _dd([
        dict(isin="DE0001", holder="H1", pct=1.0, position_date="2026-01-10",
             published_at="2026-01-11"),
        dict(isin="DE0001", holder="H2", pct=0.6, position_date="2026-01-10",
             published_at="2026-01-11"),
        # H1 exits (below-threshold filing) in March
        dict(isin="DE0001", holder="H1", pct=0.4, position_date="2026-03-10",
             published_at="2026-03-11"),
    ])
    dates = [pd.Timestamp("2026-02-01"), pd.Timestamp("2026-04-01")]
    out = short_pressure_by_date(sh, dates, im, delta_days=60)
    crowd = out["crowded"]
    cover = out["covering"]
    assert crowd[dates[0]]["raw"]["AAA"] == 1.6        # both ≥0.5 count
    assert crowd[dates[1]]["raw"]["AAA"] == 0.6        # H1 out (<0.5)
    assert cover[dates[1]]["raw"]["AAA"] == 1.0        # 1.6 → 0.6 covered
    # PIT: on the exit's publication day the filing is NOT yet visible
    same_day = short_pressure_by_date(sh, [pd.Timestamp("2026-03-11")], im,
                                      delta_days=60)
    assert same_day["crowded"][pd.Timestamp("2026-03-11")]["raw"]["AAA"] == 1.6


def test_interact_small_keeps_bottom_turnover_tercile():
    idx = pd.date_range("2026-01-31", periods=8, freq="ME")
    turn = pd.DataFrame({"AAA": 1e6, "BBB": 5e5, "CCC": 1e5}, index=idx)
    s = pd.Series({"AAA": 1.0, "BBB": 1.0, "CCC": 1.0})
    out = interact_small(s, turn, pd.Timestamp("2026-08-31"), frac=1 / 3)
    assert list(out.index) == ["CCC"]          # smallest tercile only
