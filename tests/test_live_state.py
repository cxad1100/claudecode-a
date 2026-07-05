"""Today's-action arithmetic must match the vol_managed backtest exactly."""
import numpy as np
import pandas as pd

from tools import live_state as LS
from tools import vol_forecast as VF


def test_vol_action_cap_and_band():
    a = LS.vol_action(0.05, held_w=0.0, target_vol=0.15)   # calm → cap
    assert a["w_target"] == 1.0 and a["trade"] and a["eur"] == 10_000.0
    b = LS.vol_action(0.30, held_w=0.55, target_vol=0.15)  # target 0.50, |dw|=0.05
    assert not b["trade"] and "hold" in b["instruction"]
    c = LS.vol_action(0.30, held_w=0.65, target_vol=0.15)  # |dw|=0.15 > band
    assert c["trade"] and c["eur"] < 0 and "SELL" in c["instruction"]
    d = LS.vol_action(0.0, held_w=0.4)                     # degenerate forecast
    assert not d["trade"] and "no valid forecast" in d["instruction"]


def test_vol_action_band_edge_is_strict():
    # |dw| exactly == band → no trade, same strict `>` as vol_managed
    a = LS.vol_action(0.30, held_w=0.60, target_vol=0.15, band=0.10)
    assert abs(a["dw"] + 0.10) < 1e-12 and not a["trade"]


def test_sequential_actions_reproduce_vol_managed_exposure():
    """Applying vol_action day by day must walk the exact exposure path the
    backtest walked — the panel and the backtest are one rule."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=600)
    sig = np.where((np.arange(600) // 120) % 2 == 0, 0.008, 0.03)
    r = pd.Series(rng.normal(0, 1, 600) * sig, index=idx)
    f = VF.ewma_vol(r)
    vm = VF.vol_managed(r, f, target_vol=0.15, band=0.10, cost_bps=0.0, fee_eur=0.0)
    w_tgt = (0.15 / f).clip(upper=1.0).reindex(r.index).shift(1).fillna(0.0)
    held = 0.0
    path = []
    for t in r.index:
        # feed the same already-shifted target the backtest saw (invert w = tv/σ̂;
        # capped days invert to σ̂ = tv, which re-caps to exactly 1.0)
        a = LS.vol_action(forecast_today=(0.15 / w_tgt[t]) if w_tgt[t] > 0 else 0.0,
                          held_w=held, target_vol=0.15, band=0.10)
        if a["trade"]:
            held = a["w_target"]
        path.append(held)
    assert np.allclose(path, vm["exposure"].to_numpy(), atol=1e-12)


def test_state_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    LS.save_state(0.62, path=p)
    st = LS.load_state(p)
    assert st["held_w"] == 0.62 and st["updated"]
    assert LS.load_state(tmp_path / "missing.json")["held_w"] == 0.0


def test_sleeve_status_calendar():
    idx = pd.bdate_range("2016-01-01", "2026-03-01")
    inside = LS.sleeve_status(idx, today="2025-12-20")
    assert inside["in_window"] and inside["days_left"] > 0
    outside = LS.sleeve_status(idx, today="2025-07-05")
    assert not outside["in_window"] and 0 < outside["days_to_signal"] <= 200
    jan = LS.sleeve_status(idx, today="2026-01-15")
    assert jan["in_window"]
