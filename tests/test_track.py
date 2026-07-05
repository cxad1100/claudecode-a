"""Snapshot idempotency + kill-criteria triggers."""
import numpy as np
import pandas as pd

from tools import track as TK


def test_append_snapshot_idempotent(tmp_path):
    p = tmp_path / "track.csv"
    row = dict(forecast=0.18, w_target=0.83, w_held=0.80, r_live=0.001)
    assert TK.append_snapshot("2026-07-03", row, path=p)
    assert not TK.append_snapshot("2026-07-03", dict(row, r_live=9.9), path=p)
    assert TK.append_snapshot("2026-07-04", row, path=p)
    df = TK.read_track(p)
    assert len(df) == 2 and df.loc["2026-07-03", "r_live"] == 0.001


def _track(rets):
    idx = pd.bdate_range("2026-01-01", periods=len(rets))
    return pd.DataFrame(dict(r_live=rets), index=idx)


def test_not_enough_data():
    out = TK.live_vs_backtest(_track([0.001] * 30), sharpe_lo=0.2, backtest_max_dd=-0.3)
    assert not out["enough"] and not out["kill"] and out["needed"] == 63


def test_no_kill_on_healthy_series():
    rng = np.random.default_rng(1)
    rets = rng.normal(0.002, 0.008, 150)                  # strong live Sharpe
    out = TK.live_vs_backtest(_track(rets), sharpe_lo=0.2, backtest_max_dd=-0.30)
    assert out["enough"] and not out["kill"] and out["breach_streak"] < TK.PATIENCE


def test_kill_on_persistent_sharpe_breach():
    rng = np.random.default_rng(2)
    rets = np.concatenate([rng.normal(0.002, 0.008, 80),
                           rng.normal(-0.004, 0.008, 60)])  # long losing stretch
    out = TK.live_vs_backtest(_track(rets), sharpe_lo=0.5, backtest_max_dd=-0.90)
    assert out["kill"] and any("Sharpe" in r for r in out["reasons"])


def test_kill_on_drawdown_breach():
    rets = [0.001] * 70 + [-0.05] * 10                    # −40% crash
    out = TK.live_vs_backtest(_track(rets), sharpe_lo=-5.0, backtest_max_dd=-0.20)
    assert out["kill"] and any("drawdown" in r for r in out["reasons"])
