"""Regime attribution (tools.regime_attr) — observational: is the Sharpe AI/Defense beta or
regime-timing in disguise? Pure pieces: split returns by a regime mask, and restrict the
universe by sector. Neither touches selection (re-runs use the SAME pinned config)."""
import numpy as np
import pandas as pd

from tools import regime_attr as ra


def test_conditional_performance_splits_by_mask():
    idx = pd.bdate_range("2020-01-01", periods=200)
    # first 100 days: calm up-drift (risk-ON); last 100: violent down (risk-OFF)
    r = np.concatenate([np.full(100, 0.002), np.random.default_rng(0).normal(-0.001, 0.03, 100)])
    rets = pd.Series(r, index=idx)
    mask = pd.Series([False] * 100 + [True] * 100, index=idx)   # True = risk-off
    out = ra.conditional_performance(rets, mask, ppy=252)
    assert out["on"]["n_days"] == 100 and out["off"]["n_days"] == 100
    assert out["on"]["sharpe"] > out["off"]["sharpe"]           # calm drift beats the storm
    assert out["on"]["ret_share"] > out["off"]["ret_share"]     # most gain came risk-on
    assert abs(out["on"]["ret_share"] + out["off"]["ret_share"] - 1.0) < 1e-6   # shares partition


def test_conditional_performance_all_one_regime():
    idx = pd.bdate_range("2020-01-01", periods=60)
    rets = pd.Series(np.full(60, 0.001), index=idx)
    mask = pd.Series(False, index=idx)                          # never risk-off
    out = ra.conditional_performance(rets, mask, ppy=252)
    assert out["off"]["n_days"] == 0                            # empty group is graceful
    assert out["on"]["n_days"] == 60 and out["on"]["ret_share"] == 1.0


def test_restrict_universe_drops_named_sectors():
    idx = pd.bdate_range("2020-01-01", periods=10)
    px = pd.DataFrame({c: np.arange(10, dtype=float) + 1 for c in ["A", "B", "C", "D"]}, index=idx)
    sectors = {"A": "Technology", "B": "Industrials", "C": "Healthcare", "D": "Technology"}
    kept = ra.restrict_universe(px, sectors, {"Technology", "Industrials"})
    assert list(kept.columns) == ["C"]                          # only the non-stripped sector left
    assert kept.shape[0] == 10                                  # rows untouched


def test_restrict_universe_missing_sector_is_kept():
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame({c: np.ones(5) for c in ["A", "B"]}, index=idx)
    kept = ra.restrict_universe(px, {"A": "Technology"}, {"Technology"})
    assert list(kept.columns) == ["B"]                          # B (no sector) survives the strip
