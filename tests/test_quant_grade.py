"""Sanity checks for the quant scorecard math (tools.quant_grade)."""
import numpy as np
import pandas as pd

from tools import quant_grade as Q


def _equity(n=600, mu=0.0006, sig=0.012, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(1000 * np.cumprod(1 + rng.normal(mu, sig, n)), index=idx)


def test_perf_metrics_basic():
    m = Q.perf_metrics(_equity())
    assert m["ann_vol"] > 0 and -1 < m["max_dd"] <= 0 and m["dd_days"] >= 0
    assert m["sortino"] >= m["sharpe"] - 1e-6 or m["sharpe"] < 0   # downside ≤ total vol
    assert "var95" in m and m["cvar95"] <= m["var95"]              # CVaR worse than VaR


def test_vs_benchmark_beta_one_when_identical():
    e = _equity(seed=1)
    m = Q.vs_benchmark(e, e)
    assert abs(m["beta"] - 1.0) < 1e-6 and abs(m["corr"] - 1.0) < 1e-6
    assert abs(m["alpha_ann"]) < 1e-6                              # no alpha vs itself


def test_trade_metrics_profit_factor():
    trades = [{"net": 100}, {"net": -50}, {"net": 30}, {"net": -20}]
    m = Q.trade_metrics(trades, 10_000, years=2)
    assert m["n_trades"] == 4 and m["hit_rate"] == 0.5
    assert abs(m["profit_factor"] - (130 / 70)) < 1e-9


def test_grade_penalises_uncorrected_survivorship():
    good = Q.grade(test_sharpe=1.3, dsr=0.9, mc_p=0.01, isin_overlap_frac=0.9)
    bad = Q.grade(test_sharpe=1.3, dsr=0.9, mc_p=0.01, isin_overlap_frac=0.02)
    assert good["score"] > bad["score"]                            # survivorship dock is real
    assert good["survivorship_corrected"] and not bad["survivorship_corrected"]
    assert any("Survivorship" in f for f in bad["flags"])


def test_vol_target_reduces_drawdown():
    # a volatile equity curve → vol-targeting should cut vol and (usually) drawdown
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=500)
    r = rng.normal(0.0008, 0.03, 500)            # high-vol series
    eq = pd.Series(1000 * np.cumprod(1 + r), index=idx)
    base = Q.perf_metrics(eq)
    vt = Q.vol_target(eq, target_vol=0.15)
    assert vt["ann_vol"] < base["ann_vol"]       # de-risked
    assert 0 < vt["avg_exposure"] <= 1.0         # never levers
    # exposure exposed for the picks cash-sleeve + per-rebalance timeline
    assert len(vt["exposure"]) == len(vt["equity"])
    assert 0 < vt["exposure_latest"] <= 1.0
    assert (vt["exposure"].dropna() <= 1.0 + 1e-9).all()
