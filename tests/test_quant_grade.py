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


def test_effective_bets_uncorrelated_many_bets():
    # k independent streams → many real bets (≫1), spread across factors
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=2000)
    k = 5
    rets = pd.DataFrame({f"A{i}": rng.normal(0, 0.01, 2000) for i in range(k)}, index=idx)
    m = Q.effective_bets(rets, np.full(k, 1.0 / k))
    assert m["k"] == k
    assert abs(m["n_eff_weight"] - k) < 1e-9          # equal weights → naive count = k
    assert m["n_eff_pca"] > 2.0                        # many independent bets, not one
    assert m["pc1_share"] < 0.5                        # no single factor dominates


def test_effective_bets_perfectly_correlated_is_one():
    # every name the SAME series → one factor, ~1 effective bet (naive count still says 5)
    idx = pd.bdate_range("2019-01-01", periods=500)
    base = np.random.default_rng(1).normal(0, 0.01, 500)
    rets = pd.DataFrame({f"A{i}": base for i in range(5)}, index=idx)
    m = Q.effective_bets(rets, np.full(5, 0.2))
    assert m["n_eff_pca"] < 1.05                        # collapses to a single bet
    assert m["pc1_share"] > 0.95                        # PC1 explains ~all variance
    assert abs(m["n_eff_weight"] - 5.0) < 1e-9          # weights spread → naive misses it


def test_effective_bets_weight_concentration():
    # n_eff_weight is the naive HHI of the weights, independent of the returns
    idx = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(2)
    rets = pd.DataFrame({"A": rng.normal(0, 0.01, 300),
                         "B": rng.normal(0, 0.01, 300)}, index=idx)
    m = Q.effective_bets(rets, np.array([0.9, 0.1]))
    assert abs(m["n_eff_weight"] - 1.0 / (0.81 + 0.01)) < 1e-9


def test_vol_target_turnover_cost_reduces_return():
    # vol swings → exposure (w) moves → turnover; charging it must drag net return
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2019-01-01", periods=600)
    r = np.concatenate([rng.normal(0.0005, 0.005, 150), rng.normal(0.0, 0.040, 150),
                        rng.normal(0.0005, 0.005, 150), rng.normal(0.0, 0.040, 150)])
    eq = pd.Series(1000 * np.cumprod(1 + r), index=idx)
    free = Q.vol_target(eq, target_vol=0.15, turn_cost_bps=0.0)
    costed = Q.vol_target(eq, target_vol=0.15, turn_cost_bps=25.0)
    assert costed["equity"].iloc[-1] < free["equity"].iloc[-1]      # resizing isn't free
    assert costed["ann_return"] < free["ann_return"]
    assert costed["turn_cost"] > 0                                  # reports the drag


def test_vol_target_constant_exposure_no_turnover_cost():
    # near-zero vol → target/realised pins w at the cap, constant → no resizing → no cost
    idx = pd.bdate_range("2019-01-01", periods=400)
    eq = pd.Series(1000 * np.cumprod(1 + np.full(400, 0.0003)), index=idx)
    a = Q.vol_target(eq, target_vol=0.15, turn_cost_bps=0.0)
    b = Q.vol_target(eq, target_vol=0.15, turn_cost_bps=50.0)
    assert abs(a["equity"].iloc[-1] - b["equity"].iloc[-1]) < 1e-6  # no turnover ⇒ cost is a no-op


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
