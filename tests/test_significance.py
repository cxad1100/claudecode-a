"""Pure stats checks for tools.significance — no network."""
import numpy as np
import pandas as pd

from tools.significance import (monte_carlo_null, deflated_sharpe_ratio,
                                bootstrap_sharpe_cagr_ci, strategy_period_returns,
                                period_pools, _sharpe)


def _pools(n_periods=30, pool=40, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 0.1, pool) for _ in range(n_periods)]


def test_skilled_selection_beats_noise():
    pools = _pools()
    strat = np.array([p.max() for p in pools])          # "perfect" selection each period
    out = monte_carlo_null(pools, strat, k=5, n_trials=500, seed=0)
    assert out["p_total"] < 0.01 and out["p_sharpe"] < 0.01
    assert out["strat_total"] > out["null_total_median"]


def test_random_selection_is_not_significant():
    pools = _pools()
    rng = np.random.default_rng(7)
    strat = np.array([p[rng.choice(len(p), 5, replace=False)].mean() for p in pools])
    out = monte_carlo_null(pools, strat, k=5, n_trials=500, seed=3)
    assert out["p_total"] > 0.05                          # a random book shouldn't look special


def test_deflated_sharpe_haircuts_for_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.03, 0.08, 40)                        # modest positive Sharpe
    trials = list(rng.normal(0.5, 0.6, 32))               # 32 scanned configs, dispersed
    out = deflated_sharpe_ratio(r, trials, ppy=4.0)
    assert 0.0 <= out["dsr"] <= 1.0
    assert out["sr_benchmark_annual"] > 0                 # benchmark raised by 32 trials
    assert out["n_trials"] == 32


def test_deflated_sharpe_phantom_trials_lowers_confidence():
    # "file-drawer" fix: raise the EFFECTIVE trial count above the 32 we can see (lifetime
    # pipeline iterations) → higher deflation bar → lower P(real>0). Dispersion stays from the
    # 32 observed configs; only N in the expected-max term grows.
    rng = np.random.default_rng(0)
    r = rng.normal(0.03, 0.08, 40)
    trials = list(rng.normal(0.5, 0.6, 32))
    base = deflated_sharpe_ratio(r, trials, ppy=4.0)                       # N = 32 observed
    phantom = deflated_sharpe_ratio(r, trials, ppy=4.0, n_trials_effective=32 * 10)   # ×10
    assert phantom["n_trials"] == 320 and phantom["n_trials_observed"] == 32
    assert phantom["sr_benchmark_annual"] > base["sr_benchmark_annual"]    # higher bar
    assert phantom["dsr"] < base["dsr"]                                    # honest, lower P
    assert 0.0 <= phantom["dsr"] <= 1.0


def test_deflated_sharpe_effective_none_is_baseline():
    rng = np.random.default_rng(1)
    r = rng.normal(0.03, 0.08, 40)
    trials = list(rng.normal(0.5, 0.6, 32))
    a = deflated_sharpe_ratio(r, trials, ppy=4.0)
    b = deflated_sharpe_ratio(r, trials, ppy=4.0, n_trials_effective=None)
    assert a["dsr"] == b["dsr"] and a["n_trials"] == b["n_trials"] == 32   # no override = unchanged


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(2)
    r = rng.normal(0.04, 0.07, 40)
    out = bootstrap_sharpe_cagr_ci(r, ppy=4.0, n_boot=500, seed=1)
    assert out["sharpe_lo"] <= out["sharpe"] <= out["sharpe_hi"]
    assert out["cagr_lo"] <= out["cagr"] <= out["cagr_hi"]


def test_strategy_period_returns_handles_cash():
    log = [{"ret": {"A": 0.1, "B": 0.2}}, {"ret": {}}, {"ret": {"C": -0.05}}]
    out = strategy_period_returns(log)
    assert np.isclose(out[0], 0.15) and out[1] == 0.0 and np.isclose(out[2], -0.05)


def test_period_pools_shapes():
    idx = pd.bdate_range("2020-01-01", periods=200)
    px = pd.DataFrame({f"T{i}": np.linspace(10, 20, 200) + i for i in range(6)}, index=idx)
    dates = [idx[20], idx[80], idx[140]]
    elig = {dates[0]: {"T0", "T1", "T2"}, dates[1]: {"T3", "T4"}}
    pools = period_pools(px, dates, elig, execute_lag=1)
    assert len(pools) == 2                                # n_dates - 1
    assert len(pools[0]) == 3 and len(pools[1]) == 2


def test_sharpe_zero_when_flat():
    assert _sharpe(np.zeros(10), 4.0) == 0.0
