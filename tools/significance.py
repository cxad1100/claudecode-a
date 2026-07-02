"""Significance & robustness for the momentum strategy — the tests a desk runs before
trusting a backtest, all pure (numbers in, numbers out; the report supplies the data).

Three questions:

1. **Is it better than noise?** (`monte_carlo_null`) Each rebalance, draw k *random*
   names from the SAME eligible pool, on the SAME dates — the only thing that changes is
   selection. Thousands of such random books give a null distribution of return/Sharpe;
   the p-value is where momentum lands. This automatically controls for the universe's own
   drift (the global small-cap pool ripped 2024–25): a random book harvests that drift too,
   so any excess is selection skill, not beta.

2. **Is the Sharpe real after we scanned 32 configs?** (`deflated_sharpe_ratio`) Picking the
   best of N strategies inflates Sharpe. Bailey & López de Prado's Deflated Sharpe Ratio
   haircuts the observed Sharpe for the number of trials, the return skew/kurtosis, and the
   sample length → P(true Sharpe > 0).

3. **How wide is the error bar?** (`bootstrap_sharpe_cagr_ci`) Circular block-bootstrap the
   per-rebalance returns → a confidence interval on annualised Sharpe and CAGR, so the
   headline is a range, not a falsely precise point.

Everything works in per-rebalance (gross) return space so the selection test is apples-to-
apples; costs hit momentum and a random book about equally, so gross isolates the signal.
"""
import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649015329


def period_pools(prices: pd.DataFrame, dates: list, elig_by_date: dict,
                 execute_lag: int = 1) -> list[np.ndarray]:
    """Per rebalance period, the realised holding-period return of EVERY eligible name —
    the pool the Monte-Carlo draws random books from. Mirrors the engine's execution
    (t+execute_lag entry/exit, ffill a held leg, bfill a leading-NaN entry)."""
    from tools.momentum import _exec_date
    pools = []
    for i in range(len(dates) - 1):
        d, nxt = dates[i], dates[i + 1]
        ed = _exec_date(prices.index, d, execute_lag)
        enxt = _exec_date(prices.index, nxt, execute_lag)
        elig = [t for t in elig_by_date.get(d, set()) if t in prices.columns]
        if not elig:
            pools.append(np.array([], float))
            continue
        seg = prices.loc[ed:enxt, elig].ffill().bfill()
        if len(seg) < 2:
            pools.append(np.array([], float))
            continue
        ret = (seg.iloc[-1] / seg.iloc[0] - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        pools.append(ret.to_numpy(float))
    return pools


def _sharpe(rets: np.ndarray, ppy: float) -> float:
    """Annualised Sharpe of a per-period return array (0 if degenerate)."""
    r = np.asarray(rets, float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def strategy_period_returns(holdings_log: list) -> np.ndarray:
    """Gross per-rebalance return of the actual strategy = equal-weight mean of its picks'
    realised returns each period (cash periods → 0)."""
    out = []
    for h in holdings_log:
        rv = [v for v in h.get("ret", {}).values() if v is not None and np.isfinite(v)]
        out.append(float(np.mean(rv)) if rv else 0.0)
    return np.asarray(out, float)


def monte_carlo_null(pools: list[np.ndarray], strat_rets: np.ndarray, k: int, *,
                     ppy: float = 4.0, n_trials: int = 1000, seed: int = 0) -> dict:
    """Random-selection null. `pools[i]` = realised returns of every eligible name over
    period i; `strat_rets[i]` = the strategy's gross return that period. Draw k names at
    random per period, n_trials times → null total-return and Sharpe distributions, and the
    p-value (incl. the observed, +1 smoothing) that random selection matches/beats momentum."""
    rng = np.random.default_rng(seed)
    strat_total = float(np.prod(1.0 + strat_rets) - 1.0)
    strat_sharpe = _sharpe(strat_rets, ppy)

    null_total = np.empty(n_trials)
    null_sharpe = np.empty(n_trials)
    for t in range(n_trials):
        prets = np.empty(len(pools))
        for i, arr in enumerate(pools):
            if len(arr) == 0:
                prets[i] = 0.0
            elif len(arr) <= k:
                prets[i] = arr.mean()
            else:
                prets[i] = arr[rng.choice(len(arr), size=k, replace=False)].mean()
        null_total[t] = np.prod(1.0 + prets) - 1.0
        null_sharpe[t] = _sharpe(prets, ppy)

    p_total = (np.sum(null_total >= strat_total) + 1) / (n_trials + 1)
    p_sharpe = (np.sum(null_sharpe >= strat_sharpe) + 1) / (n_trials + 1)
    return dict(strat_total=strat_total, strat_sharpe=strat_sharpe,
                null_total=null_total, null_sharpe=null_sharpe,
                p_total=float(p_total), p_sharpe=float(p_sharpe),
                null_total_median=float(np.median(null_total)),
                null_sharpe_median=float(np.median(null_sharpe)), n_trials=n_trials)


def deflated_sharpe_ratio(period_rets: np.ndarray, trial_sharpes_annual: list[float],
                          *, ppy: float = 4.0, n_trials_effective: int | None = None) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    `trial_sharpes_annual` = the annualised Sharpes of all configs we scanned (the 64-grid)
    — their dispersion sets the benchmark a lucky winner must clear. Returns the observed
    annualised Sharpe, the deflation benchmark, and DSR = P(true Sharpe > 0) after the
    multiple-testing + non-normality + length haircut.

    `n_trials_effective` addresses the *file-drawer* bias: the grid is only the trials we can
    SEE, but the pipeline (architectures, indicators, calendars) was iterated many times before
    it. Passing a larger effective N raises the expected-max deflation bar (Harvey's higher
    hurdle) while keeping the trial-Sharpe *dispersion* from the observed grid — the honest,
    strictly more-conservative estimate. None ⇒ N = the observed grid size (unchanged)."""
    r = np.asarray(period_rets, float)
    T = len(r)
    n_obs = max(len(trial_sharpes_annual), 1)
    sd = r.std(ddof=1) if T > 1 else 0.0
    if T < 3 or sd == 0:
        return dict(sharpe_annual=_sharpe(r, ppy), dsr=float("nan"),
                    sr_benchmark_annual=float("nan"), n_trials=n_obs,
                    n_trials_observed=n_obs, T=T)
    sr = r.mean() / sd                                   # per-period Sharpe
    g1 = float(stats.skew(r, bias=False))
    g2 = float(stats.kurtosis(r, fisher=False, bias=False))   # non-excess kurtosis
    sr_var = (1.0 - g1 * sr + (g2 - 1.0) / 4.0 * sr ** 2) / (T - 1)
    sr_se = np.sqrt(max(sr_var, 1e-12))

    # Effective trial count: the observed grid, unless a larger lifetime count is injected to
    # cover the file-drawer (unseen pipeline iterations). Dispersion always comes from the
    # OBSERVED grid — phantom trials borrow the same trial-to-trial Sharpe variance.
    N = max(int(n_trials_effective), 1) if n_trials_effective else n_obs
    trials_pp = np.asarray(trial_sharpes_annual, float) / np.sqrt(ppy)   # → per-period
    V = float(np.var(trials_pp, ddof=1)) if n_obs > 1 else 0.0
    sigma = np.sqrt(V) if V > 0 else sr_se
    # Expected max of N standard-normal Sharpes (Gumbel approx) × their dispersion.
    if N > 1:
        z1 = stats.norm.ppf(1.0 - 1.0 / N)
        z2 = stats.norm.ppf(1.0 - 1.0 / (N * np.e))
        sr_star = sigma * ((1.0 - EULER) * z1 + EULER * z2)
    else:
        sr_star = 0.0
    dsr = float(stats.norm.cdf((sr - sr_star) / sr_se))
    return dict(sharpe_annual=float(sr * np.sqrt(ppy)), dsr=dsr,
                sr_benchmark_annual=float(sr_star * np.sqrt(ppy)),
                skew=g1, kurtosis=g2, n_trials=N, n_trials_observed=n_obs, T=T)


def bootstrap_sharpe_cagr_ci(period_rets: np.ndarray, *, ppy: float = 4.0, n_boot: int = 2000,
                             block: int = 2, alpha: float = 0.05, seed: int = 0) -> dict:
    """Circular block-bootstrap → CI on annualised Sharpe and CAGR. Block length keeps any
    short-run autocorrelation; resamples blocks with wrap-around to the original length."""
    r = np.asarray(period_rets, float)
    T = len(r)
    if T < 3:
        s, c = _sharpe(r, ppy), float(np.prod(1.0 + r) ** (ppy / max(T, 1)) - 1.0)
        return dict(sharpe=s, sharpe_lo=s, sharpe_hi=s, cagr=c, cagr_lo=c, cagr_hi=c)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    sh, cg = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])[:T]
        rb = r[idx]
        sh[b] = _sharpe(rb, ppy)
        cg[b] = np.prod(1.0 + rb) ** (ppy / T) - 1.0
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return dict(sharpe=_sharpe(r, ppy),
                sharpe_lo=float(np.percentile(sh, lo)), sharpe_hi=float(np.percentile(sh, hi)),
                cagr=float(np.prod(1.0 + r) ** (ppy / T) - 1.0),
                cagr_lo=float(np.percentile(cg, lo)), cagr_hi=float(np.percentile(cg, hi)),
                conf=int(round((1 - alpha) * 100)))
