"""Observational regime diagnostics — a read-only lens for the strategy page.

Two signals, shown in parallel with the strategy and *never wired into selection or
sizing*:

  • `hmm_regime`  — a 2-state Gaussian Markov-switching model (statsmodels) on the
    benchmark's daily returns. The higher-variance state is tagged "risk-off". It is
    fit WALK-FORWARD (params frozen on a strictly-past expanding window at each refit
    date) and read with the FILTERED (causal) probabilities, so the label at day t uses
    only data ≤ t — no look-ahead, the same discipline as the momentum backtest.
  • `trend_state` — the daily 200d-MA trend signal the strategy already uses as its
    kill-switch (`momentum.trend_ok`), exposed as a series for the overlay / agreement.

The point of the page is the comparison: the HMM's risk-off state lands ~where the
benchmark is already below its 200d MA and where vol-targeting already de-risks — i.e.
it is largely redundant with the controls the strategy already runs. Pure functions.
"""
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def trend_state(benchmark: pd.Series, window: int = 200) -> pd.Series:
    """Daily bool: is the benchmark at/above its `window`-day simple MA? (The 200d
    trend kill-switch `momentum.trend_ok` uses, as a series.) Pre-warmup dates (no MA
    yet) are False."""
    b = benchmark.dropna()
    sma = b.rolling(window, min_periods=window).mean()
    state = b >= sma
    state[sma.isna()] = False
    return state.astype(bool)


def _fit_params(train: pd.Series, k: int):
    """Fit a k-state switching-variance Markov regression on `train`; return
    (params, risk_off_state) where risk_off_state = the highest-variance regime.
    Returns (None, None) on a failed/singular fit (the caller carries the last good
    params forward — the live benchmark feed has the odd gappy stretch)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = MarkovRegression(train, k_regimes=k, trend="c",
                                   switching_variance=True).fit(disp=False)
        sig = [float(res.params[f"sigma2[{i}]"]) for i in range(k)]
        return res.params, int(np.argmax(sig))
    except Exception:
        return None, None


def _filtered_off(seg: pd.Series, params, state: int, k: int) -> pd.Series:
    """Filtered (causal) probability of the risk-off `state` over `seg`, using frozen
    `params`. Filtering never peeks ahead: P(state_t) uses returns ≤ t only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fres = MarkovRegression(seg, k_regimes=k, trend="c",
                                switching_variance=True).filter(params)
    return fres.filtered_marginal_probabilities.iloc[:, state]


def hmm_regime(returns: pd.Series, refit_dates, k_regimes: int = 2,
               min_obs: int = 100, threshold: float = 0.5) -> pd.DataFrame:
    """Walk-forward HMM regime path on a daily return series.

    At each date in `refit_dates` (use the strategy's rebalance dates — cheap), fit the
    model on `returns.loc[:date]` (expanding, strictly past), freeze the params, and
    record the FILTERED risk-off probability for the step (date, next refit]. The label
    at any day depends only on returns up to that day and on a fit anchored no later
    than it ⇒ no look-ahead (pinned by `test_hmm_regime_no_lookahead`).

    Returns a daily frame {prob_risk_off, risk_off=prob>threshold} covering the span
    from the first usable refit onward (earlier dates have no fitted model yet, exactly
    like the backtest only starting once there is enough history)."""
    r = returns.dropna()
    dates = sorted(d for d in pd.to_datetime(list(refit_dates)) if d in r.index)
    params, state = None, None
    pieces = []
    for j, rd in enumerate(dates):
        train = r.loc[:rd]
        if len(train) >= min_obs:
            p, s = _fit_params(train, k_regimes)
            if p is not None:
                params, state = p, s
        if params is None:
            continue
        end = dates[j + 1] if j + 1 < len(dates) else r.index[-1]
        off = _filtered_off(r.loc[:end], params, state, k_regimes)
        step = off[(off.index > rd) & (off.index <= end)]
        pieces.append(step)
    if not pieces:
        return pd.DataFrame(columns=["prob_risk_off", "risk_off"])
    prob = pd.concat(pieces).sort_index()
    prob = prob[~prob.index.duplicated(keep="last")]
    return pd.DataFrame({"prob_risk_off": prob, "risk_off": prob > threshold})
