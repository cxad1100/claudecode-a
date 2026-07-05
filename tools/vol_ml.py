"""Adaptive / learned volatility forecasting — parameters fitted, not fixed.

Three learners, all walk-forward with the same t → t+1 convention as tools.vol_forecast
(value at date t = annualised vol forecast for t+1, from data ≤ t; property-tested):

1. `adaptive_ewma_vol` — the RiskMetrics decay λ is a *predictive* parameter, so it is
   QMLE-fitted on the expanding window and refit quarterly instead of pinned at 0.94.
2. `ridge_vol` — a learned forecaster: walk-forward ridge regression of log next-week
   variance on vol features (HAR components + the EWMA state), where the ridge penalty
   itself is re-chosen at every refit by a chronological train/validation split —
   the model tunes its own hyperparameter as the regime changes.
3. `ensemble_vol` — online model selection (exponentially-weighted average / Hedge):
   every day, each component forecaster's realised QLIKE loss updates a discounted
   score, and the ensemble's weights shift toward whatever has been right *lately*.
   "Which model should I trust now" stops being a fixed choice and becomes a learned,
   regime-adaptive parameter with a no-regret guarantee from online learning theory.

Why no LSTM here: ~4,000 daily observations of a noisy non-stationary series is two
orders of magnitude short of where recurrent nets reliably beat GARCH/HAR out of
sample; the failure mode is silent overfitting that this repo's whole design exists to
prevent. The harness is model-agnostic — any challenger that emits the same forecast
Series can be dropped into the QLIKE table and judged identically.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

from tools.vol_forecast import (TD, MIN_TRAIN, REFIT_EVERY, _EPS, _har_design,
                                rolling_vol, ewma_vol, garch11_vol, har_rv_vol, qlike)

BASE_COMPONENTS = ("rolling", "ewma", "garch", "har")


# ───────────────────────── adaptive-λ EWMA ─────────────────────────

def fit_ewma_lambda(r: pd.Series | np.ndarray, seed_window: int = 63) -> dict:
    """QMLE for the RiskMetrics decay on a return window: minimise the Gaussian NLL of
    the one-step EWMA variance over λ ∈ (0.80, 0.999)."""
    x = np.asarray(r, float)
    x = x[np.isfinite(x)]
    if len(x) < seed_window + 50:
        return dict(lam=0.94, loglik=np.nan, converged=False)
    x2 = x ** 2
    s2_0 = float(np.var(x[:seed_window], ddof=1))
    if s2_0 <= 0:
        return dict(lam=0.94, loglik=np.nan, converged=False)

    tail = x2[seed_window:]

    def nll(lam):
        # s2 after absorbing each x2 via lfilter (C-speed); the loss at t uses the
        # PRIOR state — s2_0 first, then the filtered sequence shifted by one.
        y, _ = lfilter([1.0 - lam], [1.0, -lam], tail, zi=np.array([lam * s2_0]))
        prior = np.concatenate(([s2_0], y[:-1]))
        return 0.5 * float(np.sum(np.log(prior) + tail / prior))

    res = minimize_scalar(nll, bounds=(0.80, 0.999), method="bounded",
                          options=dict(xatol=1e-4))
    ok = bool(res.success)
    return dict(lam=float(res.x) if ok else 0.94,
                loglik=float(-res.fun) if ok else np.nan, converged=ok)


def adaptive_ewma_vol(r: pd.Series, refit_every: int = 63,
                      min_train: int = MIN_TRAIN, seed_window: int = 63) -> pd.Series:
    """EWMA whose λ is refit on the expanding window every `refit_every` obs and applied
    frozen until the next refit — the recursion only consumes past returns, so the
    walk-forward is look-ahead-free by construction."""
    x = r.to_numpy(float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < min_train + 1:
        return pd.Series(out, index=r.index)
    x2 = x ** 2
    s2_0 = float(np.var(x[:seed_window], ddof=1))
    lam = 0.94
    for i in range(min_train, n, refit_every):
        fit = fit_ewma_lambda(x[:i], seed_window)
        if fit["converged"]:
            lam = fit["lam"]
        end = min(i + refit_every, n)
        # rerun the recursion with frozen λ (lfilter = C-speed); s2 after absorbing
        # r_t IS the t+1 conditional variance
        y, _ = lfilter([1.0 - lam], [1.0, -lam], x2[seed_window:end],
                       zi=np.array([lam * s2_0]))
        out[i:end] = np.sqrt(TD * y[i - seed_window:end - seed_window])
    return pd.Series(out, index=r.index)


# ───────────────────────── learned ridge forecaster ─────────────────────────

def _ridge_beta(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with unpenalised intercept via centring."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + alpha * len(y) * np.eye(X.shape[1])
    b = np.linalg.solve(A, Xc.T @ yc)
    b0 = ym - xm @ b
    return np.concatenate(([b0], b))


def ridge_vol(r: pd.Series, refit_every: int = REFIT_EVERY, min_train: int = MIN_TRAIN,
              h: int = 5, alphas: tuple = (0.01, 0.1, 1.0, 10.0),
              val_frac: float = 0.2, lam: float = 0.94) -> pd.Series:
    """Walk-forward ridge on log-variance features [log r², log RV_w, log RV_m,
    log EWMA-state], target = log next-`h`-day mean variance (same target as HAR).
    At every refit the penalty α is re-chosen by a chronological split: fit each α on
    the first (1-val_frac) of the training rows, score on the last val_frac, keep the
    winner, refit on everything — the regularisation adapts to the regime instead of
    being a fixed constant. Log-normal ½·resid-var bias correction on the way out."""
    x = r.to_numpy(float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < min_train + 1:
        return pd.Series(out, index=r.index)
    x2 = x ** 2
    s2 = np.full(n, np.nan)                          # fixed-λ EWMA state as a feature
    seed = 63
    if n > seed:
        s2[seed - 1] = np.var(x[:seed], ddof=1)
        for t in range(seed, n):
            s2[t] = lam * s2[t - 1] + (1.0 - lam) * x2[t]
    X_base, y_all, _, _ = _har_design(x2, h)         # same features/target as har_rv_vol
    X_all = np.column_stack([X_base, np.log(s2 + _EPS)])
    valid_x = np.isfinite(X_all).all(axis=1)
    valid_row = valid_x & np.isfinite(y_all)
    beta, rvar = None, 0.0
    for i in range(min_train, n, refit_every):
        rows = np.where(valid_row[:max(i - h, 0)])[0]
        if len(rows) >= 150:
            cut = int(len(rows) * (1.0 - val_frac))
            tr, va = rows[:cut], rows[cut:]
            best_a, best_err = alphas[0], np.inf
            for a in alphas:
                b = _ridge_beta(X_all[tr], y_all[tr], a)
                err = float(np.mean((y_all[va] - (b[0] + X_all[va] @ b[1:])) ** 2))
                if err < best_err:
                    best_a, best_err = a, err
            b = _ridge_beta(X_all[rows], y_all[rows], best_a)
            resid = y_all[rows] - (b[0] + X_all[rows] @ b[1:])
            beta, rvar = b, float(np.var(resid, ddof=1))
        if beta is None:
            continue
        end = min(i + refit_every, n)
        for t in range(i, end):
            if not valid_x[t]:
                continue
            s2_hat = float(np.exp(beta[0] + X_all[t] @ beta[1:] + 0.5 * rvar)) - _EPS
            out[t] = np.sqrt(TD * max(s2_hat, _EPS))
    return pd.Series(out, index=r.index)


# ───────────────────────── online ensemble (Hedge) ─────────────────────────

def ensemble_vol(r: pd.Series, components: dict[str, pd.Series] | None = None,
                 eta: float = 10.0, gamma: float = 0.97, warmup: int = 63,
                 return_weights: bool = False):
    """Exponentially-weighted model averaging over vol forecasters.

    Each day t the previous forecast f_m(t-1) (made FOR t) meets the realised r_t²; its
    QLIKE loss updates a discounted score L_m ← γ·L_m + (1-γ)·loss (γ=0.97 ≈ 23-day
    half-life). Weights w_m(t) ∝ exp(-η·(L_m − min L)) then blend the components'
    forecasts for t+1. Everything at t uses information ≤ t — the weights are one day
    behind the losses by construction. Pass precomputed `components` to avoid refitting
    (the report does); default components are the four base forecasters."""
    r = r.dropna()
    if components is None:
        components = dict(rolling=rolling_vol(r), ewma=ewma_vol(r),
                          garch=garch11_vol(r), har=har_rv_vol(r))
    names = list(components)
    F = pd.concat([components[m].reindex(r.index) for m in names], axis=1)
    F.columns = names
    x2 = (r.to_numpy(float)) ** 2
    fvar = (F.to_numpy(float) ** 2) / TD                 # daily-variance forecasts
    n, M = len(r), len(names)
    L = np.zeros(M)
    seen = 0
    out = np.full(n, np.nan)
    Wlog = np.full((n, M), np.nan)
    w = np.full(M, 1.0 / M)
    for t in range(n):
        row_prev = fvar[t - 1] if t > 0 else np.full(M, np.nan)
        ok_prev = np.isfinite(row_prev) & (row_prev > 0)
        if ok_prev.all():                                 # score yesterday's forecasts
            loss = np.log(row_prev) + x2[t] / row_prev
            L = gamma * L + (1.0 - gamma) * loss
            seen += 1
        row = fvar[t]
        ok = np.isfinite(row) & (row > 0)
        if not ok.all():
            continue
        if seen >= warmup:
            z = np.exp(-eta * (L - L.min()))
            w = z / z.sum()
        else:
            w = np.full(M, 1.0 / M)
        Wlog[t] = w
        out[t] = np.sqrt(TD * float(w @ row))
    fc = pd.Series(out, index=r.index)
    if return_weights:
        return fc, pd.DataFrame(Wlog, index=r.index, columns=names)
    return fc
