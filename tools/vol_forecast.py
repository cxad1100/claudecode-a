"""Volatility forecasting + the vol-managed overlay — the "predict vol, not the mean" module.

The quant-finance thesis: daily return *means* are essentially unpredictable out-of-sample,
but *volatility* clusters and is strongly predictable. So instead of forecasting returns,
forecast tomorrow's vol and size exposure `w = min(target_vol / forecast, 1.0)` — de-risk
only, remainder in cash (Trade Republic has no margin). This is the documented
Moreira–Muir / vol-targeting effect, not novel alpha: it buys Sharpe and drawdown, usually
at the cost of total return.

Alignment convention (single source of truth for the whole module): every forecaster
returns a Series indexed by date t whose value is the ANNUALISED vol forecast **for day
t+1**, formed from returns **through t**. `vol_managed()` then applies `shift(1)`
internally — exactly like `quant_grade.vol_target()` — so day t+1's return is scaled by a
weight computed from information ≤ t. No forecaster ever reads index ≥ t+1.

Pure (numbers in, numbers out). GARCH(1,1) is a hand-rolled 2-parameter QMLE with
variance targeting (no `arch` dependency); HAR-RV runs on daily squared-return proxies
(no intraday data), which is legitimately noisier than the intraday-RV original.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter

from tools import quant_grade as qg

TD = 252              # trading days / yr
RM_LAMBDA = 0.94      # RiskMetrics daily decay
REFIT_EVERY = 21      # walk-forward refit cadence (GARCH / HAR), in obs
MIN_TRAIN = 756       # 3y of daily obs before the first model-based forecast
_EPS = 1e-12          # floor under squared returns before taking logs (HAR)


# ───────────────────────────── forecasters ─────────────────────────────

def rolling_vol(r: pd.Series, lookback: int = 63) -> pd.Series:
    """Trailing realised vol — the incumbent baseline (matches `vol_target`'s estimator).
    Value at t = std of r[t-lookback+1..t], annualised."""
    return r.rolling(lookback).std(ddof=1) * np.sqrt(TD)


def ewma_vol(r: pd.Series, lam: float = RM_LAMBDA, seed_window: int = 63) -> pd.Series:
    """RiskMetrics EWMA: s2_t = lam*s2_{t-1} + (1-lam)*r_t².  s2_t already IS the t+1
    conditional variance, so the value at t is √(TD·s2_t). Seeded with the first
    `seed_window` days' sample variance (placed at t = seed_window-1)."""
    x = r.to_numpy(float)
    n = len(x)
    if n < seed_window + 1:
        return pd.Series(np.nan, index=r.index)
    s2 = np.full(n, np.nan)
    s2[seed_window - 1] = np.var(x[:seed_window], ddof=1)
    for t in range(seed_window, n):
        s2[t] = lam * s2[t - 1] + (1.0 - lam) * x[t] ** 2
    return pd.Series(np.sqrt(TD * s2), index=r.index)


def _garch_path(x2: np.ndarray, omega: float, alpha: float, beta: float,
                s2_0: float) -> np.ndarray:
    """Conditional-variance path s2[t] = omega + alpha*x2[t-1] + beta*s2[t-1], s2[0]=s2_0.
    Vectorised via lfilter (the recursion is an AR(1) filter), C-speed."""
    n = len(x2)
    u = omega + alpha * np.concatenate(([0.0], x2[:-1]))   # u[t] uses x2[t-1]
    y, _ = lfilter([1.0], [1.0, -beta], u[1:], zi=np.array([beta * s2_0]))
    return np.concatenate(([s2_0], y))


def garch11_fit(r: pd.Series | np.ndarray) -> dict:
    """GARCH(1,1) Gaussian QMLE with variance targeting: omega is pinned to
    s2_unc·(1-alpha-beta), so only (alpha, beta) are optimised (L-BFGS-B) under
    alpha+beta < 0.999. Zero-mean assumption (standard for daily returns; keeps the
    walk-forward free of any full-sample mean). Gaussian QMLE stays consistent for the
    vol path under fat tails."""
    x = np.asarray(r, float)
    x = x[np.isfinite(x)]
    s2_unc = float(np.var(x, ddof=1))
    if len(x) < 100 or s2_unc <= 0:
        return dict(alpha=0.08, beta=0.90, omega=max(s2_unc, _EPS) * 0.02,
                    persistence=0.98, loglik=np.nan, converged=False)
    x2 = x ** 2

    def nll(p):
        a, b = p
        if a + b >= 0.999:
            return 1e12 * (a + b)
        omega = s2_unc * (1.0 - a - b)
        s2 = _garch_path(x2, omega, a, b, s2_unc)
        if not np.all(s2 > 0):
            return 1e12
        return 0.5 * float(np.sum(np.log(s2) + x2 / s2))

    res = minimize(nll, x0=(0.08, 0.90), method="L-BFGS-B",
                   bounds=[(1e-6, 0.3), (0.5, 0.998)])
    a, b = (res.x if res.success else (0.08, 0.90))
    if a + b >= 0.999:                                   # penalty region → fall back
        a, b = 0.08, 0.90
    return dict(alpha=float(a), beta=float(b), omega=float(s2_unc * (1 - a - b)),
                persistence=float(a + b), loglik=float(-res.fun) if res.success else np.nan,
                converged=bool(res.success))


def garch11_vol(r: pd.Series, refit_every: int = REFIT_EVERY,
                min_train: int = MIN_TRAIN) -> pd.Series:
    """Walk-forward GARCH(1,1): refit on the expanding window every `refit_every` obs,
    apply the FROZEN params daily via the variance recursion (which only consumes past
    returns → no look-ahead). Value at t = √(TD·s2_{t+1}) with
    s2_{t+1} = omega + alpha·r_t² + beta·s2_t. A failed fit keeps the prior params."""
    x = r.to_numpy(float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < min_train + 1:
        return pd.Series(out, index=r.index)
    x2 = x ** 2
    params = None
    for i in range(min_train, n, refit_every):
        fit = garch11_fit(x[:i])
        if fit["converged"] or params is None:
            params = fit
        a, b = params["alpha"], params["beta"]
        s2_unc = float(np.var(x[:i], ddof=1))
        omega = s2_unc * (1.0 - a - b)
        end = min(i + refit_every, n)
        # recursion over the full history with the frozen params; forecasts for [i, end)
        s2 = _garch_path(x2[:end], omega, a, b, s2_unc)
        for t in range(i, end):
            out[t] = np.sqrt(TD * (omega + a * x2[t] + b * s2[t]))
    return pd.Series(out, index=r.index)


def _har_design(x2: np.ndarray, h: int = 5):
    """Shared HAR design (single source of truth for har_rv_vol AND vol_ml.ridge_vol —
    they must stay on the same features/target or their QLIKE comparison measures
    drift, not modelling): features at t = [log r²_t, log RV_w(5d), log RV_m(22d)],
    target y[t] = log mean(r²[t+1..t+h]). Returns (X, y, valid_x, valid_row)."""
    rv_w = pd.Series(x2).rolling(5).mean().to_numpy()
    rv_m = pd.Series(x2).rolling(22).mean().to_numpy()
    X = np.column_stack([np.log(x2 + _EPS), np.log(rv_w + _EPS), np.log(rv_m + _EPS)])
    y = np.log(pd.Series(x2).rolling(h).mean().shift(-h).to_numpy() + _EPS)
    valid_x = np.isfinite(X).all(axis=1)
    return X, y, valid_x, valid_x & np.isfinite(y)


def har_rv_vol(r: pd.Series, refit_every: int = REFIT_EVERY,
               min_train: int = MIN_TRAIN, h: int = 5) -> pd.Series:
    """Corsi HAR-RV from daily squared-return proxies (no intraday RV — noisier than the
    original, stated in the report). A single day's log r² is far too noisy a target
    (near-zero returns become huge negative outliers), so — standard for daily-proxy HAR
    — the regression targets the log of the NEXT-`h`-day mean variance, and the fitted
    value serves as the t+1 vol forecast (vol is persistent at that horizon). Regressors:
    log daily/weekly/monthly components; walk-forward OLS (np.linalg.lstsq) refit every
    `refit_every` obs; exponentiate with the ½·resid-var log-normal bias correction.
    At refit index i the training rows s ≤ i-1-h have targets known by i-1 → no look-ahead."""
    x2 = (r.to_numpy(float)) ** 2
    n = len(x2)
    out = np.full(n, np.nan)
    if n < min_train + 1:
        return pd.Series(out, index=r.index)
    X_base, y_all, valid_x, valid_row = _har_design(x2, h)
    X_all = np.column_stack([np.ones(n), X_base])
    beta, rvar = None, 0.0
    for i in range(min_train, n, refit_every):
        rows = np.where(valid_row[:max(i - h, 0)])[0]              # targets known by i-1
        if len(rows) >= 100:
            Xw, yw = X_all[rows], y_all[rows]
            b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            resid = yw - Xw @ b
            beta, rvar = b, float(np.var(resid, ddof=1))
        if beta is None:
            continue
        end = min(i + refit_every, n)
        for t in range(i, end):
            if not valid_x[t]:
                continue
            s2_hat = float(np.exp(X_all[t] @ beta + 0.5 * rvar)) - _EPS
            out[t] = np.sqrt(TD * max(s2_hat, _EPS))
    return pd.Series(out, index=r.index)


def parkinson_vol(high: pd.Series, low: pd.Series, lookback: int = 21) -> pd.Series:
    """Range-based realised vol (Parkinson): per-day variance = ln(H/L)²/(4·ln2), rolling
    mean over `lookback`, annualised. ~5× more efficient than squared close-close returns
    — a better *input/target* where OHLC exists (ETF/index); value at t uses days ≤ t."""
    pk = np.log(high / low) ** 2 / (4.0 * np.log(2.0))
    return np.sqrt(TD * pk.rolling(lookback).mean())


_METHODS = {"rolling": rolling_vol, "ewma": ewma_vol, "garch": garch11_vol, "har": har_rv_vol}
_ML_METHODS = ("adaptive_ewma", "ridge", "ensemble")     # live in tools.vol_ml (lazy import)


def forecast_vol(r: pd.Series, method: str = "ewma", **kw) -> pd.Series:
    """Dispatcher over the base forecasters {rolling, ewma, garch, har} plus the learned
    ones in tools.vol_ml {adaptive_ewma, ridge, ensemble} — all obey the t → t+1
    convention."""
    if method in _ML_METHODS:
        from tools import vol_ml                          # lazy: vol_ml imports this module
        fn = dict(adaptive_ewma=vol_ml.adaptive_ewma_vol, ridge=vol_ml.ridge_vol,
                  ensemble=vol_ml.ensemble_vol)[method]
        return fn(r, **kw)
    if method not in _METHODS:
        raise ValueError(f"unknown vol forecast method {method!r}; pick one of "
                         f"{sorted(_METHODS) + sorted(_ML_METHODS)}")
    return _METHODS[method](r, **kw)


# ─────────────────────────── forecast evaluation ───────────────────────────

def forward_realized_vol(r: pd.Series, horizon: int = 21) -> pd.Series:
    """Scoring target ONLY (never fed to a forecaster): value at t = realised vol over
    t+1..t+horizon, √(TD·mean r²) so horizon=1 is well-defined."""
    r2 = r ** 2
    return np.sqrt(TD * r2.rolling(horizon).mean().shift(-horizon))


def qlike(fvar: pd.Series, rvar: pd.Series) -> float:
    """Patton's robust QLIKE on VARIANCE: mean(log f + r/f). Lower is better; tolerates
    rvar = 0 (unlike the normalised form). The standard loss for vol-forecast ranking."""
    j = pd.concat([fvar, rvar], axis=1).dropna()
    f, rv = j.iloc[:, 0], j.iloc[:, 1]
    f = f[f > 0]
    rv = rv.loc[f.index]
    return float((np.log(f) + rv / f).mean())


def mse(forecast: pd.Series, realized: pd.Series) -> float:
    j = pd.concat([forecast, realized], axis=1).dropna()
    return float(((j.iloc[:, 0] - j.iloc[:, 1]) ** 2).mean())


def mincer_zarnowitz(forecast: pd.Series, realized: pd.Series) -> dict:
    """MZ regression realized = a + b·forecast: unbiased ⇒ a≈0, b≈1. Returns t-stats
    against those nulls."""
    j = pd.concat([forecast, realized], axis=1).dropna()
    if len(j) < 30:
        return {}
    res = stats.linregress(j.iloc[:, 0], j.iloc[:, 1])
    t_b1 = (res.slope - 1.0) / res.stderr if res.stderr else np.nan
    t_a0 = res.intercept / res.intercept_stderr if res.intercept_stderr else np.nan
    return dict(alpha=float(res.intercept), beta=float(res.slope), r2=float(res.rvalue ** 2),
                t_alpha0=float(t_a0), t_beta1=float(t_b1), n=len(j))


def oos_r2(pred: pd.Series, actual: pd.Series, horizon: int = 1) -> float:
    """Out-of-sample R² vs the expanding-mean benchmark (Campbell–Thompson style).
    The benchmark at t only uses actuals fully observable by t (shift by `horizon`,
    since `actual` at t is a t+1..t+h quantity)."""
    bench = actual.shift(horizon).expanding().mean()
    j = pd.concat([pred, actual, bench], axis=1).dropna()
    if len(j) < 30:
        return np.nan
    p, a, b = (j.iloc[:, k] for k in range(3))
    sse_p = float(((a - p) ** 2).sum())
    sse_b = float(((a - b) ** 2).sum())
    return 1.0 - sse_p / sse_b if sse_b > 0 else np.nan


def mean_null(r: pd.Series, refit_every: int = REFIT_EVERY, min_train: int = 252) -> dict:
    """The thesis contrast: forecast next-day *returns* (trailing means + walk-forward
    AR(1)) and report their OOS R² — ≈ 0 or negative, next to the vol forecasts' large
    positive OOS R². Benchmark = expanding mean of past returns."""
    x = r.to_numpy(float)
    n = len(x)
    actual = r.shift(-1)                                   # target: tomorrow's return
    bench = r.expanding().mean()                           # info ≤ t

    def _oos(pred: pd.Series) -> float:
        j = pd.concat([pred, actual, bench], axis=1).dropna()
        if len(j) < 30:
            return np.nan
        p, a, b = (j.iloc[:, k] for k in range(3))
        sse_b = float(((a - b) ** 2).sum())
        return 1.0 - float(((a - p) ** 2).sum()) / sse_b if sse_b > 0 else np.nan

    out = {f"mean_{lb}d": _oos(r.rolling(lb).mean()) for lb in (21, 252)}

    ar = np.full(n, np.nan)                                # walk-forward AR(1)
    phi, mu = 0.0, 0.0
    for i in range(min_train, n, refit_every):
        w = x[:i]
        mu = w.mean()
        d = w - mu
        denom = float((d[:-1] ** 2).sum())
        phi = float((d[1:] * d[:-1]).sum() / denom) if denom > 0 else 0.0
        end = min(i + refit_every, n)
        ar[i:end] = mu + phi * (x[i:end] - mu)
    out["ar1"] = _oos(pd.Series(ar, index=r.index))
    return out


# ───────────────────────────── the strategy ─────────────────────────────

def vol_managed(returns: pd.Series, forecast: pd.Series, target_vol: float = 0.15,
                cap: float = 1.0, band: float = 0.10, cost_bps: float = 5.0,
                fee_eur: float = 0.0, capital: float = 10_000.0,
                gate: pd.Series | None = None) -> dict:
    """Vol-managed overlay: exposure w = min(target_vol / forecast, cap), de-risk only,
    remainder in cash (earns 0). Generalises `quant_grade.vol_target()` to an externally
    supplied forecast — with band=0 and zero costs and forecast=rolling_vol(63) it
    reproduces it exactly.

    `forecast` follows the module convention (value at t = forecast for t+1); the shift
    happens HERE. `gate` (optional, e.g. a 200d-trend filter, info ≤ t, values 0/1)
    multiplies the target exposure before the shift.

    Rebalance band: hold the current exposure and only trade when |target − held| > band
    (path-dependent single pass) — kills the daily churn a raw daily-rebalance implies.
    Each adjustment is charged |Δw|·cost_bps/1e4 + fee_eur/capital against that day.
    """
    r = returns.dropna()
    w_tgt = (target_vol / forecast).clip(upper=cap)
    if gate is not None:
        w_tgt = w_tgt * gate.reindex(w_tgt.index)
    w_tgt = w_tgt.reindex(r.index).shift(1).fillna(0.0).clip(lower=0.0, upper=cap)

    rv = r.to_numpy(float)
    wt = w_tgt.to_numpy(float)
    n = len(rv)
    w_held, held = np.empty(n), 0.0
    net = np.empty(n)
    n_trades, turnover = 0, 0.0
    per_trade_fee = fee_eur / capital if capital else 0.0
    for t in range(n):
        dw = wt[t] - held
        cost = 0.0
        if abs(dw) > band:
            cost = abs(dw) * cost_bps / 1e4 + per_trade_fee
            held = wt[t]
            n_trades += 1
            turnover += abs(dw)
        w_held[t] = held
        net[t] = held * rv[t] - cost

    eq = pd.Series(np.cumprod(1.0 + net), index=r.index)
    years = max(len(r) / TD, 1e-9)
    m = qg.perf_metrics(eq)
    m.update(equity=eq, exposure=pd.Series(w_held, index=r.index),
             avg_exposure=float(w_held.mean()), target_vol=target_vol,
             n_trades_per_year=n_trades / years, turnover_per_year=turnover / years)
    return m
