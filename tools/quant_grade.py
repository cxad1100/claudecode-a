"""Industry-grade metrics + an honest scorecard for the momentum strategy.

Pure (numbers in, numbers out). The point is not to flatter the backtest but to grade it
the way a risk committee would: standard ratios, factor/benchmark attribution, trade
quality, tail risk, stability — and a bias audit that names what the headline does NOT
correct for (survivorship, regime, multiple-testing, capacity).
"""
import numpy as np
import pandas as pd
from scipy import stats

TD = 252  # trading days/yr


def _ret(equity: pd.Series) -> pd.Series:
    return equity.dropna().pct_change().dropna()


def perf_metrics(equity: pd.Series) -> dict:
    """Return/risk ratios from a daily equity curve."""
    r = _ret(equity)
    if len(r) < 20:
        return {}
    ann = (1 + r).prod() ** (TD / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(TD)
    downside = r[r < 0].std(ddof=1) * np.sqrt(TD)
    sharpe = ann / vol if vol else 0.0
    sortino = ann / downside if downside else 0.0
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    maxdd = float(dd.min())
    calmar = ann / abs(maxdd) if maxdd else 0.0
    # drawdown duration (longest stretch below a prior peak), in days
    underwater = dd < 0
    dur, best = 0, 0
    for u in underwater:
        dur = dur + 1 if u else 0
        best = max(best, dur)
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    omega = pos / neg if neg else 0.0
    return dict(ann_return=float(ann), ann_vol=float(vol), sharpe=float(sharpe),
                sortino=float(sortino), calmar=float(calmar), max_dd=maxdd,
                dd_days=int(best), omega=float(omega),
                skew=float(stats.skew(r, bias=False)),
                kurtosis=float(stats.kurtosis(r, fisher=True, bias=False)),
                var95=float(np.percentile(r, 5)), cvar95=float(r[r <= np.percentile(r, 5)].mean()),
                worst_day=float(r.min()), best_day=float(r.max()))


def vs_benchmark(equity: pd.Series, bench: pd.Series) -> dict:
    """CAPM-style attribution vs a benchmark equity/price series: beta, annual alpha,
    correlation, tracking error, information ratio, up/down capture."""
    rs, rb = _ret(equity), _ret(bench)
    j = pd.concat([rs, rb], axis=1, join="inner").dropna()
    if len(j) < 30:
        return {}
    a, b = j.iloc[:, 0].values, j.iloc[:, 1].values
    beta, alpha_d, r_, *_ = stats.linregress(b, a)
    active = a - b
    te = active.std(ddof=1) * np.sqrt(TD)
    ir = (active.mean() * TD) / te if te else 0.0
    up = a[b > 0].mean() / b[b > 0].mean() if (b > 0).any() and b[b > 0].mean() else np.nan
    dn = a[b < 0].mean() / b[b < 0].mean() if (b < 0).any() and b[b < 0].mean() else np.nan
    return dict(beta=float(beta), alpha_ann=float(alpha_d * TD), corr=float(r_),
                tracking_error=float(te), info_ratio=float(ir),
                up_capture=float(up), down_capture=float(dn))


def trade_metrics(trades: list, capital: float, years: float) -> dict:
    """Trade-quality stats from the per-leg trade log."""
    if not trades:
        return {}
    nets = np.array([t["net"] for t in trades], float)
    wins, losses = nets[nets > 0], nets[nets < 0]
    gp, gl = wins.sum(), -losses.sum()
    return dict(n_trades=len(trades), trades_per_year=len(trades) / max(years, 1e-9),
                hit_rate=float((nets > 0).mean()),
                profit_factor=float(gp / gl) if gl else float("inf"),
                avg_win=float(wins.mean()) if len(wins) else 0.0,
                avg_loss=float(losses.mean()) if len(losses) else 0.0,
                payoff=float(wins.mean() / -losses.mean()) if len(losses) and len(wins) else 0.0)


def rolling_sharpe(equity: pd.Series, window: int = TD) -> dict:
    """Stability of the 12-month rolling Sharpe (min / median / % windows > 0)."""
    r = _ret(equity)
    if len(r) < window + 20:
        return {}
    rs = r.rolling(window).apply(lambda x: x.mean() / x.std(ddof=1) * np.sqrt(TD)
                                 if x.std(ddof=1) else 0.0, raw=False).dropna()
    return dict(roll_sharpe_min=float(rs.min()), roll_sharpe_med=float(rs.median()),
                roll_sharpe_pos_frac=float((rs > 0).mean()))


def vol_target(equity: pd.Series, target_vol: float = 0.15, lookback: int = 63,
               cap: float = 1.0) -> dict:
    """Volatility-targeting overlay (risk-conscious): scale each day's exposure toward a
    fixed annualised `target_vol` using YESTERDAY's trailing realised vol (no look-ahead),
    capped at `cap` (1.0 = de-risk only, never lever). The un-deployed fraction sits in cash
    (earns 0). Returns the new equity curve, the average exposure, and the headline metrics.

    This is the standard institutional drawdown control: when turbulence spikes, the book
    automatically shrinks; in calm momentum tapes it runs (near-)fully invested."""
    r = _ret(equity)
    if len(r) < lookback + 5:
        return {}
    realised = r.rolling(lookback).std(ddof=1) * np.sqrt(TD)
    w = (target_vol / realised).clip(upper=cap)
    w = w.shift(1).fillna(0.0)                          # use prior day's sizing (no look-ahead)
    scaled = (r * w).dropna()
    eq = (1 + scaled).cumprod()
    eq = eq / eq.iloc[0] * float(equity.dropna().iloc[0])
    m = perf_metrics(eq)
    exposure = w.reindex(scaled.index)                  # daily deployed fraction, aligned to eq
    m.update(avg_exposure=float(exposure.mean()), target_vol=target_vol)
    m["equity"] = eq
    m["exposure"] = exposure                            # series — for the per-rebalance timeline
    m["exposure_latest"] = float(exposure.iloc[-1])     # today's invested fraction (rest = cash)
    return m


def grade(test_sharpe: float, dsr: float, mc_p: float, isin_overlap_frac: float) -> dict:
    """An honest letter grade. The headline OOS numbers earn credit; the uncorrected
    biases dock it. `isin_overlap_frac` = how much of the 'graveyard' actually belongs to
    the live universe (≈0 ⇒ survivorship is NOT corrected, the dominant penalty)."""
    score = 0.0
    score += min(test_sharpe / 1.5, 1.0) * 30          # OOS test Sharpe (capped)
    score += dsr * 25                                  # survives multiple-testing
    score += (1.0 if mc_p < 0.05 else 0.0) * 15        # beats random selection
    # biases (deductions)
    surv_corrected = isin_overlap_frac > 0.5
    score += 30 if surv_corrected else 0.0             # survivorship correction (the big one)
    flags = []
    if not surv_corrected:
        flags.append("Survivorship NOT corrected — the live universe is today's TR survivors; "
                     "the bolt-on graveyard is a near-disjoint relic, so winners that died before "
                     "today are simply absent. This inflates everything and is the dominant caveat.")
    flags.append("Regime — the result leans on the 2024–25 small-cap momentum tape; it will not repeat.")
    flags.append("Multiple testing beyond the 32-config grid — the whole pipeline (universe, "
                 "calendar, filters) was iterated many times; the true trial count is higher than DSR assumes.")
    flags.append("Known, crowded, decaying anomaly — 12-1 cross-sectional momentum is a documented "
                 "premium, not novel alpha; net of real costs and capacity it shrinks.")
    letter = ("A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55
              else "D" if score >= 40 else "F")
    return dict(score=round(score, 1), letter=letter, flags=flags,
                survivorship_corrected=surv_corrected)
