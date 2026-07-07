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


def window_metrics(equity: pd.Series, lo, hi) -> dict:
    """Canonical display metrics over the [lo, hi] slice of an equity curve:
    `perf_metrics` of the slice plus the slice's net_return. Slice boundaries follow
    the same convention as momentum_grid._stats_slice (lo/hi inclusive via .loc), so
    the canonical numbers cover exactly the dates the pre-registered selection stats
    cover — only the Sharpe basis differs (geometric here vs arithmetic there).
    Display-only: selection/adoption rules never read this."""
    eq = equity.dropna().loc[lo:hi]
    if len(eq) < 21:
        return {}
    m = perf_metrics(eq)
    if not m:
        return {}
    m["net_return"] = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    return m


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
               cap: float = 1.0, turn_cost_bps: float = 0.0) -> dict:
    """Volatility-targeting overlay (risk-conscious): scale each day's exposure toward a
    fixed annualised `target_vol` using YESTERDAY's trailing realised vol (no look-ahead),
    capped at `cap` (1.0 = de-risk only, never lever). The un-deployed fraction sits in cash
    (earns 0). Returns the new equity curve, the average exposure, and the headline metrics.

    This is the standard institutional drawdown control: when turbulence spikes, the book
    automatically shrinks; in calm momentum tapes it runs (near-)fully invested.

    Vol-targeting is NOT free: hitting the target means resizing the book (cash↔stocks) as
    realised vol moves. `turn_cost_bps` charges that daily resizing — |Δexposure| × bps —
    against the return, so the risk-conscious curve reflects its own turnover (the initial
    deployment is excluded; the base strategy already pays the entry). It models the
    proportional slippage only; the flat €/order on tiny daily resizes is extra, which is
    why a real book would band the rebalancing rather than resize every day."""
    r = _ret(equity)
    if len(r) < lookback + 5:
        return {}
    realised = r.rolling(lookback).std(ddof=1) * np.sqrt(TD)
    w_raw = (target_vol / realised).clip(upper=cap).shift(1)   # prior day's sizing (no look-ahead)
    turnover = w_raw.diff().abs().fillna(0.0)                  # daily exposure change = resizing trades
    w = w_raw.fillna(0.0)
    cost = turnover * (turn_cost_bps / 1e4)                    # charge the resizing — it isn't free
    scaled = (r * w - cost).dropna()
    eq = (1 + scaled).cumprod()
    eq = eq / eq.iloc[0] * float(equity.dropna().iloc[0])
    m = perf_metrics(eq)
    exposure = w.reindex(scaled.index)                  # daily deployed fraction, aligned to eq
    m.update(avg_exposure=float(exposure.mean()), target_vol=target_vol)
    m["equity"] = eq
    m["exposure"] = exposure                            # series — for the per-rebalance timeline
    m["exposure_latest"] = float(exposure.iloc[-1])     # today's invested fraction (rest = cash)
    m["turn_cost"] = float(cost.sum())                  # total resizing drag (fraction of book)
    return m


def effective_bets(returns: pd.DataFrame, weights) -> dict:
    """How many *independent* bets a weighted book really holds.

    Naive diversification counts names; this counts uncorrelated risk sources.
    Diagonalise the holdings' covariance Σ = Σ_i λ_i e_i e_iᵀ; the share of portfolio
    variance carried by principal component i is p_i = (wᵀe_i)² λ_i / (wᵀΣw). The
    effective number of bets is the entropy of that distribution,
    exp(−Σ p_i ln p_i) (Meucci 2009): rises with k for uncorrelated names, collapses to
    ≈1 when they all load one factor. `n_eff_weight` = the naive 1/Σw_i² for contrast —
    equal weights score k even when the book is really one bet.

    Pure/observational: covariance comes from the caller's trailing returns window (no
    look-ahead is introduced here). `weights` aligns positionally to `returns.columns`;
    flat (zero-variance) and all-NaN names are dropped and the rest renormalised."""
    w = np.asarray(weights, float)
    cols = list(returns.columns)
    keep = [i for i, c in enumerate(cols)
            if returns[c].notna().any() and returns[c].std(ddof=1) > 0]
    if len(keep) < 2:
        return {}
    r = returns.iloc[:, keep].dropna()
    w = w[keep]
    if w.sum() <= 0 or len(r) < 5:
        return {}
    w = w / w.sum()
    cov = np.cov(r.values, rowvar=False, ddof=1)
    lam, E = np.linalg.eigh(cov)                         # ascending λ, orthonormal columns
    contrib = np.clip((E.T @ w) ** 2 * lam, 0.0, None)   # variance from each PC (clip rounding)
    total = contrib.sum()
    if total <= 0:
        return {}
    p = contrib / total
    nz = p[p > 0]
    return dict(k=len(keep), n_eff_pca=float(np.exp(-(nz * np.log(nz)).sum())),
                pc1_share=float(p.max()), n_eff_weight=float(1.0 / (w ** 2).sum()))


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
    flags.append("Multiple testing beyond the 64-config grid — the whole pipeline (universe, "
                 "calendar, filters) was iterated many times; the true trial count is higher than DSR assumes.")
    flags.append("Known, crowded, decaying anomaly — 12-1 cross-sectional momentum is a documented "
                 "premium, not novel alpha; net of real costs and capacity it shrinks.")
    letter = ("A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55
              else "D" if score >= 40 else "F")
    return dict(score=round(score, 1), letter=letter, flags=flags,
                survivorship_corrected=surv_corrected)
