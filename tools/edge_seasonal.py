"""Forced-flow seasonal sleeve — the tax-loss rebound, plus the Edge-Stack combiner.

The structural-edge thesis this module implements: retail cannot out-inform, out-model or
out-speed institutions, but it CAN trade against counterparties who are *obligated* to
trade. December tax-loss selling is the cleanest such flow: names with big year-to-date
losses get force-sold into year-end for the tax asset, overshoot, and rebound through
January (documented since Wachtel 1942 / Roll 1983; alive mainly in small caps — exactly
the capacity-constrained pond the TR universe fishes in).

The sleeve: at the last session on/before Dec `entry_mmdd`, rank the year's YTD returns
among eligible liquid names, buy the bottom-k *with actually negative YTD* (no loss → no
tax-loss seller), enter t+1, hold to the last session on/before Jan `exit_mmdd`, exit to
cash for the rest of the year. Costs mirror tools.momentum: €`fee_eur` + half-spread bps
per leg, entry and exit both charged.

Honesty requirements built in: the walk-forward yields ONE observation per year — a
handful of data points, so the per-year table and the cross-sectional Monte-Carlo null
(random names from the SAME pool over the SAME window, via tools.significance) carry the
inference, never a headline Sharpe. Pure functions; the report supplies data.
"""
import numpy as np
import pandas as pd

from tools.momentum import _exec_date, eligible

TD = 252


# ───────────────────────────── calendar ─────────────────────────────

def seasonal_windows(index: pd.DatetimeIndex, entry_mmdd: str = "12-15",
                     exit_mmdd: str = "01-31") -> list[dict]:
    """One window per year with data: signal = last session on/before `year`-entry_mmdd,
    exit = last session on/before `year+1`-exit_mmdd. Windows whose signal or exit falls
    outside the index are dropped; a partial final January is kept (exit clamps to the
    last available session)."""
    out = []
    for year in sorted(set(index.year)):
        sig_target = pd.Timestamp(f"{year}-{entry_mmdd}")
        exit_target = pd.Timestamp(f"{year + 1}-{exit_mmdd}")
        pos = index.searchsorted(sig_target, side="right") - 1
        if pos < 20:                                     # need some history before the signal
            continue
        sig_date = index[pos]
        if sig_date.year != year:
            continue
        epos = index.searchsorted(exit_target, side="right") - 1
        exit_date = index[min(epos, len(index) - 1)]
        if exit_date < pd.Timestamp(year + 1, 1, 1):
            continue                    # data ends before January → a degenerate stub,
            # not a rebound window; counting it would pollute the tiny yearly sample
        out.append(dict(year=year, signal=sig_date, exit=exit_date))
    return out


def ytd_returns(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Return from the first session of `asof`'s year through `asof`, per name
    (uses data ≤ asof only)."""
    year_start = pd.Timestamp(f"{asof.year}-01-01")
    seg = prices.loc[year_start:asof].ffill()
    if len(seg) < 2:
        return pd.Series(dtype=float)
    first = seg.apply(lambda s: s.loc[s.first_valid_index()]
                      if s.first_valid_index() is not None else np.nan)
    return (seg.iloc[-1] / first - 1.0).replace([np.inf, -np.inf], np.nan).dropna()


def select_losers(ytd: pd.Series, elig: set, k: int) -> list[str]:
    """Bottom-k by YTD return among eligible names with YTD < 0 — no loss, no tax-loss
    seller, no forced flow. Fewer than k losers → take what exists."""
    pool = ytd[[t for t in ytd.index if t in elig and ytd[t] < 0.0]]
    return list(pool.sort_values().index[:k])


# ───────────────────────────── backtest ─────────────────────────────

def run_seasonal(prices: pd.DataFrame, slippage_bps: dict, *, k: int = 10,
                 capital: float = 10_000.0, fee_eur: float = 1.0,
                 entry_mmdd: str = "12-15", exit_mmdd: str = "01-31",
                 liq_max: int = 30, min_price: float = 1.0, min_obs: int = 200,
                 pit=None, execute_lag: int = 1, cost_mult: float = 1.0,
                 start: str | None = None) -> dict:
    """Walk-forward tax-loss-rebound backtest. Flat in cash outside the Dec→Jan windows
    (the sleeve's whole point is episodic forced flow, not year-round exposure).

    Returns dict(equity, trades, years=[per-year records], pools=[eligible-pool window
    returns per year, for the Monte-Carlo null], k). Each year record: year, signal,
    exit, picks, ytd (per pick), ret (per-pick window return), net_ret (portfolio, net)."""
    wins = seasonal_windows(prices.index, entry_mmdd, exit_mmdd)
    if start is not None:
        cutoff = pd.Timestamp(start)
        wins = [w for w in wins if w["signal"] >= cutoff]

    equity_val = capital
    eq_points: list[tuple] = []
    years, pools, trades = [], [], []
    for w in wins:
        d, x = w["signal"], w["exit"]
        elig = eligible(prices, d, slippage_bps, liq_max, min_obs, min_price)
        if pit is not None:
            elig = {t for t in elig if pit.listed(t, d)}
        ytd = ytd_returns(prices, d)
        picks = select_losers(ytd, elig, k)
        ed = _exec_date(prices.index, d, execute_lag)   # enter t+1; the exit date is
        # calendar-known in advance, so exiting at x's close introduces no look-ahead

        # cash plateau from the previous exit up to entry
        for day in prices.loc[:ed].index[prices.index.searchsorted(
                eq_points[-1][0] if eq_points else prices.index[0], side="right"):]:
            eq_points.append((day, equity_val))

        # the eligible pool's window returns — the Monte-Carlo null draws from this
        pool_names = [t for t in elig if t in prices.columns]
        seg_all = prices.loc[ed:x, pool_names].ffill().bfill()
        pool_ret = (seg_all.iloc[-1] / seg_all.iloc[0] - 1.0).replace(
            [np.inf, -np.inf], np.nan).dropna() if len(seg_all) >= 2 else pd.Series(dtype=float)
        pools.append(pool_ret.to_numpy(float))

        died = pit.died_between(d, x) if pit is not None else set()
        rec = dict(year=w["year"], signal=d, exit=x, picks=picks,
                   ytd={t: float(ytd[t]) for t in picks}, ret={}, net_ret=0.0,
                   dead={t for t in picks if t in died})
        if not picks:
            years.append(rec)
            continue
        wgt = equity_val / len(picks)
        cost_in = sum(fee_eur + slippage_bps.get(t, 0.0) / 1e4 * wgt for t in picks) * cost_mult
        equity_val -= cost_in
        seg = prices.loc[ed:x, picks].ffill().bfill()
        basket = (seg / seg.iloc[0]).mean(axis=1)          # equal entry weights, drift held
        for day in seg.index[1:]:
            eq_points.append((day, equity_val * float(basket[day])))
        equity_val *= float(basket.iloc[-1])
        wgt_out = equity_val / len(picks)
        cost_out = sum(fee_eur + slippage_bps.get(t, 0.0) / 1e4 * wgt_out for t in picks) * cost_mult
        equity_val -= cost_out
        if eq_points:
            eq_points[-1] = (eq_points[-1][0], equity_val)
        start_val = wgt * len(picks)                       # equity before entry costs
        rec["net_ret"] = float(equity_val / start_val - 1.0)
        for t in picks:
            r = float(seg[t].iloc[-1] / seg[t].iloc[0] - 1.0)
            rec["ret"][t] = r
            trades.append(dict(pair=t, entry=d, exit=x, gross=wgt * r,
                               net=wgt * r - (cost_in + cost_out) / len(picks) * 1.0,
                               capital=wgt))
        years.append(rec)

    # trailing cash plateau to the end of the data
    if eq_points:
        for day in prices.loc[eq_points[-1][0]:].index[1:]:
            eq_points.append((day, equity_val))
    equity = pd.Series(dict(eq_points)).sort_index() if eq_points \
        else pd.Series(dtype=float)
    return dict(equity=equity, trades=trades, years=years, pools=pools, k=k)


def seasonal_period_returns(years: list[dict]) -> np.ndarray:
    """Gross per-window return of the sleeve = equal-weight mean of its picks' window
    returns (empty window → 0), aligned with `pools` for the Monte-Carlo null."""
    out = []
    for y in years:
        rv = [v for v in y["ret"].values() if np.isfinite(v)]
        out.append(float(np.mean(rv)) if rv else 0.0)
    return np.asarray(out, float)


# ───────────────────────────── the stack ─────────────────────────────

def combine_sleeves(core_returns: pd.Series, sleeve_returns: pd.Series,
                    w_sleeve: float = 0.2) -> pd.Series:
    """Daily returns of the Edge Stack: a constant-mix blend of the capacity core
    (momentum) and the forced-flow sleeve, rebalanced daily at zero cost (the sleeve is
    cash ~11 months of the year, so the blend's drift is negligible and the simplicity
    is honest). The vol-managed overlay (tools.vol_forecast.vol_managed) then sizes the
    blend — apply it to THIS series."""
    j = pd.concat([core_returns, sleeve_returns], axis=1).fillna(0.0)
    j = j.loc[j.index.sort_values()]
    return (1.0 - w_sleeve) * j.iloc[:, 0] + w_sleeve * j.iloc[:, 1]
