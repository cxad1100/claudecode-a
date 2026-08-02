"""Regime attribution — how much of the strategy's edge is regime/sector beta, not innate alpha?

Observational only: these helpers re-slice or restrict, they never feed selection or sizing.
Two pure pieces used by the strategy page's regime-attribution section:

  * conditional_performance — split the daily return stream by a boolean regime mask (e.g. the
    HMM risk-off label) and report the Sharpe + share of total return earned in each regime.
    Answers "is the Sharpe just regime-timing?" without trading the regime.
  * restrict_universe — drop whole sectors (e.g. Technology + Industrials, the AI + defense
    tailwind) so the SAME pinned config can be re-run on the remainder. Answers "is the edge
    just AI/Defense sector beta?" A universe-restriction re-run, exactly like the existing
    never-TR-tradeable upper-bound run — no look-ahead, no config change.
"""
import numpy as np
import pandas as pd

TD = 252


def _grp(r: pd.Series, total_log: float, ppy: float) -> dict:
    """Sharpe / cumulative return / share-of-total for one regime's (non-contiguous) days."""
    if len(r) == 0:
        return dict(n_days=0, sharpe=0.0, cum_return=0.0, ret_share=0.0)
    sd = r.std(ddof=1)
    sharpe = float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0
    cum = float(np.expm1(np.log1p(r).sum()))                     # ∏(1+r) − 1 over the group's days
    share = float(np.log1p(r).sum() / total_log) if total_log else 0.0
    return dict(n_days=int(len(r)), sharpe=sharpe, cum_return=cum, ret_share=share)


def conditional_performance(returns: pd.Series, risk_off: pd.Series, ppy: float = TD) -> dict:
    """Split `returns` into risk-ON (mask False) vs risk-OFF (mask True) and grade each.

    `ret_share` attributes total compounded return between the two regimes (log-return shares,
    so they sum to 1). Pure: no look-ahead is introduced — the mask is the caller's already
    walk-forward regime label, aligned by index (unlabelled days are dropped)."""
    j = pd.concat([returns.rename("r"), risk_off.rename("off")], axis=1, join="inner").dropna()
    r, off = j["r"], j["off"].astype(bool)
    total_log = float(np.log1p(r).sum())
    return dict(on=_grp(r[~off], total_log, ppy), off=_grp(r[off], total_log, ppy),
                total_return=float(np.expm1(total_log)))


def restrict_universe(prices: pd.DataFrame, sectors: dict, drop: set) -> pd.DataFrame:
    """Return `prices` without the columns whose sector is in `drop`. Names with no sector
    (absent from `sectors`) are KEPT — dropping only removes the explicitly-tagged sectors."""
    keep = [c for c in prices.columns if sectors.get(c) not in drop]
    return prices[keep]
