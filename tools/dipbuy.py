"""Point-in-time short-term reversal ("buy the dip") tilt on the mega-cap screen
(pure functions, no I/O).

Thesis (the user's): the market over-reacts to short-term drops even in the largest,
most-efficient names, so over-weighting the biggest recent losers among the top-N
mega-caps should earn a mean-reversion premium — IF one survives after costs. This is
the classic 1-month reversal factor, restricted to a universe where reversal is
*weakest* (mega-caps), which makes it a hard, honest test rather than a flattering one.

Same walk-forward engine as momentum/megacap — the engine is never forked. A top-N
PIT market-cap screen defines eligibility (reused from `megacap.megacap_screen`); a
reversal score ranks inside it; `run_momentum` backtests. Every value at date d uses
only data with index <= d, and execution is t+1 (`execute_lag=1`, the default here)
because the reversal score reads day-d's *own* close — filling at that same close
would be look-ahead.
"""
import numpy as np
import pandas as pd

from tools.momentum import run_momentum, rebalance_dates
from tools.megacap import megacap_screen


def dip_returns(prices: pd.DataFrame, asof, window: int = 21) -> pd.Series:
    """Trailing `window`-bar simple return per ticker: price(asof) / price(asof-window) - 1.

    Uses only rows with index <= asof (no look-ahead). Empty Series when there are
    fewer than `window`+1 rows of history. inf/NaN dropped."""
    hist = prices.loc[:asof]
    if len(hist) < window + 1:
        return pd.Series(dtype=float)
    recent = hist.iloc[-1]                     # price AT asof (post-dip)
    base = hist.iloc[-(window + 1)]            # price `window` bars before
    ret = recent / base - 1.0
    return ret.replace([np.inf, -np.inf], np.nan).dropna()


def dip_scores(prices: pd.DataFrame, asof, window: int = 21) -> pd.Series:
    """Reversal score = -(trailing `window` return): the biggest recent LOSER scores
    highest, so `select_topk` buys the deepest dips first. PIT (rows <= asof only)."""
    return (-dip_returns(prices, asof, window)).replace([np.inf, -np.inf], np.nan).dropna()


def dip_scores_by_date(prices: pd.DataFrame, dates, window: int = 21) -> dict:
    """{date: {"raw": Series, "voladj": Series}} — both keys identical, matching the
    `score_by_date` contract `run_momentum` consumes (reversal has no vol-adjusted
    variant, so the toggle is a no-op)."""
    out = {}
    for d in dates:
        s = dip_scores(prices, d, window)
        out[d] = {"raw": s, "voladj": s}
    return out


def build_screen_and_scores(prices: pd.DataFrame, cap: pd.DataFrame, *,
                            n: int, window: int = 21):
    """Shared top-n cap eligibility + reversal `score_by_date`, both keyed on
    `rebalance_dates(prices.index)` so they line up with the engine's rebalance loop."""
    dates = rebalance_dates(prices.index)
    elig = megacap_screen(cap, dates, n)
    scores = dip_scores_by_date(prices, dates, window)
    return elig, scores


def run_dipbuy(prices: pd.DataFrame, slippage_bps: dict, cap: pd.DataFrame, *,
               n: int = 25, k: int = 10, window: int = 21,
               execute_lag: int = 1, **kw) -> dict:
    """Buy-the-dip reversal book on the top-n cap screen: hold the k biggest recent
    losers, equal-weight, monthly, walk-forward. Injects the same `elig_by_date`
    (top-n PIT cap) the other mega-cap arms use plus a reversal `score_by_date`; the
    engine is not forked.

    `execute_lag` defaults to 1 — the score reads day-d's close, so the fill is t+1
    (no same-bar look-ahead). Extra kwargs (capital, cost_mults, lookback, ...) pass
    through to `run_momentum`. Returns the standard
    {"runs": {mult: {equity, trades, stats}}, "holdings_log": [...], "start": iso}."""
    elig, scores = build_screen_and_scores(prices, cap, n=n, window=window)
    return run_momentum(prices, slippage_bps, k=k, elig_by_date=elig,
                        score_by_date=scores, execute_lag=execute_lag, **kw)
