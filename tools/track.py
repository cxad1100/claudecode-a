"""Live tracking + pre-registered kill criteria — the only test that settles
"profitable?": the live path against the backtest's own error bars.

`append_snapshot` writes one row per build date to local/track.csv (idempotent — a
rebuild on the same day never duplicates). `live_vs_backtest` evaluates the two kill
rules once enough live data exists:

  KILL (de-risk to cash and reassess — this repo never auto-trades) iff
    (a) the rolling 63d live Sharpe has been below the backtest bootstrap CI's lower
        bound for 21 consecutive sessions, or
    (b) the live drawdown exceeds 1.25 × the backtest max drawdown.

Both thresholds were committed before any live row existed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TD = 252
WINDOW = 63          # rolling live-Sharpe window
PATIENCE = 21        # consecutive breach sessions before KILL
DD_MULT = 1.25       # live drawdown tolerance vs backtest max DD

TRACK_PATH = Path(__file__).resolve().parent.parent / "local" / "track.csv"


def append_snapshot(date, row: dict, path: Path = TRACK_PATH) -> bool:
    """Append one dated row (forecast, w_target, w_held, r_live, …). Returns False
    without writing when the date already has a row."""
    p = Path(path)
    date = pd.Timestamp(date).normalize()
    if p.exists():
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if date in df.index:
            return False
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row], index=[date])])
    df.sort_index().to_csv(p)
    return True


def read_track(path: Path = TRACK_PATH) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, index_col=0, parse_dates=True)
    except Exception:
        return None


def live_vs_backtest(track: pd.DataFrame, sharpe_lo: float, backtest_max_dd: float,
                     window: int = WINDOW, patience: int = PATIENCE,
                     dd_mult: float = DD_MULT) -> dict:
    """Evaluate the kill rules on the live return column ('r_live')."""
    n = 0 if track is None or "r_live" not in getattr(track, "columns", []) \
        else int(track["r_live"].notna().sum())
    if n < window:
        return dict(enough=False, n=n, needed=window, kill=False, reasons=[])
    r = track["r_live"].dropna()
    roll = r.rolling(window).apply(
        lambda x: x.mean() / x.std(ddof=1) * np.sqrt(TD) if x.std(ddof=1) else 0.0,
        raw=False).dropna()
    below = (roll < sharpe_lo)
    streak = 0
    for b in below[::-1]:                      # current consecutive-breach streak
        if not b:
            break
        streak += 1
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    reasons = []
    if streak >= patience:
        reasons.append(f"rolling {window}d live Sharpe below the backtest CI lower "
                       f"bound ({sharpe_lo:.2f}) for {streak} consecutive sessions "
                       f"(limit {patience})")
    if dd < dd_mult * backtest_max_dd:         # both negative
        reasons.append(f"live drawdown {dd:.1%} exceeds {dd_mult}× backtest max DD "
                       f"({backtest_max_dd:.1%})")
    return dict(enough=True, n=n, kill=bool(reasons), reasons=reasons,
                roll_sharpe_last=float(roll.iloc[-1]), breach_streak=int(streak),
                live_dd=dd, dd_limit=float(dd_mult * backtest_max_dd))
