"""Today's action — turn the latest signals into a concrete Trade Republic instruction.

The sizing arithmetic here is the SAME rule `vol_forecast.vol_managed` backtests
(w = min(target_vol/forecast, cap), trade only when |Δw| > band) — a parity test pins
the two, so the panel can never drift from the backtest. Held exposure persists in
local/state.json, updated via `build_vol_report.py --set-exposure X` when the user
actually trades. This repo never places orders; it only says what it would do.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from tools.vol_forecast import vol_managed  # noqa: F401  (parity anchor, see tests)

STATE_PATH = Path(__file__).resolve().parent.parent / "local" / "state.json"


def load_state(path: Path = STATE_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return dict(held_w=0.0, updated=None)


def save_state(held_w: float, path: Path = STATE_PATH) -> dict:
    state = dict(held_w=float(held_w),
                 updated=datetime.now().isoformat(timespec="seconds"))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state))
    return state


def vol_action(forecast_today: float, held_w: float, *, target_vol: float = 0.15,
               cap: float = 1.0, band: float = 0.10, capital: float = 10_000.0) -> dict:
    """The exposure decision for tomorrow, from today's vol forecast — identical
    arithmetic to one step of `vol_managed` (strict `>` on the band, like the backtest)."""
    if not (forecast_today and forecast_today > 0):
        return dict(w_target=held_w, held_w=held_w, dw=0.0, trade=False, eur=0.0,
                    forecast=forecast_today,
                    instruction="no valid forecast — hold current exposure")
    w_target = min(target_vol / forecast_today, cap)
    dw = w_target - held_w
    trade = abs(dw) > band
    eur = dw * capital
    if not trade:
        instruction = (f"hold at {held_w:.0%} (target {w_target:.0%}, move "
                       f"{dw:+.1%} inside the {band:.0%} band)")
    elif dw > 0:
        instruction = f"BUY €{eur:,.0f} of the book → {w_target:.0%} invested"
    else:
        instruction = f"SELL €{-eur:,.0f} to cash → {w_target:.0%} invested"
    return dict(w_target=float(w_target), held_w=float(held_w), dw=float(dw),
                trade=bool(trade), eur=float(eur), forecast=float(forecast_today),
                instruction=instruction)


def sleeve_status(index: pd.DatetimeIndex, today=None, entry_mmdd: str = "12-15",
                  exit_mmdd: str = "01-31") -> dict:
    """Where the tax-loss sleeve stands relative to the calendar: inside a Dec→Jan
    window, or counting down to the next mid-December signal."""
    from tools.edge_seasonal import seasonal_windows
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now().normalize()
    wins = seasonal_windows(index, entry_mmdd, exit_mmdd)
    for w in wins:
        if w["signal"] <= today <= w["exit"]:
            return dict(in_window=True, window=w,
                        days_left=int((w["exit"] - today).days))
    year = today.year if today <= pd.Timestamp(f"{today.year}-{entry_mmdd}") else today.year + 1
    next_signal = pd.Timestamp(f"{year}-{entry_mmdd}")
    return dict(in_window=False, window=wins[-1] if wins else None,
                days_to_signal=int((next_signal - today).days))
