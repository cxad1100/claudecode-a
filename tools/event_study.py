"""Canonical event-study machinery (MacKinlay 1997) — the M4 primary
evidence while strategy-level out-of-sample history accrues.

Pure functions. Market-model abnormal returns with the estimation window
STRICTLY pre-event, cumulative ARs across the event window, the BMP (1991)
standardized-residual cross-sectional t (robust to event-day variance
inflation), and a calendar-time portfolio (daily EW return of names inside a
post-publication holding window) whose series feeds straight into
factors.factor_regression — the bridge from event evidence to the page's
FF5+WML spanning gate.
"""
import numpy as np
import pandas as pd

__all__ = ["market_model_ar", "car_stats", "calendar_time_portfolio"]


def market_model_ar(prices: pd.DataFrame, market: pd.Series,
                    events: pd.DataFrame, *, est_window: tuple = (-120, -21),
                    evt_window: tuple = (-5, 20)) -> dict:
    """Per-event market-model abnormal returns.

    events: frame with `ticker` and `event_date`. For each event, α/β are OLS
    on the estimation window (trading-day offsets relative to the event day,
    both strictly negative), then AR_τ = r_τ − (α + β·r_m,τ) over the event
    window. Events without enough clean estimation data are skipped.
    Returns dict(ar=DataFrame[event × τ], betas={label: β}, sd={label: σ_est},
    n=int).
    """
    r = prices.pct_change()
    rm = market.pct_change().reindex(r.index)
    lo_e, hi_e = est_window
    lo_v, hi_v = evt_window
    idx = r.index
    ars, betas, sds, labels = [], {}, {}, []
    for k, ev in events.reset_index(drop=True).iterrows():
        t = ev["ticker"]
        if t not in r.columns:
            continue
        pos_arr = idx.searchsorted(pd.Timestamp(ev["event_date"]))
        pos = int(pos_arr)
        if pos >= len(idx) or pos + lo_e < 0 or pos + hi_v >= len(idx):
            continue
        est = slice(pos + lo_e, pos + hi_e + 1)
        y, x = r[t].iloc[est], rm.iloc[est]
        ok = y.notna() & x.notna()
        if ok.sum() < 30:
            continue
        b, a = np.polyfit(x[ok], y[ok], 1)
        resid = y[ok] - (a + b * x[ok])
        evt = slice(pos + lo_v, pos + hi_v + 1)
        ar = (r[t].iloc[evt] - (a + b * rm.iloc[evt])).to_numpy()
        label = f"{t}#{k}"
        ars.append(ar)
        betas[t] = float(b)
        sds[label] = float(resid.std(ddof=2))
        labels.append(label)
    taus = list(range(lo_v, hi_v + 1))
    panel = pd.DataFrame(ars, index=labels, columns=taus) if ars else \
        pd.DataFrame(columns=taus)
    return dict(ar=panel, betas=betas, sd=sds, n=len(panel),
                evt_window=evt_window)


def car_stats(res: dict) -> dict:
    """CAR path (mean across events of cumulative AR from the window start)
    plus significance: plain cross-sectional t on the full-window CAR and the
    BMP (1991) t on standardized CARs (each event's CAR scaled by its own
    estimation σ·√L — event-day variance inflation doesn't fake significance).
    """
    panel = res["ar"]
    if not len(panel):
        return dict(car={}, t=float("nan"), bmp_t=float("nan"), n=0)
    cum = panel.fillna(0.0).cumsum(axis=1)
    car_path = cum.mean(axis=0)
    full = cum.iloc[:, -1]
    n = len(panel)
    t_plain = float(full.mean() / (full.std(ddof=1) / np.sqrt(n))) \
        if full.std(ddof=1) > 0 else float("nan")
    L = panel.shape[1]
    sd = pd.Series(res["sd"]).reindex(panel.index)
    scar = full / (sd * np.sqrt(L))
    scar = scar.replace([np.inf, -np.inf], np.nan).dropna()
    bmp = float(scar.mean() / (scar.std(ddof=1) / np.sqrt(len(scar)))) \
        if len(scar) > 2 and scar.std(ddof=1) > 0 else float("nan")
    return dict(car={int(k): float(v) for k, v in car_path.items()},
                t=t_plain, bmp_t=bmp, n=n)


def calendar_time_portfolio(prices: pd.DataFrame, events: pd.DataFrame, *,
                            hold_days: int = 21) -> pd.Series:
    """Daily return of the equal-weight book of names with a publication in
    the trailing hold window. Entry the bar AFTER `published_at` (the PIT
    rule in portfolio form); days with no holdings return 0 (cash). The
    resulting series goes straight into factors.factor_regression."""
    r = prices.pct_change()
    held = pd.DataFrame(False, index=r.index, columns=r.columns)
    for _, ev in events.iterrows():
        t = ev["ticker"]
        if t not in r.columns:
            continue
        pos_arr = r.index.searchsorted(pd.Timestamp(ev["published_at"]),
                                       side="right")
        pos = int(pos_arr)
        held.iloc[pos:pos + hold_days, held.columns.get_loc(t)] = True
    w = held.div(held.sum(axis=1).replace(0, np.nan), axis=0)
    port = (r * w).sum(axis=1, min_count=1).fillna(0.0)
    return port
