"""PIT event → cross-sectional signal layer for the regulatory stores.

Pure functions consuming data/universe/insider_dealings.csv and
short_positions.csv frames. The contract that keeps everything honest:
`pit_slice` filters STRICTLY `published_at < asof` — the market can act on a
disclosure only after it is out — and every score builder routes through it.
Score dicts are run_momentum-ready ({date: {"raw","voladj"}}, covering every
requested date), so the existing gate (MC null, DSR ladder, FF5+WML
spanning) applies unchanged.

Small-cap interaction (`interact_small`): the insider literature
(Lakonishok & Lee 2001) finds the signal concentrated in small names — and a
retail-sized book can actually trade those (capacity edge). Market cap is
not in the data layer; trailing median EUR turnover is the liquidity/size
proxy, consistent with every other gate in this repo.
"""
import numpy as np
import pandas as pd

__all__ = ["pit_slice", "insider_score_by_date", "short_pressure_by_date",
           "interact_small"]

_SHORT_FLOOR = 0.5      # EU SSR disclosure threshold: latest pct below ⇒ out


def pit_slice(events: pd.DataFrame, asof, *, lookback_days: int | None = None,
              date_col: str = "published_at") -> pd.DataFrame:
    """Rows knowable strictly BEFORE `asof` (ISO string compare), optionally
    only those published within the trailing `lookback_days` window."""
    cut = pd.Timestamp(asof).strftime("%Y-%m-%d")
    out = events[events[date_col].astype(str) < cut]
    if lookback_days is not None:
        lo = (pd.Timestamp(asof) - pd.Timedelta(days=lookback_days)) \
            .strftime("%Y-%m-%d")
        out = out[out[date_col].astype(str) >= lo]
    return out


def _series_pair(s: pd.Series) -> dict:
    """run_momentum score contract; voladj == raw here — event signals carry
    no per-name vol normalisation (the A-toggle is a no-op for them)."""
    s = s.dropna()
    return {"raw": s, "voladj": s.copy()}


def insider_score_by_date(events: pd.DataFrame, dates, isin_ticker: dict, *,
                          window_days: int = 180,
                          halflife: float = 30.0) -> dict:
    """Net insider tilt per name: Σ over disclosed dealings of
    sign(side) · 2^(−age/halflife), age in days since publication. Buys +1,
    sells −1 (other sides ignored); ISINs outside the universe map drop."""
    ev = events[events["side"].isin(["buy", "sell"])].copy()
    ev["sign"] = np.where(ev["side"] == "buy", 1.0, -1.0)
    out = {}
    for d in dates:
        d = pd.Timestamp(d)
        win = pit_slice(ev, d, lookback_days=window_days)
        if not len(win):
            out[d] = _series_pair(pd.Series(dtype=float))
            continue
        age = (d - pd.to_datetime(win["published_at"])).dt.days.astype(float)
        w = win["sign"] * np.power(2.0, -age / halflife)
        agg = w.groupby(win["isin"].map(isin_ticker)).sum()
        agg.index.name = None
        out[d] = _series_pair(agg[agg.index.notna()])
    return out


def _crowded_asof(shorts: pd.DataFrame, asof) -> pd.Series:
    """Disclosed short crowding per ISIN at `asof`: each holder's most recent
    published position; holders whose latest pct < 0.5 have left the
    disclosure regime (true position unknowable below threshold) and drop."""
    vis = pit_slice(shorts, asof)
    if not len(vis):
        return pd.Series(dtype=float)
    vis = vis.sort_values("published_at")
    last = vis.groupby(["isin", "holder"])["pct"].last()
    live = last[last >= _SHORT_FLOOR]
    return live.groupby(level="isin").sum()


def short_pressure_by_date(shorts: pd.DataFrame, dates, isin_ticker: dict, *,
                           delta_days: int = 21) -> dict:
    """Two run_momentum-ready score dicts:
    crowded  — total disclosed short % per name (a crowdedness level), and
    covering — decrease in that level over the trailing delta_days
               (positive = shorts closing = buying pressure)."""
    crowded, covering = {}, {}
    for d in dates:
        d = pd.Timestamp(d)
        now = _crowded_asof(shorts, d)
        prev = _crowded_asof(shorts, d - pd.Timedelta(days=delta_days))
        both = now.index.union(prev.index)
        cov = prev.reindex(both, fill_value=0.0) - now.reindex(both,
                                                               fill_value=0.0)
        t_now = now.groupby(now.index.map(isin_ticker)).sum()
        t_cov = cov.groupby(cov.index.map(isin_ticker)).sum()
        crowded[d] = _series_pair(t_now[t_now.index.notna()])
        covering[d] = _series_pair(t_cov[t_cov.index.notna()])
    return dict(crowded=crowded, covering=covering)


def interact_small(score: pd.Series, turnover: pd.DataFrame | None, asof, *,
                   frac: float = 1 / 3, window: int = 6) -> pd.Series:
    """Restrict a score to the bottom-`frac` trailing-turnover names — the
    capacity-edge cut: the small names institutions cannot bother with are
    where the event literature finds the effect. No turnover data → score
    unchanged (gate degrades open, like the static liquidity gate)."""
    if turnover is None or not len(score):
        return score
    med = turnover.loc[:pd.Timestamp(asof)].tail(window) \
        .reindex(columns=list(score.index)).median().dropna()
    if not len(med):
        return score
    cut = med.quantile(frac)
    keep = med[med <= cut].index
    return score[score.index.isin(keep)]
