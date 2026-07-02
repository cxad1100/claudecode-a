"""On-population survivorship test via synthetic delisting injection.

The strategy page bolts on a real graveyard, but it overlaps the live `.DE`/`.F` universe
by only ~2% — a different population — so it can't honestly answer the *holding* leak: does
the strategy hold a name into its delisting and eat the crash? This injects deaths into the
LIVE names themselves (100% representative): each name dies at a realistic annual `hazard`
with a terminal crash drawn from [loss_lo, loss_hi], then goes NaN (delisted → untradeable).
Re-running the SAME strategy on the injected prices K times bounds the drag and counts how
often momentum was actually holding a name at death (synthetic graveyard_hits).

Conservative by design: a sudden terminal crash with no pre-decline is a HARDER test than
real delistings (which usually bleed first and get down-ranked early) — if even abrupt
deaths barely dent the result, the holding leak is genuinely immaterial *for this universe*.

Pure / seeded — observational only, never wired into live selection or sizing.
"""
import numpy as np
import pandas as pd

TD = 252


def inject_delistings(prices: pd.DataFrame, hazard_annual: float = 0.05, loss_lo: float = 0.3,
                      loss_hi: float = 1.0, seed: int = 0, min_life: int = 126) -> tuple:
    """Kill names at a synthetic annual hazard and return (injected_prices, deaths).

    Each column dies at most once, no earlier than `min_life` bars in (a name needs history
    before it can be picked). On the death bar the price takes a one-day terminal crash of a
    fraction drawn from [loss_lo, loss_hi]; every bar after is NaN (untradeable). `deaths` =
    {ticker: (death_date, loss_frac)}. Deterministic in `seed`; hazard_annual=0 ⇒ a no-op."""
    out = prices.copy()
    if hazard_annual <= 0:
        return out, {}
    rng = np.random.default_rng(seed)
    vals = prices.values
    n, m = vals.shape
    p_day = 1.0 - (1.0 - hazard_annual) ** (1.0 / TD)             # per-bar death probability
    draws = rng.random((n, m))                                   # one deterministic draw block
    losses = rng.uniform(loss_lo, loss_hi, size=m)
    valid = ~np.isnan(vals)
    deaths = {}
    for j in range(m):
        elig = np.zeros(n, dtype=bool)
        if n > min_life:
            pos = np.arange(min_life, n)
            elig[pos] = valid[pos, j] & valid[pos - 1, j]        # need this + prior bar live
        hit = elig & (draws[:, j] < p_day)
        if not hit.any():
            continue
        dp = int(np.argmax(hit))                                 # first eligible death bar
        loss = float(losses[j])
        out.iloc[dp, j] = vals[dp - 1, j] * (1.0 - loss)         # terminal crash
        out.iloc[dp + 1:, j] = np.nan                           # delisted thereafter
        deaths[prices.columns[j]] = (prices.index[dp], loss)
    return out, deaths


def summarize(base_return: float, sim_returns: list, sim_hits: list, sim_deaths: list) -> dict:
    """Aggregate K injected runs into an honest survivorship-drag bound.

    `base_return` = the clean (no-injection) net return; the deltas are injected − base, so a
    near-zero `delta_mean` with the crashes present is the evidence the holding leak is
    immaterial on-population. Reports a 5–95% delta band and the mean synthetic graveyard
    hits / deaths injected per run."""
    r = np.asarray(sim_returns, float)
    d = r - base_return
    hits_mean, deaths_mean = float(np.mean(sim_hits)), float(np.mean(sim_deaths))
    # Avoidance = the share of injected deaths the strategy was NOT holding at delisting. This is
    # the STABLE, hazard-robust headline (momentum down-ranks a dyer before death); the cumulative
    # return delta is a noisy, fat-tailed corroboration best read as a pessimistic bound.
    avoidance = 1.0 - hits_mean / deaths_mean if deaths_mean > 0 else 1.0
    return dict(base_return=float(base_return), sims=int(len(r)),
                mean_return=float(r.mean()), delta_mean=float(d.mean()),
                delta_lo=float(np.percentile(d, 5)), delta_hi=float(np.percentile(d, 95)),
                hits_mean=hits_mean, deaths_mean=deaths_mean, avoidance_rate=float(avoidance))


def band(arr, lo: float = 5, hi: float = 95) -> dict:
    """mean + [lo, hi] percentile band of a 1-D sample; NaNs dropped, empty -> zeros."""
    a = np.asarray(list(arr), float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return dict(mean=0.0, lo=0.0, hi=0.0)
    return dict(mean=float(a.mean()), lo=float(np.percentile(a, lo)),
                hi=float(np.percentile(a, hi)))
