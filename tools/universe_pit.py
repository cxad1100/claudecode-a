"""Point-in-time universe view (pure): given a price frame, a map of exit dates and
(optionally) the TR snapshot store, answer who was listed / offered / tradeable / dead
at any past date.

Survivors have no entry in `delisting` (delisting_date → None). A dead name's
price column ends at its delisting date (the assembly truncates it there), so
its last bar is its last traded price — what the graveyard liquidates at.

`delisting` holds EVERY exit (deaths + ledger demotions/removals) and gates listed();
`deaths` is the subset that actually died (feeds died_between → graveyard statistics).
`membership=(store, ticker_isin)` adds the PIT snapshot gate: once a snapshot ≤ asof
exists, a name with a known ISIN must be in the latest one — names TR did not offer at
t are not pickable at t. Pre-snapshot dates and unknown ISINs carry no membership claim.
"""
import bisect

import pandas as pd


class PITUniverse:
    def __init__(self, prices: pd.DataFrame,
                 delisting: dict[str, pd.Timestamp] | None = None,
                 deaths: dict[str, pd.Timestamp] | None = None,
                 membership: tuple | None = None):
        self.prices = prices
        self._first = {t: prices[t].first_valid_index() for t in prices.columns}
        self._delist = {t: pd.Timestamp(d)
                        for t, d in (delisting or {}).items() if pd.notna(d)}
        # deaths ⊆ exits: a demoted or TR-removed name exits at market price — it did
        # not die. Default (no split known): every exit counts as a death, as before.
        self._deaths = self._delist if deaths is None else {
            t: pd.Timestamp(d) for t, d in deaths.items() if pd.notna(d)}
        # snapshot membership: {snapshot_date: isin_set}; ISO date strings sort
        # chronologically, so bisect on the raw strings is the as-of lookup.
        self._snap_dates: list[str] = []
        self._snap_sets: dict[str, set] = {}
        self._isin: dict[str, str] = {}
        if membership is not None:
            store, ticker_isin = membership
            if store is not None and len(store):
                for d, g in store.groupby("snapshot_date"):
                    self._snap_sets[str(d)] = set(g["isin"])
                self._snap_dates = sorted(self._snap_sets)
                self._isin = {t: str(z) for t, z in (ticker_isin or {}).items()
                              if pd.notna(z) and str(z)}

    def first_trade_date(self, t):
        return self._first.get(t)

    def delisting_date(self, t):
        return self._delist.get(t)                       # None for survivors

    def _offered(self, t, asof: pd.Timestamp) -> bool:
        """True unless a snapshot ≤ asof exists AND the name's known ISIN is absent
        from the latest one. No snapshots / pre-snapshot asof / unknown ISIN → True."""
        if not self._snap_dates:
            return True
        i = bisect.bisect_right(self._snap_dates, str(asof.date()))
        if i == 0:
            return True                                  # before the first snapshot
        isin = self._isin.get(t)
        if not isin:
            return True
        return isin in self._snap_sets[self._snap_dates[i - 1]]

    def listed(self, t, asof) -> bool:
        asof = pd.Timestamp(asof)
        ft = self._first.get(t)
        if ft is None or ft > asof:
            return False
        dl = self._delist.get(t)
        if dl is not None and dl <= asof:
            return False
        return self._offered(t, asof)

    def tradeable(self, asof, min_history_days: int = 273) -> set[str]:
        asof = pd.Timestamp(asof)
        hist = self.prices.loc[:asof]
        return {t for t in self.prices.columns
                if self.listed(t, asof)
                and hist[t].dropna().shape[0] >= min_history_days}

    def died_between(self, prev, asof) -> set[str]:
        prev, asof = pd.Timestamp(prev), pd.Timestamp(asof)
        return {t for t, d in self._deaths.items() if prev < d <= asof}

    def last_price(self, t, asof=None):
        col = self.prices[t].dropna()
        if asof is not None:
            col = col.loc[:pd.Timestamp(asof)]
        return float(col.iloc[-1]) if len(col) else None
