"""Append-only membership ledger for the TR universe refresh (pure, unit-tested).

`build_tr_universe.fetch` used to rebuild the live set from scratch: a name that died,
lost liquidity, or failed to download since the last refresh simply vanished — the
forward survivorship leak. reconcile() closes it: every previously-live name absent
from a fresh fetch is classified and KEPT (history intact), so the dataset only ever
gains membership events.

exit_reason classes:
  delisted — prices stopped > stale_days before the refresh: a real death; the
             graveyard liquidates at the last print.
  removed  — still printing but its ISIN left TR's enumerated list: exit at market price.
  demoted  — still printing, still offered, but failed the liquidity criteria: exit at
             market price. May resurrect at a later refresh (single-interval model:
             resurrection clears the exit; the gap is not representable — logged).
  (carried, no exit) — fetch hiccup (ISIN offered, ticker not fetched): stays LIVE on
             its old prices and is retried next refresh.

The NEW live set is stale-checked too: apply_criteria has no recency check, so a name
that died since the last refresh can pass on its frozen tail — reclassified delisted.
Deaths vs mere exits stay distinguishable downstream (universe_assemble.death_mask).
"""
import pandas as pd

STALE_DAYS = 21          # calendar days without a print ⇒ the line is dead, not idle


def _norm(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    if "exit_reason" not in meta.columns:
        meta["exit_reason"] = ""
    meta["exit_reason"] = meta["exit_reason"].fillna("")
    if "delisting_date" in meta.columns:
        meta["delisting_date"] = meta["delisting_date"].replace("", pd.NA)
    return meta


def reconcile(prev_meta: pd.DataFrame, prev_prices: pd.DataFrame,
              new_meta: pd.DataFrame, new_prices: pd.DataFrame, *,
              tr_isins: set, fetched_ok: set, asof,
              stale_days: int = STALE_DAYS) -> tuple:
    """Merge a fresh fetch (new_* = live names only) with the previous universe
    (prev_* = live + dead). Returns (meta, prices, report)."""
    asof = pd.Timestamp(asof)
    prev_meta, new_meta = _norm(prev_meta), _norm(new_meta)
    # legacy graveyard rows predate the ledger: real deaths
    legacy = prev_meta["delisting_date"].notna() & (prev_meta["exit_reason"] == "")
    prev_meta.loc[legacy, "exit_reason"] = "delisted"
    report: dict[str, list] = {k: [] for k in (
        "delisted", "removed", "demoted", "carried", "resurrected", "stale_new")}

    def _last(frame, t):
        return frame[t].last_valid_index() if t in frame.columns else None

    # a corpse can pass apply_criteria on its frozen tail — stale-check the new set
    for t in list(new_meta["ticker"]):
        lp = _last(new_prices, t)
        if lp is not None and (asof - lp).days > stale_days:
            sel = new_meta["ticker"] == t
            new_meta.loc[sel, "delisting_date"] = str(lp.date())
            new_meta.loc[sel, "exit_reason"] = "delisted"
            report["stale_new"].append(t)

    new_tickers = set(new_meta["ticker"])
    out_meta = [new_meta]
    out_px: dict = {t: new_prices[t] for t in new_prices.columns}

    for r in prev_meta.itertuples(index=False):
        t, row = r.ticker, pd.DataFrame([r._asdict()])
        if t in new_tickers:
            if pd.notna(r.delisting_date):
                report["resurrected"].append(t)     # fresh live row wins; gap lost (logged)
            continue
        if pd.notna(r.delisting_date):              # dead/exited: pass through unchanged
            out_meta.append(row)
            if t in prev_prices.columns:
                out_px[t] = prev_prices[t]
            continue
        lp = _last(prev_prices, t)                  # previously-live, absent from fetch
        if lp is None:
            continue                                # never had data — drop
        isin = "" if pd.isna(r.isin) else str(r.isin)
        if (asof - lp).days > stale_days:
            reason = "delisted"
        elif isin not in tr_isins:
            reason = "removed"
        elif t in fetched_ok:
            reason = "demoted"                      # data came back but criteria failed
        else:
            reason = ""                             # fetch hiccup: carry live, retry
            report["carried"].append(t)
        if reason:
            row["delisting_date"] = str(lp.date())
            row["exit_reason"] = reason
            report[reason].append(t)
        out_meta.append(row)
        out_px[t] = prev_prices[t].loc[:lp]

    meta = pd.concat(out_meta, ignore_index=True).drop_duplicates("ticker", keep="first")
    prices = pd.DataFrame(out_px).sort_index()
    return meta.reset_index(drop=True), prices, report
