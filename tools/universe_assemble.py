"""Pure helpers for the momentum dataset: union meta sources with a `delisting_date`
column, outer-join price frames without resurrecting dead lines, and project the
delisting map the momentum engine + PITUniverse consume.

The live universe is now built by `tools.build_market` (the single source of truth);
the old yfinance/dead-price assembler CLI that lived here has been removed. The
functions below stay because they're pure and unit-tested.
"""
import pandas as pd

_META_COLS = ["ticker", "name", "sector", "country", "currency",
              "slippage_bps", "local_id", "delisting_date"]


def assemble_meta(survivors: pd.DataFrame, misses: list[dict],
                  dead: pd.DataFrame) -> pd.DataFrame:
    """Union the three sources into one meta with a delisting_date column."""
    surv = survivors.copy()
    surv["delisting_date"] = pd.NaT
    miss = pd.DataFrame(misses)
    if not miss.empty:
        miss["delisting_date"] = pd.NaT
    dead = dead.copy()
    if not dead.empty:
        dead["delisting_date"] = pd.to_datetime(dead["delisting_date"], format="mixed")
    frames = []
    for df in (surv, miss, dead):
        if df.empty:
            continue
        for c in _META_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        frames.append(df[_META_COLS])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("ticker", keep="first").reset_index(drop=True)


def assemble_prices(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join price frames on the date index. Each frame's columns keep their
    own NaN tails (dead names already truncated at death by the dead-stock step),
    so no cross-name forward-fill resurrects a delisted line."""
    out = pd.concat(frames, axis=1).sort_index()
    return out.loc[:, ~out.columns.duplicated()]


_DEATH_REASONS = {"", "delisted"}


def death_mask(meta: pd.DataFrame) -> pd.Series:
    """Real deaths only. The ledger (tools.universe_reconcile) records demoted/removed
    exits with a delisting_date too — they gate eligibility but are NOT graveyard deaths.
    Rows with no exit_reason column/value are the pre-ledger EODHD graveyard = deaths."""
    dl = meta["delisting_date"].notna()
    if "exit_reason" not in meta.columns:
        return dl
    return dl & meta["exit_reason"].fillna("").isin(_DEATH_REASONS)


def exit_map(meta: pd.DataFrame) -> dict:
    """meta → {ticker: Timestamp} for EVERY exited name (deaths + demotions + removals)
    — the map that ends PIT eligibility (PITUniverse `delisting` input)."""
    d = meta.dropna(subset=["delisting_date"])
    return {r.ticker: pd.Timestamp(r.delisting_date) for r in d.itertuples(index=False)}


def death_map(meta: pd.DataFrame) -> dict:
    """Deaths only (PITUniverse `deaths` input — feeds died_between/graveyard stats)."""
    d = meta[death_mask(meta)]
    return {r.ticker: pd.Timestamp(r.delisting_date) for r in d.itertuples(index=False)}


def delisting_map(meta: pd.DataFrame) -> dict:
    """Back-compat alias: all exits (what gates eligibility)."""
    return exit_map(meta)
