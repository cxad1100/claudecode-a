"""Point-in-time (PIT) universe snapshots — Tier-1 survivorship fix (capture side).

The backtest universe is today's TR survivors: names that died before today are simply
absent, which inflates every momentum return on the strategy page. We can't recover the
*past* universe (TR publishes no history), but we can stop the bleed going forward by
freezing 'what TR offers' on a schedule. Once a year of snapshots accrues, a backtest can
ask `membership_asof(t)` — the names actually offered at t, including ones later delisted —
instead of pretending today's list existed all along.

Source of truth = `tools/tr_tradeable.py --enumerate` (TR's full tradeable stock universe,
isin/name/country). That fetch needs interactive 2FA every run (ephemeral by design — it
shreds creds on exit), so this is a one-command MONTHLY run, not an unattended cron: the
archive below is automatic; the human supplies the 2FA. Storage is an append-only long CSV
(one row per snapshot_date × isin), local-only (data/universe is gitignored).

Pure core (archive / membership_asof) is unit-tested; the IO + CLI are thin wrappers.
"""
import argparse
import datetime as dt
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "universe" / "tr_universe.csv"        # what --enumerate writes
STORE = ROOT / "data" / "universe" / "snapshots.csv"        # append-only PIT store (local)
COLS = ["snapshot_date", "isin", "name", "country"]


def empty_store() -> pd.DataFrame:
    """An empty PIT store with the canonical schema (string-typed)."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLS})


def archive(members: pd.DataFrame, date: str, store: pd.DataFrame | None = None) -> pd.DataFrame:
    """Append a dated membership snapshot to the store (append-only, idempotent per date).

    `members` is a TR-universe frame (isin/name/country); `date` is an ISO 'YYYY-MM-DD'.
    Re-running the same date REPLACES that day's rows (so a re-run after a partial fetch
    wins cleanly) while every earlier snapshot is preserved untouched — that history is the
    whole point. ISO date strings sort chronologically, so plain string compare is enough."""
    store = empty_store() if store is None else store
    kept = store[store["snapshot_date"] != date]                 # drop a same-day re-run
    fresh = members.loc[:, ["isin", "name", "country"]].copy()
    fresh.insert(0, "snapshot_date", date)
    return pd.concat([kept, fresh], ignore_index=True)


def membership_asof(store: pd.DataFrame, date: str) -> set:
    """The set of ISINs offered as of `date` = the latest snapshot dated on-or-before it.

    No look-ahead: a backtest at time t sees only the most recent snapshot ≤ t (and an
    empty set before the very first snapshot — we simply have no PIT knowledge yet)."""
    if store.empty:
        return set()
    eligible = store[store["snapshot_date"] <= date]
    if eligible.empty:
        return set()
    latest = eligible["snapshot_date"].max()
    return set(eligible.loc[eligible["snapshot_date"] == latest, "isin"])


# ── thin IO + CLI ─────────────────────────────────────────────────────────────
def load_store(path: pathlib.Path = STORE) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str) if pathlib.Path(path).exists() else empty_store()


def save_store(store: pd.DataFrame, path: pathlib.Path = STORE) -> None:
    store.to_csv(path, index=False)


def snapshot(src: pathlib.Path = SRC, store_path: pathlib.Path = STORE,
             date: str | None = None) -> tuple:
    """Freeze the current membership file into the PIT store under `date` (default today)."""
    date = date or dt.date.today().isoformat()
    members = pd.read_csv(src, dtype=str)
    store = archive(members, date, load_store(store_path))
    save_store(store, store_path)
    return date, int((store["snapshot_date"] == date).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Append a PIT snapshot of the TR universe.")
    ap.add_argument("--enumerate", action="store_true",
                    help="re-fetch TR's universe first (interactive 2FA) before archiving")
    ap.add_argument("--date", default=None, help="override snapshot date (ISO; default today)")
    a = ap.parse_args()
    if a.enumerate:
        from tools import tr_tradeable                          # lazy: pulls in pytr
        tr_tradeable.enumerate_universe(waf="awswaf")
    d, n = snapshot(date=a.date)
    print(f"snapshot {d}: {n} names → {STORE}", flush=True)
