"""Cross-page strategy result cache — one small artifact per strategy or variant.

The /strategy page shows every strategy the program has produced in one left menu, and
each entry must render REAL output: an equity curve, stat tiles, window metrics. But the
lab computations behind several of those strategies are far too slow to run inline on
every page build — the pairs cointegration scan alone is a ~10-minute job, and the
econophysics modules are worse. Running them per page load is not an option; showing an
empty pane is not an option either.

So each lab distills its result down to a tiny artifact here — an equity curve plus the
display metadata `strategy_registry.make_record()` needs — and the strategy page loads
those artifacts in milliseconds. `build_strategy_report.py --gather-labs` populates them.

Two invariants, both about not lying:

  * **Never fabricate.** A missing artifact means the strategy page says plainly that no
    curve is cached and links to the lab page. It never interpolates, back-fills or
    invents a curve to fill the hole.
  * **Never present stale as live.** Every artifact records the wall-clock time it was
    produced and the source that produced it, so the page can label a curve's age. A
    six-week-old pairs curve is still worth showing — silently implying it is current is
    not.

Payload shape (pickle, one file per id under local/buffer/strategy_cache/):

    {"schema": 1, "id": str, "built_at": datetime, "source": str,
     "equity": pd.Series | None, "meta": {...}}

`meta` carries whatever the record needs (name, family, status, gate, verdict, href,
cost_model, avg_exposure); it is passed through to make_record() by the caller, so this
module stays ignorant of registry semantics and never has to change when they evolve.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA = 1
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "local" / "buffer" / "strategy_cache"


def _dir(cache_dir: Path | None = None) -> Path:
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(sid: str, cache_dir: Path | None = None) -> Path:
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in sid)
    return _dir(cache_dir) / f"{safe}.pkl"


def save(sid: str, equity: pd.Series | None, *, source: str,
         cache_dir: Path | None = None, **meta) -> Path:
    """Write one strategy's artifact. `equity` may be None for a strategy that genuinely
    has no daily curve — the artifact still carries its metrics/verdict in `meta`, which
    is strictly better than no artifact at all."""
    if equity is not None:
        if not isinstance(equity, pd.Series):
            raise TypeError(f"{sid}: equity must be a pd.Series, got {type(equity)!r}")
        equity = equity.dropna()
        if len(equity) < 2:                       # a 1-point curve is not a curve
            equity = None
    payload = {"schema": SCHEMA, "id": sid, "built_at": datetime.now(),
               "source": source, "equity": equity, "meta": dict(meta)}
    path = _path(sid, cache_dir)
    tmp = path.with_suffix(".pkl.tmp")            # atomic: never a half-written artifact
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    tmp.replace(path)
    return path


def load(sid: str, cache_dir: Path | None = None) -> dict | None:
    """One artifact, or None if absent/unreadable/wrong-schema. Never raises — a corrupt
    cache entry must degrade the page to 'no curve cached', not break the build."""
    path = _path(sid, cache_dir)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return None
    return payload


def load_all(cache_dir: Path | None = None) -> dict[str, dict]:
    """Every readable artifact, keyed by id. Silently skips unreadable ones."""
    out: dict[str, dict] = {}
    for path in sorted(_dir(cache_dir).glob("*.pkl")):
        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("schema") == SCHEMA:
            sid = payload.get("id")
            if sid:
                out[sid] = payload
    return out


def age_hours(payload: dict, now: datetime | None = None) -> float | None:
    """Hours since the artifact was built — the number the page shows so a stale curve is
    always labelled as such. None when the timestamp is missing or unusable."""
    built = payload.get("built_at")
    if not isinstance(built, datetime):
        return None
    return ((now or datetime.now()) - built).total_seconds() / 3600.0


def age_label(payload: dict, now: datetime | None = None) -> str:
    """Human 'lab run 3h ago' / 'lab run 12 Jul' string for the pane header."""
    built = payload.get("built_at")
    if not isinstance(built, datetime):
        return "age unknown"
    delta = (now or datetime.now()) - built
    hrs = delta.total_seconds() / 3600.0
    if hrs < 1:
        return f"lab run {int(delta.total_seconds() // 60)}min ago"
    if hrs < 48:
        return f"lab run {int(hrs)}h ago"
    return f"lab run {built.strftime('%d %b')}"


def clear(sid: str | None = None, cache_dir: Path | None = None) -> int:
    """Drop one artifact (or all). Returns how many files were removed."""
    paths = [_path(sid, cache_dir)] if sid else list(_dir(cache_dir).glob("*.pkl"))
    n = 0
    for p in paths:
        if p.exists():
            p.unlink()
            n += 1
    return n
