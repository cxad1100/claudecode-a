"""Strategy registry — one record per strategy, uniform metrics, one comparison surface.

The framework contract: a NEW curve-bearing strategy enters the Strategy page as ONE
`make_record(...)` call in build_strategy_report.build_registry (leaderboard row, chart
trace and color all flow from the record); a new killed/cross-page idea is ONE entry in
STATIC_RECORDS. Pure (no I/O, no engine calls): the page supplies equity curves and
window boundaries; this module normalizes them through the canonical metric layer
(quant_grade.window_metrics — geometric, display-only). The pre-registered selection
and adoption math (momentum_grid, pairs_backtest.backtest_stats, tools.gates,
tools.track) is untouched by design — changing it would void the pre-registration.
"""
from dataclasses import dataclass, field

import pandas as pd

from tools import quant_grade as qg
from tools import theme

STATUSES = ("adopted", "candidate", "variant", "reference", "benchmark",
            "portfolio", "killed", "cut", "research")

# Leaderboard order: live money first, then the family variants, then context rows,
# then the ledger states (killed/cut/research render in the collapsed ledger).
_STATUS_RANK = {s: i for i, s in enumerate(
    ("adopted", "candidate", "variant", "benchmark", "reference", "portfolio",
     "killed", "cut", "research"))}

# Status badge colors (border/text of the .badge chip).
STATUS_COLOR = {
    "adopted": "#46c84e", "candidate": "#569cd6", "variant": theme.FG_DIM,
    "reference": "#d7ba7d", "benchmark": theme.FG_DIM, "portfolio": "#d4d4d4",
    "killed": "#ef4444", "cut": "#ef4444", "research": theme.FG_DIM,
}


@dataclass
class StrategyRecord:
    id: str
    name: str
    family: str                       # "momentum" | "vol-managed core" | "edge" | ...
    status: str                       # one of STATUSES
    variant_of: str | None = None
    since: str | None = None          # first date of the curve (derived when equity given)
    adopted: str | None = None        # adoption date, e.g. "2026-07-06"
    cost_model: str = ""              # short human string, e.g. "slip + €1 + 25bp resize"
    equity: pd.Series | None = None   # only for curves computed on this page
    windows: dict = field(default_factory=dict)  # name -> qg.window_metrics dict
    avg_exposure: float | None = None            # deployed fraction, where known
    gate: str | None = None           # pre-registered gate/rule status, verbatim
    verdict: str = ""                 # one-liner (ledger rows)
    live: bool = False                # this record is what local/strategy_track.csv tracks
    color: str | None = None
    href: str | None = None           # lab page for ledger rows
    flags: tuple = ()                 # ("inflated",) marks the raw reference
    holdings: tuple = ()              # per-rebalance held names + returns vs index (mega-cap arms)


def window_bounds(equity: pd.Series, train_end, val_end) -> dict:
    """The four canonical [lo, hi] windows over a curve — identical boundary convention
    to momentum_grid._stats_slice (train ≤ train_end < val ≤ val_end < test)."""
    eq = equity.dropna()
    te, ve = pd.Timestamp(train_end), pd.Timestamp(val_end)
    one = pd.Timedelta(days=1)
    return dict(train=(eq.index[0], te), val=(te + one, ve),
                test=(ve + one, eq.index[-1]), full=(eq.index[0], eq.index[-1]))


def make_record(id: str, name: str, family: str, status: str, *,
                equity: pd.Series | None = None, train_end=None, val_end=None,
                **kw) -> StrategyRecord:
    """The one-call entry point: give it a curve + the window constants and it fills
    `since` and the canonical train/val/test/full metrics. Pass `windows=` explicitly
    to reuse metrics already computed with qg.window_metrics (zero drift)."""
    if status not in STATUSES:
        raise ValueError(f"status {status!r} not in {STATUSES}")
    windows = kw.pop("windows", None) or {}
    since = kw.pop("since", None)
    if equity is not None:
        eq = equity.dropna()
        if len(eq) >= 2:
            since = since or str(eq.index[0].date())
            if not windows and train_end is not None and val_end is not None:
                windows = {k: qg.window_metrics(eq, lo, hi)
                           for k, (lo, hi) in window_bounds(eq, train_end, val_end).items()}
    return StrategyRecord(id=id, name=name, family=family, status=status,
                          equity=equity, windows=windows, since=since, **kw)


def window_labels(start, train_end, val_end) -> dict:
    """Display labels derived from the window constants — the single source for every
    'Train 2018–21 / Validation 2022–23 / Test 2024→ / Full 2018→' string on the page."""
    y0 = pd.Timestamp(start).year
    yt = pd.Timestamp(train_end).year
    yv = pd.Timestamp(val_end).year
    return dict(train=f"Train {y0}–{yt % 100:02d}",
                val=f"Validation {yt + 1}–{yv % 100:02d}",
                test=f"Test {yv + 1}→",
                full=f"Full {y0}→")


def ordered(records) -> list:
    """Leaderboard sort: adopted → candidate → variant → benchmark → reference →
    portfolio → killed/cut/research; stable within a status (insertion order)."""
    return sorted(records, key=lambda r: _STATUS_RANK.get(r.status, 99))


# Strategy statuses (a family's own rows) vs context rows appended after them.
_STRAT_STATUS = ("adopted", "candidate", "variant", "reference")


def family_ordered(records) -> list:
    """Family-contiguous ordering: strategy families render as unbroken blocks
    (ranked by each family's best status), members within a family by status rank,
    then the context rows (benchmarks, portfolio) and the ledger states. This is the
    reading order for the registry table, the matrices and the dossiers — a family's
    variants never get split by another strategy."""
    strat = [r for r in records if r.status in _STRAT_STATUS]
    rest = ordered([r for r in records if r.status not in _STRAT_STATUS])
    fam_rank = {}
    for r in strat:
        rank = _STATUS_RANK.get(r.status, 99)
        if r.family not in fam_rank or rank < fam_rank[r.family]:
            fam_rank[r.family] = rank
    fams = sorted(fam_rank, key=lambda f: fam_rank[f])
    out = []
    for f in fams:
        out.extend(sorted((r for r in strat if r.family == f),
                          key=lambda r: _STATUS_RANK.get(r.status, 99)))
    return out + rest


def assign_colors(records) -> None:
    """Fill missing colors from the theme palette, skipping colors already claimed."""
    used = {r.color for r in records if r.color}
    pool = [c for c in theme.PALETTE if c not in used]
    i = 0
    for r in records:
        if r.color is None and r.equity is not None:
            r.color = pool[i % len(pool)] if pool else theme.FG_DIM
            i += 1


# ── The ledger: killed / cut / cross-page strategies (no curves recomputed here) ──
# Verdict one-liners are copied from the lab pages' own pre-registered verdicts; the
# href is the authoritative source. Dates anchor the copy against drift.
STATIC_RECORDS = (
    StrategyRecord(
        id="edge_taxloss", name="Tax-loss rebound sleeve (Dec→Jan losers)", family="edge",
        status="cut", gate="CUT by pre-registered sleeve gate (tools/gates.py)",
        verdict="Monte-Carlo sleeve gate failed — seasonal loser-rebound edge not "
                "separable from noise (2026-07-05).", href="edge.html"),
    StrategyRecord(
        id="econo_nn", name="NN event/x-sec ranker (tanh MLP vs ridge duel)", family="econo",
        status="killed", gate="pre-registered duel, M6",
        verdict="ICs statistically twins (+0.059 vs +0.058, matched-null p 0.855); both "
                "learned rankers lose to plain momentum on validation — data budget "
                "(~26 quarterly labels), not model class (2026-07-06).", href="econo.html"),
    StrategyRecord(
        id="econo_leadlag", name="Lead-lag network momentum (M1)", family="econo",
        status="killed", gate="matched-null MC",
        verdict="Placebo ≈ signal (IC +0.001), FF5+WML α −1.1%/yr; the p 0.032 was "
                "liquidity-membership tilt — matched null p 0.55 (2026-07-04).",
        href="econo.html"),
    StrategyRecord(
        id="econo_cluster", name="Cluster-neutral selection (RMT correlation clusters, M2)",
        family="econo", status="research", gate="both-windows rule",
        verdict="Not promoted: val Sharpe 0.84 vs GICS round-robin 1.03; 252d/20-cluster "
                "labels too noisy, test-window 1.78 fluke rejected (2026-07-04).",
        href="econo.html"),
    StrategyRecord(
        id="econo_arthrottle", name="Absorption-ratio throttle (M3)", family="econo",
        status="research", gate="vs vol-target 15% incumbent",
        verdict="Observational only: loses to plain vol-targeting (val 0.99 vs 1.26, "
                "maxDD −45.9% vs −20.1%) — AR is coincident-slow (2026-07-04).",
        href="econo.html"),
    StrategyRecord(
        id="pairs", name="Pairs stat-arb (cointegration book)", family="pairs",
        status="research", gate=None,
        verdict="Research page only — never promoted to the live book.", href="pairs.html"),
    StrategyRecord(
        id="edge_stack", name="Edge stack (momentum ⊕ seasonal sleeve ⊕ EWMA overlay)",
        family="edge", status="research", gate="tools/gates.py stack_verdict",
        verdict="Sleeve cut ⇒ the stack reduces to the momentum book; page kept for the "
                "method.", href="edge.html"),
)
