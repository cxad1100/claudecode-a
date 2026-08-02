"""Verify the built /strategy page delivers on its promise: every strategy and variant
is selectable in the left menu, opens its own pane, and that pane shows real output.

Run against a built page:

    .venv/bin/python -m tools.verify_strategy_page [local/strategy.html]

It is a linter for the page's completeness contract, not a test of the numbers. It fails
loudly on the two ways the page can quietly lie: a strategy that exists in the registry
but has no way to reach it, and a pane that is empty without saying why.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_MENU = re.compile(r"data-pane='rec-([A-Za-z0-9_\-]+)'")
_PANE = re.compile(r"id='pane-rec-([A-Za-z0-9_\-]+)'")
# Each pane runs from its own id to the start of the next section (or end of document).
_SPLIT = re.compile(r"<section class='pane' id='pane-(rec-[A-Za-z0-9_\-]+|home)'")

CHART = "class='chart'"
TILES = "class='cards'"
NO_CURVE = "No equity curve is cached"


def pane_bodies(html: str) -> dict[str, str]:
    """id -> that pane's HTML, sliced between consecutive pane openings."""
    marks = [(m.group(1), m.start()) for m in _SPLIT.finditer(html)]
    out = {}
    for i, (pid, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(html)
        out[pid] = html[start:end]
    return out


def verify(html: str) -> tuple[list[str], list[str], dict]:
    """Returns (problems, notes, stats)."""
    problems, notes = [], []
    menu = list(dict.fromkeys(_MENU.findall(html)))
    panes = list(dict.fromkeys(_PANE.findall(html)))
    bodies = pane_bodies(html)

    if not menu:
        problems.append("left menu has no strategy entries at all")
    for sid in menu:
        if sid not in panes:
            problems.append(f"{sid}: menu entry with no pane — clicking it does nothing")
    for sid in panes:
        if sid not in menu:
            problems.append(f"{sid}: pane with no menu entry — unreachable")
    if "pane-home" not in bodies:
        problems.append("no default Overview pane")

    charted, honest, empty = [], [], []
    for sid in panes:
        body = bodies.get(f"rec-{sid}", "")
        if CHART in body or TILES in body:
            charted.append(sid)
        elif NO_CURVE in body:
            honest.append(sid)
        else:
            empty.append(sid)
            problems.append(f"{sid}: pane has neither output nor an explanation of the gap")

    home = bodies.get("home", "")
    for need, why in ((CHART, "the comparison chart"),
                      ("Head-to-head", "the strategies-vs-book head-to-head"),
                      ("Strategy registry", "the registry leaderboard")):
        if need not in home:
            problems.append(f"Overview is missing {why}")

    notes.append(f"{len(charted)} strategies render real output "
                 f"(curve and/or metric tiles)")
    if honest:
        notes.append(f"{len(honest)} declare a missing curve honestly: {', '.join(honest)}")
    stats = dict(menu=len(menu), panes=len(panes), charted=len(charted),
                 honest=len(honest), empty=len(empty))
    return problems, notes, stats


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "local" / "strategy.html"
    if not path.exists():
        print(f"no built page at {path} — run build_strategy_report.py first")
        return 2
    problems, notes, stats = verify(path.read_text())
    print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"  menu entries : {stats['menu']}")
    print(f"  panes        : {stats['panes']}")
    print(f"  with output  : {stats['charted']}")
    print(f"  honest gaps  : {stats['honest']}")
    for n in notes:
        print(f"  · {n}")
    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("\nOK — every strategy is reachable and every pane either shows output or "
          "says why it cannot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
