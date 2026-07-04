"""Generate the econophysics research page (LOCAL-ONLY — never a public build).

  python build_econo_report.py            # writes local/econo.html
  ECONO_LEADLAG=0 python build_econo_report.py   # skip a module (env flags)

A research lab on top of the strategy stack: new cross-sectional signals and
overlays (lead-lag networks, correlation structure, phase-transition
indicators, scraped regulatory events) run through the SAME evaluation gate as
the production momentum strategy — walk-forward `run_momentum` with costs and
PIT gates, Monte-Carlo random-book null, DSR/phantom haircut, FF5+WML spanning
alpha. Modules are guarded like the strategy page's FACTORS block: a failed or
disabled module leaves its key None and the section renders "" — the page
never breaks on a half-finished experiment. Killed experiments keep their
section (the negative result is the deliverable).
"""
import os
import webbrowser
from datetime import datetime
from pathlib import Path

from tools.report_html import page

ROOT = Path(__file__).parent
OUT = ROOT / "local" / "econo.html"

#: module key → env flag suffix; every gather block and section keys off this.
MODULES = ("leadlag", "corr", "phase", "events", "trials")


def _enabled(name: str) -> bool:
    """Module env switch, FACTORS-style: ECONO_<NAME>=0 disables, default on."""
    return os.environ.get(f"ECONO_{name.upper()}", "1") != "0"


# ── Data gathering ────────────────────────────────────────────────────────────

def gather() -> dict:
    """Run every enabled module; a disabled or failed module leaves None.

    Each block is wrapped try/except so one broken experiment (or a missing
    data file) never takes the page down — the section simply doesn't render.
    """
    d: dict = {k: None for k in MODULES}
    # Module blocks land here (M1..M5); each follows:
    #   if _enabled("leadlag"):
    #       try: d["leadlag"] = _gather_leadlag(...)
    #       except Exception as e: print(f"[econo] leadlag skipped: {e}")
    return d


# ── Sections (each returns "" when its module produced nothing) ──────────────

def sec_intro() -> str:
    return """<h1>Econophysics lab</h1>
<p class="sub">New signals and overlays, judged by the production gate:
walk-forward with costs and point-in-time universe gates, Monte-Carlo
random-book null, deflated Sharpe with phantom trials, and FF5+WML spanning
alpha (Newey–West t ≥ 2). A killed experiment stays on the page — the
negative result is the deliverable.</p>"""


def sec_leadlag(d: dict, public: bool) -> str:
    if d.get("leadlag") is None:
        return ""
    return "<h2>Lead-lag network</h2>"


def sec_corr(d: dict, public: bool) -> str:
    if d.get("corr") is None:
        return ""
    return "<h2>Correlation structure</h2>"


def sec_phase(d: dict, public: bool) -> str:
    if d.get("phase") is None:
        return ""
    return "<h2>Phase-transition indicators</h2>"


def sec_events(d: dict, public: bool) -> str:
    if d.get("events") is None:
        return ""
    return "<h2>Regulatory events</h2>"


def sec_trials(d: dict, public: bool) -> str:
    if d.get("trials") is None:
        return ""
    return "<h2>Trial ledger</h2>"


def sec_footer() -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f'<p class="footnote">Built {ts} · local-only research page</p>'


def build(d: dict, public: bool = False) -> str:
    body = "".join([
        sec_intro(),
        sec_leadlag(d, public),
        sec_corr(d, public),
        sec_phase(d, public),
        sec_events(d, public),
        sec_trials(d, public),
        sec_footer(),
    ])
    return page("Econophysics lab", body)


def main() -> None:
    d = gather()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(d), encoding="utf-8")
    print(f"wrote {OUT}")
    if os.environ.get("ECONO_OPEN"):
        webbrowser.open(OUT.as_uri())


if __name__ == "__main__":
    main()
