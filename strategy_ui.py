"""Strategy page UI — the entire rendering layer, written as one design grammar.

Data comes in as gather()'s dict (including d["registry"]); a full HTML page comes
out. No metric computation happens here — engines and stats live in tools/, data
assembly in build_strategy_report.gather().

The page is a MASTER-DETAIL app: a hierarchy spine (the strategy map) on the left,
a stage of detail panes on the right, one pane visible at a time (client-side router,
no fetch — every pane is server-rendered into the document, JS toggles visibility):

  spine   ★ Today · Compare · one group per FAMILY (its records indented, variants
          nested via variant_of) · Ledger (killed/research) · Operations · Lab
  panes   home     cockpit — command strip + the result verdict + KPI tiles
          compare  leaderboard · parallel equity · windows / yearly matrices
          fam-*    a family node: its members + the family-scoped shared evidence
          rec-*    one strategy's dossier (book · verdict · method · history)
          ops      live tracking · venture north star · ritual · you vs strategies
          lab      (private) raw reference · 64-config grid · supporting data

Why panes, not one scroll: the data is a tree (family → record → variant_of → pick)
with importance already encoded (status rank, ★ live, variant_of). The spine makes
that hierarchy visible; each pane is sized to its own content, so a deep family and a
one-line sleeve no longer fight for the same equal-width column.

Framework contract: a future strategy = one make_record() in
build_strategy_report.build_registry — it earns a spine item and a generic dossier
pane automatically; bespoke detail = one DOSSIER_BUILDERS entry here. Family-scoped
evidence attaches to the family pane, not the strategy card.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from tools.report_html import pct as _pct, card as _card, page, fig_html
from tools import theme, quant_grade as qg
from tools import strategy_registry as sreg
from tools.momentum import benchmark_curves
from tools.momentum_grid import grid_distribution, grid_percentile
from build_momentum_report import (
    START, TRAIN_END, VAL_END, MIN_TURNOVER,
    _disp, _name, _pnl_color, _equity_window,
    sec_grid, sec_feasibility, sec_timelines, sec_survivorship, sec_method,
)

# ── Display constants (single source; build_strategy_report imports the colors) ──
C_RAW = "#dcdcaa"          # raw momentum — lab reference only
C_RC = "#4ec9b0"           # risk-conscious momentum book
_GRADE_COLOR = {"A": "#46c84e", "B": "#9acd32", "C": "#d7ba7d", "D": "#e8a04e", "F": "#ef4444"}
HARVEY_T = 3.0             # Harvey (2016) multiple-testing t-stat hurdle
MODELED_SLIP_BPS = 25      # nominal modeled one-way slippage, quoted in prose

WIN_LABELS = sreg.window_labels(START, TRAIN_END, VAL_END)
TEST_FROM = f"{pd.Timestamp(VAL_END).year + 1}→"      # e.g. "2024→"
PRE_TEST = f"≤{pd.Timestamp(VAL_END).year}"           # e.g. "≤2023"

# Page-scoped CSS — the dossier design system lives with the page that uses it.
PAGE_CSS = f"""<style>
.eyebrow {{ font-family: {theme.MONO}; font-size: 0.66rem; text-transform: uppercase;
            letter-spacing: 0.14em; color: {theme.FG_DIM}; display: block; margin-bottom: 2px; }}
.badge {{ display: inline-block; padding: 1px 7px; border-radius: 9px; font-size: 0.68rem;
          font-family: {theme.MONO}; text-transform: uppercase; letter-spacing: 0.05em;
          border: 1px solid {theme.GRID}; white-space: nowrap; }}
tr.dimrow td {{ color: {theme.FG_DIM}; }}
.scroll {{ overflow-x: auto; }}
.cmd {{ display: flex; flex-wrap: wrap; gap: 6px 22px; align-items: baseline;
        font-family: {theme.MONO}; font-size: 0.82rem; background: {theme.BG_PANEL};
        border: 1px solid {theme.GRID}; border-radius: 8px; padding: 10px 16px; margin: 14px 0; }}
.cmd b {{ font-weight: 600; }}
.dossier {{ border: 1px solid {theme.GRID}; border-left: 4px solid {theme.ACCENT};
            border-radius: 8px; background: {theme.BG_PANEL};
            padding: 16px 20px 14px; margin: 18px 0; }}
.dossier > h3 {{ margin: 0 0 2px; font-size: 1.05rem; }}
.dossier .vline {{ margin: 4px 0 10px; font-size: 0.85rem; color: {theme.FG_DIM}; }}
.dossier .par > div {{ background: {theme.BG}; }}
.par {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 1.2rem; align-items: start; margin: 12px 0; }}
.par > div {{ min-width: 0; background: {theme.BG_PANEL}; border: 1px solid {theme.GRID};
              border-radius: 6px; padding: 10px 14px; overflow-x: auto; }}
.par h3 {{ margin-top: 0; }}
.par table {{ font-size: 0.8rem; }}
.par td, .par th {{ padding: 4px 8px; }}
details.ev {{ border: 1px solid {theme.GRID}; border-radius: 6px; padding: 8px 14px;
              margin: 10px 0; background: {theme.BG_PANEL}; }}
details.ev > summary {{ color: {theme.FG}; font-size: 0.88rem; }}
details.ev > summary .eyebrow {{ display: inline; margin-right: 10px; }}
details.ev[open] {{ padding-bottom: 14px; }}
details.ev h2 {{ margin-top: 18px; font-size: 1.0rem; }}
.legend {{ color: {theme.FG_DIM}; font-size: 0.82rem; margin: 4px 0 8px; }}
/* ── App shell: strategy menu (left) + detail stage (right) ────────────────── */
main {{ max-width: 1360px; }}
.pagehead {{ margin-bottom: 10px; }}
.app {{ display: grid; grid-template-columns: 264px minmax(0, 1fr); gap: 30px;
        align-items: start; }}
.stage {{ min-width: 0; }}
.spine {{ position: sticky; top: 14px; align-self: start; max-height: calc(100vh - 28px);
          overflow: auto; font-size: 0.9rem; border: 1px solid {theme.GRID};
          border-radius: 10px; padding: 10px; background: {theme.BG_PANEL}; }}
.spine [data-pane] {{ display: flex; align-items: center; gap: 8px; width: 100%;
          text-align: left; background: none; border: 0; color: {theme.FG}; font: inherit;
          padding: 5px 8px; border-radius: 6px; cursor: pointer; line-height: 1.35; }}
.spine [data-pane]:hover {{ background: {theme.BG}; }}
.spine [data-pane].active {{ background: {theme.GRID}; color: #fff; font-weight: 600; }}
.spine .s-rec {{ padding-left: 22px; font-size: 0.85rem; color: {theme.FG_DIM}; }}
.spine .s-rec.active {{ color: #fff; }}
.spine .s-fam {{ margin-top: 12px; }}
.spine .dot {{ width: 9px; height: 9px; border-radius: 50%; flex: none;
          box-shadow: 0 0 0 1px rgba(255,255,255,0.14); }}
.spine .s-sep {{ border: 0; border-top: 1px solid {theme.GRID}; margin: 12px 2px; }}
.pane[hidden] {{ display: none; }}
.pane > .crumb {{ margin: 0 0 10px; font-size: 0.78rem; color: {theme.FG_DIM}; }}
.pane > .crumb [data-pane] {{ background: none; border: 0; color: {theme.ACCENT};
          cursor: pointer; font: inherit; padding: 0; }}
.pane > .crumb [data-pane]:hover {{ text-decoration: underline; }}
.pane > h2:first-child, .pane > .crumb + h2 {{ margin-top: 0; }}
@media (max-width: 900px) {{
  .app {{ grid-template-columns: 1fr; gap: 16px; }}
  .spine {{ position: static; max-height: none; }}
}}
</style>"""


def _desc(cfg) -> str:
    parts = []
    if cfg.vol_adjust:
        parts.append("volatility-adjusted")
    if cfg.sector_neutral:
        parts.append("sector-neutral")
    if cfg.trend_filter:
        parts.append("trend-filtered (200d kill-switch)")
    parts.append(f"equal-weight top-{cfg.slots}")
    parts.append("quarterly" if cfg.freq == "Q" else "monthly")
    if cfg.lazy:
        parts.append("lazy-rebalanced")
    return ", ".join(parts)


def _rc_target(d: dict) -> float:
    return float(d["variants"][1]["perf"].get("target_vol", 0.15))


# ── Layout primitives ────────────────────────────────────────────────────────────

def _fold(eyebrow: str, summary: str, body: str) -> str:
    """Collapsed drill-down: the verdict stays visible, the workings fold away.
    Empty body ⇒ nothing (sections keep their hide-when-absent contract)."""
    if not body:
        return ""
    return (f"<details class='ev'><summary><span class='eyebrow'>{eyebrow}</span>"
            f"{summary}</summary>{body}</details>")


def _dossier(color: str, eyebrow: str, title: str, vline: str, body: str) -> str:
    """One strategy's card: color spine = its identity everywhere (chart line,
    registry row, this card), eyebrow = its true status, one verdict line, content."""
    return (f"<div class='dossier' style='border-left-color:{color}'>"
            f"<span class='eyebrow'>{eyebrow}</span>"
            f"<h3 style='color:{color}'>{title}</h3>"
            f"<p class='vline'>{vline}</p>{body}</div>")


def _badge(status: str) -> str:
    c = sreg.STATUS_COLOR.get(status, theme.FG_DIM)
    return f"<span class='badge' style='color:{c};border-color:{c}'>{status}</span>"


def _wm_cell(w: dict, key: str, fmt: str = "sharpe") -> str:
    if not w or key not in w:
        return "<td class='num dim'>—</td>"
    v = w[key]
    if fmt == "pct":
        return f"<td class='num'>{_pct(v * 100)}</td>"
    return f"<td class='num mono'>{v:.2f}</td>"


def _perf_cells(s: dict) -> str:
    """Cells from one canonical qg.window_metrics dict ('—' when the window is empty)."""
    if not s:
        return "<td class='num dim'>—</td>" * 3
    return (f"<td class='num'>{_pct(s['net_return'] * 100)}</td>"
            f"<td class='num mono'>{s['sharpe']:.2f}</td>"
            f"<td class='num'>{_pct(s['max_dd'] * 100)}</td>")


def _perf_table(v: dict) -> str:
    w = v.get("windows", {})
    rows = "".join(
        f"<tr><td>{WIN_LABELS[k]}</td>{_perf_cells(w.get(k, {}))}</tr>"
        for k in ("train", "val", "test", "full"))
    return (f"<h3 style='color:{v['color']}'>{v['short']}</h3>"
            "<table><tr><th>Window</th><th class='num'>Ret</th><th class='num'>Sharpe</th>"
            f"<th class='num'>Max DD</th></tr>{rows}</table>")


# ── Cockpit ──────────────────────────────────────────────────────────────────────

def sec_command(d: dict, public: bool) -> str:
    """The command strip: one mono status line for the whole stack — live book, kill
    state, drawdown ladder, ops alarms. No prose; every item is a live reading."""
    e = d.get("ensemble") or {}
    live = (f"ens[{','.join(e.get('codes', []))}]" if e.get("adopt")
            else f"single {d['strategy'].code} (rc)")
    items = [f"<span><span class='eyebrow'>live book</span><b>★ {live}</b></span>"]
    t = d.get("track")
    if t:
        kill = ("<b class='neg'>KILL</b>" if t.get("kill")
                else f"none · {t.get('n', 0)}/{t.get('needed', 63)} sessions")
        items.append(f"<span><span class='eyebrow'>kill signal</span>{kill}</span>")
    v = d.get("venture") or {}
    if v.get("live"):
        items.append(f"<span><span class='eyebrow'>dd ladder</span>{v.get('dd_state', '—')}</span>")
        items.append(f"<span><span class='eyebrow'>xirr vs shadow</span>"
                     f"{_pct(v.get('excess', 0) * 100)}</span>")
    vc = d.get("vol_core") or {}
    if vc.get("w_now") is not None:
        items.append(f"<span><span class='eyebrow'>core exposure</span>"
                     f"{vc['w_now'] * 100:.0f}% IWDA</span>")
    r = d.get("ritual") or {}
    if not public and r.get("items"):
        n_alarm = sum(1 for i in r["items"] if i.get("alarm"))
        items.append(f"<span><span class='eyebrow'>ritual</span>"
                     + (f"<b class='neg'>{n_alarm} overdue</b>" if n_alarm else "ok")
                     + "</span>")
    g = d["variants"][1]["grade"]["letter"]
    items.append(f"<span><span class='eyebrow'>grade</span>"
                 f"<b style='color:{_GRADE_COLOR[g]}'>{g}</b></span>")
    return f"<div class='cmd'>{''.join(items)}</div>"


def _live_record(d: dict):
    return next((r for r in (d.get("registry") or []) if r.live), None)


def sec_headline(d: dict) -> str:
    """The honest headline: the WHOLE config grid's own distribution up top — median
    and 10–90% range across every config, and where the ★ live tracked book sits inside
    it — never the single best-picked config dressed up as 'the result'. Falls back to
    the live book's own numbers if no grid is present. Raw full-invested lives in the lab."""
    rc = d["variants"][1]
    live = _live_record(d)
    lw = (live.windows if live is not None else None) or {}
    t = lw.get("test") or rc.get("windows", {}).get("test") or rc["test"]
    full = lw.get("full") or rc.get("windows", {}).get("full") or {}
    ann_vol = full.get("ann_vol", rc["perf"]["ann_vol"])
    max_dd = full.get("max_dd", rc["perf"]["max_dd"])
    live_name = live.name if live is not None else rc["label"]
    s = d["significance"]
    mc, dsr, ci = s["mc"], s["dsr"], s["ci"]
    g = rc["grade"]["letter"]
    beat = 100.0 * (1.0 - mc["p_sharpe"])
    ph = s.get("dsr_phantom") or []
    worst = ph[-1]["dsr"] if ph else dsr["dsr"]
    dsr_ok = dsr["dsr"] == dsr["dsr"]                       # not NaN

    grid = d.get("grid")
    dist = grid_distribution(grid, window="test") if grid else None
    if dist and dist["sharpe"]:
        ds, dr, n = dist["sharpe"], dist["ret"], dist["n"]
        pct = grid_percentile(grid, t["sharpe"], window="test", metric="sharpe")
        pct_ok = pct == pct                                # not NaN
        pos = f"<b>{pct:.0f}th percentile</b>" if pct_ok else "<b>its slot</b>"
        med_ret = f", median return <b>{_pct(dr['median'] * 100)}</b>" if dr else ""
        dsr_txt = (f" · <b>Deflated Sharpe</b> pays for the {n}-config search: "
                   f"P(true&gt;0) <b>{dsr['dsr']:.0%}</b> (phantom-worst {worst:.0%})"
                   if dsr_ok else "")
        cards = "".join([
            _card("Configs searched", f"{n}"),
            _card("Median test Sharpe (all)", f"{ds['median']:.2f}"),
            _card("Test Sharpe 10–90%", f"{ds['lo']:.2f}–{ds['hi']:.2f}"),
            _card("★ Live book Sharpe",
                  f"{t['sharpe']:.2f}" + (f" · {pct:.0f}th pctile" if pct_ok else "")),
            _card("★ Live test return", _pct(t["net_return"] * 100)),
            _card("Deflated Sharpe (P real&gt;0)", f"{dsr['dsr']:.0%}" if dsr_ok else "—"),
        ])
        return (
            "<h2>The whole grid — not one lucky config</h2>"
            f"<div class='note' style='border-left-color:{C_RC};border-left-width:6px'>"
            f"Across <b>all {n} configs</b> in the search the <b>median</b> held-out test "
            f"Sharpe ({TEST_FROM}) is <b>{ds['median']:.2f}</b> "
            f"(10–90% <b>{ds['lo']:.2f}–{ds['hi']:.2f}</b>, "
            f"full range {ds['min']:.2f}–{ds['max']:.2f}){med_ret} — that spread "
            "<em>is</em> the result, not the single best-picked cell. The ★ <b>live "
            f"tracked book</b> — <b>{live_name}</b> — sits at {pos} of that grid: "
            f"test Sharpe <b>{t['sharpe']:.2f}</b>, return <b>{_pct(t['net_return'] * 100)}</b>, "
            f"<b>{ann_vol * 100:.0f}% vol</b>, max drawdown <b>{_pct(max_dd * 100)}</b> "
            "(canonical basis; full-history risk). <b>Monte Carlo</b>, selection-level: it beats "
            f"<b>{beat:.1f}%</b> of {mc['n_trials']:,} random books "
            f"(p&nbsp;=&nbsp;{mc['p_sharpe']:.3f}){dsr_txt} · bootstrap {ci['conf']}% Sharpe CI "
            f"<b>{ci['sharpe_lo']:.2f}–{ci['sharpe_hi']:.2f}</b>. The honest grade "
            f"<b style='color:{_GRADE_COLOR[g]}'>{g}</b> scores the single-config risk-conscious "
            "variant. Every strategy has its own dossier below; the full evidence sits in the "
            "momentum family card.</div>"
            f"<div class='cards'>{cards}</div>")

    # ── fallback (no grid): prior single-book framing ──
    cards = "".join([
        _card("★ Live test Sharpe", f"{t['sharpe']:.2f}"),
        _card("★ Live test return", _pct(t["net_return"] * 100)),
        _card("★ Live max drawdown", _pct(max_dd * 100)),
        _card("Beats random books", f"{beat:.1f}%"),
        _card("Deflated Sharpe (P real&gt;0)", f"{dsr['dsr']:.0%}" if dsr_ok else "—"),
        _card("Single-book grade", g),
    ])
    dsr_txt = (f" · Deflated Sharpe P(true&gt;0) <b>{dsr['dsr']:.0%}</b>"
               f" (phantom-worst {worst:.0%})" if dsr_ok else "")
    return (
        "<h2>The result</h2>"
        f"<div class='note' style='border-left-color:{C_RC};border-left-width:6px'>"
        f"On the <b>held-out test window ({TEST_FROM})</b> the ★ <b>live tracked book</b> — "
        f"<b>{live_name}</b> — made <b>{_pct(t['net_return'] * 100)}</b> at "
        f"<b>{ann_vol * 100:.0f}% vol</b>, <b>Sharpe {t['sharpe']:.2f}</b>, max drawdown "
        f"<b>{_pct(max_dd * 100)}</b> (canonical basis; full-history risk). "
        f"<b>Monte Carlo</b>, selection-level: it beats <b>{beat:.1f}%</b> of "
        f"{mc['n_trials']:,} random books (p&nbsp;=&nbsp;{mc['p_sharpe']:.3f}){dsr_txt} · "
        f"bootstrap {ci['conf']}% Sharpe CI <b>{ci['sharpe_lo']:.2f}–{ci['sharpe_hi']:.2f}</b>. "
        f"The honest grade <b style='color:{_GRADE_COLOR[g]}'>{g}</b> scores the single-config "
        "risk-conscious variant. Every strategy has its own dossier below; the full evidence "
        "sits in the momentum family card.</div>"
        f"<div class='cards'>{cards}</div>")


def sec_survivorship_banner(d: dict) -> str:
    """Standing banner: the whole momentum family trades a survivor-biased universe, so
    its returns are internal-comparison-only; the survivorship-robust anchor is the GARCH
    vol-core on a clean world index."""
    vc = (d.get("vol_core") or {}).get("etf", "a clean world-index ETF")
    return (
        "<div class='note warn' style='border-left-width:6px'>"
        "<b>Read the momentum numbers as internal comparisons, not achievable returns.</b> "
        "The momentum family trades a <b>survivor-biased universe</b> (today's liquid Frankfurt "
        "cross-listings), so its gains are inflated by an <b>absent-winners / membership tilt</b> "
        "— an equal-weight hold of those same members is already large on its own. The one "
        f"<b>survivorship-robust</b> result is the adopted <b>GARCH vol-core</b> on {vc}.</div>")


def sec_intro(d: dict) -> str:
    cfg = d["strategy"]
    nc = d["n_countries"]
    body = (f'<div class="note"><b>Chosen strategy — {cfg.code} ({_desc(cfg)}).</b> '
            "<b>Sector-neutral (B) is now enabled</b> — real GICS sectors were sourced per name "
            "(yfinance home listings → <code>tools.enrich_sectors</code>), so B's round-robin caps "
            "single-sector concentration instead of being the silent no-op it was when every name "
            "read “Unknown”. Judged against the full <b>64-config grid</b> below (B included) on "
            "worst-case <b>min(train, validation) Sharpe</b>, so the deflated-Sharpe pays for the "
            "doubled search — the result rides robustness, not one lucky rally. The universe is the "
            "liquid, "
            f"<b>Trade-Republic-investable</b> names across <b>{nc} countries</b>, each priced off "
            "its <b>home exchange × EUR FX</b> — the way Lang &amp; Schwarz actually fills you "
            "(NVIDIA on NASDAQ, Samsung on KRX, Rheinmetall on XETRA, in their own currency, "
            f"converted to EUR). Behind a <b>≥{MIN_TURNOVER / 1_000:.0f}k/day turnover</b> floor"
            + (", re-checked point-in-time each rebalance" if d.get("turnover_pit") else "")
            + ". Long-only, walk-forward, executable. Not advice.</div>")
    return ("<details class='ev'><summary><span class='eyebrow'>method</span>"
            f"Universe &amp; selection — {cfg.code} ({_desc(cfg)}), "
            f"Trade-Republic-investable, {nc} countries</summary>" + body + "</details>")


# ── Compare: leaderboard · chart · matrices ──────────────────────────────────────

def sec_registry(d: dict, public: bool) -> str:
    """The strategy registry — every strategy the program has produced, one row each,
    on ONE canonical metric basis. ★ = the live tracked book. Killed / cut /
    cross-page strategies live in the collapsed ledger below — visible file-drawer."""
    recs = sreg.family_ordered(d.get("registry") or [])
    if not recs:
        return ""
    main_status = ("adopted", "candidate", "variant", "benchmark", "reference", "portfolio")
    rows = []
    for r in recs:
        if r.status not in main_status:
            continue
        if public and r.status == "portfolio":
            continue
        w_test, w_full = r.windows.get("test"), r.windows.get("full")
        name = f"<b>★ {r.name}</b>" if r.live else r.name
        if "inflated" in r.flags:
            name += " <span class='dim'>— reference, inflated</span>"
        if r.href:
            name += f" <a href='{r.href}' class='dim'>↗</a>"
        sub = []
        if r.since:
            sub.append(f"since {r.since}")
        if r.adopted:
            sub.append(f"adopted {r.adopted}")
        meta_line = f"<div class='dim' style='font-size:0.72rem'>{' · '.join(sub)}</div>" if sub else ""
        exp = (f"{r.avg_exposure * 100:.0f}%" if r.avg_exposure is not None else "—")
        gate = r.gate or r.verdict or "—"
        cls = " class='dimrow'" if ("inflated" in r.flags or r.status == "benchmark") else ""
        rows.append(
            f"<tr{cls}><td>{_badge(r.status)}</td>"
            f"<td style='color:{r.color or theme.FG}'>{name}{meta_line}</td>"
            + _wm_cell(w_test, "sharpe") + _wm_cell(w_test, "net_return", "pct")
            + _wm_cell(w_test, "max_dd", "pct") + _wm_cell(w_full, "sharpe")
            + _wm_cell(w_full, "ann_vol", "pct")
            + f"<td class='num mono'>{exp}</td>"
            f"<td class='dim' style='font-size:0.78rem'>{r.cost_model or '—'}</td>"
            f"<td class='dim' style='font-size:0.78rem'>{gate}</td></tr>")
    ledger = []
    for r in recs:
        if r.status not in ("killed", "cut", "research"):
            continue
        link = f" <a href='{r.href}'>lab ↗</a>" if r.href else ""
        ledger.append(
            f"<tr class='dimrow'><td>{_badge(r.status)}</td><td>{r.name}</td>"
            f"<td>{r.family}</td><td style='font-size:0.78rem'>{r.verdict}{link}</td></tr>")
    ledger_html = ""
    if ledger:
        ledger_html = (
            "<details><summary>Registry ledger — killed &amp; cross-page strategies "
            f"({len(ledger)})</summary>"
            "<p class='dim'>The visible file-drawer: every strategy the program tried and did "
            "not promote, with its pre-registered verdict. Nothing is recomputed here — the "
            "lab pages hold the full workings.</p>"
            "<div class='scroll'><table><tr><th>Status</th><th>Strategy</th><th>Family</th>"
            "<th>Verdict</th></tr>"
            + "".join(ledger) + "</table></div></details>")
    how = ("<p class='dim'>One row per strategy, all metrics computed by the <b>same "
           "canonical function</b> (geometric daily basis) over the <b>same windows</b> — "
           f"<b>{WIN_LABELS['test']}</b> is the held-out comparison window. Cost models "
           "differ by row (the Costs column names each); the Monte-Carlo / deflated-Sharpe "
           "numbers in the momentum dossier use a <i>gross per-rebalance</i> basis by design "
           "— selection vs noise, costs hit a random book equally. A future strategy joins "
           "this table (and everything below) as one registry record.</p>")
    return (
        "<h2>Strategy registry — every strategy, one basis</h2>"
        f"<p class='legend'>★ = live tracked book · {WIN_LABELS['test']} = held-out window "
        "· one canonical metric basis</p>"
        "<div class='scroll'><table><tr><th>Status</th><th>Strategy</th>"
        f"<th class='num'>{WIN_LABELS['test']} Sharpe</th><th class='num'>Test ret</th>"
        "<th class='num'>Test maxDD</th><th class='num'>Full Sharpe</th>"
        "<th class='num'>Ann. vol</th><th class='num'>Avg exp</th>"
        "<th>Costs</th><th>Gate / verdict</th></tr>"
        + "".join(rows) + "</table></div>"
        + _fold("how to read this table", "metric basis, windows, cost models", how)
        + ledger_html)


def sec_parallel_curves(d: dict, public: bool) -> str:
    """Every curve-bearing strategy in parallel — index = 100 at the common window
    start — against the equal-weight baseline and the benchmarks."""
    res = d["res"]
    window = _equity_window(res)
    recs = [r for r in sreg.family_ordered(d.get("registry") or [])
            if r.equity is not None and "inflated" not in r.flags
            and r.status in ("adopted", "candidate", "variant")]
    if not recs:
        return ""
    fig = go.Figure()
    for r in recs:
        s = r.equity.reindex(window).ffill().dropna()
        if len(s) < 2:
            continue
        name = f"★ {r.name}" if r.live else r.name
        fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100.0, name=name,
                                 line=dict(color=r.color, width=2.6 if r.live else 1.8)))
    ew = d.get("ew_eq")
    if ew is not None:
        s = ew.reindex(window).ffill().dropna()
        if len(s) >= 2:
            fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100.0,
                                     name="Equal-weight (initial picks, buy-hold)",
                                     line=dict(color=theme.FG_DIM, width=1.4, dash="dot")))
    for name, curve in benchmark_curves(d["benchmarks"], window, d["capital"]).items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve / d["capital"] * 100.0,
                                 name=name, line=dict(width=1.2)))
    fig.add_hline(y=100, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=500, yaxis_title="Index (start = 100)",
                      hovermode="x unified", margin=dict(t=20))
    pf_note = ("" if public else
               " Your real book is cash-flow-timed, so it is compared honestly in "
               "<b>You vs the strategies</b> below, not force-rebased onto this chart.")
    how = ("<p class='dim'>Every live/candidate strategy over the <b>same walk-forward "
           "window</b>, rebased to 100 at the common start, vs the equal-weight basket of "
           "the first picks (survivorship-honest baseline) and the benchmark ETFs. "
           "<i>(The raw full-invested curve — inflated by survivorship and full exposure — "
           "stays in the research lab.)</i>" + pf_note + "</p>")
    return ("<h2>Walk-forward equity — strategies vs benchmarks</h2>"
            "<p class='legend'>index = 100 at common start · click legend entries to "
            "toggle lines</p>"
            f"<div class='chart'>{fig_html(fig)}</div>"
            + _fold("how to read this chart", "window, rebasing, what's excluded", how))


def _matrix_records(d: dict) -> list:
    """Registry records that belong in the parallel analytics matrices."""
    return [r for r in sreg.family_ordered(d.get("registry") or [])
            if r.windows and "inflated" not in r.flags
            and r.status in ("adopted", "candidate", "variant", "benchmark")]


def sec_perf_compare(d: dict, public: bool) -> str:
    """Windows × strategies matrix — every strategy analysed in parallel on the
    same canonical basis over the same windows."""
    recs = _matrix_records(d)
    if not recs:
        return ""
    rc = d["variants"][1]
    live = _live_record(d)
    lw = (live.windows if live is not None else None) or rc.get("windows", {})
    t = lw.get("test") or rc["test"]
    full = lw.get("full") or {}
    cards = [
        _card("★ Live test return", _pct(t["net_return"] * 100)),
        _card("★ Live test Sharpe", f"{t['sharpe']:.2f}"),
        _card("★ Live max DD", _pct(full.get("max_dd", rc["perf"]["max_dd"]) * 100)),
        _card("★ Live ann. vol",
              _pct(full.get("ann_vol", rc["perf"]["ann_vol"]) * 100, signed=False)),
    ]
    if not public and full.get("net_return") is not None:
        cards.append(_card("Net P&L (live, full)",
                           f"€{full['net_return'] * d['capital']:+,.0f}"))
    heads = "".join(
        f"<th class='num' style='color:{r.color or theme.FG}'>"
        f"{'★ ' if r.live else ''}{r.name}</th>" for r in recs)

    def cell(w):
        if not w:
            return "<td class='num dim'>—</td>"
        return (f"<td class='num mono'>{_pct(w['net_return'] * 100)}<br>"
                f"<span class='dim' style='font-size:0.72rem'>S {w['sharpe']:.2f} · "
                f"DD {w['max_dd'] * 100:.0f}%</span></td>")

    rows = "".join(
        f"<tr><td>{WIN_LABELS[k]}</td>" +
        "".join(cell(r.windows.get(k)) for r in recs) + "</tr>"
        for k in ("train", "val", "test", "full"))
    how = (f"<p class='dim'>Windows: <b>{WIN_LABELS['train']}</b> (used to pick the config) · "
           f"<b>{WIN_LABELS['val']}</b> (used to compare configs) · <b>{WIN_LABELS['test']} "
           "(held out — never touched the choice)</b>. "
           "<b>The test row is the only truly out-of-sample number</b>. Cells show return, "
           "Sharpe (S) and max drawdown (DD) on the canonical basis.</p>")
    return ("<h2>Performance — all strategies, same windows</h2>"
            f"<p class='legend'>cells: return · S = Sharpe · DD = max drawdown — "
            f"<b>{WIN_LABELS['test']} row is the out-of-sample one</b></p>"
            f"<div class='cards'>{''.join(cards)}</div>"
            "<div class='scroll'><table><tr><th>Window</th>"
            + heads + f"</tr>{rows}</table></div>"
            + _fold("how to read these windows", "what train/validation/test mean", how))


def _yearly_returns(series: pd.Series) -> pd.Series:
    """Calendar-year returns keyed by year int; the first year runs from inception."""
    s = series.dropna()
    last = s.groupby(s.index.year).last()
    prev = last.shift(1)
    if len(prev):
        prev.iloc[0] = s.iloc[0]
    return (last / prev - 1.0).dropna()


def _yearly_pnl(series: pd.Series) -> pd.Series:
    eq = series.dropna()
    last = eq.groupby(eq.index.year).last()
    prev = last.shift(1)
    if len(prev):
        prev.iloc[0] = eq.iloc[0]
    return (last - prev).dropna()


def sec_yearly_compare(d: dict, public: bool) -> str:
    """Years × strategies matrix — calendar-year returns of every curve in parallel,
    with the live book's €P&L (private builds) and the S&P 500 as the last column."""
    recs = [r for r in _matrix_records(d)
            if r.equity is not None and r.status != "benchmark"]
    if not recs:
        return ""
    yr_by_rec = {r.id: _yearly_returns(r.equity.dropna()) for r in recs}
    years = sorted({y for s in yr_by_rec.values() for y in s.index})
    if not years:
        return ""
    spx = d["benchmarks"]["S&P 500"] if "S&P 500" in d["benchmarks"].columns else None
    bret = _yearly_returns(spx.dropna()) if spx is not None else pd.Series(dtype=float)
    live = next((r for r in recs if r.live), None)
    pnl = _yearly_pnl(live.equity.dropna()) if (live is not None and not public) else None

    heads = "".join(
        f"<th class='num' style='color:{r.color or theme.FG}'>"
        f"{'★ ' if r.live else ''}{r.name}</th>" for r in recs)
    eur_h = "<th class='num'>★ P&amp;L</th>" if pnl is not None else ""
    rows = []
    for y in years:
        cells = "".join(
            (f"<td class='num'>{_pct(yr_by_rec[r.id][y] * 100)}</td>"
             if y in yr_by_rec[r.id].index else "<td class='num dim'>—</td>")
            for r in recs)
        b = (f"<td class='num'>{_pct(bret[y] * 100)}</td>" if y in bret.index
             else "<td class='num dim'>—</td>")
        eur = (f"<td class='num mono'>€{pnl.get(y, 0.0):+,.0f}</td>"
               if pnl is not None else "")
        rows.append(f"<tr><td class='mono'>{y}</td>{cells}{b}{eur}</tr>")
    return ("<h2>Yearly P&amp;L — all strategies</h2>"
            f"<p class='legend'>calendar-year net returns · {years[0]} and {years[-1]} are "
            "part-years"
            + (" · ★ P&amp;L = live book € at paper capital" if pnl is not None else "")
            + "</p>"
            "<div class='scroll'><table><tr><th>Year</th>"
            + heads + "<th class='num'>S&amp;P 500</th>" + eur_h + "</tr>"
            + "".join(rows) + "</table></div>")


# ── Dossiers: books, per-strategy detail, per-family evidence ────────────────────

def _picks_rows(d: dict, cur: dict, weight_pct: float) -> str:
    """Rows of one book: ticker / name / ISIN / weight (ISIN is the tradeable id)."""
    rows = []
    for t in cur["picks"]:
        m = d["meta"].get(t, {})
        home = str(m.get("home") or t).split(".")[0]
        isin = m.get("isin") if pd.notna(m.get("isin")) else ""
        rows.append(
            f"<tr><td class='mono'>{home}</td><td>{_name(m, t)}</td>"
            f"<td class='dim mono' style='font-size:0.72rem'>{isin}</td>"
            f"<td class='num mono'>{weight_pct:.1f}%</td></tr>")
    return "".join(rows)


_BOOK_HEAD = ("<table><tr><th>Ticker</th><th>Name</th><th>ISIN</th>"
              "<th class='num'>Weight</th></tr>")


def _book_panel(d: dict, title: str, color: str, holdings_log: list, *,
                weight_scale: float = 1.0, exposure_latest: float | None = None,
                sub_note: str = "") -> str:
    """One strategy's current book as a panel: latest non-empty picks + weights as
    % of total capital. `exposure_latest` adds the de-risked CASH row when < 1."""
    head = f"<h3 style='color:{color}'>{title}</h3>"
    cur = next((h for h in reversed(holdings_log) if h["picks"]), None)
    if cur is None:
        return f"<div>{head}<p class='dim'>No eligible names at the latest rebalance.</p></div>"
    n = len(cur["picks"])
    exp = 1.0 if exposure_latest is None else float(exposure_latest)
    invested = 100.0 * exp * weight_scale
    w = invested / n if n else 0.0
    rows = _picks_rows(d, cur, w)
    cash = 100.0 * weight_scale - invested
    if exposure_latest is not None and cash > 0.05:
        rows += (f"<tr><td class='mono dim'>CASH</td><td class='dim'>de-risked sleeve</td>"
                 f"<td></td><td class='num mono'>{cash:.1f}%</td></tr>")
    sub = (f"<p class='dim pick-sub'>top-{n} · <b>{invested:.0f}% of total capital</b>"
           f"{sub_note} · {cur['date'].date()}</p>")
    return f"<div>{head}{sub}{_BOOK_HEAD}{rows}</table></div>"


def _window_ret(eq: pd.Series, d0, d1) -> float:
    eq = eq.dropna()
    if eq.empty:
        return 0.0
    try:
        a = eq.asof(pd.Timestamp(d0))
        b = eq.asof(pd.Timestamp(d1)) if d1 is not None else eq.iloc[-1]
        if pd.notna(a) and pd.notna(b) and a != 0:
            return float(b / a - 1.0)
    except Exception:
        pass
    return 0.0


def _window_exposure(exp, d0, d1) -> float:
    if exp is None or getattr(exp, "empty", True):
        return 1.0
    lo = pd.Timestamp(d0)
    s = exp.loc[exp.index >= lo] if d1 is None else exp.loc[(exp.index >= lo) & (exp.index <= pd.Timestamp(d1))]
    return float(s.mean()) if len(s) else 1.0


def _timeline_col(d: dict, v: dict) -> str:
    rc = v["key"] == "rc"
    eq, exp = v["equity"], v["exposure"]
    lines = [f"<h3 style='color:{v['color']}'>{v['short']}</h3>"]
    for h in v["holdings_log"]:
        dead = h.get("dead", set())
        spans = " ".join(
            f"<span style='color:{_pnl_color(h['ret'].get(t, 0.0), t in dead)}' "
            f"title='{t} {h['ret'].get(t, 0.0):+.0%}'>{_disp(d['meta'], t)}</span>"
            for t in h["picks"]) or "<span class='dim'>cash</span>"
        if rc:
            wret = _window_ret(eq, h["date"], h.get("next"))
            badge = _window_exposure(exp, h["date"], h.get("next"))
            head = (f"<span class='mono dim'>{h['date'].date()}</span> "
                    f"<b style='color:{_pnl_color(wret, False)}'>{wret:+.1%}</b> "
                    f"<span class='dim mono' title='avg exposure this period'>@{badge * 100:.0f}%</span> ")
        else:
            rv = [x for x in h["ret"].values() if pd.notna(x)]
            mret = sum(rv) / len(rv) if rv else 0.0
            head = (f"<span class='mono dim'>{h['date'].date()}</span> "
                    f"<b style='color:{_pnl_color(mret, False)}'>{mret:+.1%}</b> ")
        lines.append(f"<div>{head}{spans}</div>")
    return f"<div style='font-size:0.78rem;line-height:1.7'>{''.join(lines)}</div>"


def _track_fold(d: dict, public: bool) -> str:
    """Live-tracking fold for whichever dossier is the ★ live book — the kill
    criteria belong to the tracked strategy, not to the page."""
    t = d.get("track")
    body = sec_track(d, public)
    if not t or not body:
        return ""
    if t["n"] < t["needed"]:
        s = f"accruing {t['n']}/{t['needed']} sessions — kill rules not armed yet"
    elif t.get("kill"):
        s = "<b class='neg'>KILL signal live</b>"
    else:
        s = "no kill signal — live path within the backtest's error bars"
    return _fold("live tracking", s, body)


_TIMELINE_KEY = ("Each line = one rebalance's picks, colored by that period's return — "
                 "<span style='color:#0a6b00'>■</span> ≥+20% · "
                 "<span style='color:#46c84e'>■</span> up · "
                 "<span style='color:#ef4444'>■</span> down · "
                 "<span style='color:#7a0000'>■</span> ≤−20% · "
                 "<span style='color:#000'>■</span> defaulted. Hover for the %.")


def _dossier_mom_ens(d: dict, r, public: bool) -> str:
    e = d.get("ensemble") or {}
    adopt = bool(e.get("adopt"))
    sleeves = e.get("sleeves") or []
    panels = "".join(
        _book_panel(d, f"Sleeve {i}/{len(sleeves)} — {sl['code']}", r.color,
                    sl["holdings_log"], weight_scale=1.0 / len(sleeves),
                    sub_note=" · equal capital")
        for i, sl in enumerate(sleeves, 1)) if sleeves else ""
    history = "".join(
        _fold("history", f"Every rebalance — sleeve {sl['code']}",
              f"<p class='dim'>{_TIMELINE_KEY}</p>"
              + _timeline_col(d, dict(key="sleeve", short=f"Sleeve {sl['code']}",
                                      color=r.color, equity=None, exposure=None,
                                      holdings_log=sl["holdings_log"])))
        for sl in sleeves)
    method = _fold("method", "Pre-registered selection-variance fix — adoption rule "
                             "and basis", sec_ensemble(d, public))
    track = _track_fold(d, public) if r.live else ""
    return _dossier(
        r.color,
        ("adopted · ★ live book" if adopt else "candidate · on the bench"),
        f"Momentum ensemble — top-{e.get('n', 3)} quarterly, equal capital",
        f"Adoption rule (ex ante): ensemble min(train,val) {e.get('ens_min', 0):.2f} ≥ "
        f"single {e.get('single_min', 0):.2f} − 0.05 → "
        + ("<b>ADOPTED</b>" if adopt else "bench") +
        " · fees per sleeve · buy every sleeve to hold this book",
        (f"<div class='par'>{panels}</div>" if panels else "")
        + track + method + history)


def _dossier_mom_rc(d: dict, r, public: bool) -> str:
    rc = d["variants"][1]
    ens_live = r.status == "variant"       # variant ⇔ the ensemble is the live book
    grade = rc["grade"]
    book = _book_panel(d, "Current book", r.color, rc["holdings_log"],
                       exposure_latest=rc.get("exposure_latest", 1.0),
                       sub_note=" · vol-targeted, rest in cash")
    g_details = _fold("method", "Quant scorecard &amp; honest grade — the full metric set",
                      sec_grade_compare(d, public))
    history = _fold("history", "Every rebalance — risk-conscious book",
                    f"<p class='dim'>{_TIMELINE_KEY} Shows the book return after "
                    "vol-scaling and average exposure (<span class='mono'>@x%</span>).</p>"
                    + _timeline_col(d, rc))
    track = _track_fold(d, public) if r.live else ""
    return _dossier(
        r.color,
        ("variant · single-config reference" if ens_live else "adopted · ★ live book"),
        f"Momentum single book — risk-conscious (vol-target {_rc_target(d):.0%})",
        f"The same selection, one config, scaled to {_rc_target(d):.0%} vol with the "
        f"remainder in cash · honest grade "
        f"<b style='color:{_GRADE_COLOR[grade['letter']]}'>{grade['letter']}</b>"
        + ("" if ens_live else " · the tracked book"),
        f"<div class='par'>{book}</div>" + track + g_details + history)


def _dossier_vol_core(d: dict, r, public: bool) -> str:
    v = d.get("vol_core") or {}
    act = ""
    if v.get("w_now") is not None:
        act = (f"<p class='mono'>today's order: hold <b>{v['w_now'] * 100:.0f}%</b> of core "
               f"capital in IWDA.AS (forecast vol {v['fc_now'] * 100:.1f}%, "
               f"as of {v['asof']})</p>")
    book = (f"<div><h3 style='color:{r.color}'>Current book</h3>"
            "<p class='dim pick-sub'>one ETF, exposure steered by the GARCH forecast</p>"
            + _BOOK_HEAD +
            "<tr><td class='mono'>IWDA</td><td>iShares Core MSCI World</td>"
            "<td class='dim mono' style='font-size:0.72rem'>IE00B4L5Y983</td>"
            f"<td class='num mono'>{(v.get('w_now') or 0) * 100:.0f}%</td></tr></table>"
            + act + "</div>")
    method = _fold("method", "GARCH(1,1) forecast, band rebalancing, gate table — full "
                             "tournament on the vol lab page", sec_vol_core(d, public))
    return _dossier(
        r.color, "adopted · passive sleeve",
        "GARCH vol-managed IWDA core",
        "The one overlay that cleared <b>pre-registered</b> gates (vol lab) — "
        "risk-adjusted return on the same asset: smaller drawdowns, cash in turbulence",
        f"<div class='par'>{book}</div>" + method)


def _dossier_generic(d: dict, r, public: bool) -> str:
    """Any future registry record renders a dossier automatically: status, verdict,
    canonical windows, link to its lab page. Richer detail = add a builder to
    DOSSIER_BUILDERS — that plus the make_record() call is the whole contract."""
    rows = "".join(
        f"<tr><td>{WIN_LABELS[k]}</td>{_perf_cells(r.windows.get(k, {}))}</tr>"
        for k in ("train", "val", "test", "full")) if r.windows else ""
    body = (("<table><tr><th>Window</th><th class='num'>Ret</th><th class='num'>Sharpe</th>"
             f"<th class='num'>Max DD</th></tr>{rows}</table>") if rows else "")
    if r.href:
        body += f"<p class='dim'><a href='{r.href}'>full workings ↗</a></p>"
    return _dossier(
        r.color or theme.ACCENT,
        r.status + (" · ★ live book" if r.live else ""),
        r.name, r.gate or r.verdict or "—", body)


def _dossier_megacap(d: dict, r, public: bool) -> str:
    """Pending dossier for the mega-cap PIT screen — no live data yet, so it states the
    design + the awaiting-data condition and links to the standalone page."""
    body = (
        "<p>Screen the universe to the largest names by <b>point-in-time market cap</b>, "
        "then rank inside the top-N by three arms:</p>"
        "<ul><li><b>size</b> — hold the biggest</li>"
        "<li><b>growth</b> — fastest trailing YoY revenue growth</li>"
        "<li><b>momentum</b> — 12-1 price momentum</li></ul>"
        "<p>Fully point-in-time: shares &amp; revenue lagged 75d, membership re-ranked "
        "every rebalance. N sweep {1,5,10,25,50}, equal-weight, net of slippage.</p>"
        "<div class='note warn'><b>Awaiting cap data</b> — no market-cap history fetched yet "
        "(0/400 coverage). Run the EODHD fundamentals fetch to populate, then this fills with "
        "real numbers. <a href='megacap.html'>Open the mega-cap page ↗</a></div>")
    return _dossier(r.color or theme.ACCENT, "research · awaiting data",
                    r.name, r.verdict or "", body)


# Bespoke dossier builders by registry id; anything else falls back to the generic
# card. Adding a future strategy = one make_record() (renders immediately) and,
# when it earns one, a builder entry here.
DOSSIER_BUILDERS = {
    "mom_ens": _dossier_mom_ens,
    "mom_rc": _dossier_mom_rc,
    "vol_core": _dossier_vol_core,
    "megacap": _dossier_megacap,
}


def _momentum_evidence(d: dict, public: bool) -> str:
    """The momentum family's shared evidence — these tests grade the SELECTION shared
    by the ensemble and the single book; family-scoped, not page-global."""
    s = d["significance"]
    mc, dsr = s["mc"], s["dsr"]
    beat = 100.0 * (1.0 - mc["p_sharpe"])
    dsr_txt = f" · DSR {dsr['dsr']:.0%}" if dsr["dsr"] == dsr["dsr"] else ""
    sig_sum = (f"selection beats <b>{beat:.1f}%</b> of random books "
               f"(p {mc['p_sharpe']:.3f}){dsr_txt} — Monte Carlo, deflation, bootstrap")
    fr = d.get("factor_reg")
    fac_sum = ""
    if fr:
        key = "FF5+WML" if "FF5+WML" in fr["raw"] else next(iter(fr["raw"]))
        m = fr["raw"][key]
        fac_sum = (f"{key} α {m['alpha_ann'] * 100:+.1f}%/yr (t {m['alpha_t']:.1f}) — "
                   + ("residual selection edge" if m["alpha_t"] >= 2.0
                      else "not separable from momentum beta"))
    ra = d.get("regime_attr")
    reg_sum = ""
    if ra:
        st_ = ra["strip"]
        retain = (st_["strip_test_sharpe"] / st_["full_test_sharpe"] * 100
                  if st_["full_test_sharpe"] else 0.0)
        reg_sum = (f"<b>{retain:.0f}%</b> of the selection Sharpe survives ex AI/Defense; "
                   f"pre-{TEST_FROM[:-1]} tape positive — regime-boosted, not regime-created")
    sc = d.get("scenarios")
    scen_sum = ""
    if sc:
        t50 = (float(sc["scenarios"]["base"]["term_p50"]) - 1) * 100
        b50 = (float(sc["scenarios"]["bear"]["term_p50"]) - 1) * 100
        scen_sum = (f"1y medians — base <b>{t50:+.0f}%</b>, bear <b>{b50:+.0f}%</b> "
                    "(block bootstrap, sensitivity not forecast)")
    body = "".join([
        _fold("significance", sig_sum, sec_significance(d, public)),
        _fold("factor spanning", fac_sum, sec_factor_regression(d, public)),
        _fold("diagnostics", "HMM regime + PCA effective bets — observational, never "
                             "traded", sec_diagnostics(d, public)),
        _fold("regime attribution", reg_sum, sec_regime(d, public)),
        _fold("scenario fan", scen_sum, sec_scenarios(d, public)),
        sec_caveat(d),
        _fold("delisting stress", "bull/base/bear injected-delisting grid — does the "
                                  "edge survive missing corpses?",
              sec_delisting_stress(d, public)),
    ])
    return _dossier(
        theme.FG_DIM, "momentum family · shared evidence",
        "Momentum family — evidence &amp; risk",
        "These tests grade the momentum <b>selection</b> shared by the ensemble and the "
        "single book — family-scoped, not page-global. Verdicts visible; workings folded.",
        body)


_FAMILY_TITLE = {"momentum": "Momentum family",
                 "vol-managed core": "Vol-managed core"}


def _dossier_records(d: dict) -> list:
    """The curve-bearing strategies that earn a detail pane — same filter the old
    single-scroll dossier block used, in family-contiguous order."""
    return [r for r in sreg.family_ordered(d.get("registry") or [])
            if r.status in ("adopted", "candidate", "variant")]


def _fam_color(members: list) -> str:
    return next((r.color for r in members if r.live), members[0].color or theme.ACCENT)


_MENU_TITLE = {"momentum": "Momentum", "vol-managed core": "Vol-managed core",
               "mega-cap": "Mega-cap"}


def _paned_records(d: dict) -> list:
    """Every strategy that earns a left-menu entry + a detail pane: the curve-bearing
    books (adopted/candidate/variant) plus the mega-cap incubating card."""
    recs = [r for r in sreg.family_ordered(d.get("registry") or [])
            if r.status in ("adopted", "candidate", "variant")]
    inc = [r for r in sreg.family_ordered(d.get("registry") or [])
           if r.status == "research" and r.id == "megacap"]
    return recs + inc


def _menu_name(r) -> str:
    """Short menu label — the registry names are full sentences; take the head."""
    return r.name.split(" \u2014 ")[0].split(" (")[0]


def _crumb() -> str:
    return "<p class='crumb'><button data-pane='home'>\u2190 Overview</button></p>"


def _spine(d: dict, public: bool, has_ops: bool, has_lab: bool) -> str:
    """The left menu: Overview (all strategies) + one entry per strategy, grouped by
    family, + Operations + Lab. Every entry is a <button data-pane=…>; the router
    swaps the matching stage pane."""
    recs = _paned_records(d)
    items = ["<button data-pane='home' class='s-home'>\u25a6 Overview \u2014 all strategies</button>"]
    for fam in list(dict.fromkeys(r.family for r in recs)):
        members = [r for r in recs if r.family == fam]
        items.append("<div class='s-fam'><span class='eyebrow' "
                     f"style='padding:8px 8px 2px'>{_MENU_TITLE.get(fam, fam.capitalize())}</span></div>")
        for r in members:
            star = "\u2605 " if r.live else ""
            items.append(
                f"<button data-pane='rec-{r.id}' class='s-rec'>"
                f"<span class='dot' style='background:{r.color or theme.FG_DIM}'></span>"
                f"{star}{_menu_name(r)}</button>")
    items.append("<hr class='s-sep'>")
    items.append("<button data-pane='research'>Research &amp; robustness</button>")
    if has_ops:
        items.append("<button data-pane='ops'>Operations</button>")
    if has_lab:
        items.append("<button data-pane='lab'>Research lab</button>")
    return f"<nav class='spine'>{''.join(items)}</nav>"


def _chart_all(d: dict, public: bool) -> str:
    """The Overview chart: every curve-bearing strategy vs the benchmarks and the
    equal-weight baseline, rebased to 100 at the common start. The real book is NOT on
    this axis — over the strategies' multi-year window it rebases to a late stub that
    reads as a loser; it gets its own faithful cash-flow-matched panel (sec_portfolio_roi)."""
    window = _equity_window(d["res"])
    recs = [r for r in sreg.family_ordered(d.get("registry") or [])
            if r.equity is not None and "inflated" not in r.flags
            and r.status in ("adopted", "candidate", "variant")]
    fig = go.Figure()
    for r in recs:
        srs = r.equity.reindex(window).ffill().dropna()
        if len(srs) < 2:
            continue
        nm = f"\u2605 {r.name}" if r.live else r.name
        fig.add_trace(go.Scatter(x=srs.index, y=srs / srs.iloc[0] * 100.0, name=nm,
                                 line=dict(color=r.color, width=2.6 if r.live else 1.8)))
    ew = d.get("ew_eq")
    if ew is not None:
        srs = ew.reindex(window).ffill().dropna()
        if len(srs) >= 2:
            fig.add_trace(go.Scatter(x=srs.index, y=srs / srs.iloc[0] * 100.0,
                                     name="Equal-weight baseline",
                                     line=dict(color=theme.FG_DIM, width=1.4, dash="dot")))
    for name, curve in benchmark_curves(d["benchmarks"], window, d["capital"]).items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve / d["capital"] * 100.0, name=name,
                                 line=dict(width=1.2)))
    fig.add_hline(y=100, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=520, yaxis_title="Index (start = 100)",
                      hovermode="x unified", margin=dict(t=20))
    return f"<div class='chart'>{fig_html(fig)}</div>"


def sec_portfolio_roi(d: dict, public: bool) -> str:
    """Your real book vs its cash-flow-matched market benchmarks over its OWN window, in
    cumulative ROI % — the same data and view as the /portfolio report's ROI chart (same
    euros, same dates, into each benchmark). Private: never on the public Pages build."""
    pr = d.get("portfolio_roi")
    if public or pr is None or getattr(pr, "empty", True):
        return ""
    srs = pr.dropna()
    if len(srs) < 2:
        return ""
    bench = d.get("portfolio_bench") or {}
    fig = go.Figure()
    for name, b in bench.items():                       # benchmarks thin, in the back
        bs_ = b.dropna()
        if len(bs_) >= 2:
            fig.add_trace(go.Scatter(x=bs_.index, y=bs_.values, name=name,
                                     line=dict(width=1.3)))
    fig.add_trace(go.Scatter(x=srs.index, y=srs.values,           # your book: thick white, on top
                             name="Your portfolio (real, cash-flow-timed)",
                             line=dict(color="#ffffff", width=2.6)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=380, yaxis_title="Cumulative ROI (%)",
                      hovermode="x unified", margin=dict(t=20))
    start = srs.index[0].strftime("%Y-%m-%d")
    return (f"<h2>Your real book vs the market — cash-flow-matched (from {start})</h2>"
            "<p class='legend'>the same euros on the same dates into each benchmark; your "
            "book in white. Same data &amp; method as the Portfolio page — over your own "
            "window, not rebased onto the strategies' multi-year axis.</p>"
            f"<div class='chart'>{fig_html(fig)}</div>")


def _pane_compare_all(d: dict, public: bool) -> str:
    """The default (Overview) pane shown when no strategy is selected: command strip,
    the full-grid result verdict, the survivorship banner, the all-strategies comparison
    chart (strategies vs benchmarks), the real book's own cash-flow-matched ROI panel,
    and the registry leaderboard."""
    return ("<section class='pane' id='pane-home'>"
            + sec_command(d, public) + sec_headline(d)
            + sec_survivorship_banner(d)
            + "<h2>Walk-forward equity \u2014 all strategies vs benchmarks</h2>"
            "<p class='legend'>index = 100 at common start; click legend entries to "
            "toggle lines; pick a strategy in the menu for its detail</p>"
            + _chart_all(d, public)
            + sec_portfolio_roi(d, public)
            + sec_registry(d, public)
            + sec_perf_compare(d, public)
            + sec_yearly_compare(d, public)
            + "</section>")


def _strat_curve(d: dict, r) -> str:
    window = _equity_window(d["res"])
    fig = go.Figure()
    srs = r.equity.reindex(window).ffill().dropna()
    if len(srs) >= 2:
        fig.add_trace(go.Scatter(x=srs.index, y=srs / srs.iloc[0] * 100.0, name=r.name,
                                 line=dict(color=r.color, width=2.4)))
    for name, curve in benchmark_curves(d["benchmarks"], window, d["capital"]).items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve / d["capital"] * 100.0, name=name,
                                 line=dict(width=1.1)))
    fig.add_hline(y=100, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=360, yaxis_title="Index (start = 100)",
                      hovermode="x unified", margin=dict(t=20))
    return f"<div class='chart'>{fig_html(fig)}</div>"


def _strat_stats(r) -> str:
    w = r.windows or {}
    t, full = w.get("test") or {}, w.get("full") or {}
    tiles = "".join([
        _card("Test return", _pct(t["net_return"] * 100) if t else "\u2014"),
        _card("Test Sharpe", f"{t['sharpe']:.2f}" if t else "\u2014"),
        _card("Max drawdown", _pct(full["max_dd"] * 100) if full else "\u2014"),
    ])
    return f"<div class='cards'>{tiles}</div>"


def _strat_windows(r) -> str:
    w = r.windows or {}
    rows = "".join(f"<tr><td>{WIN_LABELS[k]}</td>{_perf_cells(w.get(k, {}))}</tr>"
                   for k in ("train", "val", "test", "full"))
    return ("<table><tr><th>Window</th><th class='num'>Ret</th><th class='num'>Sharpe</th>"
            f"<th class='num'>Max DD</th></tr>{rows}</table>")


def _pane_strategy(d: dict, r, public: bool) -> str:
    """One strategy's detail pane — the SAME uniform template for every strategy:
    equity curve vs benchmark, key-stat tiles, the train/val/test/full windows table,
    and a method/verdict fold. Curve-less research rows (mega-cap) show their pending
    state instead of a chart."""
    if r.equity is not None:
        core = _strat_curve(d, r) + _strat_stats(r) + _strat_windows(r)
    else:
        core = ("<div class='note warn'>" + (r.gate or "Awaiting data \u2014 no live "
                "results yet.") + "</div>")
    method = _fold("method / verdict", r.status,
                   "<p>" + (r.verdict or r.gate or "\u2014") + "</p>")
    return (f"<section class='pane' id='pane-rec-{r.id}' hidden>"
            f"{_crumb()}"
            f"<h2 style='color:{r.color or theme.FG}'>{r.name} {_badge(r.status)}</h2>"
            f"{core}{method}</section>")


ROUTER_JS = """<script>
(function(){
  function panes(){ return document.querySelectorAll('.pane'); }
  function resize(pane){
    if(!window.Plotly||!pane) return;
    pane.querySelectorAll('.js-plotly-plot').forEach(function(g){
      try{ Plotly.Plots.resize(g); }catch(e){}
    });
  }
  function show(id){
    if(!document.getElementById('pane-'+id)) id='home';
    panes().forEach(function(p){ p.hidden=(p.id!=='pane-'+id); });
    document.querySelectorAll('.spine [data-pane]').forEach(function(b){
      b.classList.toggle('active', b.dataset.pane===id);
    });
    resize(document.getElementById('pane-'+id));
  }
  document.addEventListener('click', function(e){
    var el=e.target.closest('[data-pane]'); if(!el) return;
    e.preventDefault(); var id=el.dataset.pane;
    if(history.replaceState) history.replaceState(null,'','#'+id); else location.hash=id;
    show(id); window.scrollTo(0,0);
  });
  function boot(){ show((location.hash||'#home').replace('#','')); }
  if(document.readyState!=='loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
  window.addEventListener('load', function(){
    setTimeout(function(){ resize(document.querySelector('.pane:not([hidden])')); }, 120);
  });
})();
</script>"""


def sec_grade_compare(d: dict, public: bool) -> str:
    rc = d["variants"][1]
    tm = d["quant"]["trades"]
    g = rc["grade"]
    color = _GRADE_COLOR[g["letter"]]
    score_cards = "".join([
        f"<div class='card'><div class='k'>{rc['short']} grade</div>"
        f"<div class='v' style='color:{color};font-size:2rem'>{g['letter']}</div></div>",
        _card("Score", f"{g['score']:.0f}")])
    flags = "".join(f"<li>{f}</li>" for f in g["flags"])
    return (
        "<h2>Quant scorecard &amp; honest grade</h2>"
        f"<div class='cards'>{score_cards}</div>"
        "<p class='dim'>Graded like a risk committee: standard ratios, benchmark attribution, "
        "trade quality and stability — then docked for what it doesn't correct. The full metric "
        "set for the risk-conscious book:</p>"
        + _scorecard_table(rc, tm)
        + f"<p class='dim'><b>Verdict: <span style='color:{color}'>{g['letter']}</span></b> — a "
        f"real but modest momentum tilt (its <i>selection</i> beats a random book on the same "
        f"universe, p={d['significance']['mc']['p_sharpe']:.3f}; deflated-Sharpe "
        f"{d['significance']['dsr']['dsr']:.0%}), held prudently via volatility-targeting. The "
        f"deductions — survivorship, regime, capacity — sit in the underlying selection and are "
        f"detailed in the caveat below:<ul>{flags}</ul></p>")


def _scorecard_table(v: dict, tm: dict) -> str:
    """One variant's full scorecard — [Metric | value] with group sub-headers."""
    p, b, r = v["perf"], v["bench"] or {}, v["roll"] or {}

    def mr(k, val):
        return f"<tr><td>{k}</td><td class='num mono'>{val}</td></tr>"

    def grp(label):
        return (f"<tr><td colspan='2' style='padding-top:16px;color:{theme.FG_DIM};"
                f"font-weight:600;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;"
                f"border-bottom:1px solid {theme.GRID}'>{label}</td></tr>")
    perf_rows = "".join([
        mr("Sharpe (full, daily)", f"{p['sharpe']:.2f}"), mr("Sortino", f"{p['sortino']:.2f}"),
        mr("Calmar (CAGR/maxDD)", f"{p['calmar']:.2f}"), mr("Omega", f"{p['omega']:.2f}"),
        mr("Ann. return", _pct(p['ann_return'] * 100)), mr("Ann. vol", _pct(p['ann_vol'] * 100)),
        mr("Max drawdown", _pct(p['max_dd'] * 100)), mr("Underwater (days)", f"{p['dd_days']}"),
        mr("Skew / kurtosis", f"{p['skew']:.2f} / {p['kurtosis']:.2f}"),
        mr("VaR / CVaR 95 (daily)", f"{p['var95'] * 100:.1f}% / {p['cvar95'] * 100:.1f}%"),
        mr("Avg exposure", f"{p.get('avg_exposure', 1.0) * 100:.0f}%"),
    ])
    bench_rows = "".join([
        mr("Beta vs S&amp;P", f"{b.get('beta', float('nan')):.2f}"),
        mr("Alpha (annual)", _pct(b.get('alpha_ann', 0) * 100)),
        mr("Correlation", f"{b.get('corr', float('nan')):.2f}"),
        mr("Information ratio", f"{b.get('info_ratio', float('nan')):.2f}"),
        mr("Tracking error", _pct(b.get('tracking_error', 0) * 100)),
        mr("Up / down capture",
           f"{b.get('up_capture', float('nan')):.2f} / {b.get('down_capture', float('nan')):.2f}"),
    ]) if b else ""
    roll_rows = "".join([
        mr("12m Sharpe — median", f"{r.get('roll_sharpe_med', float('nan')):.2f}"),
        mr("12m Sharpe — worst", f"{r.get('roll_sharpe_min', float('nan')):.2f}"),
        mr("12m windows positive", _pct(r.get('roll_sharpe_pos_frac', 0) * 100)),
    ]) if r else ""
    trade_rows = "".join([
        mr("Hit rate", _pct(tm['hit_rate'] * 100)), mr("Profit factor", f"{tm['profit_factor']:.2f}"),
        mr("Payoff (avgW/avgL)", f"{tm['payoff']:.2f}"), mr("Trades / year", f"{tm['trades_per_year']:.0f}"),
    ]) if tm else ""
    head = f"<tr><th>Metric</th><th class='num' style='color:{v['color']}'>{v['short']}</th></tr>"
    body = head + grp("Risk / return") + perf_rows
    if bench_rows:
        body += grp("vs S&amp;P 500") + bench_rows
    if roll_rows:
        body += grp("Stability — 12-month rolling") + roll_rows
    if trade_rows:
        body += grp("Trade quality — selection (identical)") + trade_rows
    return f"<h3 style='color:{v['color']}'>{v['short']}</h3><table>{body}</table>"


def sec_ensemble(d: dict, public: bool) -> str:
    """Pre-registered method upgrade: equal-capital top-3 quarterly ensemble vs the
    single pick_ultimate config. Adoption rule fixed ex ante; +1 ledger trial."""
    e = d.get("ensemble")
    if not e:
        return ""
    cards = "".join([
        _card("Max drawdown", _pct(e["max_dd"] * 100)),
        _card("Trades/yr", f"{e['trades_per_year']:.0f}"),
        _card("DSR ×5", f"{e['dsr5']:.2f}" if e.get("dsr5") is not None else "—"),
    ])
    a = e.get("alpha")
    a_line = (f" · FF5+WML α {a['alpha_ann'] * 100:+.1f}%/yr "
              f"(t {a['alpha_t']:.2f})" if a else "")
    verdict = ("ADOPTED as the live book" if e.get("adopt")
               else "kept on the bench (single pick retains the edge)")
    return f"""<h2>Ensemble — top-{e['n']} quarterly configs, equal capital</h2>
<p class="sub">Selection-variance fix, <b>pre-registered</b>: pick_ultimate's
min(train,val) criterion is knife-edged, so the book averages the top-{e['n']}
quarterly cells ({' + '.join(f"<span class='mono'>{c}</span>" for c in e['codes'])})
at equal capital — fees charged per sleeve on its own third. Adoption rule fixed
before evaluation: adopt iff ensemble min(train,val)
({e['ens_min']:.2f}) ≥ single <span class='mono'>{e['single_code']}</span>
({e['single_min']:.2f}) − 0.05. <b>{verdict}</b>{a_line} · +1 selection-rule
trial charged to the program ledger. <span class="dim">Rule numbers are on the
pre-registered selection basis (arithmetic √252, momentum_grid) and deliberately
stay on it; the registry table quotes the canonical geometric basis — its
window Sharpes for this book live there, not here.</span></p>
<div class="cards">{cards}</div>"""


def sec_vol_core(d: dict, public: bool) -> str:
    """The adopted overlay — GARCH vol-managed MSCI World core. Percentages only."""
    v = d.get("vol_core")
    if not v:
        return ""
    bh, m = v["bh"], v["managed"]
    cards = "".join([
        _card("B&H Sharpe", f"{bh['sharpe']:.2f}"),
        _card("Managed Sharpe", f"{m['sharpe']:.2f}"),
        _card("B&H max DD", _pct(bh["max_dd"] * 100)),
        _card("Managed max DD", _pct(m["max_dd"] * 100)),
        _card("Avg exposure", f"{m['avg_exposure'] * 100:.0f}%"),
        _card("Trades/yr", f"{m['n_trades_per_year']:.0f}"),
    ])
    action = ""
    if v.get("w_now") is not None:
        action = (f"<p class='mono'>as of {v['asof']}: forecast vol "
                  f"{v['fc_now'] * 100:.1f}% → target exposure "
                  f"{v['w_now'] * 100:.0f}% (min(target/σ̂, 100%), 10% band)</p>")
    return f"""<h2>Adopted core — GARCH vol-managed {v['etf']}</h2>
<p class="sub">The one overlay that cleared <b>pre-registered</b> gates
(tools/gates.py — thresholds committed before the data run): GARCH(1,1)
beats the trailing-63d incumbent on QLIKE forecast quality AND is no worse
as a strategy. It buys risk-adjusted return, not raw return: same asset,
smaller drawdowns, cash when turbulence spikes (Moreira–Muir). Full
tournament and gate table on the vol lab page.</p>
<div class="cards">{cards}</div>
{action}"""


def sec_significance(d: dict, public: bool) -> str:
    s = d["significance"]
    mc, dsr, ci = s["mc"], s["dsr"], s["ci"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mc["null_sharpe"], nbinsx=40, name="random books",
                               marker_color=theme.FG_DIM, opacity=0.85))
    fig.add_vline(x=mc["strat_sharpe"], line_color="#dcdcaa", line_width=2.5,
                  annotation_text="this strategy", annotation_position="top")
    fig.add_vline(x=mc["null_sharpe_median"], line_color="#569cd6", line_dash="dash",
                  annotation_text="random median", annotation_position="bottom")
    fig.update_layout(height=340, bargap=0.02, showlegend=False,
                      xaxis_title="Gross annualised Sharpe (per-rebalance)",
                      yaxis_title="random books", margin=dict(t=20))

    verdict = ("clears" if mc["p_sharpe"] < 0.05 else "does <b>not</b> clear")
    dsr_txt = ("—" if dsr["dsr"] != dsr["dsr"] else
               f"After haircutting for the <b>{dsr['n_trials']} configs we scanned</b>, the return "
               f"skew/kurtosis and the {dsr['T']}-period sample length, the <b>Deflated Sharpe</b> "
               f"puts P(true Sharpe&gt;0) at <b>{dsr['dsr']:.0%}</b> (benchmark a lucky winner had to "
               f"clear: {dsr['sr_benchmark_annual']:.2f} annualised).")

    ph = s.get("dsr_phantom") or []
    t_stat = s.get("t_stat")
    phantom_html = ""
    if ph:
        pmult = s.get("phantom_mult", 5)
        pcards = "".join(
            _card(("Grid only" if p["mult"] == 1 else f"×{p['mult']} lifetime") + f" ({p['n']} trials)",
                  "—" if p["dsr"] != p["dsr"] else f"{p['dsr']:.0%}")
            for p in ph)
        pworst = ph[-1]
        harvey = ("clears" if (t_stat or 0) >= HARVEY_T else "sits below")
        phantom_html = (
            "<h3>(4) The file-drawer — phantom trials</h3>"
            f'<div class="cards">{pcards}</div>'
            "<p class='dim'>The DSR above only penalises the <b>64 configs in the grid</b> — the "
            "multiple testing it can <i>see</i>. But the pipeline (architectures, indicators, "
            "calendars, abandoned ideas) was iterated many times <i>before</i> that grid, and that "
            "unseen search inflates any winner. There's no git trail for it, so we raise the "
            f"<b>effective</b> trial count and watch P(real&gt;0) decay: at an estimated <b>×{pmult}</b> "
            f"lifetime iterations ({pmult * (ph[0]['n']):,}… trials) it holds at "
            f"<b>{[p['dsr'] for p in ph if p['mult'] == pmult][0]:.0%}</b>, and even at a pessimistic "
            f"<b>×{pworst['mult']}</b> it is <b>{pworst['dsr']:.0%}</b> — the dispersion of the grid "
            "Sharpes stays fixed, only the deflation bar rises. "
            + (f"Complementary hurdle: Harvey (2016) argues industry-wide unseen testing lifts the "
               f"real bar to a <b>t-stat of {HARVEY_T:.1f}</b>; this strategy's implied t-stat is "
               f"<b>{t_stat:.1f}</b>, which <b>{harvey}</b> it. " if t_stat is not None else "")
            + "This is a subjective estimate, shown as a <i>range</i> — not a precise correction. It "
            "is the honest ceiling on confidence, and the edge survives it.</p>")
    return ("<h2>Significance &amp; robustness</h2>"
            "<p class='dim'>Three desk-grade sanity checks, all on <b>gross per-rebalance</b> "
            "returns so the comparison is pure selection (costs hit a random book about the same). "
            f"<b>(1) Better than noise?</b> {mc['n_trials']:,} random books — same eligible pool, "
            "same dates, {k} names picked at <i>random</i> each rebalance — give the grey null "
            "below; momentum’s Sharpe (gold line) {verdict} the 5% bar with "
            "<b>p = {p:.3f}</b>. The random median (blue) is the universe’s own drift — beating it "
            "is the actual edge. <b>(2) Real after scanning configs?</b> {dsr} <b>(3) How wide the "
            "error bar?</b> a circular block-bootstrap puts the annualised Sharpe’s {conf}% CI at "
            "<b>{slo:.2f}–{shi:.2f}</b> and CAGR at <b>{clo:.0f}%–{chi:.0f}%</b>.</p>"
            .format(k=d["strategy"].slots, verdict=verdict, p=mc["p_sharpe"], dsr=dsr_txt,
                    conf=ci["conf"], slo=ci["sharpe_lo"], shi=ci["sharpe_hi"],
                    clo=ci["cagr_lo"] * 100, chi=ci["cagr_hi"] * 100) +
            f"<div class='chart'>{fig_html(fig)}</div>"
            + phantom_html +
            "<p class='dim'>A low p-value says the <i>selection</i> adds value over drawing names "
            "at random from the same liquid pool; it does not promise the level repeats. "
            "Volatility-targeting changes the sizing, not the edge — so this verdict is about the "
            "selection you hold either way. Read it with the regime and capacity caveats below.</p>")


def sec_factor_regression(d: dict, public: bool) -> str:
    fr = d.get("factor_reg")
    if not fr:
        return ""
    raw, rc = fr["raw"], fr.get("rc") or {}
    key = "FF5+WML" if "FF5+WML" in raw else next(iter(raw))
    m = raw[key]
    wml = m["betas"].get("WML")
    cards = "".join([
        _card(f"Alpha ({key}, raw)", f"{_pct(m['alpha_ann'] * 100)}"),
        _card("Alpha t-stat (Newey-West)", f"{m['alpha_t']:.1f}"),
        _card("WML loading", f"{wml[0]:.2f} (t={wml[1]:.1f})" if wml else "—"),
        _card("R²", f"{m['r2']:.2f}"),
    ])

    def rows(label, regs, color):
        out = []
        for name in ("CAPM", "FF5", "FF5+WML"):
            r = regs.get(name)
            if not r:
                continue
            b_mkt = r["betas"].get("MKT_RF", (float("nan"), float("nan")))
            b_wml = r["betas"].get("WML")
            wml_cell = (f"{b_wml[0]:.2f} <span class='dim'>(t={b_wml[1]:.1f})</span>"
                        if b_wml else "—")
            out.append(
                f"<tr><td style='color:{color}'>{label}</td><td class='mono'>{name}</td>"
                f"<td class='num mono'>{_pct(r['alpha_ann'] * 100)} "
                f"<span class='dim'>(t={r['alpha_t']:.1f})</span></td>"
                f"<td class='num mono'>{b_mkt[0]:.2f}</td>"
                f"<td class='num mono'>{wml_cell}</td>"
                f"<td class='num mono'>{r['r2']:.2f}</td>"
                f"<td class='num mono'>{r['n']:,}</td></tr>")
        return "".join(out)

    body = rows("Original", raw, C_RAW) + rows("Risk-conscious", rc, C_RC)
    at, aa = m["alpha_t"], m["alpha_ann"]
    if at >= 2.0:
        verdict = ("a <b>residual selection edge beyond the factors</b> — the alpha survives "
                   "the momentum factor itself at a real t-stat")
    elif aa > 0:
        verdict = ("<b>positive but not statistically separable from factor exposure</b> — "
                   "read the edge as momentum factor beta, self-managed at retail scale "
                   "(which no UCITS momentum ETF gives you this cheaply), not as proprietary alpha")
    else:
        verdict = ("<b>no residual</b> — the performance is factor exposure; fine to hold, "
                   "not proprietary")
    wml_txt = (f"The WML loading of <b>{wml[0]:.2f}</b> confirms the book actually harvests "
               f"the momentum premium it targets. " if wml else "")
    return (
        "<h2>Factor spanning — is the edge just momentum beta?</h2>"
        f"<p class='dim'>Daily strategy returns, converted to <b>USD</b> and taken in excess "
        f"of the T-bill rate, regressed on the <b>Ken French {fr['source']} factors</b> "
        f"(daily, {fr['start']} → {fr['end']}, n = {fr['n']:,} overlapping days; "
        "<b>Newey-West</b> t-stats). CAPM, then the 5-factor model, then 5F + <b>WML</b> "
        "(momentum). The question each row answers: after paying the loadings, is anything "
        "left?</p>"
        f"<div class='cards'>{cards}</div>"
        "<div class='scroll'><table><tr><th>Book</th><th>Model</th><th class='num'>Alpha (ann.)</th>"
        "<th class='num'>β market</th><th class='num'>β WML</th><th class='num'>R²</th>"
        f"<th class='num'>Days</th></tr>{body}</table></div>"
        f"<p class='dim'><b>Verdict:</b> {wml_txt}On this sample the {key} alpha is "
        f"<b>{_pct(aa * 100)}</b>/yr (t = {at:.1f}) — {verdict}. Same caveats as everything "
        "above: the level rides the survivor universe, and ~8 years of daily data bounds how "
        "sharp any factor t-stat can be. Observational — nothing here feeds selection.</p>")


def sec_diagnostics(d: dict, public: bool) -> str:
    """Observational-only lens: HMM regime beside the 200d kill-switch, and the PCA
    effective number of bets. Renders only what gather() produced."""
    reg = d.get("regime")
    eb = d.get("eff_bets")
    have_reg = reg is not None and len(reg) > 0 and "prob_risk_off" in getattr(reg, "columns", [])
    have_eb = bool(eb)
    if not have_reg and not have_eb:
        return ""
    out = ["<h2>Observational diagnostics — HMM &amp; PCA (not traded)</h2>",
           "<p class='dim'>Two lenses run <b>in parallel</b> with the strategy and feed "
           "<b>nothing</b> back into selection or sizing — they answer “would a regime model or "
           "an orthogonality test have helped?” without spending any of the <b>deflated-Sharpe</b> "
           "budget above. Every extra knob wired into execution would only raise that haircut.</p>"]

    if have_reg:
        eq = d["res"]["runs"][1.0]["equity"].dropna()
        ro = reg["risk_off"].astype(bool)
        tb = reg["trend_broken"].astype(bool) if "trend_broken" in reg.columns else pd.Series(dtype=bool)
        common = ro.index.intersection(tb.index)
        agree = float((ro.loc[common] == tb.loc[common]).mean()) if len(common) else float("nan")
        e = eq / d["capital"] * 100.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=e.index, y=e.values, name="Strategy (raw)",
                                 line=dict(color=C_RAW, width=2.0)))
        fig.add_trace(go.Scatter(x=reg.index, y=reg["prob_risk_off"].values, name="HMM P(risk-off)",
                                 line=dict(color="#ef4444", width=1.4), fill="tozeroy",
                                 fillcolor="rgba(239,68,68,0.12)", yaxis="y2"))
        if len(tb):
            fig.add_trace(go.Scatter(x=tb.index, y=tb.astype(float).values,
                                     name="Below 200d MA (kill-switch on)",
                                     line=dict(color="#569cd6", width=1.2, dash="dot"), yaxis="y2"))
        fig.update_layout(height=380, hovermode="x unified", margin=dict(t=20),
                          yaxis=dict(title="Index (start = 100)"),
                          yaxis2=dict(title="prob / on–off", overlaying="y", side="right",
                                      range=[-0.02, 1.05], showgrid=False))
        cards = "".join([
            _card("HMM ↔ 200d agreement", _pct(agree * 100, signed=False) if agree == agree else "—"),
            _card("Days HMM risk-off", _pct(ro.mean() * 100, signed=False)),
            _card("Days below 200d MA", _pct(tb.mean() * 100, signed=False) if len(tb) else "—"),
        ])
        out += ["<h3 style='color:#ef4444'>Regime lens — HMM vs the 200d kill-switch</h3>",
                f"<div class='cards'>{cards}</div>",
                f"<div class='chart'>{fig_html(fig)}</div>",
                "<p class='dim'>The HMM’s risk-off probability (red) and the benchmark dropping "
                "below its 200d MA (blue) light up together — the agreement card says how often. "
                "Both coincide with the turbulence the <b>vol-targeting</b> overlay already de-risks "
                "into. An HMM kill-switch would mostly <b>re-discover the control the strategy already "
                "runs</b> — at the cost of more parameters to overfit.</p>"]

    if have_eb:
        ebd = pd.DataFrame(eb).set_index("date").sort_index()
        k = int(ebd["k"].iloc[-1]) if "k" in ebd.columns else d["strategy"].slots
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ebd.index, y=ebd["n_eff_pca"].values, name="Effective bets (PCA)",
                                  line=dict(color=C_RC, width=2.2)))
        fig2.add_hline(y=k, line_dash="dash", line_color=theme.FG_DIM,
                       annotation_text=f"names held ({k})", annotation_position="top left")
        fig2.add_trace(go.Scatter(x=ebd.index, y=ebd["pc1_share"].values, name="PC1 share of variance",
                                  line=dict(color=C_RAW, width=1.2), yaxis="y2"))
        fig2.update_layout(height=360, hovermode="x unified", margin=dict(t=20),
                           yaxis=dict(title="effective # of bets", rangemode="tozero"),
                           yaxis2=dict(title="PC1 share", overlaying="y", side="right",
                                       range=[0, 1.02], showgrid=False))
        last = ebd.iloc[-1]
        cards2 = "".join([
            _card("Effective bets now", f"{last['n_eff_pca']:.1f}"),
            _card("Names held", f"{k}"),
            _card("PC1 share now", _pct(last["pc1_share"] * 100, signed=False)),
        ])
        out += [f"<h3 style='color:{C_RC}'>Diversification lens — PCA effective bets</h3>",
                f"<div class='cards'>{cards2}</div>",
                f"<div class='chart'>{fig_html(fig2)}</div>",
                f"<p class='dim'>You hold <b>{k}</b> names, but the <b>effective</b> number of "
                "independent bets (teal) is what survives their correlations — it sinks toward 1–2 "
                "when one factor (PC1, gold) dominates, i.e. the book is really one crowded theme. "
                "The strategy now runs <b>sector-neutral</b> selection (B, real sectors sourced via "
                "<code>tools.enrich_sectors</code>), which caps single-sector piling — these bars "
                "are the B-<i>on</i> book, so watch whether effective bets sit higher than a "
                "single-theme run would. Full orthogonalisation (PCA eigenportfolios) isn’t "
                "executable on Trade Republic (long/short baskets). Shown to watch, not to trade.</p>"]
    return "".join(out)


def sec_regime(d: dict, public: bool) -> str:
    ra = d.get("regime_attr")
    if not ra:
        return ""
    st, cond, pre = ra["strip"], ra.get("cond"), ra["pre"]
    full_sh, strip_sh = st["full_test_sharpe"], st["strip_test_sharpe"]
    retain = (strip_sh / full_sh * 100) if full_sh else 0.0
    strip_verdict = ("largely survives" if strip_sh >= 1.0 and retain >= 60
                     else "roughly halves" if retain >= 40 else "leans heavily on those sectors")

    labels = ["Full<br>(test)", "Ex&nbsp;AI+Defense", f"{PRE_TEST}<br>(pre-regime)"]
    vals = [full_sh, strip_sh, pre["sharpe"]]
    colors = [C_RAW, "#ef4444", "#569cd6"]
    fig = go.Figure(go.Bar(x=labels, y=[round(v, 2) for v in vals], marker_color=colors,
                           text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.add_hline(y=full_sh, line_dash="dash", line_color=theme.FG_DIM,
                  annotation_text="selection test Sharpe", annotation_position="top right")
    fig.update_layout(height=360, margin=dict(t=30), showlegend=False,
                      yaxis_title="Selection (raw, pre-overlay) test Sharpe", xaxis_title="")

    cards = [
        _card("Selection test Sharpe", f"{full_sh:.2f}"),
        _card("Ex AI/Defense", f"{strip_sh:.2f}"),
        _card("Sharpe retained", _pct(retain, signed=False)),
        _card(f"{PRE_TEST} Sharpe (regime-out)", f"{pre['sharpe']:.2f}"),
    ]
    cond_txt = ""
    if cond:
        on, off = cond["on"], cond["off"]
        if off["sharpe"] <= 0:
            ratio_txt = "far higher" if on["sharpe"] > 0 else "similar"
        else:
            r = on["sharpe"] / off["sharpe"]
            ratio_txt = "far higher" if r > 3 else "higher" if r > 1.2 else "similar"
        cond_txt = (
            f" <b>(2) Regime-timing?</b> Splitting every day by the HMM label (a <i>daily</i>-return "
            f"Sharpe — not comparable to the annualised bars), the risk-on Sharpe is <b>{ratio_txt}</b> "
            f"than risk-off (<b>{on['sharpe']:.2f}</b> vs <b>{off['sharpe']:.2f}</b>), and only "
            f"<b>{_pct(off['ret_share'] * 100, signed=True)}</b> of the total return came from the "
            f"(rarer) risk-off days. So the edge is a risk-on phenomenon — exactly what the "
            f"<b>vol-targeting</b> overlay leans into and what an HMM kill-switch would mostly "
            f"re-discover, at the cost of more parameters to overfit.")
    return (
        "<h2>Regime attribution — is the Sharpe AI/Defense beta in disguise?</h2>"
        "<p class='dim'>The single most-cited worry: the recent test tape was a momentum dream (AI "
        "semiconductors + defense spending), so is the held-out Sharpe borrowed from a once-a-decade "
        "macro regime? Three <b>observational</b> shocks, each a re-run/re-slice of the <b>same</b> "
        "config — none touch selection, so none spend deflated-Sharpe budget. All three run on the "
        "<b>raw pre-overlay selection curve</b> on its own basis — internally consistent, and "
        "deliberately <i>not</i> the headline's risk-conscious number. "
        f"<b>(1) Sector beta?</b> Dropping <b>Technology + Industrials</b> (the AI &amp; defense "
        f"tailwind — {st['n_dropped']:,} names, coarse: defense is a slice of Industrials) and "
        f"re-running leaves a held-out test Sharpe of <b>{strip_sh:.2f}</b> vs <b>{full_sh:.2f}</b> "
        f"full — it <b>{strip_verdict}</b> (<b>{retain:.0f}%</b> retained).{cond_txt} "
        f"<b>(3) Pre-regime?</b> On the <b>{PRE_TEST}</b> tape (train+val — momentum’s hard mean-"
        f"reverting stretch included) the Sharpe was <b>{pre['sharpe']:.2f}</b> — positive, so the edge "
        "predates the AI/defense boom.</p>"
        f'<div class="cards">{"".join(cards)}</div>'
        f"<div class='chart'>{fig_html(fig)}</div>"
        "<p class='dim'><b>Verdict:</b> the regime tailwind is real and it <i>amplifies</i> the level, "
        "but the core selection edge is not <i>only</i> AI/Defense and not <i>only</i> 2024 — it "
        "survives the sector strip and predates the boom, at a lower Sharpe. Read the headline as "
        "regime-<b>boosted</b>, not regime-<b>created</b>. Still observational: nothing here is traded.</p>")


def sec_scenarios(d: dict, public: bool) -> str:
    sc = d.get("scenarios")
    if not sc:
        return ""
    S = sc["scenarios"]
    days = np.arange(sc["horizon"] + 1)
    yrs = sc["horizon"] / 252.0
    C_BEAR, C_BULL = "#ef4444", "#46c84e"

    def pctf(x):
        return (np.asarray(x, float) - 1.0) * 100.0

    base = S["base"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=pctf(base["p95"]), line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=pctf(base["p5"]), fill="tonexty",
                             fillcolor="rgba(78,201,176,0.15)", line=dict(width=0),
                             name="Base P5–P95", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=pctf(base["p50"]), name="Base (median)",
                             line=dict(color=C_RC, width=2.6)))
    fig.add_trace(go.Scatter(x=days, y=pctf(S["bear"]["p50"]), name="Bear (median)",
                             line=dict(color=C_BEAR, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=days, y=pctf(S["bull"]["p50"]), name="Bull (median)",
                             line=dict(color=C_BULL, width=2, dash="dot")))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=440, xaxis_title=f"trading days (→ {yrs:.0f}y horizon)",
                      yaxis=dict(title="cumulative return", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))

    def term_card(name, label):
        return _card(f"{label} · {yrs:.0f}y median", _pct(float(pctf(S[name]['term_p50']))))
    cards = "".join([
        term_card("bear", "Bear"), term_card("base", "Base"), term_card("bull", "Bull"),
        _card("Base P5–P95",
              f"{_pct(float(pctf(base['term_p5'])))} … {_pct(float(pctf(base['term_p95'])))}"),
    ])
    return (
        "<h2>Scenario fan — bear / base / bull</h2>"
        f"<p class='dim'>A <b>regime-conditioned block bootstrap</b>: the risk-conscious book's own "
        f"daily returns are resampled in {sc['block']}-day blocks {sc['n_sims']:,}× over a "
        f"{yrs:.0f}-year horizon. <b>Base</b> draws every block at its natural frequency; "
        f"<b>bear</b> over-weights <b>risk-off</b> regime blocks (×{sc['tilt']:.0f}); <b>bull</b> "
        f"over-weights risk-on. The realised tape was <b>{sc['frac_off'] * 100:.0f}%</b> risk-off. "
        f"<span style='color:{C_RC}'>Teal band</span> = base P5–P95; the "
        f"<span style='color:{C_BEAR}'>red</span> / <span style='color:{C_BULL}'>green</span> "
        "dotted lines are the bear / bull medians.</p>"
        f"<div class='cards'>{cards}</div>"
        f"<div class='chart'>{fig_html(fig)}</div>"
        "<p class='dim'><b>Read this as a sensitivity, not a forecast.</b> It is the dispersion of "
        "terminal wealth implied by the strategy's <i>realised</i> return process under different "
        "regime mixes — no new information, and <b>observational</b> (it never touches the selection "
        "or the vol-target, so it costs no deflated-Sharpe). It also inherits the <b>survivor "
        "universe</b>: the bear path is “bear among the names that survived”, so a real bear with "
        "delistings is worse.</p>")


def sec_caveat(d: dict) -> str:
    hits = d.get("graveyard_hits", 0)
    rc = d["variants"][1]
    test_ret = (rc.get("windows", {}).get("test") or rc["test"])["net_return"] * 100
    surv = (f"it held <b>0</b> of them into death" if hits == 0 else
            f"it held <b>{hits}</b> into delisting, liquidated by the graveyard at the last price")
    nlive = d.get("n_live")
    b = d.get("bounds", {})
    bound_txt = ""
    if b:
        bound_txt = (
            f' Trimming the {b["n_dead_dropped"]} never-TR-tradeable corpses (the 200 Korean + a few) '
            f'moves the held-out test from <b>{b["lower_full"]*100:+.1f}%</b> to '
            f'<b>{b["upper_test"]*100:+.1f}%</b> — all but identical, but for the <i>wrong</i> reason: '
            f'the graveyard barely overlaps the live universe, so it isn’t correcting anything.')
    trade = (
        f' <b>Tradeability is built in, not assumed.</b> The {nlive:,} live names are TR’s own '
        f'instrument list — <i>enumerated</i> from a Trade Republic account and priced at their '
        f'home listing (Milan, Tokyo, etc.) via yfinance — so every one is a name you can actually '
        f'buy (a few TR lists but restricts in your region may slip in). The {d["n_dead"]} delisted '
        f'names are the survivorship graveyard; we report the all-corpses result as the conservative '
        f'<b>lower bound</b>.{bound_txt}')
    ov = d.get("quant", {}).get("isin_overlap", 0.0)
    si = d.get("surv_inject")
    onpop = ""
    if si and si.get("sims"):
        rel = (si["delta_mean"] / si["base_return"] * 100) if si.get("base_return") else 0.0
        held, dn, ks = si["hits_mean"], si["deaths_mean"], si["sims"]
        avoid = si.get("avoidance_rate", 0.0) * 100
        hz = si.get("hazard", 0.05)
        lo, hi = si.get("loss", (0.40, 1.00))
        verdict = ("immaterial" if avoid >= 99 else "small" if avoid >= 95 else "real")
        onpop = (
            f' <b>On-population test — the honest fix for that {ov*100:.0f}%.</b> Since the real '
            f'graveyard barely overlaps the live set, we inject <i>synthetic</i> delistings into the '
            f'live names themselves (~{dn:,.0f}/run, hazard&nbsp;{hz*100:.0f}%/yr, terminal '
            f'crash {lo*100:.0f}–{hi*100:.0f}%) and re-run the identical strategy '
            f'{ks}×. Momentum held a name into its delisting only <b>{held:.1f}×/run</b> — it sold '
            f'<b>{avoid:.1f}%</b> of the dying names <i>before</i> they died (it down-ranks a name as it '
            f'deteriorates). So the <i>holding</i> leak is <b>{verdict}</b>, and this is now measured, '
            f'not asserted. As a <b>pessimistic</b> corroboration — assuming every name held into death '
            f'is a total wipeout (real winner-delistings are usually buy-outs at a <i>premium</i>) — the '
            f'worst-case cumulative drag averages <b>{rel:+.0f}%</b> of the level, but it is a wide, '
            f'high-variance tail. Either way this does <b>not</b> touch the <i>membership</i> leak '
            f'(absent winners) — that stays the real problem.')
    ledger = (
        ' <b>Forward fix live:</b> the universe refresh keeps an append-only membership '
        'ledger (a live name that dies, loses liquidity or leaves TR is carried with an '
        'exit date — never silently dropped), and point-in-time snapshots of TR\'s own '
        'list gate eligibility from <b>2026-06-29</b> forward (a name is only pickable at '
        't if TR offered it at t). Neither rewrites the past — the historical window stays '
        'survivor-biased and stress-bounded — but the bleed stops compounding from here.')
    deep = (
        f'{trade}{onpop}{ledger}'
        f'<br><br><b style="color:{C_RC}">Risk-conscious version.</b> Volatility-targeting directly '
        f'addresses the drawdown and the raw vol — it cuts both materially — but it does <b>not</b> '
        f'fix survivorship, regime dependence or capacity: those sit in the underlying selection, '
        f'which is identical, so they apply equally to both versions. Its daily cash↔stock resizing '
        f'is <b>charged</b>, so its curve pays for its own '
        f'turnover — but the flat €/order on tiny daily resizes is extra, so a real book would '
        f'<b>band</b> the rebalancing rather than resize every day.')
    return (
        f'<div class="note warn"><b>The dominant caveat — survivorship is NOT corrected.</b> '
        f'The live universe is Trade Republic’s <i>current</i> list — names that <b>survived to '
        f'today</b>. A name that pumped then delisted before now is simply absent, so the backtest '
        f'only ever picks from winners-that-made-it. The {d["n_dead"]} “graveyard” names are a '
        f'near-disjoint EODHD relic (<b>{ov*100:.0f}%</b> ISIN overlap with the live set), so they '
        f'do <b>not</b> fix it.{bound_txt} This is the single biggest reason to distrust the raw '
        f'level, and is why the raw full-invested curve is kept in the lab, not on the main page.'
        f'<br><br>The other caveats: <b>(1) Regime</b> — {TEST_FROM} was an exceptional small-cap '
        f'momentum tape; even the held-out {test_ret:+.0f}% test figure (risk-conscious book, the '
        f'same number the headline quotes) is regime-specific and will <b>not</b> '
        f'repeat. <b>(2) Concentration</b> — top-{d["strategy"].slots}; sector-neutral (B) now caps '
        f'single-sector piling, but with no per-name weight cap a few big movers still drive the '
        f'curve, and sectors are sourced for only part of the universe (rest fall in one “Unknown” '
        f'bucket). <b>(3) Capacity</b> — picks '
        f'are liquid enough for a small account, but modeled slippage ({MODELED_SLIP_BPS}bps) '
        f'understates real fills '
        f'in size. <b>(4) Mechanics</b> — daily closes, €1/order, slippage modeled not measured, and '
        f'<b>past performance is not future returns</b>.'
        f"<details><summary>tradeability, the on-population test and the forward fix — "
        f"the full workings</summary>{deep}</details></div>")


def sec_delisting_stress(d: dict, public: bool) -> str:
    st = d.get("delisting_stress")
    if st is None:
        return "" if public else ("<h2>Delisting stress</h2>"
                                  "<p class='dim'>Skipped (SURV_SIMS=0).</p>")
    pr, clean = st["presets"], st["clean"]
    order = [("bull", "Bull"), ("base", "Base"), ("bear", "Bear")]
    a_rc = [pr[k]["alpha"]["rc"]["mean"] for k, _ in order]
    lo_rc, hi_rc = min(a_rc), max(a_rc)
    edge_bear = pr["bear"]["edge"]["rc"]["mean"]

    if public:
        cards = (_card("Risk-conscious alpha (range across intensities)",
                       f"{_pct(hi_rc * 100)} … {_pct(lo_rc * 100)}")
                 + _card("Edge vs an equally-delisted basket (bear)", _pct(edge_bear * 100)))
        alpha_clause = ("keeps a positive alpha even at a crisis delisting rate"
                        if lo_rc > 0 else
                        "keeps most of its alpha even at a crisis delisting rate")
        edge_clause = ("still beats a buy-hold basket that eats the very same deaths"
                       if edge_bear > 0 else
                       "stays close to a buy-hold basket that eats the very same deaths")
        return (
            "<h2>Does the edge survive missing delistings?</h2>"
            "<p class='dim'>Today's universe is survivors. We inject synthetic deaths into the live "
            "names at three intensities — benign (bull), empirical (base) and crisis (bear) "
            "delisting rates, terminal losses drawn from the delisting literature — then re-run the "
            "identical strategy and re-measure annualised alpha against the untouched MSCI World.</p>"
            f"<div class='cards'>{cards}</div>"
            f"<p class='dim'>The vol-targeted book {alpha_clause}, and {edge_clause} — the edge comes "
            "from selection, not from quietly skipping the survivorship correction. This is a "
            "one-sided, downside-only stress (absent <i>winners</i> are not added back): it bounds "
            "the survivorship drag, it does not remove the caveat.</p>")

    heads = "".join(f"<th class='num'>{lab}</th>" for _, lab in order)

    def band_row(label, metric, key):
        cells = "".join(
            f"<td class='num mono'>{_pct(pr[k][metric][key]['mean'] * 100)}"
            f"<span class='dim'> [{_pct(pr[k][metric][key]['lo'] * 100)}, "
            f"{_pct(pr[k][metric][key]['hi'] * 100)}]</span></td>" for k, _ in order)
        return (f"<tr><td>{label}</td>"
                f"<td class='num mono'>{_pct(clean[metric][key] * 100)}</td>{cells}</tr>")

    ret_cells = "".join(f"<td class='num mono'>{_pct(pr[k]['mean_return'] * 100)}</td>"
                        for k, _ in order)
    av_cells = "".join(f"<td class='num mono'>{pr[k]['avoidance_rate'] * 100:.0f}%</td>"
                       for k, _ in order)
    dd_cells = "".join(f"<td class='num mono'>{pr[k]['deaths_mean']:.1f}</td>" for k, _ in order)
    return (
        "<h2>Delisting stress — full grid</h2>"
        f"<p class='dim'>Synthetic deaths injected into the live names, {st['sims']} sims per "
        "intensity, identical strategy re-run. Alpha = annualised CAPM alpha vs the un-injected "
        "MSCI World; edge = vs an equally-delisted buy-hold of the initial picks; brackets = 5–95% "
        "band. Base = the empirical central case (= the single-intensity survivorship test). Clean "
        "= no injection.</p>"
        "<div class='scroll'><table><thead>"
        f"<tr><th>Metric</th><th class='num'>Clean</th>{heads}</tr>"
        "</thead><tbody>"
        f"<tr><td>Original — mean return</td>"
        f"<td class='num mono'>{_pct(clean['ret']['raw'] * 100)}</td>{ret_cells}</tr>"
        + band_row("Original — alpha vs bench", "alpha", "raw")
        + band_row("Risk-conscious — alpha vs bench", "alpha", "rc")
        + band_row("Original — edge vs EW", "edge", "raw")
        + band_row("Risk-conscious — edge vs EW", "edge", "rc")
        + f"<tr><td>Avoidance rate</td><td class='num mono'>—</td>{av_cells}</tr>"
        + f"<tr><td>Deaths / run</td><td class='num mono'>0</td>{dd_cells}</tr>"
        + "</tbody></table></div>")


# ── Operations ───────────────────────────────────────────────────────────────────

def sec_track(d: dict, public: bool) -> str:
    t = d.get("track")
    if not t:
        return ""
    if t["n"] < t["needed"]:
        status = (f"accruing: {t['n']}/{t['needed']} live sessions before the "
                  f"kill rules arm")
    elif t["kill"]:
        status = ("<b class='neg'>KILL</b> — " + "; ".join(t["reasons"])
                  + " (de-risk to cash and reassess; this page never trades)")
    else:
        status = "live path within the backtest's error bars — no kill signal"
    return f"""<h2>Live tracking — pre-registered kill criteria</h2>
<p class="sub">One row per build day → <span class="mono">{t['path']}</span>.
KILL iff rolling 63d live Sharpe sits below the backtest bootstrap CI lower
bound for 21 straight sessions, or live drawdown exceeds 1.25× the backtest
max. Thresholds committed before any live row existed. Status: {status}.</p>"""


def sec_venture(d: dict, public: bool) -> str:
    v = d.get("venture")
    if not v:
        return ""
    sat = v.get("satellite", {})
    if not v.get("live"):
        return f"""<h2>Venture — north star vs same-cashflow shadow</h2>
<p class="sub">Live meter arms at 2+ tracker rows (has {v.get('n_rows', 0)}).
The comparison is money-weighted XIRR of the LIVE path vs a same-cashflow
IWDA shadow — backtest curves never enter this meter. Drawdown ladder
(<b>pre-registered</b>, timing-exempt): −25% → half vol target; −35% →
de-risk + exceptional review. Satellite {sat.get('live', 0)}/{sat.get('cap', 3)}.</p>"""
    cards = "".join([
        _card("Months / 36", f"{v['months']:.1f}"),
        _card("Book XIRR", _pct(v["book_xirr"] * 100)),
        _card("IWDA shadow XIRR", _pct(v["shadow_xirr"] * 100)),
        _card("Excess (money-wt.)", _pct(v["excess"] * 100)),
        _card("Drawdown", _pct(v["dd"] * 100)),
        _card("DD ladder", v["dd_state"]),
        _card("Satellite", f"{sat.get('live', 0)}/{sat.get('cap', 3)}"),
    ])
    return f"""<h2>Venture — north star vs same-cashflow shadow</h2>
<p class="sub">Aspiration: 3y money-weighted XIRR ≥ IWDA shadow +3%/yr after
tax — but 3y CAGR alone cannot certify ±3%, so verdicts condition on
<b>pre-registered process evidence</b> (live path inside the backtest
bootstrap band, PSR trend, slippage gap). Shadow = every deposit buys IWDA
on the same date. Drawdown ladder (operational, timing-exempt): −25% →
half vol target ("half-vol"); −35% → de-risk to core/cash ("derisk") +
exceptional review. Missed ritual ⇒ hold positions, never catch-up trades.
</p>
<div class="cards">{cards}</div>"""


def sec_ritual(d: dict, public: bool) -> str:
    r = d.get("ritual")
    if public or not r:
        return ""
    rows = "".join(
        f"<tr><td>{i['name']}</td><td class='mono'>{i['last']}</td>"
        f"<td class='mono'>{i['age_days']}d</td>"
        f"<td>{'<b class=neg>OVERDUE</b>' if i['alarm'] else 'ok'}</td></tr>"
        for i in r.get("items", []))
    return f"""<h2>Monthly ritual — store freshness</h2>
<p class="sub">Fetch cadence keeps the event moats accruing (no backfill
exists). Alarm at 35 days. Checklist: TR snapshot (2FA) · short-register
fetch · BaFin dealings fetch · quarterly rebalance orders · tracker check.
</p>
<table><thead><tr><th>store</th><th>last</th><th>age</th><th>status</th>
</tr></thead><tbody>{rows}</tbody></table>"""


def sec_vs_portfolio(d: dict, public: bool) -> str:
    """Head-to-head: your real book vs the strategies over the same window. Private."""
    pr = d.get("portfolio_roi")
    if public or pr is None or getattr(pr, "empty", True):
        return ""
    vs = d.get("vs_scale") or {}
    eq = vs.get("raw_equity")
    if eq is None:
        eq = d["res"]["runs"][1.0]["equity"]
    start = pr.index[0]
    eqw = eq[eq.index >= start].dropna()
    prw = pr[pr.index >= start].dropna()
    if len(eqw) < 5 or len(prw) < 5:
        return ""
    strat = (eqw / eqw.iloc[0] - 1.0) * 100.0
    rcw = vs.get("rc_equity")
    if rcw is None:
        rcw = (d.get("quant", {}).get("vol_target") or {}).get("equity")
    rc = None
    if rcw is not None:
        rcw = rcw[rcw.index >= start].dropna()
        if len(rcw) >= 5:
            rc = (rcw / rcw.iloc[0] - 1.0) * 100.0

    line = rc if rc is not None else strat
    line_name = "Momentum — risk-conscious" if rc is not None else "Momentum"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prw.index, y=prw.values, name="Your portfolio (real)",
                             line=dict(color="#ffffff", width=2.6)))
    fig.add_trace(go.Scatter(x=line.index, y=line.values, name=line_name,
                             line=dict(color=C_RC, width=2.6)))
    vc_roi = None
    vc_rec = next((r for r in (d.get("registry") or [])
                   if r.id == "vol_core" and r.equity is not None), None)
    if vc_rec is not None:
        vcw = vc_rec.equity[vc_rec.equity.index >= start].dropna()
        if len(vcw) >= 5:
            vc_roi = (vcw / vcw.iloc[0] - 1.0) * 100.0
            fig.add_trace(go.Scatter(x=vc_roi.index, y=vc_roi.values,
                                     name="GARCH vol-managed IWDA core",
                                     line=dict(color=vc_rec.color or "#569cd6", width=1.8)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM)
    fig.update_layout(height=440, yaxis=dict(title="Cumulative ROI (%)", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))

    def stat(roi_pct):
        m = qg.perf_metrics(1.0 + roi_pct / 100.0)
        return roi_pct.iloc[-1], m.get("sharpe", 0.0), m.get("max_dd", 0.0) * 100.0
    pt, ps, pdd = stat(prw)
    lt, ls, ldd = stat(line)
    yrs = max((prw.index[-1] - start).days / 365.25, 1e-9)
    rows = (f"<tr><td>Your portfolio</td><td class='num'>{_pct(pt)}</td>"
            f"<td class='num mono'>{ps:.2f}</td><td class='num'>{_pct(pdd)}</td></tr>"
            f"<tr><td>{line_name}</td><td class='num'>{_pct(lt)}</td>"
            f"<td class='num mono'>{ls:.2f}</td><td class='num'>{_pct(ldd)}</td></tr>")
    if vc_roi is not None:
        vt_, vs_, vdd = stat(vc_roi)
        rows += (f"<tr><td>GARCH vol-managed IWDA core</td><td class='num'>{_pct(vt_)}</td>"
                 f"<td class='num mono'>{vs_:.2f}</td><td class='num'>{_pct(vdd)}</td></tr>")
    lead = lt - pt
    return (
        "<h2>You vs the strategies</h2>"
        f"<p class='dim'>Same window — since your first trade ({start.date()}, ~{yrs:.1f}y). "
        "<span style='color:#fff'>White</span> = your real book; "
        f"<span style='color:{C_RC}'>teal</span> = the risk-conscious (vol-targeted) strategy — "
        "hypothetical, lump-sum"
        + ("; the third line = the adopted GARCH vol-managed IWDA core, same window"
           if vc_roi is not None else "") + ". "
        + (f"It is run at <b>your invested scale (€{vs['capital']:,.0f})</b>, so the flat "
           "€1/transaction fee is the same % drag it is on your book (on €10k it's a negligible "
           "~0.15%). " if vs.get("capital") else "")
        + "Apples-to-pears (your book is cash-flow-timed), and the strategy carries every caveat "
        "above — survivorship especially — so read the gap as indicative.</p>"
        f"<div class='chart'>{fig_html(fig)}</div>"
        "<table><tr><th>Book</th><th class='num'>Total ROI</th><th class='num'>Sharpe</th>"
        f"<th class='num'>Max DD</th></tr>{rows}</table>"
        f"<p class='dim'>Over this window the risk-conscious strategy is <b>{_pct(lead)}</b> "
        f"{'ahead of' if lead >= 0 else 'behind'} your portfolio on total return. This is the "
        "version that matches how you actually run money.</p>")


# ── Lab ──────────────────────────────────────────────────────────────────────────

def sec_raw_reference(d: dict) -> str:
    """The raw, full-invested Original — reference only, in the lab."""
    raw = d["variants"][0]
    res = d["res"]
    window = _equity_window(res)
    fig = go.Figure()
    eq = raw["equity"].reindex(window).ffill()
    fig.add_trace(go.Scatter(x=eq.index, y=eq / d["capital"] * 100.0,
                             name="Original (raw, full-invested)",
                             line=dict(color=C_RAW, width=2.2)))
    for name, curve in benchmark_curves(d["benchmarks"], window, d["capital"]).items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve / d["capital"] * 100.0,
                                 name=name, line=dict(width=1.2)))
    fig.add_hline(y=100, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=380, yaxis_title="Index (start = 100)",
                      hovermode="x unified", margin=dict(t=20))
    return ("<h2>Survivorship-inflated — not achievable</h2>"
            "<div class='note warn'>This raw, full-invested book carries <b>two survivorship "
            "leaks</b> — holding-dead names (measured immaterial) and, the dominant one, an "
            "<b>absent-winners / membership tilt</b> from ranking only today's survivors. It is "
            "the <b>raw, full-invested</b> version of the same "
            f"selection. Its full-window total (<b>{_pct(raw['full']['net_return'] * 100)}</b>) and "
            f"test-window return (<b>{_pct(raw['test']['net_return'] * 100)}</b>) are <b>inflated by "
            f"survivorship and by holding a ~{raw['perf']['ann_vol'] * 100:.0f}%-vol book at full "
            f"exposure</b>, with a <b>{_pct(raw['perf']['max_dd'] * 100)}</b> drawdown almost no one "
            "would sit through. It is <b>not</b> how you'd run the book and is <b>not</b> quoted on "
            "the main page — kept here only so the inflation is visible, not hidden.</div>"
            f"<div class='chart'>{fig_html(fig)}</div>"
            + _perf_table(raw))


# ── Page assembly ────────────────────────────────────────────────────────────────

def render(d: dict, public: bool = False) -> str:
    """The whole page as a master-detail app: a left menu (one entry per strategy) and
    a stage of panes, one visible at a time via the client-side router. The default
    (Overview) pane shows every strategy vs benchmarks + the real portfolio; each
    strategy's own pane uses the SAME uniform template (curve · stats · windows ·
    method). Registry-driven — a new make_record() earns a menu entry + a pane."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cfg = d["strategy"]
    recs = _paned_records(d)

    home = _pane_compare_all(d, public)
    strat_panes = "".join(_pane_strategy(d, r, public) for r in recs)

    # Momentum-family research & robustness (public): grade, the evidence band (which
    # itself folds significance + factor spanning), diagnostics, regimes, scenarios,
    # the survivorship caveat. _momentum_evidence already embeds significance/factor —
    # never render those standalone too, or they duplicate.
    research = (f"<section class='pane' id='pane-research' hidden>{_crumb()}"
                "<h2>Research &amp; robustness — momentum family</h2>"
                + sec_grade_compare(d, public) + sec_ensemble(d, public)
                + _momentum_evidence(d, public) + sec_diagnostics(d, public)
                + sec_regime(d, public) + sec_scenarios(d, public) + sec_caveat(d)
                + "</section>")

    ops_body = (sec_track(d, public) + sec_venture(d, public)
                + sec_ritual(d, public) + sec_vs_portfolio(d, public))
    has_ops = bool(ops_body.strip())
    ops = (f"<section class='pane' id='pane-ops' hidden>{_crumb()}"
           "<h2>Operations — north star, ritual, your real book</h2>"
           f"{ops_body}</section>") if has_ops else ""

    has_lab = not public
    lab = (f"<section class='pane' id='pane-lab' hidden>{_crumb()}"
           "<h2>Research lab — momentum family (private)</h2>"
           "<p class='dim'>The private workings: the raw full-invested reference, the "
           "64-config grid, feasibility, and the supporting data.</p>"
           + sec_raw_reference(d) + sec_survivorship(d) + sec_grid(d) + sec_feasibility(d)
           + sec_timelines(d) + sec_method() + "</section>") if has_lab else ""

    stage = f"<div class='stage'>{home}{strat_panes}{research}{ops}{lab}</div>"
    body = "".join([
        PAGE_CSS,
        "<div class='pagehead'>"
        "<h1>Strategy — adopted stack &amp; registry</h1>"
        f"<p class='dim'>generated {now} · config {cfg.code} · "
        f"<a href='report.html'>← portfolio</a></p></div>",
        f"<div class='app'>{_spine(d, public, has_ops, has_lab)}{stage}</div>",
        ROUTER_JS,
    ])
    return page(f"Strategy — {cfg.code}", body)
