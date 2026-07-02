"""Generate the static momentum HTML report.

  python build_momentum_report.py            # writes local/momentum.html + docs/momentum.html
  python build_momentum_report.py --refresh  # force re-download of price data

Long-only cross-sectional momentum (12-1, top-k, monthly rebalance) over the
broker-tradeable universe, walk-forward with Trade Republic-style costs.
Mirrors build_pairs_report.py. Public build shows percentages only (the €1
fee text excepted) — the universe is today's survivors, so the survivorship
caveat stays.
"""

import argparse
import re
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from tools import theme
from tools.report_html import fig_html, pct as _pct, card as _card, page
from tools.momentum import (run_momentum, benchmark_curves, equal_weight_curve,
                            winsorize_prices, to_xetra_calendar)
from tools.pairs_universe import UNIVERSE, fetch_prices
from tools.portfolio_tools import BENCHMARKS
from tools.data_buffer import cached_price_history
from tools.universe_pit import PITUniverse
from tools.universe_assemble import delisting_map
from tools.momentum_grid import run_grid, feasibility, ALL_CONFIGS

# ── Settings (one place) ──────────────────────────────────────────────────────
K            = 15
LOOKBACK     = 252
SKIP         = 21
REBAL        = "M"
START        = "2018-01-01"   # first rebalance (MiFID era); full 9y survivorship-corrected window
TRAIN_END    = "2021-12-31"   # train ≤ this (in-sample, used to choose the config)
VAL_END      = "2023-12-31"   # validation = train_end→here; test = after (never informs the pick)
LIQ_MAX      = 30
MIN_PRICE    = 1.0        # drop sub-€1 penny listings (12-1 momentum = tick noise there)
MIN_TURNOVER = 100_000    # €/day median turnover floor — drops thin/ghost listings whose
                          #   German-exchange feed is stale/broken (Aker BP ARC.F €0/day,
                          #   Seagate 847.F €893/day) and only the real prices survive
MAX_TURNOVER = 50_000_000_000   # €/day ceiling — no real stock clears €50B/day; above = glitch
WINSOR_CAP   = 0.5        # clip daily returns ±50% — kills split-adjustment glitches
EXEC_LAG     = 1          # trade t+1 (next bar after the signal), not the signal-day close
CAPITAL      = 10_000.0   # paper account, EUR
FEE_EUR      = 1.0        # Trade Republic per-order fee
COST_MULTS   = (0.0, 1.0, 2.0)

REBAL_LABEL = {"M": "monthly", "W": "weekly", "Q": "quarterly"}

ROOT = Path(__file__).parent
PRICES_CSV = ROOT / "data" / "universe" / "universe_prices.csv"
META_CSV = ROOT / "data" / "universe" / "universe_meta.csv"
# The universe is now TR-native (tools.tr_tradeable --enumerate → tools.build_tr_universe):
# every live name is tradeable on TR by construction, so there is no separate tradeability filter.


def _broker(t: str) -> str:
    """Broker label for a yfinance ticker: 'WKN · Name' so the pick is findable
    in the app. Falls back to the ticker when the universe lacks broker IDs."""
    m = UNIVERSE.get(t, {})
    wkn, name = m.get("local_id", ""), m.get("name", t)
    return f"{wkn} · {name}" if wkn else name


def _disp(meta: dict, t: str) -> str:
    """Display ticker: the liquid HOME symbol (STX.US → STX), never the dead .F
    German shadow that keys the row."""
    m = meta.get(t, {})
    return str(m.get("home") or t).split(".")[0]


def _name(m: dict, t: str) -> str:
    """Clean display name — EODHD's German listing name often trails the German
    ticker in parens ('AXT  (AHV.SG)'); strip that, keep real parentheticals."""
    return re.sub(r"\s*\([A-Z0-9][A-Z0-9.]*\)\s*$", "", str(m.get("name", t))).strip() or t


# ── Data assembly ─────────────────────────────────────────────────────────────

def _slip(m) -> int:
    v = m.get("slippage_bps")
    return int(v) if pd.notna(v) else 30


def gather(force: bool = False, refresh: bool | None = None, with_grid: bool = True) -> dict:
    """Load the survivorship-corrected dataset (survivors ∪ 270 dead), run the
    walk-forward with the active graveyard, and (when `with_grid`) the 64-config
    matrix. `force`/`refresh` only re-fetch the benchmark series."""
    refresh = force if refresh is None else refresh
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices = to_xetra_calendar(prices)                         # L&S/XETRA sessions only (tradeable days)
    prices = winsorize_prices(prices, cap=WINSOR_CAP)          # de-glitch the raw feed
    meta_df = pd.read_csv(META_CSV)                            # TR-native universe (already tradeable)
    meta = {r["ticker"]: dict(r) for _, r in meta_df.iterrows()}
    slip = {t: _slip(m) for t, m in meta.items() if t in prices.columns}
    sectors = {t: m.get("sector") for t, m in meta.items()}     # real GICS now (tools.enrich_sectors)
    pit = PITUniverse(prices, delisting_map(meta_df))

    benches = {n: v for n, v in BENCHMARKS.items() if n != "Bitcoin"}   # equities/bonds only
    bench_tickers = [tk for tk, _ in benches.values()]
    bench_raw = cached_price_history(bench_tickers, period="9y", force=refresh)
    bench = bench_raw.rename(columns={tk: name for name, (tk, _) in benches.items()})
    spx = bench["S&P 500"] if "S&P 500" in bench.columns else bench.iloc[:, 0]

    # Sectors are real now (tools.enrich_sectors) → sector-neutral (B) configs join the grid.
    res = run_momentum(prices, slip, k=K, lookback=LOOKBACK, skip=SKIP, capital=CAPITAL,
                       cost_mults=COST_MULTS, freq=REBAL, liq_max=LIQ_MAX, fee_eur=FEE_EUR,
                       min_price=MIN_PRICE, start=START, pit=pit, execute_lag=EXEC_LAG)
    grid = (run_grid(prices, slip, sectors=sectors, benchmark=spx, pit=pit, start=START,
                     configs=ALL_CONFIGS,
                     train_end=TRAIN_END, val_end=VAL_END, capital=CAPITAL,
                     lookback=LOOKBACK, skip=SKIP, execute_lag=EXEC_LAG)
            if with_grid else None)

    return dict(prices=prices, res=res, benchmarks=bench, capital=CAPITAL, meta=meta,
                grid=grid, n_dead=int(meta_df["delisting_date"].notna().sum()),
                n_countries=len({m.get("country") for m in meta.values()} - {"—", None}))


def _equity_window(res: dict):
    eq = res["runs"][1.0]["equity"]
    return eq.index[1:] if len(eq) > 1 else eq.index


# ── Sections ──────────────────────────────────────────────────────────────────

def sec_intro() -> str:
    return f"""
<div class="note">
<b>Cross-sectional momentum (long-only)</b> — ranks {len(UNIVERSE)} broker-tradeable
stocks by 12-1 momentum (trailing 12 months, skipping the most recent month), holds
the equal-weight top-{K}, rebalanced monthly. Walk-forward, no look-ahead, with
Trade Republic-style costs (€{FEE_EUR:.0f}/order + live half-spread). Unlike the
pairs page, this is long-only and <b>executable</b> on Trade Republic. Not financial
advice.
</div>"""


def sec_holdings(d: dict) -> str:
    log = d["res"]["holdings_log"]
    if not log:
        return ("<h2>Current top picks</h2>"
                "<p class='dim'>No rebalance with enough history yet.</p>")
    cur = log[-1]
    picks = cur["picks"]
    out = ["<h2>Current top picks</h2>"]
    out.append(f"<p class='dim'>Equal-weight top-{len(picks)} as of {cur['date'].date()}, "
               "ranked by 12-1 momentum score. Each leg shows its home ticker, name and ISIN "
               "— search the ISIN or name in Trade Republic to trade it.</p>")
    if not picks:
        out.append("<p class='dim'>No eligible names at the latest rebalance.</p>")
        return "".join(out)
    rows = []
    w = 100.0 / len(picks)
    for t in picks:
        m = d["meta"].get(t, {})
        home = str(m.get("home") or t).split(".")[0]          # STX.US → STX (search this in TR)
        isin = m.get("isin") if pd.notna(m.get("isin")) else ""
        rows.append(
            f"<tr><td class='mono'>{home}</td>"
            f"<td>{_name(m, t)}</td>"
            f"<td class='dim mono' style='font-size:0.72rem'>{isin}</td>"
            f"<td>{m.get('country','—')}</td>"
            f"<td class='num mono'>{cur['scores'].get(t, float('nan')) * 100:+.1f}%</td>"
            f"<td class='num mono'>{w:.1f}%</td></tr>")
    out.append("<table><tr><th>Ticker</th><th>Name</th><th>ISIN</th><th>Country</th>"
               "<th class='num'>12-1 momentum</th><th class='num'>Weight</th></tr>"
               + "".join(rows) + "</table>")
    return "".join(out)


def sec_curve(d: dict) -> str:
    res = d["res"]
    log = res["holdings_log"]
    window = _equity_window(res)
    out = ["<h2>Walk-forward equity vs benchmarks</h2>"]
    out.append("<p class='dim'>Equity since the first rebalance with enough history, "
               "compared to a buy-hold equal-weight basket of today's top picks "
               "(survivorship-honest baseline) and the MSCI World / S&amp;P 500.</p>")

    fig = go.Figure()
    eq = res["runs"][1.0]["equity"].reindex(window).ffill()
    fig.add_trace(go.Scatter(x=eq.index, y=eq / d["capital"] * 100.0,
                             name=f"Momentum (top-{K})",
                             line=dict(color=theme.ACCENT, width=2.4)))

    # Use the first rebalance that actually has eligible picks — the very first
    # date(s) can have an empty pick list if there isn't yet `lookback+skip` days
    # of history for any name (equal_weight_curve can't build a basket from []).
    first_picks = next((h["picks"] for h in log if h["picks"]), [])
    if first_picks:
        ew = equal_weight_curve(d["prices"], first_picks, window, d["capital"])
        fig.add_trace(go.Scatter(x=ew.index, y=ew / d["capital"] * 100.0,
                                 name="Equal-weight (initial picks, buy-hold)",
                                 line=dict(color=theme.FG_DIM, width=1.4, dash="dot")))

    for name, curve in benchmark_curves(d["benchmarks"], window, d["capital"]).items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve / d["capital"] * 100.0,
                                 name=name, line=dict(width=1.4)))

    fig.add_hline(y=100, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=460, yaxis_title="Index (start = 100)",
                      hovermode="x unified", margin=dict(t=20))
    out.append(f"<div class='chart'>{fig_html(fig)}</div>")
    return "".join(out)


def sec_stats(d: dict, public: bool) -> str:
    runs = d["res"]["runs"]
    base = runs[1.0]["stats"]
    out = ["<h2>Performance</h2>"]
    out.append(f"<p class='dim'>Walk-forward since {d['res']['start']}, "
               f"{REBAL_LABEL[REBAL]} rebalances, t+1 execution, costs as above.</p>")

    cards = [
        _card("Net return", _pct(base["net_return"] * 100)),
        _card("Sharpe", f"{base['sharpe']:.2f}"),
        _card("Max drawdown", _pct(base["max_drawdown"] * 100)),
        _card("Win rate", _pct(base["win_rate"] * 100, signed=False)
              if base["win_rate"] is not None else "—"),
    ]
    if not public:
        cards.append(_card("Net P&L", f"€{base['net_return'] * d['capital']:+,.0f}"))
        cards.append(_card("Costs paid", f"€{base['total_costs']:,.0f}"))
    else:
        cards.append(_card("Costs / capital",
                           _pct(base["total_costs"] / d["capital"] * 100, signed=False)))
    out.append(f'<div class="cards">{"".join(cards)}</div>')

    rows = "".join(
        f"<tr><td class='mono'>{m:.0f}×</td>"
        f"<td class='num'>{_pct(runs[m]['stats']['net_return'] * 100)}</td>"
        f"<td class='num mono'>{runs[m]['stats']['sharpe']:.2f}</td>"
        f"<td class='num'>{_pct(runs[m]['stats']['max_drawdown'] * 100)}</td></tr>"
        for m in sorted(runs))
    out.append("<h3>Cost sensitivity</h3>"
               "<p class='dim'>Identical holdings schedule, re-priced at 0×, 1× and 2× "
               f"the assumed €{FEE_EUR:.0f}/order + half-spread frictions.</p>"
               "<table><tr><th>Costs</th><th class='num'>Net return</th>"
               "<th class='num'>Sharpe</th><th class='num'>Max DD</th></tr>"
               + rows + "</table>")
    return "".join(out)


def sec_rebalance_log(d: dict) -> str:
    log = d["res"]["holdings_log"]
    if not log:
        return ""
    rows = []
    for h in reversed(log[-12:]):
        picks_label = ", ".join(h["picks"]) if h["picks"] else "—"
        rows.append(f"<tr><td class='mono'>{h['date'].date()}</td>"
                    f"<td class='num mono'>{len(h['picks'])}</td>"
                    f"<td class='mono' style='font-size:0.8rem'>{picks_label}</td></tr>")
    return ("<details><summary>Rebalance log (last 12)</summary>"
           "<table><tr><th>Date</th><th class='num'>Holdings</th><th>Picks</th></tr>"
           + "".join(rows) + "</table></details>")


def sec_caveat(d: dict) -> str:
    nc = d.get("n_countries", 0)
    return f"""
<div class="note warn">
<b>Read the result as relative, not absolute — but not because of survivorship.</b>
The universe is survivorship-<i>corrected</i> (delisted/collapsed names are carried and
liquidated by the graveyard, below), and momentum barely feels it anyway: it buys
<i>winners</i>, so it almost never holds a name into its death. The real reasons the
headline is optimistic are <b>regime</b> (2023→ was an exceptional momentum tape) and
<b>concentration</b> (a top-k that a few explosive names dominate; sector-neutral B now
caps single-sector piling, but there is no per-name weight cap). The universe is the liquid,
<b>Trade-Republic-investable</b> set across
{nc} countries, each priced off its <b>home exchange × EUR FX</b> (the Lang &amp; Schwarz
model — NVIDIA on NASDAQ, Samsung on KRX, in their own currency converted to EUR), behind a
≥100k/day turnover floor. Membership uses peak turnover over the whole window, so there is a
mild liquidity look-ahead, and TR-routability is assumed from liquidity, not verified. Daily
closes only — intraday execution and borrow costs (for any future short overlay) are ignored.
</div>"""


def sec_method() -> str:
    return f"""
<h2>How it works</h2>
<details open><summary>12-1 momentum (the core idea)</summary>
<p>Momentum is the empirical tendency for assets that have performed well over the
past ~12 months to keep outperforming over the next month, and for recent losers to
keep lagging. The "12-1" convention skips the most recent month: trailing-month
returns show short-term <i>reversal</i> rather than momentum, so including it would
fight the signal. Score = price(t − {SKIP}d) / price(t − {LOOKBACK}d) − 1, using only
data available at the rebalance date.</p></details>
<details><summary>Eligibility &amp; selection</summary>
<p>At each {REBAL_LABEL[REBAL]} rebalance, a name is eligible if its assumed
half-spread is ≤ {LIQ_MAX} bps and it has ≥ {LOOKBACK + SKIP} trading days of
history with a positive last price. The top-{K} eligible names by 12-1 score are
held equal-weight until the next rebalance — turnover is whatever set difference
results from re-ranking, no forced full rotation.</p></details>
<details><summary>Costs &amp; execution</summary>
<p>Entries and exits at each rebalance are charged €{FEE_EUR:.0f}/order plus the
ticker's live half-spread (in bps of the position size); names held across a
rebalance are not re-charged. The cost-sensitivity table re-prices the identical
holdings schedule at 0×, 1× and 2× these frictions — selection never depends on the
multiplier.</p></details>
<details><summary>No look-ahead, by construction</summary>
<p>Scores at a rebalance date use only price history up to and including that date;
the resulting positions accrue returns strictly <i>after</i> it. Unit tests assert
that truncating future data leaves past scores and past holdings unchanged.</p>
</details>
"""


def sec_survivorship(d: dict) -> str:
    n = d.get("n_dead", 0)
    if not n:
        return ""
    return (f'<div class="note warn"><b>Survivorship-corrected.</b> The universe '
            f'includes <b>{n}</b> EUR-listed names that delisted/died 2018→now '
            f'(e.g. Wirecard, peak €195 → €0.40), held until their delisting date and '
            f'liquidated by the graveyard at the last traded price — so the backtest '
            f'can buy a name that later goes to zero and eat the loss.</div>')


def sec_grid(d: dict) -> str:
    g = d.get("grid")
    if not g:
        return ""
    has_test = any("test" in c for c in g["cells"])
    rows = []
    for c in sorted(g["cells"], key=lambda c: c["val"]["sharpe"], reverse=True):
        test_cells = (f"<td class='num'>{_pct(c['test']['net_return'] * 100)}</td>"
                      f"<td class='num mono'>{c['test']['sharpe']:.2f}</td>") if has_test else ""
        rows.append(
            f"<tr><td class='mono'>{c['code']}</td>"
            f"<td class='num'>{_pct(c['train']['net_return'] * 100)}</td>"
            f"<td class='num mono'>{c['train']['sharpe']:.2f}</td>"
            f"<td class='num'>{_pct(c['val']['net_return'] * 100)}</td>"
            f"<td class='num mono'>{c['val']['sharpe']:.2f}</td>"
            f"{test_cells}"
            f"<td class='num mono'>{c['trades_per_year']:.0f}</td></tr>")
    test_hdr = "<th class='num'>Test ret</th><th class='num'>Test Sh</th>" if has_test else ""
    return ("<h2>64-config grid (A·B·C·D·E·F)</h2>"
            "<p class='dim'>A vol-adj · B sector-neutral · C trend-filter · D 10-slot · E quarterly "
            "· F lazy (B now live — real GICS sectors sourced via tools.enrich_sectors). Ranked by "
            "<b>validation</b> Sharpe; train = 2018–21 "
            "(picks the config), validation = 2022–23, <b>test = 2024→ (held out, never "
            "informs the pick)</b>. A config you'd trust holds up across all three — "
            "especially test.</p>"
            "<table><tr><th>Cfg</th><th class='num'>Train ret</th><th class='num'>Train Sh</th>"
            "<th class='num'>Val ret</th><th class='num'>Val Sh</th>" + test_hdr +
            "<th class='num'>Trades/yr</th></tr>" + "".join(rows) + "</table>")


def sec_feasibility(d: dict) -> str:
    g = d.get("grid")
    if not g:
        return ""
    best = max(g["cells"], key=lambda c: c["val"]["sharpe"])
    f = feasibility(best, capital=d["capital"], fee_eur=FEE_EUR)
    return (f"<h3>Small-account feasibility (best val cell {best['code']})</h3>"
            f"<p class='dim'>{best['trades_per_year']:.0f} trades/yr × €{FEE_EUR:.0f} = "
            f"€{f['annual_fee_eur']:.0f}/yr = {f['fee_drag_pct']:.2f}% of €{d['capital']:,.0f}. "
            f"Pays for itself: <b>{'yes' if f['pays_for_itself'] else 'no'}</b>.</p>")


def _pnl_color(ret: float, dead: bool) -> str:
    if dead:
        return "#000000"                  # defaulted / delisted
    if ret >= 0.20:
        return "#0a6b00"                  # a lot positive
    if ret >= 0.0:
        return "#46c84e"                  # positive
    if ret > -0.20:
        return "#ef4444"                  # negative
    return "#7a0000"                       # a lot negative


def sec_timelines(d: dict) -> str:
    g = d.get("grid")
    if not g:
        return ""
    blocks = []
    for c in sorted(g["cells"], key=lambda c: c["val"]["sharpe"], reverse=True):
        lines = []
        for row in c["timeline"]:
            dead = set(row["dead"])
            spans = " ".join(
                f"<span style='color:{_pnl_color(r, t in dead)}' "
                f"title='{t} {r:+.0%}'>{_disp(d['meta'], t)}</span>"
                for t, r in row["ret"].items())
            spans = spans or "<span class='dim'>cash</span>"
            rv = [v for v in row["ret"].values() if pd.notna(v)]
            mret = sum(rv) / len(rv) if rv else 0.0
            lines.append(f"<div><span class='mono dim'>{row['date']}</span> "
                         f"<b style='color:{_pnl_color(mret, False)}'>{mret:+.1%}</b> {spans}</div>")
        blocks.append(
            f"<details><summary>{c['code']} · val Sharpe {c['val']['sharpe']:.2f} · "
            f"{c['trades_per_year']:.0f} tr/yr</summary>"
            f"<div style='font-size:0.78rem;line-height:1.7'>{''.join(lines)}</div></details>")
    return ("<h2>Monthly picks per variation</h2>"
            "<p class='dim'>Each line is one rebalance's equal-weight picks, colored by that "
            "holding period's return — <span style='color:#0a6b00'>■</span> ≥+20% · "
            "<span style='color:#46c84e'>■</span> up · <span style='color:#ef4444'>■</span> down · "
            "<span style='color:#7a0000'>■</span> ≤−20% · <span style='color:#000'>■</span> "
            "defaulted. Hover a ticker for its %. All 64 variations, collapsed.</p>"
            + "".join(blocks))


# ── Assembly ──────────────────────────────────────────────────────────────────

def build(d: dict, public: bool = False) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "Momentum Lab" + ("" if public else " — private")
    back = "index.html" if public else "report.html"
    badge = "public build — percentages only" if public else "private build"
    body = "".join([
        f"<h1>{title}</h1>",
        f"<p class='dim'>generated {now} · {badge} · "
        f"<a href='{back}'>← portfolio monitor</a></p>",
        sec_intro(),
        sec_holdings(d),
        sec_curve(d),
        sec_stats(d, public),
        sec_rebalance_log(d),
        sec_caveat(d),
        sec_survivorship(d) if not public else "",
        sec_grid(d) if not public else "",
        sec_feasibility(d) if not public else "",
        sec_timelines(d) if not public else "",
        sec_method(),
    ])
    return page(title, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the local report")
    ap.add_argument("--refresh", action="store_true", help="force price re-download")
    args = ap.parse_args()

    print("gathering momentum data (yfinance)...")
    d = gather(refresh=args.refresh)

    # The momentum lab is now the lower half of the Strategy page; this standalone
    # build is debug-only and writes the local snapshot only (no docs/ export).
    local = ROOT / "local/momentum.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d, public=False))
    print(f"wrote {local}  (debug; the live lab is the Strategy page's lower half)")

    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
