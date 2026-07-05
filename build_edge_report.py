"""Edge Stack — the structural-edge composite report.

  python build_edge_report.py            # writes local/edge.html (local-only page)
  python build_edge_report.py --open

The framing: retail cannot out-inform hedge funds, out-model quants, or out-speed HFT —
and doesn't need to. The durable retail edges are structural: (1) CAPACITY — trade the
small/illiquid pond institutions cannot enter at size (the existing momentum core);
(2) FORCED FLOWS — trade against counterparties obligated to trade (the December
tax-loss-rebound sleeve, tools.edge_seasonal); (3) HORIZON — hold through what
career-risked managers cannot (the vol-managed overlay makes holding survivable);
(4) COSTS/TAXES — lazy rebalancing, bands, €1-fee modeling: the only guaranteed alpha.

Inference honesty: the seasonal sleeve produces ONE observation per year. The per-year
table and the cross-sectional Monte-Carlo null (random names, same pool, same window)
carry the evidence — never a headline Sharpe off ~8 data points.
"""
import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tools import theme, significance as sig, quant_grade as qg, vol_forecast as vf
from tools import edge_seasonal as es
from tools.report_html import pct as _pct, card as _card, page, fig_html
from tools.universe_panel import load_universe_panel, momentum_net_returns
from tools import gates, live_state

ROOT = Path(__file__).parent

K = 10                      # sleeve slots
ENTRY_MMDD = "12-15"        # signal date; entry executes t+1
EXIT_MMDD = "01-31"
W_SLEEVE = 0.20             # forced-flow sleeve weight in the stack
TARGET_VOL = 0.15           # matches the strategy page's overlay
CAPITAL = 10_000.0


def gather(force: bool = False, refresh: bool | None = None) -> dict:
    refresh = force if refresh is None else refresh
    d: dict = dict(seasonal=None, mc=None, stack=None)

    uni = load_universe_panel()                     # prepared ONCE, shared with the core
    if uni is not None:
        res = es.run_seasonal(uni["prices"], uni["slip"], k=K, capital=CAPITAL,
                              fee_eur=uni["fee_eur"], entry_mmdd=ENTRY_MMDD,
                              exit_mmdd=EXIT_MMDD, liq_max=uni["liq_max"],
                              min_price=uni["min_price"], pit=uni["pit"], execute_lag=1)
        d["seasonal"] = res
        strat = es.seasonal_period_returns(res["years"])
        if len(strat) >= 3:
            # the null's random books match the sleeve's TYPICAL breadth, not the k
            # cap — in thin-loser years the sleeve holds fewer than K names and a
            # fixed-K null would be better-diversified than the strategy it judges
            picked = [len(y["picks"]) for y in res["years"] if y["picks"]]
            k_eff = max(1, int(round(np.mean(picked)))) if picked else K
            d["mc"] = sig.monte_carlo_null(res["pools"], strat, k=k_eff, ppy=1.0,
                                           n_trials=2000, seed=0)
            d["mc"]["k_eff"] = k_eff
        d["sleeve_verdict"] = gates.sleeve_verdict(d["mc"], res["years"])
        d["stack_verdict"] = gates.stack_verdict(d["sleeve_verdict"],
                                                 w_sleeve=W_SLEEVE, target_vol=TARGET_VOL)
        d["sleeve_status"] = live_state.sleeve_status(uni["prices"].index,
                                                      entry_mmdd=ENTRY_MMDD,
                                                      exit_mmdd=EXIT_MMDD)
        d["meta"] = uni["meta"]

    # the stack: momentum core ⊕ seasonal sleeve, then the vol-managed overlay.
    # No blanket except here — a corrupt panel should fail the build loudly (serve.py
    # falls back to the last snapshot); only the missing-panel case degrades to None.
    core = momentum_net_returns(panel=uni) if uni is not None else None
    if core is not None and d["seasonal"] is not None and len(core) > 900:
        sleeve = d["seasonal"]["equity"].pct_change().fillna(0.0)
        blend = es.combine_sleeves(core, sleeve, w_sleeve=W_SLEEVE)
        blend = blend.loc[core.index.intersection(blend.index)]
        fc = vf.ewma_vol(blend)
        stack = vf.vol_managed(blend, fc, target_vol=TARGET_VOL, band=0.10,
                               cost_bps=5.0, fee_eur=1.0, capital=CAPITAL)
        core_eq = (1.0 + core).cumprod()
        d["stack"] = dict(overlay=stack, core=dict(qg.perf_metrics(core_eq), equity=core_eq),
                          blend_eq=(1.0 + blend).cumprod())
    return d


# ───────────────────────────── sections ─────────────────────────────

def sec_edges(d: dict) -> str:
    return (
        '<div class="note"><b>The premise — structural edge, not informational edge.</b> '
        'Retail cannot out-inform funds (news is priced before you read it), out-model '
        'quants, or out-speed HFT. What survives are edges that come from <i>who you are</i>, '
        'not what you know: <b>(1)&nbsp;Capacity</b> — a €10k book trades names a $1B fund '
        'cannot touch; the momentum core lives here. <b>(2)&nbsp;Forced flows</b> — trade '
        'against counterparties <i>obligated</i> to trade; the December tax-loss-rebound '
        'sleeve below. <b>(3)&nbsp;Horizon</b> — no redemptions, no career risk; the '
        'vol-managed overlay exists to make holding through drawdowns survivable. '
        '<b>(4)&nbsp;Costs &amp; taxes</b> — lazy rebalancing, bands, the €1 fee modelled: '
        'the only guaranteed alpha. Every sleeve maps to one edge; nothing here claims to '
        'know tomorrow&rsquo;s price. '
        '<br><br><b>Expectations, in writing.</b> The momentum core is the only return '
        'engine; the overlays buy Sharpe and drawdown and usually cost total return. '
        'Base case if the momentum premium persists: market-like or somewhat better '
        'returns with materially shallower drawdowns. Downside case: an index fund minus '
        'costs. Which one you get is decided by the pre-registered gates below and by '
        'live-vs-backtest tracking (/vol) — never by a backtest headline. Not advice.</div>')


def _gate_table(checks: list[dict]) -> str:
    rows = "".join(
        f"<tr><td>{c['name']}</td><td class='num mono'>{c['value']}</td>"
        f"<td class='num mono dim'>{c['threshold']}</td>"
        f"<td class='num' style='color:{'#46c84e' if c['passed'] else '#ef4444'}'>"
        f"{'PASS' if c['passed'] else 'FAIL'}</td></tr>" for c in checks)
    return ("<table><tr><th>Gate</th><th class='num'>Measured</th>"
            f"<th class='num'>Bar</th><th class='num'>Result</th></tr>{rows}</table>")


def sec_verdict(d: dict) -> str:
    sv = d.get("sleeve_verdict")
    if not sv:
        return ""
    stk = d.get("stack_verdict", {})
    color = "#46c84e" if sv["verdict"] == "KEEP" else "#ef4444"
    return ("<h2>Verdict — pre-registered gates</h2>"
            "<p class='dim'>Thresholds committed to code before the first real-data "
            "run (tools/gates.py); the verdict is a binding decision, not a story "
            "fitted to the outcome.</p>"
            f"<h3>Tax-loss sleeve: <span style='color:{color}'>{sv['verdict']}</span></h3>"
            + _gate_table(sv["checks"]) +
            f"<p><b>{stk.get('statement', '')}</b> <span class='dim'>(forecast method "
            "per the /vol verdict)</span></p>")


def sec_today(d: dict) -> str:
    st = d.get("sleeve_status")
    if not st:
        return ""
    out = ["<h2>Sleeve calendar — today</h2>"]
    if st["in_window"]:
        w = st["window"]
        out.append(f"<p><b>IN WINDOW</b> — signal was {w['signal'].date()}, exit "
                   f"{w['exit'].date()} ({st['days_left']} days left).</p>")
        res = d.get("seasonal")
        rec = res["years"][-1] if res and res["years"] else None
        if rec and rec["picks"]:
            meta = d.get("meta", {})
            rows = []
            for t in rec["picks"]:
                m = meta.get(t, {})
                isin = m.get("isin") if pd.notna(m.get("isin", np.nan)) else ""
                rows.append(f"<tr><td class='mono'>{str(m.get('home') or t).split('.')[0]}</td>"
                            f"<td>{m.get('name', t)}</td>"
                            f"<td class='dim mono' style='font-size:0.72rem'>{isin}</td>"
                            f"<td class='num'>{_pct(rec['ytd'].get(t, 0) * 100)}</td></tr>")
            out.append("<p class='dim'>Current sleeve picks — search the ISIN in Trade "
                       "Republic:</p><table><tr><th>Ticker</th><th>Name</th><th>ISIN</th>"
                       "<th class='num'>YTD at signal</th></tr>" + "".join(rows) + "</table>")
    else:
        out.append(f"<p>Out of window — flat in cash. <b>Next signal in "
                   f"{st['days_to_signal']} days</b> (mid-December); the sleeve then "
                   f"buys the bottom-{K} negative-YTD liquid names and exits end of "
                   "January.</p>")
    return "".join(out)


def sec_seasonal(d: dict) -> str:
    res = d.get("seasonal")
    if not res:
        return ("<h2>Forced-flow sleeve — the tax-loss rebound</h2>"
                "<div class='note'>The universe price panel (data/universe/"
                "universe_prices.csv) isn't on this machine, so the sleeve backtest was "
                "skipped. Run the universe build first, then refresh this page.</div>")
    rows = []
    for y in res["years"]:
        if not y["picks"]:
            rows.append(f"<tr><td class='mono'>{y['year']}</td><td class='num dim'>—</td>"
                        "<td class='num dim'>no losers</td><td class='num dim'>—</td>"
                        "<td class='num dim'>—</td></tr>")
            continue
        avg_ytd = float(np.mean(list(y["ytd"].values())))
        n_dead = len(y.get("dead", set()))
        dead_cell = (f"<td class='num mono' style='color:#ef4444'>{n_dead}</td>"
                     if n_dead else "<td class='num mono'>0</td>")
        rows.append(f"<tr><td class='mono'>{y['year']}→{y['year'] + 1}</td>"
                    f"<td class='num mono'>{len(y['picks'])}</td>"
                    f"<td class='num'>{_pct(avg_ytd * 100)}</td>"
                    f"<td class='num'>{_pct(y['net_ret'] * 100)}</td>{dead_cell}</tr>")
    eq = res["equity"]
    roi = (eq / eq.iloc[0] - 1.0) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roi.index, y=roi.values, name="tax-loss sleeve",
                             line=dict(color="#4ec9b0", width=2.0)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM)
    fig.update_layout(height=340, yaxis=dict(title="Cumulative ROI (%)", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))
    mc = d.get("mc")
    cards = ""
    if mc:
        cards = "".join([
            _card("Windows (years)", f"{len(res['years'])}"),
            _card("p-value vs random picks", f"{mc['p_sharpe']:.3f}"),
            _card("Sleeve total (gross)", _pct(mc["strat_total"] * 100)),
            _card("Random-book median", _pct(mc["null_total_median"] * 100)),
        ])
        cards = f'<div class="cards">{cards}</div>'
    return (
        "<h2>Forced-flow sleeve — the tax-loss rebound</h2>"
        f"<p class='dim'>Each year: rank YTD returns among eligible liquid names at the "
        f"last session on/before Dec&nbsp;{ENTRY_MMDD[3:]}, buy the bottom-{K} <b>with "
        "negative YTD only</b> (no loss → no forced seller), enter t+1, exit end of "
        "January, cash the rest of the year. The inference is the <b>per-year table</b> "
        "and the <b>Monte-Carlo null</b> — random books from the same pool over the same "
        "windows, drawn at the sleeve's <i>average</i> breadth so thin-loser years don't "
        "face an artificially diversified null — never a Sharpe off a handful of yearly "
        "points.</p>"
        + cards +
        "<table><tr><th>Window</th><th class='num'>Picks</th><th class='num'>Avg YTD of "
        "picks</th><th class='num'>Window net return</th><th class='num'>Died in "
        "window</th></tr>" + "".join(rows) + "</table>"
        f"<div class='chart'>{fig_html(fig)}</div>")


def sec_stack(d: dict) -> str:
    st = d.get("stack")
    if not st:
        return ("<h2>The Edge Stack</h2>"
                "<div class='note'>Needs both the momentum core and the seasonal sleeve — "
                "unavailable on this machine (see above).</div>")
    ov, core = st["overlay"], st["core"]
    ce, oe = core["equity"], ov["equity"]
    croi = (ce / ce.iloc[0] - 1.0) * 100.0
    oroi = (oe / oe.iloc[0] - 1.0) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=croi.index, y=croi.values, name="Momentum core (raw)",
                             line=dict(color="#808080", width=1.6)))
    fig.add_trace(go.Scatter(x=oroi.index, y=oroi.values,
                             name=f"Edge Stack ({W_SLEEVE:.0%} sleeve, vol-managed)",
                             line=dict(color="#4ec9b0", width=2.4)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM)
    fig.update_layout(height=420, yaxis=dict(title="Cumulative ROI (%)", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))

    def row(label, m, exp=None):
        e = f"{exp * 100:.0f}%" if exp is not None else "100%"
        return (f"<tr><td>{label}</td><td class='num'>{_pct(m.get('ann_return', 0) * 100)}</td>"
                f"<td class='num mono'>{m.get('ann_vol', 0) * 100:.1f}%</td>"
                f"<td class='num mono'>{m.get('sharpe', 0):.2f}</td>"
                f"<td class='num'>{_pct(m.get('max_dd', 0) * 100)}</td>"
                f"<td class='num mono'>{e}</td></tr>")
    return (
        "<h2>The Edge Stack</h2>"
        f"<p class='dim'>Capacity core ({1 - W_SLEEVE:.0%} momentum) ⊕ forced-flow sleeve "
        f"({W_SLEEVE:.0%} tax-loss rebound), sized by the forecast-based vol overlay to "
        f"{TARGET_VOL:.0%} target vol (de-risk only). The sleeve is cash ~11 months/year, "
        "so it mostly adds its January episode plus dry powder.</p>"
        f"<div class='chart'>{fig_html(fig)}</div>"
        "<table><tr><th>Book</th><th class='num'>Ann. return</th><th class='num'>Vol</th>"
        "<th class='num'>Sharpe</th><th class='num'>Max DD</th><th class='num'>Avg exp.</th></tr>"
        + row("Momentum core (raw)", core)
        + row("Edge Stack (vol-managed)", ov, ov.get("avg_exposure")) + "</table>")


def sec_method(d: dict) -> str:
    return (
        '<div class="note warn"><b>Honesty box.</b> '
        '<b>(1) Sample size.</b> The seasonal sleeve has one observation per year — a '
        'single-digit number of windows. The Monte-Carlo selection null is the real test; '
        'the equity curve is illustration, not evidence. '
        '<b>(1b) Survivorship is AMPLIFIED here, not just inherited.</b> The sleeve '
        'deliberately buys the year&rsquo;s most-crushed names — the cohort with the highest '
        'delisting hazard — on a universe of today&rsquo;s survivors. Deep losers that '
        'subsequently died are missing at a higher rate than random names, so the missing-'
        'graveyard inflation is worse for this sleeve than for momentum (which buys '
        'winners); the &ldquo;died in window&rdquo; column above shows only the deaths the '
        'graveyard actually captures. '
        '<b>(2) Documented, not secret.</b> The January/tax-loss effect is in the '
        'literature since Wachtel (1942) and Roll (1983); it has weakened in large caps '
        'and survives mainly in small/illiquid names — which is exactly the capacity '
        'argument, but also means execution costs decide everything. '
        '<b>(3) Geography.</b> The strongest evidence is US (Dec 31 tax year). The TR '
        'universe is global: US names carry the classic effect; German investors face '
        'immediate withholding without a wash-sale rule, so the local motive is real but '
        'thinner-documented. '
        '<b>(4) Crowding.</b> Anticipatory buying has pulled part of the rebound into '
        'late December — the entry sits mid-December for that reason, and the edge can '
        'decay further. '
        '<b>(5) The stack inherits every caveat of its parts</b> — the momentum core&rsquo;s '
        'survivorship notes (see /strategy) and the vol overlay&rsquo;s not-alpha framing '
        '(see /vol). Past performance is not future returns.</div>')


def build(d: dict, public: bool = False) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "".join([
        "<h1>Edge Stack — structural edges only</h1>",
        f"<p class='dim'>generated {now} · <a href='report.html'>← portfolio</a> · "
        "<a href='strategy.html'>strategy</a> · <a href='vol.html'>vol lab</a></p>",
        sec_edges(d),
        sec_verdict(d),
        sec_today(d),
        sec_seasonal(d),
        sec_stack(d),
        sec_method(d),
    ])
    return page("Edge Stack", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    d = gather(refresh=args.refresh)
    local = ROOT / "local/edge.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d))                            # local-only — no docs/ export
    print(f"wrote {local}")
    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
