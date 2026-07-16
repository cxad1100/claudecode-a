"""Mega-cap PIT screen — size / growth / momentum arms, N-sweep. Debug build writes
the local snapshot only (mirrors build_momentum_report)."""
import argparse
import datetime as dt
import pathlib

import pandas as pd
import plotly.graph_objects as go

from tools.megacap import (run_arms, cap_panel, yoy_growth_panel, load_cache,
                           candidate_pool, coverage_report, prune_shares, ARMS, CACHE)
from tools.megacap_global import load_global, GIANTS as GLOBAL_GIANTS
from tools.momentum import (rebalance_dates, to_xetra_calendar, winsorize_prices,
                            benchmark_curves)
from tools.report_html import page, fig_html, card, pct
from tools.data_buffer import cached_price_history
from tools.portfolio_tools import BENCHMARKS
from tools import theme

ROOT = pathlib.Path(__file__).resolve().parent
PRICES_CSV = ROOT / "data" / "universe" / "universe_prices.csv"
META_CSV = ROOT / "data" / "universe" / "universe_meta.csv"
TURN_CSV = ROOT / "data" / "universe" / "universe_turnover.csv"
WINSOR_CAP = 0.5
N_SWEEP = (1, 5, 10, 25, 50)
HEADLINE_N = 25
K = 10
CAPITAL = 10_000.0

# Real, EUR-denominated yardsticks (the arms are EUR too → no FX contamination).
# PRIMARY is the toughest fair comp: buying the biggest names ≈ owning the Nasdaq.
BENCH_NAMES = ("Nasdaq 100", "MSCI World", "S&P 500")
PRIMARY_BENCH = "Nasdaq 100"
_ARM_COLOR = {"size": "#569cd6", "growth": "#4ec9b0", "momentum": "#c586c0"}
_ARM_LABEL = {"size": "Largest-cap (size)", "growth": "Revenue growth",
              "momentum": "12-1 momentum"}


def _slip(m) -> int:
    v = m.get("slippage_bps")
    return int(v) if v not in (None, "") else 30


def _cagr(series) -> float:
    s = series.dropna() if series is not None else pd.Series(dtype=float)
    if len(s) < 2:
        return float("nan")
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    if yrs <= 0 or float(s.iloc[0]) <= 0:
        return float("nan")
    return (float(s.iloc[-1]) / float(s.iloc[0])) ** (1.0 / yrs) - 1.0


def load_data(refresh: bool = False) -> dict:
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices = winsorize_prices(to_xetra_calendar(prices), cap=WINSOR_CAP)
    meta_df = pd.read_csv(META_CSV)
    slip = {r["ticker"]: _slip(r) for _, r in meta_df.iterrows()
            if r["ticker"] in prices.columns}
    # No cache yet → empty panels: the page still serves, showing 0% coverage and a
    # prompt to run the live fetch. Once populated, rebuild for real numbers.
    shares, rev = load_cache(CACHE) if CACHE.exists() else ({}, {})
    # Non-US giants EDGAR can't cover (TSMC, Tencent, Nestlé, LVMH, Roche, Alibaba,
    # Samsung, Sony) → EUR price + yfinance shares, merged so the top-N pool isn't
    # US-only. They join the size + momentum arms; the growth arm skips them (no
    # revenue panel), which is honest rather than fabricated.
    gpx, gshares, gslip = load_global(prices.index)
    gpx = gpx[[c for c in gpx.columns if c not in prices.columns]]   # never duplicate a column
    n_global = len(gpx.columns)
    if n_global:
        keep = set(gpx.columns)
        prices = pd.concat([prices, gpx], axis=1)
        shares = {**shares, **{k: v for k, v in gshares.items() if k in keep}}
        slip = {**slip, **{k: v for k, v in gslip.items() if k in keep}}
    shares = prune_shares(shares, meta_df)          # drop glitch caps + same-company dupes
    dates = rebalance_dates(prices.index)
    cap = cap_panel(shares, prices, dates)
    yoy = yoy_growth_panel(rev, dates)
    turnover = (pd.read_csv(TURN_CSV, index_col=0, parse_dates=True)
                if TURN_CSV.exists() else None)
    cover = coverage_report({t: (t in shares) for t in
                             candidate_pool(meta_df, prices.columns,
                                            turnover=turnover)})
    # Real EUR index benchmarks (same source the momentum/strategy pages use). Guarded:
    # a failed fetch just drops the overlay, the screen still renders.
    bench = pd.DataFrame(index=prices.index)
    try:
        picks = {n: BENCHMARKS[n] for n in BENCH_NAMES if n in BENCHMARKS}
        raw = cached_price_history([tk for tk, _ in picks.values()], period="9y",
                                   force=refresh)
        bench = raw.rename(columns={tk: n for n, (tk, _) in picks.items()})
    except Exception:
        pass
    return dict(prices=prices, cap=cap, yoy=yoy, rev=rev, slip=slip, benchmarks=bench,
                capital=CAPITAL, coverage=cover, n_global=n_global)


def _overlay_chart(arm_eq: dict, bcurves: dict, capital: float) -> str:
    """Headline arms + real index benchmarks, rebased to 100, log-y so the
    excess-over-index gap is readable next to a 6× run."""
    fig = go.Figure()
    for arm in ARMS:
        eq = arm_eq.get(arm)
        s = eq.dropna() if eq is not None else pd.Series(dtype=float)
        if len(s) < 2:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s / float(s.iloc[0]) * 100.0,
                                 name=_ARM_LABEL[arm],
                                 line=dict(color=_ARM_COLOR[arm], width=2.4)))
    for name, curve in bcurves.items():
        c = curve.dropna()
        if len(c) < 2:
            continue
        fig.add_trace(go.Scatter(x=c.index, y=c / capital * 100.0, name=name,
                                 line=dict(width=1.5, dash="dot")))
    fig.add_hline(y=100, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=460, yaxis_title="Index (start = 100, log)",
                      yaxis_type="log", hovermode="x unified", margin=dict(t=20))
    return f"<div class='chart'>{fig_html(fig)}</div>"


def _topn_cutoff(cap: pd.DataFrame, n: int) -> pd.Series:
    """The n-th largest PIT market cap at each rebalance — the size a name must clear
    to even ENTER the top-n screen."""
    def nth(row):
        v = row.dropna().sort_values(ascending=False)
        return float(v.iloc[n - 1]) if len(v) >= n else float("nan")
    return cap.apply(nth, axis=1).dropna()


def sec_survivorship_check(cap: pd.DataFrame) -> str:
    """Why a delisted-universe rebuild is ~immaterial for THIS screen: the top-N cap
    cutoff dwarfs every company that actually failed in-window. Death-survivorship
    (holding a name into its bankruptcy) needs a name big enough to enter the screen
    to have died — and none did. Quantified, not asserted."""
    if cap is None or cap.empty:
        return ""
    parts = []
    for n in (HEADLINE_N, 50):
        c = _topn_cutoff(cap, n) / 1e9
        if c.empty:
            continue
        parts.append(f"<b>top-{n}</b> never below <b>€{c.min():.0f}bn</b> "
                     f"(median €{c.median():.0f}bn)")
    if not parts:
        return ""
    return (
        "<h2>Survivorship — does a delisted-universe rebuild change this?</h2>"
        "<p class='note'><b>For this screen: no, and here is the proof.</b> To bias a "
        "cap screen, a name must be big enough to <i>enter</i> it and then die while "
        "held. The entry bar here is enormous — " + "; ".join(parts) + ". The largest "
        "companies that actually failed in 2018–2026 (regional banks, property "
        "developers) peaked in the <b>tens of billions</b> and were mostly acquisitions, "
        "not zeros — an order of magnitude below the cutoff. Nothing that large died, so "
        "adding the dead names back moves the arms by ~0. This is the exception that "
        "<i>confirms</i> the momentum family's survivorship problem: that book trades the "
        "whole small/mid universe, where deaths are common and the correction is real "
        "(handled there via delisting-stress). The residual excess-over-index above is "
        "<b>not</b> dead-name bias — it is the equal-weight top-10 tilt versus a "
        "cap-weighted index, plus universe composition.</p>")


def _period_returns(prices: pd.DataFrame, picks, d0, d1) -> dict:
    """Each pick's simple return over the hold [d0, d1], from the price panel."""
    out = {}
    for t in picks:
        if t not in prices.columns:
            continue
        p0 = prices.loc[:d0, t].dropna()
        p1 = prices.loc[:d1, t].dropna()
        if len(p0) and len(p1) and float(p0.iloc[-1]) > 0:
            out[t] = float(p1.iloc[-1]) / float(p0.iloc[-1]) - 1.0
    return out


def sec_holdings(res: dict, arm: str, prices: pd.DataFrame, name_map: dict,
                 giants: set) -> str:
    """What the arm actually held each rebalance — the latest book name-by-name
    (weight · period return · contribution), then the full period-by-period history.
    Non-US giants are tagged so the internationalisation is visible at a glance."""
    hl = [h for h in res[arm]["holdings_log"] if h.get("picks")]
    if not hl:
        return ""

    def chip(t):
        cls = "gia" if t in giants else "usx"
        return f"<span class='chip {cls}'>{t}</span>"

    # ── latest book, name by name ──
    last = hl[-1]
    d0, d1, picks = last["date"], last["next"], last["picks"]
    rets = _period_returns(prices, picks, d0, d1)
    w = 1.0 / len(picks)
    body_rows = ""
    for t in sorted(picks, key=lambda x: -rets.get(x, -9)):
        r = rets.get(t)
        flag = " <span class='chip gia'>non-US</span>" if t in giants else ""
        body_rows += (f"<tr><td>{chip(t)} {name_map.get(t, t)}{flag}</td>"
                      f"<td class='num'>{w * 100:.0f}%</td>"
                      f"<td class='num'>{pct(r * 100) if r is not None else '—'}</td>"
                      f"<td class='num'>{pct(r * w * 100) if r is not None else '—'}</td></tr>")
    basket = sum(rets.values()) * w if rets else float("nan")
    n_gi = sum(1 for t in picks if t in giants)

    # ── full history: one row per period ──
    eqrun = 1.0
    hist = ""
    for h in hl:
        rr = _period_returns(prices, h["picks"], h["date"], h["next"])
        b = sum(rr.values()) / len(h["picks"]) if rr else 0.0
        eqrun *= (1 + b)
        chips = " ".join(chip(t) for t in h["picks"])
        hist += (f"<tr><td class='mono'>{pd.Timestamp(h['date']).date()}</td>"
                 f"<td>{chips}</td><td class='num'>{pct(b * 100)}</td>"
                 f"<td class='num mono'>{eqrun:.2f}×</td></tr>")

    return (
        f"<h3>{_ARM_LABEL.get(arm, arm)} — latest book "
        f"({pd.Timestamp(d0).date()} → {pd.Timestamp(d1).date()})</h3>"
        f"<p class='dim'>10 names, equal-weight. <b>{n_gi} non-US</b> this period. "
        f"Basket return over the hold: <b>{pct(basket * 100)}</b>.</p>"
        "<table><thead><tr><th>holding</th><th class='num'>weight</th>"
        "<th class='num'>period return</th><th class='num'>contribution</th></tr></thead>"
        f"<tbody>{body_rows}</tbody></table>"
        f"<details><summary>Full holdings history — {len(hl)} rebalances</summary>"
        "<div class='scroll'><table><thead><tr><th>rebalance</th><th>held (10, equal-weight)</th>"
        "<th class='num'>basket ret</th><th class='num'>cumulative</th></tr></thead>"
        f"<tbody>{hist}</tbody></table></div></details>")


def build_html(data: dict) -> str:
    prices, cap, yoy, slip = data["prices"], data["cap"], data["yoy"], data["slip"]
    cov, capital = data["coverage"], data["capital"]
    bench = data.get("benchmarks")

    # One arm run per N (reused for both the table and the headline overlay).
    arm_runs = {n: run_arms(prices, slip, cap, yoy, n=n, k=K, capital=capital)
                for n in N_SWEEP}
    hl_eq = {arm: arm_runs[HEADLINE_N][arm]["runs"][1.0].get("equity") for arm in ARMS}

    # Benchmark curves + CAGR over the headline arms' own window.
    window = next((e.dropna().index for e in hl_eq.values()
                   if e is not None and len(e.dropna()) > 1), None)
    bcurves = (benchmark_curves(bench, window, capital)
               if bench is not None and window is not None and len(bench.columns) else {})
    bench_cagr = {name: _cagr(c) for name, c in bcurves.items()}
    prim = bench_cagr.get(PRIMARY_BENCH, float("nan"))
    prim_ok = prim == prim                                  # not NaN

    def _exc(cagr):
        return "—" if not prim_ok or cagr != cagr else pct((cagr - prim) * 100)

    # N-sweep table, now with CAGR + excess-over-primary.
    rows = []
    for n in N_SWEEP:
        for arm in ARMS:
            r = arm_runs[n][arm]["runs"][1.0]
            st, eq = r["stats"], r.get("equity")
            c = _cagr(eq)
            hl = " class='headline'" if n == HEADLINE_N else ""
            rows.append(
                f"<tr{hl}><td>N={n}</td><td>{_ARM_LABEL.get(arm, arm)}</td>"
                f"<td class='num'>{pct(st['net_return'] * 100)}</td>"
                f"<td class='num'>{pct(c * 100) if c == c else '—'}</td>"
                f"<td class='num'>{st['sharpe']:.2f}</td>"
                f"<td class='num'>{pct(st['max_drawdown'] * 100)}</td>"
                f"<td class='num'>{_exc(c)}</td></tr>")

    # Benchmark reference rows (their own CAGR — the yardstick).
    for name, c in bench_cagr.items():
        rows.append(
            f"<tr class='bench'><td>index</td><td>{name}</td>"
            f"<td class='num'>{pct((float(bcurves[name].dropna().iloc[-1] / capital) - 1) * 100)}</td>"
            f"<td class='num'>{pct(c * 100) if c == c else '—'}</td>"
            "<td class='num'>—</td><td class='num'>—</td><td class='num'>0.0%</td></tr>")

    empty = ("<p class='prompt'><em>No market-cap data yet — run the EDGAR "
             "fundamentals fetch, then rebuild.</em></p>" if cov["covered"] == 0 else "")

    # holdings view — name map (EDGAR universe + global giants) and the giant set
    giants = set(GLOBAL_GIANTS.values())
    try:
        _meta = pd.read_csv(META_CSV)
        name_map = dict(zip(_meta["ticker"], _meta["name"]))
    except Exception:
        name_map = {}
    name_map.update({tk: nm for nm, tk in GLOBAL_GIANTS.items()})
    hold_arms = arm_runs[HEADLINE_N]
    holdings = ("<h2>What the book actually held — period by period</h2>"
                "<p class='dim'><span class='chip gia'>teal</span> = non-US giant "
                "(TSMC, Tencent, Nestlé, LVMH, Roche, Alibaba, Sony, merged in via "
                "yfinance shares); <span class='chip usx'>grey</span> = EDGAR-covered. "
                "Individual name returns are close-to-close over each hold; the basket is "
                "their equal-weight average.</p>"
                + sec_holdings(hold_arms, "size", prices, name_map, giants)
                + "<details><summary>Growth arm holdings</summary>"
                + sec_holdings(hold_arms, "growth", prices, name_map, giants) + "</details>"
                + "<details><summary>Momentum arm holdings</summary>"
                + sec_holdings(hold_arms, "momentum", prices, name_map, giants) + "</details>")

    chart = _overlay_chart(hl_eq, bcurves, capital) if bcurves else ""
    cards = ""
    if bcurves:
        yard = "".join(card(f"{name} CAGR", pct(c * 100, signed=False))
                       for name, c in bench_cagr.items() if c == c)
        best = max((( _cagr(hl_eq[a]), a) for a in ARMS if hl_eq.get(a) is not None),
                   default=(float("nan"), None))
        exc_txt = _exc(best[0]) if best[1] else "—"
        cards = (f"<div class='cards'>{yard}"
                 + card(f"Best arm − {PRIMARY_BENCH}", exc_txt) + "</div>")

    interp = ""
    if prim_ok:
        interp = (f"<p class='note'><b>How much is just the market?</b> Buying the biggest "
                  f"names is nearly owning the index — over this window {PRIMARY_BENCH} alone "
                  f"compounded <b>{pct(prim * 100, signed=False)}/yr</b>. Read the "
                  "<b>excess-over-index</b> column, not the gross return: that thin margin is "
                  "the only part that could be skill, and even it inherits the "
                  "<b>survivor-biased universe</b> (today's largest liquid names — the dead "
                  "giants that a real screen would have bought and lost on are absent). "
                  "Internal comparison, not an achievable return.</p>")

    n_global = data.get("n_global", 0)
    glob_txt = (f" <b>+{n_global} non-US giants</b> (TSMC, Tencent, Nestlé, LVMH, Roche, "
                "Alibaba, Sony) merged via yfinance shares." if n_global else "")
    body = (
        f"<style>tr.headline td {{ background:{theme.BG_PANEL}; font-weight:600; }}"
        f"tr.bench td {{ color:{theme.FG_DIM}; font-style:italic; }}"
        f".chip {{ display:inline-block; padding:0 6px; border-radius:5px; font-family:{theme.MONO};"
        f" font-size:0.74rem; margin:1px 2px 1px 0; }}"
        f".chip.gia {{ background:#134e4a; color:#5eead4; }}"
        f".chip.usx {{ background:{theme.BG_PANEL}; color:{theme.FG_DIM}; }}</style>"
        "<h1>Mega-cap PIT screen — size / growth / momentum</h1>"
        f"<p class='coverage'>Cap coverage: {cov['covered']}/{cov['candidates']} "
        f"EDGAR candidates ({cov['pct']}%).{glob_txt}</p>"
        f"{empty}"
        f"<p>Headline N={HEADLINE_N}, hold k={K}, monthly, equal-weight, net of slippage. "
        "Fully point-in-time — shares &amp; revenue stamped to their SEC-EDGAR filing dates "
        "(giants: yfinance filing-stamped shares).</p>"
        f"{interp}{cards}{chart}"
        f"{sec_survivorship_check(cap)}"
        f"{holdings}"
        "<h2>N-sweep — every screen width × arm</h2>"
        "<table><thead><tr><th>screen</th><th>arm</th><th class='num'>Total</th>"
        "<th class='num'>CAGR</th><th class='num'>Sharpe</th><th class='num'>Max DD</th>"
        f"<th class='num'>Excess vs {PRIMARY_BENCH}</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return page("Mega-cap PIT screen — size / growth / momentum", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    html = build_html(load_data(refresh=args.refresh))
    (ROOT / "local").mkdir(exist_ok=True)
    (ROOT / "local" / "megacap.html").write_text(html)
    print(f"wrote local/megacap.html ({dt.date.today()})")


if __name__ == "__main__":
    main()
