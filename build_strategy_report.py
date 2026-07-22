"""The chosen production strategy — one config (the 'ultimate' extracted from the
32-config grid), on the survivorship-corrected universe.

  python build_strategy_report.py            # writes local/strategy.html + docs/strategy.html
  python build_strategy_report.py --open

Unlike the momentum *lab* (which renders the whole grid), this page commits to a
single config and shows it cleanly: which strategy + why, current picks, equity vs
benchmarks, train/validation/full performance, the PnL-colored monthly timeline,
and the (now small) honest caveats.
"""
import argparse
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tools.report_html import pct as _pct, card as _card, page, fig_html
from tools import theme, significance as sig, quant_grade as qg
import build_vol_report as V     # vol overlay strategy (its own gather + rich sections)
import build_edge_report as E    # structural stack + tax-loss sleeve
import build_pairs_report as P   # market-neutral pairs
from tools.momentum import (run_momentum, winsorize_prices, to_xetra_calendar,
                            precompute_eligibility)
from tools.universe_pit import PITUniverse
from tools.universe_assemble import delisting_map
from tools.momentum_grid import MomentumConfig, _stats_slice, run_grid, ALL_CONFIGS
from tools.portfolio_tools import BENCHMARKS, parse_portfolio
from tools.portfolio_analytics import build_roi_timeseries
from tools.data_buffer import cached_price_history
from build_momentum_report import (
    PRICES_CSV, META_CSV, ROOT, LOOKBACK, SKIP, START, LIQ_MAX, MIN_PRICE, CAPITAL,
    FEE_EUR, COST_MULTS, TRAIN_END, VAL_END, WINSOR_CAP, EXEC_LAG,
    _slip, _broker, _disp, _pnl_color, sec_holdings, sec_curve,
    sec_grid, sec_feasibility, sec_timelines, sec_survivorship, sec_method,
)

# The chosen strategy — A···EF = vol-adjusted, equal-weight top-15, quarterly, lazy.
# Picked from the 32-config grid (sector-neutral B is excluded — the global universe has no
# sector data, so it would be a silent no-op), run on the XETRA / Lang & Schwarz trading
# calendar (the only days you can actually fill), for the highest worst-case robustness
# min(train, validation) Sharpe among configs that pay for their own costs and are positive in
# both windows. A···EF is top-tier on robustness (train 0.71 / val 0.87) AND holds up out-of-sample
# (test 1.27) — preferred over the marginally-higher-robust A····F, which overfits (test 0.65);
# quarterly + lazy keep turnover (and the €1/order drag) low. Survivorship-clean by construction —
# momentum buys winners, dying names rank last, so it holds ~0 into death.
STRATEGY = MomentumConfig(vol_adjust=True, slots=15, freq="Q", lazy=True)

# Risk-conscious overlay: volatility-target the book to this annualised vol (de-risk only,
# park the rest in cash). Cuts the raw strategy's ~32% vol / −44% drawdown to a moderate
# profile while lifting the Sharpe — the prudent way to actually run momentum.
RISK_TARGET_VOL = 0.15


def _desc(cfg: MomentumConfig) -> str:
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


def gather(force: bool = False, refresh: bool | None = None) -> dict:
    refresh = force if refresh is None else refresh
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices = to_xetra_calendar(prices)                         # L&S/XETRA sessions only (tradeable days)
    prices = winsorize_prices(prices, cap=WINSOR_CAP)          # de-glitch the raw feed
    meta_df = pd.read_csv(META_CSV)                            # universe is pre-filtered at build time
    # Universe is now TR-native (tools.tr_tradeable --enumerate → tools.build_tr_universe):
    # every live name is one you can trade on TR, by construction — no separate filter.
    n_live = int(meta_df["delisting_date"].isna().sum())
    meta = {r["ticker"]: dict(r) for _, r in meta_df.iterrows()}
    slip = {t: _slip(m) for t, m in meta.items() if t in prices.columns}
    pit = PITUniverse(prices, delisting_map(meta_df))

    benches = {n: v for n, v in BENCHMARKS.items() if n != "Bitcoin"}   # equities/bonds only
    bench_tickers = [tk for tk, _ in benches.values()]
    bench_raw = cached_price_history(bench_tickers, period="9y", force=refresh)
    bench = bench_raw.rename(columns={tk: name for name, (tk, _) in benches.items()})
    spx = bench["S&P 500"] if "S&P 500" in bench.columns else bench.iloc[:, 0]

    # No sector data on the global universe → sectors=None (sector-neutral configs excluded).
    res = run_momentum(prices, slip, lookback=LOOKBACK, skip=SKIP, capital=CAPITAL,
                       cost_mults=COST_MULTS, start=START, liq_max=LIQ_MAX, fee_eur=FEE_EUR,
                       min_price=MIN_PRICE, sectors=None, benchmark=spx, pit=pit,
                       execute_lag=EXEC_LAG, **STRATEGY.kwargs())
    eq, tr = res["runs"][1.0]["equity"], res["runs"][1.0]["trades"]
    te, ve = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
    train = _stats_slice(eq, tr, eq.index[0], te, CAPITAL)
    val = _stats_slice(eq, tr, te + pd.Timedelta(days=1), ve, CAPITAL)
    test = _stats_slice(eq, tr, ve + pd.Timedelta(days=1), eq.index[-1], CAPITAL)
    hits = sum(len(h.get("dead", set())) for h in res["holdings_log"])

    # ── Upper bound: drop dead names that were never TR-tradeable (ISIN domicile absent
    #    from the live TR set — e.g. the 200 Korean corpses TR never offered). Including them
    #    only adds forced death-losses, so removing them lifts the result: the all-dead run is
    #    the lower bound, this the upper. Re-run the SAME strategy on the trimmed graveyard.
    live_cc = {str(i)[:2] for i, dl in zip(meta_df["isin"], meta_df["delisting_date"])
               if pd.isna(dl) and isinstance(i, str) and len(str(i)) >= 2}
    keep = [(pd.isna(dl) or (isinstance(i, str) and str(i)[:2] in live_cc))
            for i, dl in zip(meta_df["isin"], meta_df["delisting_date"])]
    ub_meta = meta_df[keep].reset_index(drop=True)
    n_dead_dropped = int(meta_df["delisting_date"].notna().sum() - ub_meta["delisting_date"].notna().sum())
    ub_tickers = set(ub_meta["ticker"])
    ub_prices = prices[[c for c in prices.columns if c in ub_tickers]]
    ub_pit = PITUniverse(ub_prices, delisting_map(ub_meta))
    ub_res = run_momentum(ub_prices, {t: slip[t] for t in ub_prices.columns if t in slip},
                          lookback=LOOKBACK, skip=SKIP, capital=CAPITAL, cost_mults=(1.0,),
                          start=START, liq_max=LIQ_MAX, fee_eur=FEE_EUR, min_price=MIN_PRICE,
                          sectors=None, benchmark=spx, pit=ub_pit, execute_lag=EXEC_LAG,
                          **STRATEGY.kwargs())
    ub_eq, ub_tr = ub_res["runs"][1.0]["equity"], ub_res["runs"][1.0]["trades"]
    bounds = dict(lower_full=test["net_return"], n_dead_dropped=n_dead_dropped,
                  upper_full=ub_res["runs"][1.0]["stats"]["net_return"],
                  lower_full_all=res["runs"][1.0]["stats"]["net_return"],
                  upper_test=_stats_slice(ub_eq, ub_tr, ve + pd.Timedelta(days=1),
                                          ub_eq.index[-1], CAPITAL)["net_return"])
    grid = run_grid(prices, slip, sectors=None, benchmark=spx, pit=pit, start=START,
                    configs=[c for c in ALL_CONFIGS if not c.sector_neutral],
                    train_end=TRAIN_END, val_end=VAL_END, capital=CAPITAL,
                    lookback=LOOKBACK, skip=SKIP, execute_lag=EXEC_LAG)
    # ── Significance & robustness: random-selection null, deflated Sharpe, bootstrap CI ──
    hl = res["holdings_log"]
    rb_dates = [h["date"] for h in hl] + [hl[-1]["next"]]
    elig = precompute_eligibility(prices, slip, rb_dates, liq_max=LIQ_MAX,
                                  min_obs=LOOKBACK + SKIP, min_price=MIN_PRICE, pit=pit)
    pools = sig.period_pools(prices, rb_dates, elig, execute_lag=EXEC_LAG)
    strat_rets = sig.strategy_period_returns(hl)
    ppy = {"Q": 4.0, "M": 12.0, "W": 52.0}.get(STRATEGY.freq, 12.0)
    mc = sig.monte_carlo_null(pools, strat_rets, k=STRATEGY.slots, ppy=ppy, n_trials=1000, seed=0)
    dsr = sig.deflated_sharpe_ratio(strat_rets, [c["full"]["sharpe"] for c in grid["cells"]], ppy=ppy)
    ci = sig.bootstrap_sharpe_cagr_ci(strat_rets, ppy=ppy, seed=0)

    # ── Quant scorecard: industry metrics + an honest letter grade ──
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    li = {str(r["isin"]) for i, (r, dl) in enumerate(zip(meta_df.to_dict("records"),
          meta_df["delisting_date"])) if pd.isna(dl) and str(r["isin"])}
    di = {str(r["isin"]) for r, dl in zip(meta_df.to_dict("records"), meta_df["delisting_date"])
          if pd.notna(dl) and str(r["isin"])}
    overlap = len(li & di) / max(len(di), 1)
    quant = dict(perf=qg.perf_metrics(eq), bench=qg.vs_benchmark(eq, spx),
                 trades=qg.trade_metrics(tr, CAPITAL, years), roll=qg.rolling_sharpe(eq),
                 grade=qg.grade(test["sharpe"], dsr["dsr"], mc["p_sharpe"], overlap),
                 isin_overlap=overlap,
                 vol_target=qg.vol_target(eq, target_vol=RISK_TARGET_VOL))

    # ── Your real portfolio's ROI (cumulative %), for the head-to-head ──
    portfolio_roi = None
    pf_csv = ROOT / "input" / "portfolio.csv"
    if pf_csv.exists():
        try:
            txns = parse_portfolio(pf_csv)["transactions"]
            pr, _ = build_roi_timeseries(txns)
            if pr is not None and not pr.empty:
                portfolio_roi = pr
        except Exception:
            portfolio_roi = None

    n_countries = len({m.get("country") for m in meta.values()} - {"—", None})
    return dict(prices=prices, res=res, benchmarks=bench, capital=CAPITAL, meta=meta, quant=quant,
                portfolio_roi=portfolio_roi,
                strategy=STRATEGY, train=train, val=val, test=test, graveyard_hits=hits,
                grid=grid, n_dead=int(meta_df["delisting_date"].notna().sum()),
                n_countries=n_countries,
                n_live=n_live, bounds=bounds,
                significance=dict(mc=mc, dsr=dsr, ci=ci, ppy=ppy))


def sec_intro(d: dict) -> str:
    cfg = d["strategy"]
    nc = d["n_countries"]
    return (f'<div class="note"><b>Chosen strategy — {cfg.code} ({_desc(cfg)}).</b> '
            "Picked from the <b>32-config grid</b> for the highest worst-case "
            "<b>min(train, validation) Sharpe</b> among configs that pay for their own trading "
            "costs and are positive in both windows — so the result rides robustness, not one "
            "lucky rally. (Sector-neutral is excluded: the global universe carries no sector "
            "data, so it would be a silent no-op.) The universe is the liquid, "
            f"<b>Trade-Republic-investable</b> names across <b>{nc} countries</b>, each priced off "
            "its <b>home exchange × EUR FX</b> — the way Lang &amp; Schwarz actually fills you "
            "(NVIDIA on NASDAQ, Samsung on KRX, Rheinmetall on XETRA, in their own currency, "
            "converted to EUR). Behind a <b>≥100k/day turnover</b> floor. Long-only, walk-forward, "
            "executable. Not advice.</div>")


def sec_perf(d: dict, public: bool) -> str:
    full = d["res"]["runs"][1.0]["stats"]
    test = d["test"]
    out = ["<h2>Performance</h2>",
           "<p class='dim'>Train = 2018–21 (used to pick the config), validation = 2022–23 "
           "(used to compare configs), <b>test = 2024→ (held out — never touched the "
           "choice)</b>. The <b>test</b> column is the only truly out-of-sample number; "
           "trust it over the eye-popping full-window total.</p>"]
    cards = [
        _card("Test net return", _pct(test["net_return"] * 100)),
        _card("Test Sharpe", f"{test['sharpe']:.2f}"),
        _card("Test max DD", _pct(test["max_drawdown"] * 100)),
        _card("Full net return", _pct(full["net_return"] * 100)),
    ]
    if not public:
        cards.append(_card("Net P&L", f"€{full['net_return'] * d['capital']:+,.0f}"))
    out.append(f'<div class="cards">{"".join(cards)}</div>')
    rows = "".join(
        f"<tr><td>{name}</td><td class='num'>{_pct(s['net_return'] * 100)}</td>"
        f"<td class='num mono'>{s['sharpe']:.2f}</td>"
        f"<td class='num'>{_pct(s['max_drawdown'] * 100)}</td></tr>"
        for name, s in (("Train 2018–21", d["train"]), ("Validation 2022–23", d["val"]),
                        ("Test 2024→ (held out)", test), ("Full 2018→", full)))
    out.append("<table><tr><th>Window</th><th class='num'>Net return</th>"
               "<th class='num'>Sharpe</th><th class='num'>Max DD</th></tr>" + rows + "</table>")
    return "".join(out)


def _yearly_returns(series: pd.Series) -> pd.Series:
    """Calendar-year returns keyed by year int: each year from the prior year's last
    close to this year's last close; the first year runs from inception."""
    s = series.dropna()
    last = s.groupby(s.index.year).last()
    prev = last.shift(1)
    if len(prev):
        prev.iloc[0] = s.iloc[0]          # first year measured from inception
    return (last / prev - 1.0).dropna()


def sec_yearly(d: dict, public: bool) -> str:
    eq = d["res"]["runs"][1.0]["equity"].dropna()
    if len(eq) < 2:
        return ""
    sret = _yearly_returns(eq)
    bench = d["benchmarks"]
    spx = bench["S&P 500"] if "S&P 500" in bench.columns else None
    bret = _yearly_returns(spx.reindex(eq.index).ffill()) if spx is not None else pd.Series(dtype=float)
    # € P&L per year — the actual paper-account change (private builds only)
    last = eq.groupby(eq.index.year).last()
    prev = last.shift(1)
    prev.iloc[0] = eq.iloc[0]
    pnl = (last - prev).dropna()

    rows = []
    for y, r in sret.items():
        eur = f"<td class='num mono'>€{pnl.get(y, 0.0):+,.0f}</td>" if not public else ""
        b = (f"<td class='num'>{_pct(bret[y] * 100)}</td>"
             if y in bret.index else "<td class='num dim'>—</td>")
        rows.append(f"<tr><td class='mono'>{y}</td>"
                    f"<td class='num'>{_pct(r * 100)}</td>{eur}{b}</tr>")
    eur_hdr = "<th class='num'>P&amp;L (€10k)</th>" if not public else ""
    pnl_note = ("the €10k paper account's actual P&amp;L, and " if not public else "and ")
    return ("<h2>Yearly P&amp;L</h2>"
            "<p class='dim'>Calendar-year net return of the strategy (first year from "
            f"inception), {pnl_note}the S&amp;P 500 over the same year. 2018 and 2026 are "
            "part-years.</p>"
            "<table><tr><th>Year</th><th class='num'>Strategy</th>" + eur_hdr +
            "<th class='num'>S&amp;P 500</th></tr>" + "".join(rows) + "</table>")


def sec_risk_conscious(d: dict, public: bool) -> str:
    """The risk-conscious variant: the same picks, volatility-targeted so the book de-risks
    into turbulence. Cuts drawdown and lifts the Sharpe vs the raw strategy."""
    vt = d.get("quant", {}).get("vol_target")
    base = d.get("quant", {}).get("perf")
    if not vt or not base or "equity" not in vt:
        return ""
    be, ve = d["res"]["runs"][1.0]["equity"], vt["equity"]
    broi = (be / be.iloc[0] - 1.0) * 100.0
    vroi = (ve / ve.iloc[0] - 1.0) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=broi.index, y=broi.values, name="Raw strategy (A·"+"··EF)",
                             line=dict(color="#808080", width=1.6)))
    fig.add_trace(go.Scatter(x=vroi.index, y=vroi.values, name="Risk-conscious (vol-targeted)",
                             line=dict(color="#4ec9b0", width=2.4)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM)
    fig.update_layout(height=420, yaxis=dict(title="Cumulative ROI (%)", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))

    def r(label, m, exp=None):
        e = f"<td class='num mono'>{exp*100:.0f}%</td>" if exp is not None else "<td class='num dim'>100%</td>"
        return (f"<tr><td>{label}</td><td class='num'>{_pct(m['ann_return']*100)}</td>"
                f"<td class='num mono'>{m['ann_vol']*100:.0f}%</td><td class='num mono'>{m['sharpe']:.2f}</td>"
                f"<td class='num'>{_pct(m['max_dd']*100)}</td>{e}</tr>")
    return (
        "<h2>Risk-conscious version</h2>"
        f"<p class='dim'>Same momentum picks, but the book is <b>volatility-targeted to "
        f"{RISK_TARGET_VOL:.0%}</b>: each day it scales exposure toward that vol using the prior "
        "day's realised vol (no look-ahead, no leverage — it only ever de-risks) and parks the rest "
        "in cash. When momentum gets turbulent the book automatically shrinks. The raw strategy's "
        "−44% drawdown is the price of its raw return; this is how you'd actually run it.</p>"
        f"<div class='chart'>{fig_html(fig)}</div>"
        "<table><tr><th>Version</th><th class='num'>Ann. return</th><th class='num'>Vol</th>"
        "<th class='num'>Sharpe</th><th class='num'>Max DD</th><th class='num'>Avg exposure</th></tr>"
        f"{r('Raw (full-invested)', base)}{r('Risk-conscious', vt, vt['avg_exposure'])}</table>"
        f"<p class='dim'>Vol-targeting cuts the drawdown from <b>{_pct(base['max_dd']*100)}</b> to "
        f"<b>{_pct(vt['max_dd']*100)}</b> and lifts the Sharpe from <b>{base['sharpe']:.2f}</b> to "
        f"<b>{vt['sharpe']:.2f}</b>, at ~<b>{vt['avg_exposure']*100:.0f}%</b> average exposure "
        "(the rest in cash) — lower absolute return, far better risk-adjusted. Same survivorship "
        "caveats apply to the underlying signal.</p>")


def sec_vs_portfolio(d: dict, public: bool) -> str:
    """Head-to-head: your real Trade Republic portfolio vs the momentum strategy over the
    same window. Private only (it's your actual book)."""
    pr = d.get("portfolio_roi")
    if public or pr is None or getattr(pr, "empty", True):
        return ""
    eq = d["res"]["runs"][1.0]["equity"]
    start = pr.index[0]
    eqw = eq[eq.index >= start].dropna()
    prw = pr[pr.index >= start].dropna()
    if len(eqw) < 5 or len(prw) < 5:
        return ""
    strat = (eqw / eqw.iloc[0] - 1.0) * 100.0          # strategy cumulative ROI % from your start
    # risk-conscious (vol-targeted) curve over the same window
    vt = d.get("quant", {}).get("vol_target") or {}
    rcw = vt.get("equity")
    rc = None
    if rcw is not None:
        rcw = rcw[rcw.index >= start].dropna()
        if len(rcw) >= 5:
            rc = (rcw / rcw.iloc[0] - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prw.index, y=prw.values, name="Your portfolio (real)",
                             line=dict(color="#ffffff", width=2.6)))
    fig.add_trace(go.Scatter(x=strat.index, y=strat.values, name="Momentum — raw",
                             line=dict(color="#dcdcaa", width=1.8)))
    if rc is not None:
        fig.add_trace(go.Scatter(x=rc.index, y=rc.values, name="Momentum — risk-conscious",
                                 line=dict(color="#4ec9b0", width=2.4)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM)
    fig.update_layout(height=440, yaxis=dict(title="Cumulative ROI (%)", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))

    def stat(roi_pct):
        m = qg.perf_metrics(1.0 + roi_pct / 100.0)
        return roi_pct.iloc[-1], m.get("sharpe", 0.0), m.get("max_dd", 0.0) * 100.0
    pt, ps, pdd = stat(prw)
    st, ss, sdd = stat(strat)
    yrs = max((prw.index[-1] - start).days / 365.25, 1e-9)
    rows = (f"<tr><td>Your portfolio</td><td class='num'>{_pct(pt)}</td>"
            f"<td class='num mono'>{ps:.2f}</td><td class='num'>{_pct(pdd)}</td></tr>"
            f"<tr><td>Momentum — raw</td><td class='num'>{_pct(st)}</td>"
            f"<td class='num mono'>{ss:.2f}</td><td class='num'>{_pct(sdd)}</td></tr>")
    if rc is not None:
        rt, rs, rdd = stat(rc)
        rows += (f"<tr><td>Momentum — risk-conscious</td><td class='num'>{_pct(rt)}</td>"
                 f"<td class='num mono'>{rs:.2f}</td><td class='num'>{_pct(rdd)}</td></tr>")
    lead = st - pt
    return (
        "<h2>You vs the strategy</h2>"
        f"<p class='dim'>Same window — since your first trade ({start.date()}, ~{yrs:.1f}y). "
        "<span style='color:#fff'>White</span> = your real book; "
        "<span style='color:#dcdcaa'>gold</span> = the raw momentum strategy; "
        "<span style='color:#4ec9b0'>teal</span> = the risk-conscious (vol-targeted) version — all "
        "hypothetical, lump-sum. Apples-to-pears (your book is cash-flow-timed), and the strategies "
        "carry every caveat below — survivorship especially — so read the gap as indicative.</p>"
        f"<div class='chart'>{fig_html(fig)}</div>"
        "<table><tr><th>Book</th><th class='num'>Total ROI</th><th class='num'>Sharpe</th>"
        f"<th class='num'>Max DD</th></tr>{rows}</table>"
        f"<p class='dim'>Over this window the raw strategy is <b>{_pct(lead)}</b> "
        f"{'ahead of' if lead >= 0 else 'behind'} your portfolio on total return — but watch the "
        "drawdown and Sharpe columns: the risk-conscious version is the fairer comparison to how "
        "you actually run money.</p>")


def sec_grade(d: dict, public: bool) -> str:
    q = d["quant"]
    p, bm, tm, rl, g = q["perf"], q["bench"], q["trades"], q["roll"], q["grade"]
    gcolor = {"A": "#46c84e", "B": "#9acd32", "C": "#d7ba7d", "D": "#e8a04e", "F": "#ef4444"}[g["letter"]]

    def row(k, v):
        return f"<tr><td>{k}</td><td class='num mono'>{v}</td></tr>"
    perf_rows = "".join([
        row("Sharpe (full, daily)", f"{p['sharpe']:.2f}"), row("Sortino", f"{p['sortino']:.2f}"),
        row("Calmar (CAGR/maxDD)", f"{p['calmar']:.2f}"), row("Omega", f"{p['omega']:.2f}"),
        row("Ann. return", _pct(p['ann_return']*100)), row("Ann. vol", _pct(p['ann_vol']*100)),
        row("Max drawdown", _pct(p['max_dd']*100)), row("Underwater (days)", f"{p['dd_days']}"),
        row("Skew / kurtosis", f"{p['skew']:.2f} / {p['kurtosis']:.2f}"),
        row("VaR 95 / CVaR 95 (daily)", f"{p['var95']*100:.1f}% / {p['cvar95']*100:.1f}%")])
    bench_rows = "".join([
        row("Beta vs S&amp;P", f"{bm['beta']:.2f}"), row("Alpha (annual)", _pct(bm['alpha_ann']*100)),
        row("Correlation", f"{bm['corr']:.2f}"), row("Information ratio", f"{bm['info_ratio']:.2f}"),
        row("Tracking error", _pct(bm['tracking_error']*100)),
        row("Up / down capture", f"{bm['up_capture']:.2f} / {bm['down_capture']:.2f}")]) if bm else ""
    trade_rows = "".join([
        row("Hit rate", _pct(tm['hit_rate']*100)), row("Profit factor", f"{tm['profit_factor']:.2f}"),
        row("Payoff (avgW/avgL)", f"{tm['payoff']:.2f}"), row("Trades / year", f"{tm['trades_per_year']:.0f}")]) if tm else ""
    roll_rows = "".join([
        row("12m Sharpe — median", f"{rl['roll_sharpe_med']:.2f}"),
        row("12m Sharpe — worst", f"{rl['roll_sharpe_min']:.2f}"),
        row("12m windows positive", _pct(rl['roll_sharpe_pos_frac']*100))]) if rl else ""

    flags = "".join(f"<li>{f}</li>" for f in g["flags"])
    score_card = _card("Score / 100", f"{g['score']:.0f}")
    sharpe_card = _card("Full Sharpe", f"{p['sharpe']:.2f}")
    dd_card = _card("Max DD", _pct(p["max_dd"] * 100))
    return (
        "<h2>Quant scorecard &amp; honest grade</h2>"
        f"<div class='cards'>"
        f"<div class='card'><div class='k'>Grade</div>"
        f"<div class='v' style='color:{gcolor};font-size:2rem'>{g['letter']}</div></div>"
        f"{score_card}{sharpe_card}{dd_card}</div>"
        "<p class='dim'>Graded like a risk committee: standard ratios, benchmark attribution, "
        "trade quality and stability — then the headline is <b>docked for what it doesn't "
        "correct</b>. The full-window daily Sharpe (<b>"
        f"{p['sharpe']:.2f}</b>) and the −{abs(p['max_dd'])*100:.0f}% max drawdown are the sober "
        "view; the eye-popping test total is regime + survivorship.</p>"
        "<div style='display:flex;flex-wrap:wrap;gap:1.5rem'>"
        f"<table><tr><th>Risk / return</th><th class='num'>Value</th></tr>{perf_rows}</table>"
        f"<table><tr><th>vs S&amp;P 500</th><th class='num'>Value</th></tr>{bench_rows}</table>"
        f"<table><tr><th>Trade quality</th><th class='num'>Value</th></tr>{trade_rows}</table>"
        f"<table><tr><th>Stability</th><th class='num'>Value</th></tr>{roll_rows}</table>"
        "</div>"
        "<div class='note warn'><b>Bias audit — why this is <i>not</i> clean alpha.</b> "
        f"Is it real? Partly. Momentum-<i>selection</i> beats a random book on the same universe "
        f"(p={d['significance']['mc']['p_sharpe']:.3f}, deflated-Sharpe "
        f"{d['significance']['dsr']['dsr']:.0%}) — a genuine, modest tilt. But the <i>level</i> is "
        f"inflated, and the honest verdict is a <b>{g['letter']}</b>:<ul>{flags}</ul>"
        "Bottom line: a real but small momentum tilt riding survivorship + a small-cap regime — "
        "a known, decaying premium, not novel alpha. If it looks too easy, it is.</div>")


def sec_significance(d: dict, public: bool) -> str:
    s = d["significance"]
    mc, dsr, ci = s["mc"], s["dsr"], s["ci"]
    pct_beat = 100.0 * (1.0 - mc["p_sharpe"])

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

    cards = [
        _card("p-value vs random", f"{mc['p_sharpe']:.3f}"),
        _card("Beats random books", f"{pct_beat:.1f}%"),
        _card("Deflated Sharpe (P real>0)", "—" if dsr['dsr'] != dsr['dsr'] else f"{dsr['dsr']:.0%}"),
        _card(f"Sharpe {ci['conf']}% CI", f"{ci['sharpe_lo']:.2f} – {ci['sharpe_hi']:.2f}"),
    ]
    verdict = ("clears" if mc["p_sharpe"] < 0.05 else "does <b>not</b> clear")
    dsr_txt = ("—" if dsr["dsr"] != dsr["dsr"] else
               f"After haircutting for the <b>{dsr['n_trials']} configs we scanned</b>, the return "
               f"skew/kurtosis and the {dsr['T']}-period sample length, the <b>Deflated Sharpe</b> "
               f"puts P(true Sharpe&gt;0) at <b>{dsr['dsr']:.0%}</b> (benchmark a lucky winner had to "
               f"clear: {dsr['sr_benchmark_annual']:.2f} annualised).")
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
            f'<div class="cards">{"".join(cards)}</div>'
            f"<div class='chart'>{fig_html(fig)}</div>"
            "<p class='dim'>A low p-value says the <i>selection</i> adds value over drawing names "
            "at random from the same liquid pool; it does not promise the level repeats. Read it "
            "with the regime and capacity caveats below.</p>")


def sec_timeline(d: dict) -> str:
    lines = []
    for h in d["res"]["holdings_log"]:
        dead = h.get("dead", set())
        spans = " ".join(
            f"<span style='color:{_pnl_color(h['ret'].get(t, 0.0), t in dead)}' "
            f"title='{t} {h['ret'].get(t, 0.0):+.0%}'>{_disp(d['meta'], t)}</span>"
            for t in h["picks"])
        spans = spans or "<span class='dim'>cash</span>"
        rv = [v for v in h["ret"].values() if pd.notna(v)]
        mret = sum(rv) / len(rv) if rv else 0.0
        lines.append(f"<div><span class='mono dim'>{h['date'].date()}</span> "
                     f"<b style='color:{_pnl_color(mret, False)}'>{mret:+.1%}</b> {spans}</div>")
    return ("<h2>Every rebalance, colored by outcome</h2>"
            "<p class='dim'>Each line is one rebalance’s picks, colored by that holding "
            "period’s return — <span style='color:#0a6b00'>■</span> ≥+20% · "
            "<span style='color:#46c84e'>■</span> up · <span style='color:#ef4444'>■</span> down · "
            "<span style='color:#7a0000'>■</span> ≤−20% · <span style='color:#000'>■</span> "
            "defaulted (delisted/died). Hover for the %.</p>"
            f"<div style='font-size:0.78rem;line-height:1.7'>{''.join(lines)}</div>")


def sec_caveat(d: dict) -> str:
    hits = d.get("graveyard_hits", 0)
    test_ret = d["test"]["net_return"] * 100
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
    return (
        f'<div class="note warn"><b>The dominant caveat — survivorship is NOT corrected.</b> '
        f'The live universe is Trade Republic’s <i>current</i> list — names that <b>survived to '
        f'today</b>. A name that pumped then delisted before now is simply absent, so the backtest '
        f'only ever picks from winners-that-made-it. The {d["n_dead"]} “graveyard” names are a '
        f'near-disjoint EODHD relic (<b>{ov*100:.0f}%</b> ISIN overlap with the live set), so they '
        f'do <b>not</b> fix it.{bound_txt} This inflates the headline and is the single biggest reason '
        f'to distrust the level — see the bias audit in the scorecard above.{trade}'
        f'<br><br>The other caveats: <b>(1) Regime</b> — 2024→ was an exceptional small-cap momentum '
        f'tape; even the held-out {test_ret:+.0f}% test figure is regime-specific and will <b>not</b> '
        f'repeat. <b>(2) Concentration</b> — top-{d["strategy"].slots}, no sector/geographic cap, so '
        f'the book can pile into one theme; a few names drive the curve. <b>(3) Capacity</b> — picks '
        f'are liquid enough for a small account, but modeled slippage (25bps) understates real fills '
        f'in size. <b>(4) Mechanics</b> — daily closes, €1/order, slippage modeled not measured, and '
        f'<b>past performance is not future returns</b>.</div>')


def build(d: dict, public: bool = False) -> str:
    """One page: the chosen strategy up top (intro → picks → equity → perf → timeline →
    caveats), then a <hr> and the research lab below (the 32-config grid, feasibility, every
    variation's timeline, the survivorship note, and the method). Local-only output."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cfg = d["strategy"]
    body = "".join([
        f"<h1>Momentum strategy — {cfg.code}</h1>",
        f"<p class='dim'>generated {now} · <a href='report.html'>← portfolio</a></p>",
        # ── the chosen strategy ──
        sec_intro(d),
        sec_holdings(d),
        sec_curve(d),
        sec_perf(d, public),
        sec_risk_conscious(d, public),
        sec_vs_portfolio(d, public),
        sec_grade(d, public),
        sec_significance(d, public),
        sec_yearly(d, public),
        sec_timeline(d),
        sec_caveat(d),
        # ── the lab (private/live only): how this config was chosen + all the rest ──
        ("".join([
            "<hr style='margin:3rem 0;border:0;border-top:2px solid #333'>",
            "<h1>Research lab</h1><p class='dim'>How the config above was chosen — the whole "
            "32-config grid it was picked from, and the supporting data. Skip unless you "
            "want the workings.</p>",
            sec_survivorship(d), sec_grid(d), sec_feasibility(d), sec_timelines(d), sec_method(),
        ]) if not public else ""),
    ])
    return page(f"Strategy — {cfg.code}", body)


# ════════════════════════════════════════════════════════════════════════════
#  STRATEGIES HUB — every strategy on one page, ordered by conviction, each at
#  full depth by reusing that strategy's own section functions (no duplication).
# ════════════════════════════════════════════════════════════════════════════

_HUB_CSS = """
.hubgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:18px 0 8px}
.hubcard{display:block;text-decoration:none;color:inherit;background:#252526;border:1px solid #333;
 border-radius:10px;padding:16px 18px;position:relative;overflow:hidden;transition:border-color .15s}
.hubcard:hover{border-color:#569cd6}
.hubcard::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--bar,#569cd6)}
.hubcard .hn{font-size:15px;font-weight:700;letter-spacing:-.01em;margin-bottom:2px;color:#e8e8e8}
.hubcard .ht{font-size:12.5px;color:#9a9a9a;line-height:1.45;margin:4px 0 12px;min-height:2.6em}
.hubcard .hs{display:flex;gap:16px;font-size:12px;color:#b5b5b5;flex-wrap:wrap}
.hubcard .hs b{color:#e8e8e8;font-variant-numeric:tabular-nums}
.hbadge{display:inline-block;font:600 10px ui-monospace,Menlo,monospace;letter-spacing:.06em;
 padding:3px 8px;border-radius:20px;color:#0b0b0b;margin-bottom:8px}
.stratband{margin:3rem 0 0;padding:14px 0 2px;border-top:2px solid #333}
.stratband h1{margin:0;font-size:1.7rem}
.stratband .role{font:600 11px ui-monospace,Menlo,monospace;letter-spacing:.12em;text-transform:uppercase;color:#808080}
.hubstub{background:#2a2a2b;border:1px solid #3a3a3a;border-radius:8px;padding:14px 16px;color:#b5b5b5;font-size:13.5px}
"""

_BADGE = dict(good="#46c84e", warn="#d7ba7d", crit="#ef4444", neutral="#569cd6", teal="#4ec9b0")


def _badge(text, kind="neutral"):
    return f'<span class="hbadge" style="background:{_BADGE[kind]}">{text}</span>'


def _g(dct, *path, default=None):
    """Defensive nested get: _g(d,'a','b') → d['a']['b'] or default."""
    cur = dct
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


_PAIRS_PKL = ROOT / "local" / "buffer" / "hub_pairs.pkl"


def _cached_pairs(populate: bool = False):
    """Pairs runs thousands of Engle-Granger cointegration tests (10+ min) — far too slow
    to ever run inside a page build. So the hub NEVER scans inline: it reads a pickle if
    one exists (best-effort, any age), else returns None and the section shows a stub.
    The cache is populated out-of-band by `python build_strategy_report.py --pairs`
    (background it once); its result moves slowly, so a stale-ish cache is fine."""
    import pickle
    if not populate:
        try:
            with open(_PAIRS_PKL, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None                                  # no cache → stub, no scan
    d = P.gather(force=True)                             # the slow scan, only on --pairs
    try:
        _PAIRS_PKL.parent.mkdir(parents=True, exist_ok=True)
        with open(_PAIRS_PKL, "wb") as f:
            pickle.dump(d, f)
    except Exception:
        pass
    return d


def gather_all(force: bool = False, refresh: bool | None = None) -> dict:
    """Gather every strategy's data. Each is independent and guarded — one failing
    (e.g. pairs needs a live fetch) never sinks the page; its card shows the reason.
    Pairs is served from an 18h pickle cache (its scan is minutes)."""
    refresh = force if refresh is None else refresh
    out: dict = {}
    for key, fn in (("mom", lambda: gather(force=force, refresh=refresh)),
                    ("vol", lambda: V.gather(force=force, refresh=refresh)),
                    ("edge", lambda: E.gather(force=force, refresh=refresh)),
                    ("pairs", lambda: _cached_pairs())):    # cache-only, never scans inline
        try:
            out[key] = fn()
        except Exception as exc:                       # noqa: BLE001 — report, don't crash
            out[key] = None
            out[f"{key}_err"] = f"{type(exc).__name__}: {exc}"
            print(f"  [hub] {key} unavailable — {out[f'{key}_err']}")
    if out.get("pairs") is None and "pairs_err" not in out:
        out["pairs_err"] = ("cache empty — populate once with "
                            "<span class='mono'>python build_strategy_report.py --pairs</span> "
                            "(~10-min cointegration scan), or open the <a href='/pairs' "
                            "target='_top'>dedicated /pairs page</a>")
    return out


def _hub_cards(d: dict) -> str:
    mom, vol, edge, pairs = d.get("mom"), d.get("vol"), d.get("edge"), d.get("pairs")
    cards = []

    # 1 — momentum core (return engine)
    if mom:
        test = mom.get("test", {})
        full = _g(mom, "res", "runs", 1.0, "stats", default={})
        grade = _g(mom, "quant", "grade", "letter", default="—")
        cards.append(_card_html("momentum", "Momentum core", "teal",
            _badge("RETURN ENGINE", "teal"),
            "12-1 cross-sectional on small/illiquid names funds can't crowd — the capacity edge.",
            [("Test Sharpe", f"{test.get('sharpe', 0):.2f}"),
             ("Full return", _pct(full.get("net_return", 0) * 100)),
             ("Grade", grade)]))
    else:
        cards.append(_stub_card("momentum", "Momentum core", d.get("mom_err")))

    # 2 — vol overlay (ADOPT)
    if vol:
        v = _g(vol, "verdicts", "etf", "verdict", default="—")
        garch = _g(vol, "etf", "variants", "garch", default={})
        bh = _g(vol, "etf", "bh", default={})
        kind = "good" if str(v).startswith("ADOPT") else "warn"
        cards.append(_card_html("vol", "Volatility overlay", kind, _badge(v.upper(), kind),
            "Forecast tomorrow's risk (GARCH/learned), size exposure to it. De-risk only, no leverage.",
            [("ETF Sharpe", f"{bh.get('sharpe', 0):.2f}→{garch.get('sharpe', 0):.2f}"),
             ("Max DD", f"{bh.get('max_dd', 0) * 100:.0f}%→{garch.get('max_dd', 0) * 100:.0f}%"),
             ("Avg exp", f"{garch.get('avg_exposure', 1) * 100:.0f}%")]))
    else:
        cards.append(_stub_card("vol", "Volatility overlay", d.get("vol_err")))

    # 3 — structural stack (composite)
    ov = _g(edge, "stack", "overlay", default={}) if edge else {}
    if ov:
        cards.append(_card_html("edge", "Structural stack", "neutral", _badge("COMPOSITE", "neutral"),
            "Capacity + horizon: momentum core, vol-managed to target vol. The book you'd actually run.",
            [("Ann. return", _pct(ov.get("ann_return", 0) * 100)),
             ("Sharpe", f"{ov.get('sharpe', 0):.2f}"),
             ("Max DD", f"{ov.get('max_dd', 0) * 100:.0f}%")]))
    else:
        cards.append(_stub_card("edge", "Structural stack", d.get("edge_err") or "needs momentum core"))

    # 4 — pairs (market-neutral)
    if pairs:
        st = _g(pairs, "bt", "runs", 1.0, "stats", default={})
        cards.append(_card_html("pairs", "Pairs trading", "neutral", _badge("MARKET-NEUTRAL", "neutral"),
            "Cointegration spreads, walk-forward, α/β/μ/σ frozen on the formation window.",
            [("Live pairs", str(len(pairs.get("live", [])))),
             ("BT Sharpe", f"{st.get('sharpe', 0):.2f}"),
             ("BT return", _pct(st.get("net_return", 0) * 100))]))
    else:
        cards.append(_stub_card("pairs", "Pairs trading", d.get("pairs_err")))

    # 5 — tax-loss sleeve (verdict last, whatever it is)
    if edge:
        sv = _g(edge, "sleeve_verdict", "verdict", default="—")
        mc = _g(edge, "mc", default={})
        kind = "good" if sv == "KEEP" else "crit"
        cards.append(_card_html("taxloss", "Tax-loss sleeve", kind, _badge(sv, kind),
            "Forced-flow seasonal: buy December's crushed losers, sell in January. Judged by a random-selection null.",
            [("Verdict", sv),
             ("MC p-value", f"{mc.get('p_sharpe', float('nan')):.2f}"),
             ("vs random", f"{mc.get('strat_total', 0) * 100:+.0f}% / {mc.get('null_total_median', 0) * 100:+.0f}%")]))
    else:
        cards.append(_stub_card("taxloss", "Tax-loss sleeve", d.get("edge_err")))

    return ('<div class="note">Every strategy in the book, ordered by weight of evidence — '
            'return engines first, the automatically-cut sleeve last. Each card jumps to its '
            'full section.</div>'
            f'<div class="hubgrid">{"".join(cards)}</div>')


def _card_html(anchor, name, kind, badge, thesis, stats):
    bar = _BADGE[kind]
    sh = "".join(f'<div><span class="dim">{k}</span> <b>{v}</b></div>' for k, v in stats)
    return (f'<a href="#{anchor}" class="hubcard" style="--bar:{bar}">{badge}'
            f'<div class="hn">{name}</div><div class="ht">{thesis}</div>'
            f'<div class="hs">{sh}</div></a>')


def _stub_card(anchor, name, err):
    return (f'<a href="#{anchor}" class="hubcard" style="--bar:#666">{_badge("UNAVAILABLE", "warn")}'
            f'<div class="hn">{name}</div><div class="ht">Not built this run — {err or "no data"}.</div>'
            f'<div class="hs dim">see section below</div></a>')


def _band(anchor, title, role):
    return f'<div class="stratband" id="{anchor}"><div class="role">{role}</div><h1>{title}</h1></div>'


def _safe(fn, *args, label=""):
    """Render a section, converting any failure into an inline note instead of a 500."""
    try:
        return fn(*args)
    except Exception as exc:                            # noqa: BLE001
        return f'<div class="hubstub">Section {label or fn.__name__} failed: {type(exc).__name__}: {exc}</div>'


def build_all(d: dict, public: bool = False) -> str:
    """The strategies hub: index cards, then every strategy full-depth in conviction order."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mom, vol, edge, pairs = d.get("mom"), d.get("vol"), d.get("edge"), d.get("pairs")
    parts = [f"<style>{_HUB_CSS}</style>",
             "<h1>Strategies</h1>",
             f"<p class='dim'>generated {now} · <a href='report.html'>← portfolio</a> · "
             "all strategies, ordered by conviction · <b>not advice</b></p>",
             _hub_cards(d)]

    # 1 — Momentum core (flagship detail)
    if mom:
        parts += [_band("momentum", f"Momentum core — {mom['strategy'].code}", "1 · return engine · capacity edge"),
                  _safe(sec_intro, mom), _safe(sec_holdings, mom), _safe(sec_curve, mom),
                  _safe(sec_perf, mom, public), _safe(sec_risk_conscious, mom, public),
                  _safe(sec_vs_portfolio, mom, public), _safe(sec_grade, mom, public),
                  _safe(sec_significance, mom, public), _safe(sec_yearly, mom, public),
                  _safe(sec_timeline, mom), _safe(sec_caveat, mom)]
    else:
        parts += [_band("momentum", "Momentum core", "1 · return engine"),
                  f'<div class="hubstub">Unavailable — {d.get("mom_err")}</div>']

    # 2 — Volatility overlay (full artifact depth, reusing the vol page's own sections)
    if vol:
        parts += [_band("vol", "Volatility overlay", "2 · risk control · forecast the variance, not the mean"),
                  _safe(V.sec_thesis, vol), _safe(V.sec_verdict, vol), _safe(V.sec_today, vol),
                  _safe(V.sec_forecast_eval, vol), _safe(V.sec_mean_null, vol), _safe(V.sec_etf, vol),
                  _safe(V.sec_momentum, vol), _safe(V.sec_significance, vol), _safe(V.sec_costs, vol),
                  _safe(V.sec_method, vol)]
    else:
        parts += [_band("vol", "Volatility overlay", "2 · risk control"),
                  f'<div class="hubstub">Unavailable — {d.get("vol_err")}</div>']

    # 3 — Structural stack (the composite you'd run)
    if edge:
        parts += [_band("edge", "Structural stack", "3 · composite · capacity + horizon, vol-managed"),
                  _safe(E.sec_edges, edge), _safe(E.sec_stack, edge)]
    else:
        parts += [_band("edge", "Structural stack", "3 · composite"),
                  f'<div class="hubstub">Unavailable — {d.get("edge_err")}</div>']

    # 4 — Pairs (market-neutral)
    if pairs:
        parts += [_band("pairs", "Pairs trading", "4 · market-neutral · statistical arbitrage"),
                  _safe(P.sec_intro), _safe(P.sec_snapshot, pairs), _safe(P.sec_pair_charts, pairs),
                  _safe(P.sec_backtest, pairs, public), _safe(P.sec_costs, pairs), _safe(P.sec_method)]
    else:
        parts += [_band("pairs", "Pairs trading", "4 · market-neutral"),
                  f'<div class="hubstub">Unavailable — {d.get("pairs_err")}</div>']

    # 5 — Tax-loss sleeve (the verdict — last)
    if edge:
        parts += [_band("taxloss", "Tax-loss sleeve", "5 · forced-flow seasonal · pre-registered verdict"),
                  _safe(E.sec_verdict, edge), _safe(E.sec_today, edge), _safe(E.sec_seasonal, edge),
                  _safe(E.sec_method, edge)]
    else:
        parts += [_band("taxloss", "Tax-loss sleeve", "5 · forced-flow seasonal"),
                  f'<div class="hubstub">Unavailable — {d.get("edge_err")}</div>']

    return page("Strategies", "".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--solo", action="store_true", help="momentum-only page (the old /strategy)")
    ap.add_argument("--pairs", action="store_true",
                    help="populate the pairs cache (the slow one-time cointegration scan)")
    args = ap.parse_args()
    local = ROOT / "local/strategy.html"
    local.parent.mkdir(exist_ok=True)
    if args.pairs:
        d = _cached_pairs(populate=True)
        print(f"pairs cache written: {len(d['live'])} live pairs, {d['n_tested']} tested")
        return
    if args.solo:
        local.write_text(build(gather(refresh=args.refresh)))
        print(f"wrote {local}  (momentum solo {STRATEGY.code})")
    else:
        d = gather_all(refresh=args.refresh)
        local.write_text(build_all(d))                  # live/local only — no docs/ export
        built = [k for k in ("mom", "vol", "edge", "pairs") if d.get(k)]
        print(f"wrote {local}  (strategies hub: {', '.join(built)})")
    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
