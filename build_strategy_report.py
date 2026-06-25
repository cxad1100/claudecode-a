"""The chosen production strategy — one config (the 'ultimate' extracted from the
32-config grid), on the survivorship-corrected universe, shown two ways:

  • Original     — the raw config, full-invested.
  • Risk-conscious — the SAME selection, volatility-targeted (de-risk only).

The two are presented side by side (picks, equity, performance, scorecard, yearly
and every rebalance) as a head-to-head: the risk-conscious version is a valid
alternative and gets the full analysis the original gets.

  python build_strategy_report.py            # writes local/strategy.html
  python build_strategy_report.py --open

Unlike the momentum *lab* (which renders the whole grid), this page commits to a
single config + its risk-conscious variation. The research lab (how the config was
chosen) is appended on private/live builds only.
"""
import argparse
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tools.report_html import pct as _pct, card as _card, page, fig_html
from tools import theme, significance as sig, quant_grade as qg
from tools.momentum import (run_momentum, winsorize_prices, to_xetra_calendar,
                            precompute_eligibility, benchmark_curves, equal_weight_curve)
from tools.universe_pit import PITUniverse
from tools.universe_assemble import delisting_map
from tools.momentum_grid import MomentumConfig, _stats_slice, run_grid, ALL_CONFIGS
from tools.portfolio_tools import BENCHMARKS, parse_portfolio
from tools.portfolio_analytics import build_roi_timeseries
from tools.data_buffer import cached_price_history
from build_momentum_report import (
    PRICES_CSV, META_CSV, ROOT, LOOKBACK, SKIP, START, LIQ_MAX, MIN_PRICE, CAPITAL,
    FEE_EUR, COST_MULTS, TRAIN_END, VAL_END, WINSOR_CAP, EXEC_LAG, K,
    _slip, _broker, _disp, _name, _pnl_color, _equity_window,
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

# Head-to-head colors — gold = original (raw), teal = risk-conscious (vol-targeted).
C_RAW = "#dcdcaa"
C_RC = "#4ec9b0"
_GRADE_COLOR = {"A": "#46c84e", "B": "#9acd32", "C": "#d7ba7d", "D": "#e8a04e", "F": "#ef4444"}


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


def build_variants(res: dict, vt: dict, spx, train: dict, val: dict, test: dict,
                   quant: dict, capital: float, *, dsr: float, mc_p: float, overlap: float,
                   train_end=TRAIN_END, val_end=VAL_END) -> list[dict]:
    """Two parallel 'variant bundles' of identical shape so every section can render
    either: the raw strategy and its volatility-targeted (risk-conscious) twin.

    Selection is IDENTICAL across both (same holdings_log / trades) — vol-targeting
    scales the whole book, it does not change which names are picked. So the rc bundle
    re-derives its windowed stats / quant metrics / grade from the vol-targeted equity
    curve, but shares the picks and the per-name trade record."""
    tr = res["runs"][1.0]["trades"]
    raw = dict(
        label=f"Original (raw, {STRATEGY.code})", short="Original", key="raw", color=C_RAW,
        equity=res["runs"][1.0]["equity"], holdings_log=res["holdings_log"], trades=tr,
        train=train, val=val, test=test, full=res["runs"][1.0]["stats"],
        perf=quant["perf"], bench=quant["bench"], roll=quant["roll"], grade=quant["grade"],
        exposure=None, exposure_latest=1.0)

    rc_eq = vt["equity"]
    te, ve = pd.Timestamp(train_end), pd.Timestamp(val_end)
    one = pd.Timedelta(days=1)
    rc_test = _stats_slice(rc_eq, tr, ve + one, rc_eq.index[-1], capital)
    rc = dict(
        label=f"Risk-conscious (vol-target {RISK_TARGET_VOL:.0%})", short="Risk-conscious",
        key="rc", color=C_RC,
        equity=rc_eq, holdings_log=res["holdings_log"], trades=tr,
        train=_stats_slice(rc_eq, tr, rc_eq.index[0], te, capital),
        val=_stats_slice(rc_eq, tr, te + one, ve, capital),
        test=rc_test,
        full=_stats_slice(rc_eq, tr, rc_eq.index[0], rc_eq.index[-1], capital),
        perf=vt,
        bench=qg.vs_benchmark(rc_eq, spx) if spx is not None else {},
        roll=qg.rolling_sharpe(rc_eq),
        grade=qg.grade(rc_test["sharpe"], dsr, mc_p, overlap),
        exposure=vt.get("exposure"),
        exposure_latest=vt.get("exposure_latest", vt.get("avg_exposure", 1.0)))
    return [raw, rc]


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

    # ── The two variant bundles (original + risk-conscious), full parity ──
    variants = build_variants(res, quant["vol_target"], spx, train, val, test, quant, CAPITAL,
                              dsr=dsr["dsr"], mc_p=mc["p_sharpe"], overlap=overlap)

    # ── Your real portfolio's ROI (cumulative %) + a same-scale strategy run, for the head-to-head ──
    portfolio_roi, vs_scale = None, None
    pf_csv = ROOT / "input" / "portfolio.csv"
    if pf_csv.exists():
        try:
            txns = parse_portfolio(pf_csv)["transactions"]
            pr, _ = build_roi_timeseries(txns)
            if pr is not None and not pr.empty:
                portfolio_roi = pr
            invested = sum(float(t["price"]) for t in txns if t["action"] == "buy")
            if invested > 0:
                # Re-run the SAME strategy at YOUR invested scale, so the flat €1/order fee is the
                # same % drag it is on your real (small) book — on €10k it's a negligible ~0.15%.
                vsr = run_momentum(prices, slip, lookback=LOOKBACK, skip=SKIP, capital=invested,
                                   cost_mults=(1.0,), start=START, liq_max=LIQ_MAX, fee_eur=FEE_EUR,
                                   min_price=MIN_PRICE, sectors=None, benchmark=spx, pit=pit,
                                   execute_lag=EXEC_LAG, **STRATEGY.kwargs())
                vse = vsr["runs"][1.0]["equity"]
                vs_scale = dict(capital=invested, raw_equity=vse,
                                rc_equity=qg.vol_target(vse, target_vol=RISK_TARGET_VOL).get("equity"))
        except Exception:
            portfolio_roi = None

    n_countries = len({m.get("country") for m in meta.values()} - {"—", None})
    return dict(prices=prices, res=res, benchmarks=bench, capital=CAPITAL, meta=meta, quant=quant,
                portfolio_roi=portfolio_roi, vs_scale=vs_scale, variants=variants,
                strategy=STRATEGY, train=train, val=val, test=test, graveyard_hits=hits,
                grid=grid, n_dead=int(meta_df["delisting_date"].notna().sum()),
                n_countries=n_countries,
                n_live=n_live, bounds=bounds,
                significance=dict(mc=mc, dsr=dsr, ci=ci, ppy=ppy))


# ── Shared layout helper ────────────────────────────────────────────────────────

def _cmp(left: str, right: str) -> str:
    """Two-column side-by-side block (Original | Risk-conscious); collapses on mobile."""
    return f"<div class='cmp'><div>{left}</div><div>{right}</div></div>"


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


def sec_summary(d: dict) -> str:
    """Both versions summarised on top: equal-height intro panels + one shared headline
    compare table, so the key numbers sit at the same level (symmetric, not nested cards)."""
    raw, rc = d["variants"]

    def panel(v, blurb):
        return (f"<div class='note' style='border-left-color:{v['color']}'>"
                f"<b style='color:{v['color']}'>{v['label']}.</b> {blurb}</div>")

    def grade_cell(v):
        g = v["grade"]["letter"]
        return f"<b style='color:{_GRADE_COLOR[g]};font-size:1.05rem'>{g}</b>"

    metrics = [
        ("Test return", lambda v: _pct(v["test"]["net_return"] * 100)),
        ("Test Sharpe", lambda v: f"{v['test']['sharpe']:.2f}"),
        ("Max drawdown", lambda v: _pct(v["perf"]["max_dd"] * 100)),
        ("Ann. vol", lambda v: _pct(v["perf"]["ann_vol"] * 100, signed=False)),
        ("Grade", grade_cell),
    ]

    def mtable(v):
        body = "".join(f"<tr><td>{k}</td><td class='num mono'>{fn(v)}</td></tr>" for k, fn in metrics)
        return (f"<table><tr><th>Headline (test / full)</th>"
                f"<th class='num' style='color:{v['color']}'>{v['short']}</th></tr>{body}</table>")

    raw_blurb = ("The chosen config — buys the 12-1 winners and holds them full-invested. The "
                 f"highest raw return, but the −{abs(raw['perf']['max_dd']) * 100:.0f}% drawdown and "
                 f"~{raw['perf']['ann_vol'] * 100:.0f}% vol are the price.")
    rc_blurb = (f"The same picks, volatility-targeted to {RISK_TARGET_VOL:.0%}: it scales exposure "
                "toward that vol using the prior day’s realised vol (de-risk only, no leverage) and "
                "parks the rest in cash, so the book shrinks in turbulence — how you’d actually run it.")
    return ("<h2>Two ways to run it — side by side</h2>"
            "<p class='dim'>This page compares the <b>original</b> strategy (raw, full-invested) "
            "against a <b>risk-conscious</b> variation (same selection, volatility-targeted) as a "
            "valid alternative — picks, equity, performance, scorecard, yearly P&amp;L and every "
            "rebalance, head to head. The headline:</p>"
            f"<div class='cmp stretch'>{panel(raw, raw_blurb)}{panel(rc, rc_blurb)}</div>"
            + _cmp(mtable(raw), mtable(rc)))


# ── Picks (side by side) ────────────────────────────────────────────────────────

def _picks_table(d: dict, v: dict) -> str:
    log = v["holdings_log"]
    cur = next((h for h in reversed(log) if h["picks"]), None)
    head = f"<h3 style='color:{v['color']}'>{v['short']}</h3>"
    if cur is None:
        return head + "<p class='dim'>No eligible names at the latest rebalance.</p>"
    picks = cur["picks"]
    n = len(picks)
    exp = v["exposure_latest"] if v["exposure_latest"] is not None else 1.0
    invested = 100.0 * exp
    w = invested / n if n else 0.0
    rows = []
    for t in picks:
        m = d["meta"].get(t, {})
        home = str(m.get("home") or t).split(".")[0]
        isin = m.get("isin") if pd.notna(m.get("isin")) else ""
        sc = cur["scores"].get(t, float("nan"))
        rows.append(
            f"<tr><td class='mono'>{home}</td><td>{_name(m, t)}</td>"
            f"<td class='dim mono' style='font-size:0.72rem'>{isin}</td>"
            f"<td>{m.get('country', '—')}</td>"
            f"<td class='num mono'>{sc * 100:+.1f}%</td>"
            f"<td class='num mono'>{w:.1f}%</td></tr>")
    cash = 100.0 - invested
    if v["key"] == "rc" and cash > 0.05:
        rows.append(
            f"<tr><td class='mono dim'>CASH</td><td class='dim'>de-risked sleeve</td>"
            f"<td></td><td></td><td class='num dim'>—</td>"
            f"<td class='num mono'>{cash:.1f}%</td></tr>")
    if v["key"] == "rc":
        sub = (f"<p class='dim pick-sub'>top-{n} → <b>{invested:.0f}% invested</b> "
               f"({cash:.0f}% cash) at current vol · {cur['date'].date()}</p>")
    else:
        sub = f"<p class='dim pick-sub'>top-{n} equal-weight, full-invested · {cur['date'].date()}</p>"
    return (head + sub +
            "<table><tr><th>Ticker</th><th>Name</th><th>ISIN</th><th>Country</th>"
            "<th class='num'>12-1 mom</th><th class='num'>Weight</th></tr>" + "".join(rows) + "</table>")


def sec_picks_compare(d: dict) -> str:
    raw, rc = d["variants"]
    cur = next((h for h in reversed(raw["holdings_log"]) if h["picks"]), None)
    n = len(cur["picks"]) if cur else 0
    return ("<h2>Current top picks</h2>"
            f"<p class='dim'>Identical selection — both versions hold the same equal-weight top-{n} "
            "ranked by 12-1 momentum. The risk-conscious version simply scales the whole book toward "
            "its vol target, parking the remainder in cash; the names are the same. Each leg shows "
            "its home ticker, name and ISIN — search the ISIN or name in Trade Republic to trade "
            "it.</p>"
            + _cmp(_picks_table(d, raw), _picks_table(d, rc)))


# ── Equity (one overlaid chart) ─────────────────────────────────────────────────

def sec_curve_compare(d: dict) -> str:
    res = d["res"]
    window = _equity_window(res)
    fig = go.Figure()
    for v in d["variants"]:
        eq = v["equity"].reindex(window).ffill()
        fig.add_trace(go.Scatter(x=eq.index, y=eq / d["capital"] * 100.0, name=v["short"],
                                 line=dict(color=v["color"], width=2.4)))
    first_picks = next((h["picks"] for h in res["holdings_log"] if h["picks"]), [])
    if first_picks:
        ew = equal_weight_curve(d["prices"], first_picks, window, d["capital"])
        fig.add_trace(go.Scatter(x=ew.index, y=ew / d["capital"] * 100.0,
                                 name="Equal-weight (initial picks, buy-hold)",
                                 line=dict(color=theme.FG_DIM, width=1.4, dash="dot")))
    for name, curve in benchmark_curves(d["benchmarks"], window, d["capital"]).items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve / d["capital"] * 100.0,
                                 name=name, line=dict(width=1.4)))
    fig.add_hline(y=100, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=480, yaxis_title="Index (start = 100)",
                      hovermode="x unified", margin=dict(t=20))
    return ("<h2>Walk-forward equity vs benchmarks</h2>"
            "<p class='dim'>Both strategies since the first rebalance with enough history, vs a "
            "buy-hold equal-weight basket of today's top picks (survivorship-honest baseline) and "
            "the MSCI World / S&amp;P 500. "
            f"<span style='color:{C_RAW}'>Gold</span> = original (raw), "
            f"<span style='color:{C_RC}'>teal</span> = risk-conscious (vol-targeted). The teal line "
            "rides lower in calm rallies (it holds cash) but falls far less in the drawdowns.</p>"
            f"<div class='chart'>{fig_html(fig)}</div>")


# ── Performance (merged compare table) ──────────────────────────────────────────

def _perf_cells(s: dict) -> str:
    return (f"<td class='num'>{_pct(s['net_return'] * 100)}</td>"
            f"<td class='num mono'>{s['sharpe']:.2f}</td>"
            f"<td class='num'>{_pct(s['max_drawdown'] * 100)}</td>")


def _perf_table(v: dict) -> str:
    rows = "".join(
        f"<tr><td>{label}</td>{_perf_cells(v[k])}</tr>"
        for label, k in (("Train 2018–21", "train"), ("Validation 2022–23", "val"),
                         ("Test 2024→", "test"), ("Full 2018→", "full")))
    return (f"<h3 style='color:{v['color']}'>{v['short']}</h3>"
            "<table><tr><th>Window</th><th class='num'>Ret</th><th class='num'>Sharpe</th>"
            f"<th class='num'>Max DD</th></tr>{rows}</table>")


def sec_perf_compare(d: dict, public: bool) -> str:
    raw, rc = d["variants"]
    cards = [
        _card("Test return — orig", _pct(raw["test"]["net_return"] * 100)),
        _card("Test return — risk-con", _pct(rc["test"]["net_return"] * 100)),
        _card("Max DD — orig", _pct(raw["perf"]["max_dd"] * 100)),
        _card("Max DD — risk-con", _pct(rc["perf"]["max_dd"] * 100)),
    ]
    if not public:
        cards.append(_card("Net P&L — orig", f"€{raw['full']['net_return'] * d['capital']:+,.0f}"))
        cards.append(_card("Net P&L — risk-con", f"€{rc['full']['net_return'] * d['capital']:+,.0f}"))
    return ("<h2>Performance</h2>"
            "<p class='dim'>Train = 2018–21 (used to pick the config), validation = 2022–23 "
            "(used to compare configs), <b>test = 2024→ (held out — never touched the choice)</b>. "
            "The test column is the only truly out-of-sample number; trust it over the eye-popping "
            "full-window total. Both versions share the same selection — the risk-conscious curve is "
            f"that selection scaled to a {RISK_TARGET_VOL:.0%} vol target.</p>"
            f"<div class='cards'>{''.join(cards)}</div>"
            + _cmp(_perf_table(raw), _perf_table(rc)))


# ── Quant scorecard & grade (merged compare) ────────────────────────────────────

def _grade_card(v: dict) -> str:
    g = v["grade"]
    color = _GRADE_COLOR[g["letter"]]
    return (f"<div class='card'><div class='k'>{v['short']} grade</div>"
            f"<div class='v' style='color:{color};font-size:2rem'>{g['letter']}</div></div>")


def _scorecard_table(v: dict, tm: dict) -> str:
    """One variant's full scorecard — [Metric | value] with group sub-headers. Two of these
    sit side by side (Original | Risk-conscious), same left/right zones as every other section."""
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


def sec_grade_compare(d: dict, public: bool) -> str:
    raw, rc = d["variants"]
    tm = d["quant"]["trades"]                       # trade quality = selection (identical for both)
    g = raw["grade"]
    score_cards = "".join([
        _grade_card(raw), _grade_card(rc),
        _card("Score — orig", f"{raw['grade']['score']:.0f}"),
        _card("Score — risk-con", f"{rc['grade']['score']:.0f}"),
    ])
    flags = "".join(f"<li>{f}</li>" for f in g["flags"])
    return (
        "<h2>Quant scorecard &amp; honest grade</h2>"
        f"<div class='cards'>{score_cards}</div>"
        "<p class='dim'>Graded like a risk committee: standard ratios, benchmark attribution, "
        "trade quality and stability — then the headline is <b>docked for what it doesn't "
        "correct</b>. The risk-conscious version earns the same (or better) grade by trading some "
        "raw return for a much smaller drawdown and a higher risk-adjusted Sharpe; the bias "
        "deductions below apply to the shared selection, so they hit both equally. (Trade quality is "
        "selection-driven, so it is identical in both columns.)</p>"
        + _cmp(_scorecard_table(raw, tm), _scorecard_table(rc, tm))
        + "<div class='note warn'><b>Bias audit — why this is <i>not</i> clean alpha.</b> "
        f"Is it real? Partly. Momentum-<i>selection</i> beats a random book on the same universe "
        f"(p={d['significance']['mc']['p_sharpe']:.3f}, deflated-Sharpe "
        f"{d['significance']['dsr']['dsr']:.0%}) — a genuine, modest tilt. But the <i>level</i> is "
        f"inflated, and the honest verdict is a <b>{raw['grade']['letter']}</b> (original) / "
        f"<b>{rc['grade']['letter']}</b> (risk-conscious):<ul>{flags}</ul>"
        "Bottom line: a real but small momentum tilt riding survivorship + a small-cap regime — "
        "a known, decaying premium, not novel alpha. Vol-targeting improves how you <i>hold</i> it, "
        "not what it <i>is</i>. If it looks too easy, it is.</div>")


# ── Yearly P&L (merged compare table) ───────────────────────────────────────────

def _yearly_returns(series: pd.Series) -> pd.Series:
    """Calendar-year returns keyed by year int: each year from the prior year's last
    close to this year's last close; the first year runs from inception."""
    s = series.dropna()
    last = s.groupby(s.index.year).last()
    prev = last.shift(1)
    if len(prev):
        prev.iloc[0] = s.iloc[0]          # first year measured from inception
    return (last / prev - 1.0).dropna()


def _yearly_pnl(series: pd.Series) -> pd.Series:
    eq = series.dropna()
    last = eq.groupby(eq.index.year).last()
    prev = last.shift(1)
    if len(prev):
        prev.iloc[0] = eq.iloc[0]
    return (last - prev).dropna()


def sec_yearly_compare(d: dict, public: bool) -> str:
    raw, rc = d["variants"]
    req, ceq = raw["equity"].dropna(), rc["equity"].dropna()
    if len(req) < 2:
        return ""
    spx = d["benchmarks"]["S&P 500"] if "S&P 500" in d["benchmarks"].columns else None
    bret = _yearly_returns(spx.reindex(req.index).ffill()) if spx is not None else pd.Series(dtype=float)

    def ytable(v, eq):
        sret, pnl = _yearly_returns(eq), _yearly_pnl(eq)
        rows = []
        for y in sorted(sret.index):
            b = (f"<td class='num'>{_pct(bret[y] * 100)}</td>" if y in bret.index
                 else "<td class='num dim'>—</td>")
            eur = f"<td class='num mono'>€{pnl.get(y, 0.0):+,.0f}</td>" if not public else ""
            rows.append(f"<tr><td class='mono'>{y}</td>"
                        f"<td class='num'>{_pct(sret[y] * 100)}</td>{b}{eur}</tr>")
        eur_h = "<th class='num'>P&amp;L</th>" if not public else ""
        return (f"<h3 style='color:{v['color']}'>{v['short']}</h3>"
                "<table><tr><th>Year</th><th class='num'>Return</th>"
                f"<th class='num'>S&amp;P</th>{eur_h}</tr>" + "".join(rows) + "</table>")

    pnl_note = ("each version's actual €P&amp;L, and " if not public else "and ")
    return ("<h2>Yearly P&amp;L</h2>"
            "<p class='dim'>Calendar-year net return of each version (first year from inception), "
            f"{pnl_note}the S&amp;P 500 over the same year. 2018 and 2026 are part-years.</p>"
            + _cmp(ytable(raw, req), ytable(rc, ceq)))


# ── Every rebalance, colored (two columns) ──────────────────────────────────────

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


def sec_timeline_compare(d: dict) -> str:
    raw, rc = d["variants"]
    return ("<h2>Every rebalance, colored by outcome</h2>"
            "<p class='dim'>Each line is one rebalance’s picks (identical for both versions), colored "
            "by that holding period’s return — <span style='color:#0a6b00'>■</span> ≥+20% · "
            "<span style='color:#46c84e'>■</span> up · <span style='color:#ef4444'>■</span> down · "
            "<span style='color:#7a0000'>■</span> ≤−20% · <span style='color:#000'>■</span> "
            "defaulted (delisted/died). Hover for the %. The risk-conscious column shows each period’s "
            "book return after vol-scaling and its average exposure (<span class='mono'>@x%</span> "
            "invested).</p>"
            + _cmp(_timeline_col(d, raw), _timeline_col(d, rc)))


# ── Shared sections (both versions) ─────────────────────────────────────────────

def sec_vs_portfolio(d: dict, public: bool) -> str:
    """Head-to-head: your real Trade Republic portfolio vs the momentum strategy over the
    same window. Private only (it's your actual book)."""
    pr = d.get("portfolio_roi")
    if public or pr is None or getattr(pr, "empty", True):
        return ""
    # Use the strategy run sized to YOUR invested capital (flat €1/order then hits the same %
    # it does on your book); fall back to the €10k run if the portfolio scale is unavailable.
    vs = d.get("vs_scale") or {}
    eq = vs.get("raw_equity")
    if eq is None:
        eq = d["res"]["runs"][1.0]["equity"]
    start = pr.index[0]
    eqw = eq[eq.index >= start].dropna()
    prw = pr[pr.index >= start].dropna()
    if len(eqw) < 5 or len(prw) < 5:
        return ""
    strat = (eqw / eqw.iloc[0] - 1.0) * 100.0          # strategy cumulative ROI % from your start
    # risk-conscious (vol-targeted) curve over the same window, same scale
    rcw = vs.get("rc_equity")
    if rcw is None:
        rcw = (d.get("quant", {}).get("vol_target") or {}).get("equity")
    rc = None
    if rcw is not None:
        rcw = rcw[rcw.index >= start].dropna()
        if len(rcw) >= 5:
            rc = (rcw / rcw.iloc[0] - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prw.index, y=prw.values, name="Your portfolio (real)",
                             line=dict(color="#ffffff", width=2.6)))
    fig.add_trace(go.Scatter(x=strat.index, y=strat.values, name="Momentum — raw",
                             line=dict(color=C_RAW, width=1.8)))
    if rc is not None:
        fig.add_trace(go.Scatter(x=rc.index, y=rc.values, name="Momentum — risk-conscious",
                                 line=dict(color=C_RC, width=2.4)))
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
        f"<span style='color:{C_RAW}'>gold</span> = the raw momentum strategy; "
        f"<span style='color:{C_RC}'>teal</span> = the risk-conscious (vol-targeted) version — all "
        "hypothetical, lump-sum. "
        + (f"The strategies are run at <b>your invested scale (€{vs['capital']:,.0f})</b>, so the flat "
           "€1/transaction fee is the same % drag it is on your book (on €10k it's a negligible "
           "~0.15%). " if vs.get("capital") else "")
        + "Apples-to-pears (your book is cash-flow-timed), and the strategies carry every caveat "
        "below — survivorship especially — so read the gap as indicative.</p>"
        f"<div class='chart'>{fig_html(fig)}</div>"
        "<table><tr><th>Book</th><th class='num'>Total ROI</th><th class='num'>Sharpe</th>"
        f"<th class='num'>Max DD</th></tr>{rows}</table>"
        f"<p class='dim'>Over this window the raw strategy is <b>{_pct(lead)}</b> "
        f"{'ahead of' if lead >= 0 else 'behind'} your portfolio on total return — but watch the "
        "drawdown and Sharpe columns: the risk-conscious version is the fairer comparison to how "
        "you actually run money.</p>")


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
            "at random from the same liquid pool; it does not promise the level repeats. "
            "<b>Selection is identical for the original and the risk-conscious versions</b>, so this "
            "verdict applies to both — vol-targeting changes the sizing, not the edge. Read it with "
            "the regime and capacity caveats below.</p>")


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
        f'<b>past performance is not future returns</b>.'
        f'<br><br><b style="color:{C_RC}">Risk-conscious version.</b> Volatility-targeting directly '
        f'addresses the drawdown and the raw vol — it cuts both materially — but it does <b>not</b> '
        f'fix survivorship, regime dependence or capacity: those sit in the underlying selection, '
        f'which is identical, so they apply equally to both versions.</div>')


def build(d: dict, public: bool = False) -> str:
    """One page, two strategies side by side: a shared intro + on-top summary, then the
    side-by-side compare (picks → equity → performance → scorecard → yearly → every
    rebalance), then the shared head-to-head / significance / caveats, then (private only)
    the research lab — the 32-config grid the config was chosen from + supporting data."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cfg = d["strategy"]
    body = "".join([
        "<h1>Momentum strategy — Original vs Risk-conscious</h1>",
        f"<p class='dim'>generated {now} · config {cfg.code} · "
        f"<a href='report.html'>← portfolio</a></p>",
        # ── both strategies, framed + summarised on top ──
        sec_intro(d),
        sec_summary(d),
        # ── side-by-side compare (full parity) ──
        sec_picks_compare(d),
        sec_curve_compare(d),
        sec_perf_compare(d, public),
        sec_grade_compare(d, public),
        sec_yearly_compare(d, public),
        sec_timeline_compare(d),
        # ── shared (both versions) ──
        sec_vs_portfolio(d, public),
        sec_significance(d, public),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    d = gather(refresh=args.refresh)
    local = ROOT / "local/strategy.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d))                          # live/local only — no docs/ export
    print(f"wrote {local}  (strategy {STRATEGY.code}: original + risk-conscious + lab)")
    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
