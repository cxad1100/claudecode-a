"""Vol lab — the "predict volatility, not the mean" strategy report.

  python build_vol_report.py            # writes local/vol.html (local-only page)
  python build_vol_report.py --open

Thesis: daily return MEANS are ~unpredictable out-of-sample; daily VOLATILITY is strongly
predictable (clustering). So the strategy forecasts tomorrow's vol (EWMA / GARCH(1,1) /
HAR-RV vs the incumbent trailing-vol baseline) and sizes exposure
w = min(target_vol / forecast, 1) — de-risk only, remainder in cash. Applied to (a) the
MSCI World ETF (clean test) and (b) the production momentum strategy's equity curve.

Honest framing baked in: this is the documented Moreira–Muir vol-targeting effect — NOT
novel alpha. It buys Sharpe/drawdown and usually gives up total return under the
no-leverage cap. The report shows the losses too (2020 whipsaw), not just the wins.
"""
import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tools import theme, significance as sig, quant_grade as qg, vol_forecast as vf
from tools import vol_ml as vml
from tools.report_html import pct as _pct, card as _card, page, fig_html

ROOT = Path(__file__).parent

TARGET_VOL = 0.15                     # matches build_strategy_report.RISK_TARGET_VOL
ETF = "IWDA.AS"                       # iShares Core MSCI World, EUR — TR-tradeable
ETF_NAME = "MSCI World (IWDA.AS)"
PROXY = "^GSPC"                       # research-only long history (USD price index)
PROXY_NAME = "S&P 500 (^GSPC)"
PROXY_START = "1995-01-01"
METHODS = ("rolling", "ewma", "garch", "har",            # fixed-parameter forecasters
           "adaptive_ewma", "ridge", "ensemble")         # learned (tools.vol_ml)
METHOD_LABEL = dict(rolling="Trailing 63d (incumbent)", ewma="EWMA (RiskMetrics)",
                    garch="GARCH(1,1)", har="HAR-RV (daily proxy)",
                    adaptive_ewma="EWMA, learned λ", ridge="Ridge (learned, auto-α)",
                    ensemble="Online ensemble (Hedge)")
COST_BPS = 5.0                        # liquid-ETF half-spread
FEE_EUR = 1.0                         # Trade Republic per-order fee
BAND = 0.10                           # only rebalance when |Δw| > band
CAPITAL = 10_000.0
_COLORS = dict(rolling="#808080", ewma="#4ec9b0", garch="#569cd6", har="#c586c0",
               adaptive_ewma="#d7ba7d", ridge="#e8a04e", ensemble="#46c84e",
               trend="#dcdcaa", bh="#d4d4d4")


# ───────────────────────────── data ─────────────────────────────

def _fetch_ohlc(ticker: str, period: str = "max") -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    return raw[["Close", "High", "Low"]].dropna(how="all")


def _cached_ohlc(ticker: str, force: bool = False, ttl_hours: float = 12,
                 _fetch=_fetch_ohlc) -> pd.DataFrame:
    """OHLC needs its own cache — data_buffer.cached_price_history stores Close only."""
    import time
    d = ROOT / "local" / "buffer"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"ohlc_{ticker.replace('^', 'i').replace('.', '_')}.pkl"
    if not force and path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
    df = _fetch(ticker)
    try:
        df.to_pickle(path)
    except Exception:
        pass
    return df


def _momentum_returns(refresh: bool = False) -> pd.Series | None:
    """Net daily returns of the production momentum strategy (one run_momentum call,
    no grid/MC). None when the universe price panel isn't on disk."""
    from build_momentum_report import (PRICES_CSV, META_CSV, LOOKBACK, SKIP, START,
                                       LIQ_MAX, MIN_PRICE, CAPITAL as MCAP, FEE_EUR as MFEE,
                                       WINSOR_CAP, EXEC_LAG, _slip)
    if not PRICES_CSV.exists():
        return None
    from build_strategy_report import STRATEGY
    from tools.momentum import run_momentum, winsorize_prices, to_xetra_calendar
    from tools.universe_pit import PITUniverse
    from tools.universe_assemble import delisting_map
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices = winsorize_prices(to_xetra_calendar(prices), cap=WINSOR_CAP)
    meta_df = pd.read_csv(META_CSV)
    meta = {r["ticker"]: dict(r) for _, r in meta_df.iterrows()}
    slip = {t: _slip(m) for t, m in meta.items() if t in prices.columns}
    pit = PITUniverse(prices, delisting_map(meta_df))
    res = run_momentum(prices, slip, lookback=LOOKBACK, skip=SKIP, capital=MCAP,
                       cost_mults=(1.0,), start=START, liq_max=LIQ_MAX, fee_eur=MFEE,
                       min_price=MIN_PRICE, sectors=None, benchmark=None, pit=pit,
                       execute_lag=EXEC_LAG, **STRATEGY.kwargs())
    eq = res["runs"][1.0]["equity"]
    return eq.dropna().pct_change().dropna()


# ───────────────────────────── compute (pure) ─────────────────────────────

def compute_underlying(r: pd.Series, name: str, price: pd.Series | None = None,
                       ohlc: pd.DataFrame | None = None, trend_combo: bool = False) -> dict:
    """Everything the report shows for one underlying, from its daily returns alone
    (price/OHLC optional extras). Pure — testable with synthetic data."""
    r = r.dropna()
    forecasts: dict[str, pd.Series] = {}
    for m in METHODS:
        if m == "ensemble":                       # reuse the base fits — no double compute
            comp = {k: forecasts[k] for k in vml.BASE_COMPONENTS if k in forecasts}
            forecasts[m] = vml.ensemble_vol(r, components=comp)
        else:
            forecasts[m] = vf.forecast_vol(r, method=m)
    valid = pd.concat(forecasts.values(), axis=1).dropna().index
    if len(valid) < 300:
        return {}
    oos_start = valid[0]
    r_oos = r.loc[oos_start:]

    fwd1, fwd21 = vf.forward_realized_vol(r, 1), vf.forward_realized_vol(r, 21)
    pk_fwd = None
    if ohlc is not None and {"High", "Low"}.issubset(ohlc.columns):
        pk_daily = (np.log(ohlc["High"] / ohlc["Low"]) ** 2 / (4 * np.log(2.0))).reindex(r.index)
        pk_fwd = np.sqrt(vf.TD * pk_daily.rolling(21).mean().shift(-21))

    stats_, variants = {}, {}
    for m, f in forecasts.items():
        fo = f.loc[oos_start:]
        s = dict(qlike_h1=vf.qlike(fo ** 2, (fwd1.loc[oos_start:]) ** 2),
                 qlike_h21=vf.qlike(fo ** 2, (fwd21.loc[oos_start:]) ** 2),
                 mse_h21=vf.mse(fo, fwd21.loc[oos_start:]),
                 oos_r2=vf.oos_r2(fo, fwd21.loc[oos_start:], horizon=21),
                 mz=vf.mincer_zarnowitz(fo, fwd21.loc[oos_start:]))
        if pk_fwd is not None:
            s["qlike_pk"] = vf.qlike(fo ** 2, (pk_fwd.loc[oos_start:]) ** 2)
        stats_[m] = s
        variants[m] = vf.vol_managed(r_oos, f, target_vol=TARGET_VOL, band=BAND,
                                     cost_bps=COST_BPS, fee_eur=FEE_EUR, capital=CAPITAL)
    winner = min(METHODS, key=lambda m: stats_[m]["qlike_h21"])

    if trend_combo and price is not None:
        gate = (price > price.rolling(200).mean()).astype(float).reindex(r.index).ffill()
        variants["trend"] = vf.vol_managed(r_oos, forecasts[winner], target_vol=TARGET_VOL,
                                           band=BAND, cost_bps=COST_BPS, fee_eur=FEE_EUR,
                                           capital=CAPITAL, gate=gate)
    bh_eq = (1.0 + r_oos).cumprod()
    return dict(name=name, returns=r, r_oos=r_oos, oos_start=oos_start,
                forecasts=forecasts, fwd21=fwd21, stats=stats_, variants=variants,
                winner=winner, bh=dict(qg.perf_metrics(bh_eq), equity=bh_eq),
                price=price)


def cost_grid(r_oos: pd.Series, forecast: pd.Series) -> list[dict]:
    out = []
    for cb in (0.0, 5.0, 10.0, 25.0):
        for bd in (0.05, 0.10, 0.20):
            m = vf.vol_managed(r_oos, forecast, target_vol=TARGET_VOL, band=bd,
                               cost_bps=cb, fee_eur=FEE_EUR, capital=CAPITAL)
            out.append(dict(cost_bps=cb, band=bd, sharpe=m.get("sharpe", 0.0),
                            ann_return=m.get("ann_return", 0.0),
                            trades=m.get("n_trades_per_year", 0.0)))
    return out


def gather(force: bool = False, refresh: bool | None = None) -> dict:
    refresh = force if refresh is None else refresh
    etf_ohlc = _cached_ohlc(ETF, force=refresh)
    spx_ohlc = _cached_ohlc(PROXY, force=refresh)
    spx_ohlc = spx_ohlc[spx_ohlc.index >= PROXY_START]

    etf_close = etf_ohlc["Close"].dropna()
    spx_close = spx_ohlc["Close"].dropna()
    etf = compute_underlying(etf_close.pct_change().dropna(), ETF_NAME, price=etf_close,
                             ohlc=etf_ohlc, trend_combo=True)
    spx = compute_underlying(spx_close.pct_change().dropna(), PROXY_NAME, price=spx_close,
                             ohlc=spx_ohlc)
    if not etf or not spx:
        raise RuntimeError("not enough price history for the vol lab")

    mom = None
    try:
        mr = _momentum_returns(refresh)
        if mr is not None and len(mr) > 900:
            mom = compute_underlying(mr, "Momentum strategy (net)")
            if mom:
                # the incumbent /strategy overlay for the head-to-head (no band/costs)
                mom["incumbent"] = qg.vol_target((1.0 + mom["r_oos"]).cumprod(),
                                                 target_vol=TARGET_VOL)
    except Exception:
        mom = None

    mean_null = vf.mean_null(spx["returns"])

    # significance over EVERY variant scanned (methods × underlyings + trend combo)
    all_variants = []
    for u in (etf, spx, mom):
        if u:
            all_variants += [(u["name"], m, v) for m, v in u["variants"].items()]
    trial_sharpes = [v.get("sharpe", 0.0) for _, _, v in all_variants]
    best_name, best_m, best = max(all_variants, key=lambda t: t[2].get("sharpe", -9))
    best_rets = best["equity"].pct_change().dropna().to_numpy()
    dsr = sig.deflated_sharpe_ratio(best_rets, trial_sharpes, ppy=252.0)
    ci = sig.bootstrap_sharpe_cagr_ci(best_rets, ppy=252.0, block=21, seed=0)

    grid = cost_grid(etf["r_oos"], etf["forecasts"][etf["winner"]])
    return dict(etf=etf, spx=spx, mom=mom, mean_null=mean_null, grid=grid,
                significance=dict(dsr=dsr, ci=ci, best=f"{best_m} on {best_name}"))


# ───────────────────────────── sections ─────────────────────────────

def sec_thesis(d: dict) -> str:
    return (
        '<div class="note"><b>The thesis — predict volatility, not the mean.</b> '
        'Daily return <i>means</i> are essentially unpredictable out-of-sample (the table '
        'below shows their OOS R&sup2; &asymp; 0). Daily <i>volatility</i> clusters and is strongly '
        'predictable. So instead of forecasting returns, forecast tomorrow&rsquo;s vol and size '
        f'exposure <b>w = min({TARGET_VOL:.0%} / forecast&nbsp;vol, 100%)</b> — de-risk only, '
        'never lever (Trade Republic has no margin), remainder in cash. Rebalance only when '
        f'the target moves more than {BAND:.0%} (kills daily churn); each adjustment pays '
        f'{COST_BPS:.0f}&nbsp;bps + &euro;{FEE_EUR:.0f}. This is the documented Moreira&ndash;Muir '
        'vol-targeting effect — <b>not novel alpha</b>: it buys Sharpe and drawdown, and under '
        'the no-leverage cap it usually gives up total return in bull tapes. Not advice.</div>')


def sec_forecast_eval(d: dict) -> str:
    rows = []
    for u in (d["spx"], d["etf"]):
        for m in METHODS:
            s = u["stats"][m]
            mz = s["mz"] or {}
            star = " ★" if m == u["winner"] else ""
            pk = f"{s['qlike_pk']:.3f}" if "qlike_pk" in s else "—"
            rows.append(
                f"<tr><td>{u['name']}</td><td>{METHOD_LABEL[m]}{star}</td>"
                f"<td class='num mono'>{s['qlike_h1']:.3f}</td>"
                f"<td class='num mono'>{s['qlike_h21']:.3f}</td>"
                f"<td class='num mono'>{pk}</td>"
                f"<td class='num mono'>{s['oos_r2']:+.2f}</td>"
                f"<td class='num mono'>{mz.get('beta', float('nan')):.2f}</td>"
                f"<td class='num mono'>{mz.get('r2', float('nan')):.2f}</td></tr>")

    u = d["spx"]
    f = u["forecasts"][u["winner"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u["fwd21"].index, y=u["fwd21"].values * 100,
                             name="realised (next 21d)", line=dict(color=_COLORS["bh"], width=1.2)))
    fig.add_trace(go.Scatter(x=f.index, y=f.values * 100,
                             name=f"forecast — {METHOD_LABEL[u['winner']]}",
                             line=dict(color=_COLORS[u["winner"]], width=1.8)))
    fig.add_hline(y=TARGET_VOL * 100, line_dash="dash", line_color=theme.FG_DIM,
                  annotation_text="target vol")
    fig.update_layout(height=380, yaxis=dict(title="Annualised vol (%)", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))
    return (
        "<h2>Is volatility actually predictable? Yes.</h2>"
        "<p class='dim'>Every forecaster only sees returns through day t and predicts day "
        "t+1 (property-tested — mutating the future leaves past forecasts untouched). "
        "Scored out-of-sample against the next-day and next-21-day realised vol: "
        "<b>QLIKE</b> (the standard robust loss — lower is better), QLIKE against the "
        "range-based <b>Parkinson</b> target where OHLC exists (a ~5&times; more efficient vol "
        "proxy), <b>OOS R&sup2;</b> vs an expanding-mean benchmark, and the "
        "<b>Mincer&ndash;Zarnowitz</b> regression (unbiased &rArr; &beta;&nbsp;&asymp;&nbsp;1). "
        "★ = QLIKE-h21 winner per underlying. ^GSPC runs from 1995 so the score covers "
        "2000&ndash;02, 2008 and 2020 — as a USD research proxy only; the tradeable backtest "
        "below is the EUR ETF.</p>"
        "<table><tr><th>Underlying</th><th>Model</th><th class='num'>QLIKE h1</th>"
        "<th class='num'>QLIKE h21</th><th class='num'>QLIKE h21 (PK)</th>"
        "<th class='num'>OOS R&sup2;</th><th class='num'>MZ &beta;</th>"
        "<th class='num'>MZ R&sup2;</th></tr>" + "".join(rows) + "</table>"
        f"<div class='chart'>{fig_html(fig)}</div>")


def sec_mean_null(d: dict) -> str:
    mn = d["mean_null"]
    label = {"mean_21d": "Trailing 21d mean", "mean_252d": "Trailing 252d mean",
             "ar1": "AR(1), walk-forward refit"}
    vol_r2 = max(s["oos_r2"] for s in d["spx"]["stats"].values())
    rows = "".join(f"<tr><td>{label.get(k, k)}</td><td class='num mono'>{v:+.3f}</td></tr>"
                   for k, v in mn.items())
    return (
        "<h2>…and the mean is not</h2>"
        f"<p class='dim'>The same walk-forward machinery pointed at next-day <i>returns</i> "
        f"instead of vol, on the same {PROXY_NAME} history. OOS R&sup2; vs the expanding mean:</p>"
        f"<table><tr><th>Return forecast</th><th class='num'>OOS R&sup2;</th></tr>{rows}</table>"
        f"<p class='dim'>&asymp; 0 across the board, vs <b>{vol_r2:+.2f}</b> for the best vol "
        "forecast above — that asymmetry is the entire strategy. Any edge here comes from "
        "predicting risk, not from predicting direction.</p>")


def _roi_fig(u: dict) -> go.Figure:
    fig = go.Figure()
    bh = u["bh"]["equity"]
    fig.add_trace(go.Scatter(x=bh.index, y=(bh / bh.iloc[0] - 1) * 100, name="Buy & hold",
                             line=dict(color=_COLORS["bh"], width=1.4)))
    for m, v in u["variants"].items():
        e = v["equity"]
        nm = METHOD_LABEL.get(m, f"Trend-filtered {METHOD_LABEL[u['winner']]}")
        w = 2.4 if m == u["winner"] else 1.4
        fig.add_trace(go.Scatter(x=e.index, y=(e / e.iloc[0] - 1) * 100, name=nm,
                                 line=dict(color=_COLORS.get(m, "#dcdcaa"), width=w)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM)
    fig.update_layout(height=440, yaxis=dict(title="Cumulative ROI (%)", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))
    return fig


def _dd_fig(u: dict) -> go.Figure:
    fig = go.Figure()
    for label, eq, color in ([("Buy & hold", u["bh"]["equity"], _COLORS["bh"])] +
                             [(METHOD_LABEL.get(m, "Trend combo"), v["equity"],
                               _COLORS.get(m, "#dcdcaa"))
                              for m, v in u["variants"].items() if m in (u["winner"], "trend")]):
        dd = (eq / eq.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=label,
                                 line=dict(color=color, width=1.6)))
    fig.update_layout(height=300, yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))
    return fig


def _metrics_table(u: dict, extra_rows: str = "") -> str:
    def row(label, m, exp=None, tr=None):
        e = f"{exp * 100:.0f}%" if exp is not None else "100%"
        t = f"{tr:.0f}" if tr is not None else "—"
        return (f"<tr><td>{label}</td><td class='num'>{_pct(m.get('ann_return', 0) * 100)}</td>"
                f"<td class='num mono'>{m.get('ann_vol', 0) * 100:.1f}%</td>"
                f"<td class='num mono'>{m.get('sharpe', 0):.2f}</td>"
                f"<td class='num mono'>{m.get('calmar', 0):.2f}</td>"
                f"<td class='num'>{_pct(m.get('max_dd', 0) * 100)}</td>"
                f"<td class='num mono'>{e}</td><td class='num mono'>{t}</td></tr>")
    rows = row("Buy & hold", u["bh"])
    for m, v in u["variants"].items():
        rows += row(METHOD_LABEL.get(m, f"Trend-filtered {METHOD_LABEL[u['winner']]}"), v,
                    v.get("avg_exposure"), v.get("n_trades_per_year"))
    return ("<table><tr><th>Variant</th><th class='num'>Ann. return</th><th class='num'>Vol</th>"
            "<th class='num'>Sharpe</th><th class='num'>Calmar</th><th class='num'>Max DD</th>"
            "<th class='num'>Avg exp.</th><th class='num'>Trades/yr</th></tr>"
            + rows + extra_rows + "</table>")


def sec_etf(d: dict) -> str:
    u = d["etf"]
    win = u["variants"][u["winner"]]
    exp = win["exposure"]
    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(x=exp.index, y=exp.values * 100, name="exposure",
                                 line=dict(color=_COLORS[u["winner"]], width=1.2, shape="hv"),
                                 fill="tozeroy"))
    fig_exp.update_layout(height=240, yaxis=dict(title="Exposure (%)", ticksuffix="%",
                                                 range=[0, 105]), margin=dict(t=20))

    out = [f"<h2>Applied to the tradeable ETF — {u['name']}</h2>",
           f"<p class='dim'>All variants start {u['oos_start'].date()} (after the 3y model "
           f"warm-up), net of {COST_BPS:.0f} bps + &euro;{FEE_EUR:.0f} per adjustment, "
           f"{BAND:.0%} rebalance band. The trend combo multiplies the winning forecast's "
           "exposure by a 200-day trend gate — the standard patch for vol-targeting's "
           "V-recovery whipsaw.</p>",
           f"<div class='chart'>{fig_html(_roi_fig(u))}</div>",
           _metrics_table(u),
           f"<div class='chart'>{fig_html(_dd_fig(u))}</div>",
           f"<p class='dim'>Exposure over time — {METHOD_LABEL[u['winner']]}:</p>",
           f"<div class='chart'>{fig_html(fig_exp)}</div>"]

    # the 2020 whipsaw, shown honestly
    if u["r_oos"].index[0] <= pd.Timestamp("2019-06-01"):
        px = u["bh"]["equity"]
        seg = slice(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
        pxs, exps = px.loc[seg], exp.loc[seg]
        if len(pxs) > 50:
            f20 = go.Figure()
            f20.add_trace(go.Scatter(x=pxs.index, y=(pxs / pxs.iloc[0] - 1) * 100,
                                     name="ETF (buy & hold)", line=dict(color=_COLORS["bh"], width=1.8)))
            f20.add_trace(go.Scatter(x=exps.index, y=exps.values * 100, name="strategy exposure",
                                     line=dict(color=_COLORS[u["winner"]], width=1.8, shape="hv"),
                                     yaxis="y2"))
            f20.update_layout(height=320, hovermode="x unified", margin=dict(t=20),
                              yaxis=dict(title="ETF ROI 2020 (%)", ticksuffix="%"),
                              yaxis2=dict(title="Exposure (%)", overlaying="y", side="right",
                                          range=[0, 105], ticksuffix="%"))
            out += ["<h3>The 2020 whipsaw — the weakness, not hidden</h3>",
                    "<p class='dim'>Vol spikes <i>after</i> the first leg down, so the overlay "
                    "sells into the crash and re-buys after the V-recovery is underway — it "
                    "protects the depth of the drawdown but gives back part of the rebound. "
                    "This is the known failure mode of vol targeting; the trend combo softens "
                    "but does not remove it.</p>",
                    f"<div class='chart'>{fig_html(f20)}</div>"]
    return "".join(out)


def sec_momentum(d: dict) -> str:
    u = d.get("mom")
    if not u:
        return ("<h2>Applied to the momentum strategy</h2>"
                "<div class='note'>The universe price panel (data/universe/"
                "universe_prices.csv) isn't on this machine, so the momentum leg was "
                "skipped. Run the universe build first, then refresh this page.</div>")
    inc = u.get("incumbent") or {}
    extra = ""
    if inc:
        extra = (f"<tr><td>Incumbent /strategy overlay (trailing 63d, no band, no costs)</td>"
                 f"<td class='num'>{_pct(inc.get('ann_return', 0) * 100)}</td>"
                 f"<td class='num mono'>{inc.get('ann_vol', 0) * 100:.1f}%</td>"
                 f"<td class='num mono'>{inc.get('sharpe', 0):.2f}</td>"
                 f"<td class='num mono'>{inc.get('calmar', 0):.2f}</td>"
                 f"<td class='num'>{_pct(inc.get('max_dd', 0) * 100)}</td>"
                 f"<td class='num mono'>{inc.get('avg_exposure', 0) * 100:.0f}%</td>"
                 f"<td class='num mono'>—</td></tr>")
    return ("<h2>Applied to the momentum strategy</h2>"
            "<p class='dim'>The same forecast-based overlay on the production momentum "
            "book's net equity curve. Caveat first: the momentum config already "
            "vol-adjusts its <i>scores</i> per name, and /strategy already applies a "
            "trailing-vol overlay — this section asks whether a <i>forecast-based</i> "
            "overlay beats that incumbent, not whether overlays help at all. Stacking "
            "all three risks over-damping.</p>"
            f"<div class='chart'>{fig_html(_roi_fig(u))}</div>"
            + _metrics_table(u, extra))


def sec_significance(d: dict) -> str:
    s = d["significance"]
    dsr, ci = s["dsr"], s["ci"]
    dsr_v = "—" if dsr["dsr"] != dsr["dsr"] else f"{dsr['dsr']:.0%}"
    cards = [
        _card("Best variant", s["best"]),
        _card("Deflated Sharpe (P real>0)", dsr_v),
        _card(f"Sharpe {ci['conf']}% CI", f"{ci['sharpe_lo']:.2f} – {ci['sharpe_hi']:.2f}"),
        _card(f"CAGR {ci['conf']}% CI", f"{ci['cagr_lo']*100:+.1f}% – {ci['cagr_hi']*100:+.1f}%"),
    ]
    return ("<h2>Significance</h2>"
            f"<p class='dim'>The best variant's Sharpe, haircut for the <b>{dsr['n_trials']} "
            "variants scanned</b> on this page (methods &times; underlyings + the trend combo), "
            "return skew/kurtosis and sample length (Bailey&ndash;L&oacute;pez de Prado deflated "
            "Sharpe), plus a circular block-bootstrap (21-day blocks) error bar. Remember what "
            "is being tested: vol-managed <i>beta</i>, not stock selection — a high DSR here "
            "says the risk-adjusted improvement is unlikely to be luck, not that the overlay "
            "prints money.</p>"
            f'<div class="cards">{"".join(cards)}</div>')


def sec_costs(d: dict) -> str:
    by_band: dict[float, list] = {}
    for g in d["grid"]:
        by_band.setdefault(g["band"], []).append(g)
    head = "<tr><th>Rebalance band</th>" + "".join(
        f"<th class='num'>{int(c)} bps</th>" for c in (0, 5, 10, 25)) + "</tr>"
    rows = ""
    for bd in sorted(by_band):
        cells = "".join(f"<td class='num mono'>{g['sharpe']:.2f} "
                        f"<span class='dim'>({g['trades']:.0f}/yr)</span></td>"
                        for g in sorted(by_band[bd], key=lambda g: g["cost_bps"]))
        rows += f"<tr><td class='mono'>{bd:.0%}</td>{cells}</tr>"
    return ("<h2>Cost &amp; churn sensitivity</h2>"
            f"<p class='dim'>Sharpe of the winning ETF variant ({METHOD_LABEL[d['etf']['winner']]}) "
            "across per-adjustment cost (columns) and rebalance band (rows); trades/yr in grey. "
            "The overlay survives realistic ETF costs because the band keeps turnover low — "
            "but every rebalance is also a taxable event (see caveats).</p>"
            f"<table>{head}{rows}</table>")


def sec_method(d: dict) -> str:
    return (
        '<div class="note warn"><b>What this is — and is not.</b> '
        '<b>(1) Not alpha.</b> Vol targeting is the published Moreira&ndash;Muir (2017) effect; '
        'Cederburg et&nbsp;al. (2020) showed real-time versions fail on many factors. It is most '
        'robust exactly where applied here — the equity market and momentum (Barroso&ndash;'
        'Santa-Clara) — but it is beta timing on a known anomaly, not an edge others lack. '
        '<b>(2) The forecast is the strong link; the payoff is the weak one.</b> Vol is genuinely '
        'predictable (table above), but converting that into returns needs high vol to coincide '
        'with poor risk-adjusted returns — noisier, regime-dependent. '
        '<b>(3) No leverage = truncated upside.</b> Much of the published gain comes from levering '
        'calm markets; capped at 100%, expect better Sharpe/drawdown and usually <i>lower</i> total '
        'return than buy &amp; hold in bull tapes. '
        '<b>(4) Whipsaw.</b> The 2020 panel shows the failure mode explicitly. '
        '<b>(5) Taxes are NOT modelled.</b> Every de-risk sale realises German capital-gains tax '
        '(26.375%); the band keeps trades rare but the drag is real. '
        '<b>(6) Data.</b> IWDA.AS starts ~2009 (bull-heavy sample; flatters buy &amp; hold and '
        'understates crash protection); its Amsterdam close vs the US session adds stale-close '
        'echo to daily vol; ^GSPC is a USD price index used only to score forecasts. HAR here '
        'runs on daily squared-return proxies (next-week target), noisier than the intraday-RV '
        'original. <b>(7) Learned vs fixed parameters.</b> Three forecasters learn their '
        'parameters walk-forward (tools/vol_ml.py): the EWMA λ is QMLE-refit quarterly, the '
        'ridge model re-chooses its own regularisation at every refit by a chronological '
        'validation split, and the online ensemble (Hedge) reallocates trust across models '
        'daily by realised QLIKE. No LSTM by choice: ~4k noisy non-stationary daily '
        'observations is far below where recurrent nets beat GARCH/HAR out of sample, and '
        'the failure mode is silent overfitting — any challenger emitting the same forecast '
        'Series can be added to the QLIKE table and judged identically. '
        '<b>(8) Future work.</b> Downside semi-vol targeting; a ^VIX vol-risk-premium '
        'signal; leverage via UCITS-leveraged ETFs. <b>Past performance is not future returns.</b>'
        '</div>')


def build(d: dict, public: bool = False) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "".join([
        "<h1>Vol lab — predict volatility, not the mean</h1>",
        f"<p class='dim'>generated {now} · <a href='report.html'>← portfolio</a> · "
        "<a href='strategy.html'>strategy</a></p>",
        sec_thesis(d),
        sec_forecast_eval(d),
        sec_mean_null(d),
        sec_etf(d),
        sec_momentum(d),
        sec_significance(d),
        sec_costs(d),
        sec_method(d),
    ])
    return page("Vol lab", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    d = gather(refresh=args.refresh)
    local = ROOT / "local/vol.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d))                            # local-only — no docs/ export
    print(f"wrote {local}")
    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
