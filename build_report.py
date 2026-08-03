"""Generate the static HTML dashboard.

  python build_report.py            # writes local/report.html (full, € amounts)
                                    #    and docs/index.html  (public, percentages only)

No server. Open local/report.html directly; docs/ deploys via GitHub Pages.
Public build hides: euro values, share counts, costs, transactions.
"""

import argparse
import webbrowser
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tools import theme
from tools.report_html import fig_html, pct, card, page
from tools import equity_log
from tools.data_buffer import (
    cached_current_prices,
    cached_market_caps,
    cached_price_history,
)
from tools.portfolio_tools import (
    COMPANY_NAMES as NAMES,
    parse_portfolio,
    compute_portfolio_summary,
)
from tools.portfolio_analytics import (
    build_roi_timeseries,
    compute_quant_metrics,
    compute_correlation_matrix,
    xirr,
)
from tools.portfolio_tools import TICKER_MAP
from tools.optimizer import (
    fetch_price_history,
    fetch_market_caps,
    to_returns,
    annualize,
    optimize,
    risk_parity,
    risk_contributions,
    hrp,
    black_litterman,
    max_return_at_vol,
    efficient_frontier,
    random_portfolios,
    portfolio_perf,
)

# ── Settings (one place) ──────────────────────────────────────────────────────
LOOKBACK_DAYS = 365      # window for the COVARIANCE estimate (the trustworthy input)
RF            = 0.045    # risk-free rate (annual)
LONG_ONLY     = True
MAX_W         = 0.35     # max single-position weight
BL_DELTA      = 2.5      # Black-Litterman market risk-aversion (standard ≈ 2.5)
BL_TAU        = 0.05     # Black-Litterman prior uncertainty scale
# Optional subjective views for Black-Litterman. Empty = pure market-implied.
# Example: [{"assets": {"NVD.F": 1}, "ret": 0.12, "confidence": 0.5}]
BL_VIEWS: list[dict] = []

ROOT = Path(__file__).parent


_fig_html, _pct, _card = fig_html, pct, card


# ── Data assembly ─────────────────────────────────────────────────────────────

def gather(force: bool = False) -> dict:
    portfolio = parse_portfolio(ROOT / "input/portfolio.csv")
    prices, stale, as_of = cached_current_prices(portfolio["holdings"], force=force)
    summary = compute_portfolio_summary(portfolio, prices)
    positions = [p for p in summary["positions"] if p["position_value"] > 0]
    txns = portfolio["transactions"]
    tot = summary["totals"]

    # ── Transparent accounting — every line reconciles to the next ──────────────
    gross_deposits = sum(t["price"] for t in txns if t["action"] == "buy")
    cash_returned  = sum(t["price"] for t in txns if t["action"] == "sell")
    realized       = tot["realized_pnl"]
    cost_of_sold   = cash_returned - realized          # cost basis of shares sold
    net_cost_basis = gross_deposits - cost_of_sold     # cost basis still at work (open)
    current_value  = tot["current_value"]              # live market value of open
    unrealized     = current_value - net_cost_basis
    total_pnl      = realized + unrealized
    net_invested   = gross_deposits - cash_returned    # out-of-pocket cash
    simple_roi     = total_pnl / gross_deposits * 100 if gross_deposits else 0.0

    # Money-weighted return (XIRR): dated cash flows, buys negative, value today positive
    flows = [(date.fromisoformat(t["date"]), -t["price"] if t["action"] == "buy" else t["price"])
             for t in txns]
    flows.append((date.today(), current_value))
    mwr = xirr(flows)
    hold_years = (date.today() - min(f[0] for f in flows)).days / 365.0
    mwr_cumulative = ((1 + mwr) ** hold_years - 1) * 100 if mwr is not None else None

    acct = dict(gross_deposits=gross_deposits, cash_returned=cash_returned, realized=realized,
                cost_of_sold=cost_of_sold, net_cost_basis=net_cost_basis,
                current_value=current_value, unrealized=unrealized, total_pnl=total_pnl,
                net_invested=net_invested, simple_roi=simple_roi, mwr=mwr,
                mwr_cumulative=mwr_cumulative, hold_years=hold_years)

    # Log today's snapshot (auditable daily equity series; gitignored local/)
    equity_log.append_snapshot(ROOT / "local/equity_log.csv", dict(
        date=str(date.today()), current_value=round(current_value, 2),
        net_cost_basis=round(net_cost_basis, 2), gross_deposits=round(gross_deposits, 2),
        cash_returned=round(cash_returned, 2), realized_pnl=round(realized, 2),
        unrealized_pnl=round(unrealized, 2), total_pnl=round(total_pnl, 2)))

    deployed = gross_deposits  # back-compat alias
    roi_series, bm_series, asset_values = build_roi_timeseries(txns)
    metrics = compute_quant_metrics(roi_series, bm_series.get("S&P 500"))
    correlation = compute_correlation_matrix(portfolio["holdings"], TICKER_MAP)

    # Optimizer inputs
    tickers = [p["ticker"] for p in positions]
    hist = cached_price_history(tickers, period="5y")   # TTL only — daily data,
                                                        # "Update to now" re-pulls live prices, not 5y history
    rets = to_returns(hist)
    universe = [t for t in tickers if t in rets.columns]
    window = rets.loc[rets.index >= rets.index[-1] - pd.Timedelta(days=LOOKBACK_DAYS)]
    mean_ann, cov_ann = annualize(window[universe])
    mu, sig = mean_ann.values, cov_ann.values

    val = {p["ticker"]: p["position_value"] for p in positions}
    tot_val = sum(val[t] for t in universe) or 1.0
    cur_w = np.array([val[t] / tot_val for t in universe])

    # ── Market-cap weights → Black-Litterman implied returns (the return model) ──
    caps = cached_market_caps(universe)                 # TTL only (caps barely move)
    # ETFs/missing names have no single-stock cap (and an ETF's AUM is on a wildly
    # different scale than trillion-€ stocks). Give them the MEDIAN fetched cap so
    # they sit as a typical holding in the prior — never crushed to ~0, never dominant.
    avail = [caps[t] for t in universe if t in caps]
    fallback = float(np.median(avail)) if avail else 1.0
    cap_vec = np.array([caps.get(t, fallback) for t in universe])
    mkt_w = cap_vec / cap_vec.sum()
    pi = black_litterman(cov_ann, mkt_w, delta=BL_DELTA, tau=BL_TAU, views=BL_VIEWS)
    mean_bl = RF + pi                       # total expected return = rf + implied excess
    mu_bl = mean_bl.values

    cur_ret, cur_vol, cur_sharpe = portfolio_perf(cur_w, mu_bl, sig, RF)
    cur_rc = risk_contributions(cur_w, cov_ann)

    kw = dict(long_only=LONG_ONLY, max_w=MAX_W)
    # Covariance-only (no return forecast)
    w_minvar = optimize(mean_ann, cov_ann, objective="min_var", rf=RF, **kw)
    w_rp     = risk_parity(cov_ann, max_w=MAX_W)
    w_hrp    = hrp(cov_ann)   # uncapped by design — clustering controls concentration
    # Black-Litterman return-based
    w_bl_sharpe = optimize(mean_bl, cov_ann, objective="sharpe", rf=RF, **kw)
    w_bl_same   = max_return_at_vol(mean_bl, cov_ann, cur_vol, **kw)

    frontier = efficient_frontier(mean_bl, cov_ann, n_points=40, **kw)
    cloud = random_portfolios(mean_bl, cov_ann, n=2500, rf=RF)

    # YTD from ROI series
    ytd = None
    if not roi_series.empty:
        yr = datetime.now().year
        this_year = roi_series[roi_series.index.year == yr]
        prev = roi_series[roi_series.index.year < yr]
        if not this_year.empty:
            base = (1 + float(prev.iloc[-1]) / 100) if not prev.empty else 1.0
            ytd = ((1 + float(this_year.iloc[-1]) / 100) / base - 1) * 100

    return dict(positions=positions, summary=summary, deployed=deployed, txns=txns,
                acct=acct,
                roi_series=roi_series, bm_series=bm_series, metrics=metrics,
                asset_values=asset_values,
                correlation=correlation, universe=universe, sig=sig,
                cov_ann=cov_ann, cur_w=cur_w, tot_val=tot_val, mkt_w=mkt_w,
                mu_bl=mu_bl, pi=pi.values, cur_rc=cur_rc,
                cur_perf=(cur_ret, cur_vol, cur_sharpe),
                w_minvar=w_minvar, w_rp=w_rp, w_hrp=w_hrp,
                w_bl_sharpe=w_bl_sharpe, w_bl_same=w_bl_same,
                frontier=frontier, cloud=cloud, ytd=ytd,
                stale=stale, as_of=as_of)


# ── Sections ──────────────────────────────────────────────────────────────────

def _eur(x, signed=False):
    s = f"€{x:+,.2f}" if signed else f"€{x:,.2f}"
    cls = "pos" if (signed and x > 0) else ("neg" if (signed and x < 0) else "")
    return f'<span class="mono {cls}">{s}</span>'


def _ledger(title, rows, total):
    """rows: list of (label, value_html, op) where op ∈ {'', '+', '−', '='}."""
    body = ""
    for label, val, op in rows:
        opc = f"<span class='dim' style='display:inline-block;width:1em'>{op}</span>"
        bold = " style='font-weight:600;border-top:1px solid #2d2d2d'" if op == "=" else ""
        body += f"<tr{bold}><td>{opc}{label}</td><td class='num'>{val}</td></tr>"
    return f"<h3>{title}</h3><table class='ledger'>{body}</table>"


def sec_summary(d: dict, public: bool) -> str:
    a = d["acct"]
    m = d["metrics"]

    # ── Headline cards (returns, always shown) ──────────────────────────────────
    cards = []
    if not public:
        cards.append(_card("Current Value", _eur(a["current_value"])))
        cards.append(_card("Total P&L", _eur(a["total_pnl"], signed=True)))
    cards.append(_card("Simple ROI", _pct(a["simple_roi"])))
    if a["mwr"] is not None:
        cards.append(_card("Money-Weighted (IRR)", f'{_pct(a["mwr"]*100)}<span class="dim"> /yr</span>'))
    cards.append(_card("YTD", _pct(d["ytd"]) if d["ytd"] is not None else "—"))
    if m:
        cards.append(_card("Sharpe", f'{m["sharpe"]:.2f}'))
        cards.append(_card("Max Drawdown", _pct(m["max_drawdown"])))
    cards.append(_card("Positions", str(len(d["positions"]))))
    out = ["<h2>Summary</h2>", f'<div class="cards">{"".join(cards)}</div>']

    if public:
        out.append("<p class='dim'>Euro amounts hidden in the public build. "
                   "<b>Simple ROI</b> = total profit ÷ all capital ever deployed (cumulative). "
                   "<b>Money-Weighted (IRR)</b> = annualised internal rate of return on your actual "
                   "dated cash flows. <b>YTD</b> = calendar-year return.</p>")
        return "".join(out)

    # ── Transparent accounting waterfall (private only) ─────────────────────────
    out.append("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px'>")
    out.append("<div>" + _ledger("Capital flow (cash)", [
        ("Gross deposits — all buys", _eur(a["gross_deposits"]), ""),
        ("Cash returned — all sells", _eur(-a["cash_returned"], signed=True), "−"),
        ("Net invested (out of pocket)", _eur(a["net_invested"]), "="),
    ], None) + "</div>")
    out.append("<div>" + _ledger("Position accounting (where it is now)", [
        ("Net cost basis (open positions)", _eur(a["net_cost_basis"]), ""),
        ("Unrealized P&L", _eur(a["unrealized"], signed=True), "+"),
        ("Current market value", _eur(a["current_value"]), "="),
    ], None) + "</div>")
    out.append("<div>" + _ledger("Profit &amp; loss (where it came from)", [
        ("Realized P&L (closed trades)", _eur(a["realized"], signed=True), ""),
        ("Unrealized P&L (open positions)", _eur(a["unrealized"], signed=True), "+"),
        ("Total P&L", _eur(a["total_pnl"], signed=True), "="),
    ], None) + "</div>")
    out.append("</div>")

    # Returns, fully labelled
    mwr_line = ""
    if a["mwr"] is not None:
        mwr_line = (f"<tr><td>Money-Weighted Return (XIRR)</td>"
                    f"<td class='num'>{_pct(a['mwr']*100)} <span class='dim'>/yr</span></td>"
                    f"<td class='dim'>IRR on dated cash flows; ≈{_pct(a['mwr_cumulative'])} over "
                    f"{a['hold_years']:.1f}y. Rewards deploying early.</td></tr>")
    out.append(
        "<h3>Returns — what each percentage means</h3><table>"
        f"<tr><td>Simple ROI</td><td class='num'>{_pct(a['simple_roi'])}</td>"
        f"<td class='dim'>Total P&amp;L ÷ gross deposits = "
        f"{_eur(a['total_pnl'], signed=True)} ÷ {_eur(a['gross_deposits'])}. Cumulative, "
        "ignores timing.</td></tr>"
        f"{mwr_line}"
        f"<tr><td>YTD (calendar)</td><td class='num'>{_pct(d['ytd']) if d['ytd'] is not None else '—'}</td>"
        "<td class='dim'>This calendar year, flow-adjusted from the daily equity series.</td></tr>"
        "</table>")

    out.append(
        f"<div class='note'>Two cash 'net' figures differ on purpose: <b>net invested</b> "
        f"({_eur(a['net_invested'])}) is out-of-pocket; <b>net cost basis</b> "
        f"({_eur(a['net_cost_basis'])}) is the cost still at work in open positions. The "
        f"{_eur(a['net_cost_basis']-a['net_invested'], signed=True)} gap is realized profit "
        f"({_eur(a['realized'], signed=True)}) you already banked, which lowered your out-of-pocket "
        "below cost basis. Cash returned from sells = cost of those shares "
        f"({_eur(a['cost_of_sold'])}) + realized gain ({_eur(a['realized'], signed=True)}).</div>")
    return "".join(out)


def sec_weights_now(d: dict, public: bool) -> str:
    """The actionable core: four portfolios, none relying on trailing-mean returns."""
    universe, cur_w = d["universe"], d["cur_w"]
    mu_bl, sig = d["mu_bl"], d["sig"]
    cur_ret, cur_vol, cur_sharpe = d["cur_perf"]
    out = ["<h2>Weights to use now</h2>"]

    out.append(f"""
<div class="note">
None of these use trailing returns as a forecast (that's the part that doesn't work).
The three <b>robust</b> portfolios use only the <b>covariance matrix</b> — how your assets move
together — which is statistically stable. The two <b>Black-Litterman</b> portfolios get expected
returns from <b>market-cap weights</b> (what the market collectively bets), not from your assets'
recent winning streaks. All are long-only and sum to 100%. Min-Var and Risk-Parity (and both BL
portfolios) cap each position at max {MAX_W:.0%}; <b>HRP is uncapped by design</b> — its clustering
controls concentration instead of an arbitrary limit.
</div>""")

    cols = [("Min-Var", d["w_minvar"]), ("Risk-Parity", d["w_rp"]), ("HRP", d["w_hrp"]),
            ("BL Max-Sharpe", d["w_bl_sharpe"]), ("BL Same-Risk", d["w_bl_same"])]
    head = "<tr><th>Ticker</th><th>Name</th><th class='num'>Current</th>" + \
        "".join(f"<th class='num'>{c[0]}</th>" for c in cols) + \
        ("<th class='num'>Mkt-cap</th>" if True else "") + "</tr>"
    rows = []
    for i, t in enumerate(universe):
        cells = f"<td class='num mono'>{cur_w[i]*100:.1f}%</td>"
        for _, w in cols:
            cells += (f"<td class='num mono'>{w[i]*100:.1f}%</td>" if w is not None
                      else "<td class='num'>—</td>")
        cells += f"<td class='num mono dim'>{d['mkt_w'][i]*100:.1f}%</td>"
        rows.append(f"<tr><td class='mono'>{t}</td><td>{NAMES.get(t, t)}</td>{cells}</tr>")
    out.append(f"<table>{head}{''.join(rows)}</table>")

    def perf_cards(w, label, desc, eur=False):
        if w is None:
            return f"<div class='note warn'><b>{label}</b>: no feasible solution under constraints.</div>"
        w = np.asarray(w)
        r, v, s = portfolio_perf(w, mu_bl, sig, RF)
        shift = ""
        if eur and not public:
            biggest = sorted(((w[i] - cur_w[i]) * d["tot_val"], universe[i]) for i in range(len(universe)))
            up = [f"{tk} €{e:+,.0f}" for e, tk in biggest[-2:][::-1] if e > 1]
            dn = [f"{tk} €{e:+,.0f}" for e, tk in biggest[:2] if e < -1]
            if up or dn:
                shift = "<div class='dim mono' style='margin-top:6px'>shift: " + ", ".join(up + dn) + "</div>"
        return (f"<h3>{label}</h3><p class='dim'>{desc}</p><div class='cards'>"
                + _card("Volatility", _pct(v * 100, signed=False))
                + _card("vs current vol", _pct((v - cur_vol) * 100))
                + _card("Exp. return*", _pct(r * 100))
                + _card("Sharpe*", f"{s:.2f}")
                + "</div>" + shift)

    out.append("<h3 style='color:#6a9955'>Robust — no return forecast (covariance only)</h3>")
    out.append(perf_cards(d["w_minvar"], "Minimum Variance",
                          "The single lowest-risk mix of your assets. The most reliable output of "
                          "the whole method — depends on nothing but the covariance matrix.", eur=True))
    out.append(perf_cards(d["w_rp"], "Risk Parity (Equal Risk Contribution)",
                          "Every asset contributes the same share of total portfolio risk — no single "
                          "name dominates your risk. The institutional answer to 'I don't trust return "
                          "forecasts'.", eur=True))
    out.append(perf_cards(d["w_hrp"], "Hierarchical Risk Parity (HRP)",
                          "Clusters your assets by correlation, then splits risk down the tree. "
                          "Recognises that the US mega-caps move together and treats them as one risk "
                          "unit, so they can't individually dominate — diversification without a single "
                          "hardcoded cap, and it doesn't lean into mega-caps the way Black-Litterman "
                          "does.", eur=True))

    out.append("<h3 style='color:#569cd6'>Black-Litterman — market-implied returns</h3>")
    out.append(perf_cards(d["w_bl_sharpe"], "BL Max-Sharpe",
                          "Highest return per unit of risk, where 'return' is what the market implies "
                          "from cap weights (δ·Σ·w_market), not your holdings' past performance.", eur=True))
    out.append(perf_cards(d["w_bl_same"], "BL Same-Risk Max-Return",
                          "Keeps your current volatility exactly; maximizes the market-implied expected "
                          "return. A pure upgrade at today's risk level.", eur=True))

    out.append(f"""
<div class="note warn">
<b>* Expected return &amp; Sharpe</b> use Black-Litterman market-implied returns, not trailing history.
They're a coherent relative ranking, not a promise — the market can be wrong. <b>Volatility and risk
contributions are the solid numbers</b>; the covariance is estimated from {LOOKBACK_DAYS} days of prices
and is far more stable than any return forecast. Not financial advice.
</div>""")
    return "".join(out)


def sec_risk_contrib(d: dict) -> str:
    """The killer insight: where your risk actually comes from vs where your money is."""
    universe, cur_w, rc = d["universe"], d["cur_w"], d["cur_rc"]
    order = np.argsort(-rc)
    rows = []
    for i in order:
        t = universe[i]
        gap = rc[i] - cur_w[i]
        cls = "neg" if gap > 0.02 else ("pos" if gap < -0.02 else "dim")
        rows.append(f"<tr><td class='mono'>{t}</td><td>{NAMES.get(t, t)}</td>"
                    f"<td class='num mono'>{cur_w[i]*100:.1f}%</td>"
                    f"<td class='num mono'>{rc[i]*100:.1f}%</td>"
                    f"<td class='num mono {cls}'>{gap*100:+.1f}pp</td></tr>")
    return ("<h2>Where your risk actually comes from</h2>"
            "<p class='dim'>Capital weight is how your money is split; risk contribution is how your "
            "<i>volatility</i> is split (weight × how much it moves with the rest). A name with risk "
            "far above its weight is silently driving your portfolio. Risk Parity above equalises the "
            "right-hand column.</p>"
            "<table><tr><th>Ticker</th><th>Name</th><th class='num'>Capital</th>"
            "<th class='num'>Risk</th><th class='num'>Risk − Capital</th></tr>"
            f"{''.join(rows)}</table>")


def sec_frontier(d: dict) -> str:
    mu, sig = d["mu_bl"], d["sig"]
    cloud, frontier = d["cloud"], d["frontier"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cloud["vol"] * 100, y=cloud["ret"] * 100, mode="markers",
        marker=dict(size=4, color=cloud["sharpe"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Sharpe", thickness=12)),
        name="Random mixes", opacity=0.4,
        hovertemplate="vol %{x:.1f}%<br>ret %{y:.1f}%<extra></extra>"))
    if frontier:
        fig.add_trace(go.Scatter(x=[f["vol"] * 100 for f in frontier],
                                 y=[f["ret"] * 100 for f in frontier],
                                 mode="lines", line=dict(color=theme.FG, width=2), name="Frontier"))

    def pt(w, name, symbol, color):
        if w is None:
            return
        r, v, s = portfolio_perf(np.array(w), mu, sig, RF)
        # markers only — identity comes from the legend above the plot; in-plot
        # text labels collided when points clustered (Current/BL/Risk-Parity).
        fig.add_trace(go.Scatter(x=[v * 100], y=[r * 100], mode="markers",
                                 marker=dict(size=15, color=color, symbol=symbol,
                                             line=dict(width=1, color="#000")),
                                 name=name,
                                 hovertemplate=f"{name}: Sharpe {s:.2f}<extra></extra>"))

    pt(d["cur_w"], "Current", "circle", "#c586c0")
    pt(d["w_minvar"], "Min-Var", "square", "#6a9955")
    pt(d["w_rp"], "Risk-Parity", "triangle-up", "#b5cea8")
    pt(d["w_hrp"], "HRP", "hexagon", "#9cdcfe")
    pt(d["w_bl_sharpe"], "BL Max-Sharpe", "star", "#dcdcaa")
    pt(d["w_bl_same"], "BL Same-Risk", "diamond", "#4ec9b0")
    fig.update_layout(height=470, xaxis_title="Volatility (annual %)",
                      yaxis_title="Expected return — Black-Litterman implied (annual %)",
                      margin=dict(t=20))
    return ("<h2>Efficient frontier</h2>"
            "<p class='dim'>Each grey dot is a random mix of your assets. The white line is the "
            "frontier — best return at every risk level. The vertical axis uses Black-Litterman "
            "market-implied returns (not trailing history); the horizontal axis (volatility) is the "
            "reliable one. Min-Var sits at the far left (lowest risk).</p>"
            f"<div class='chart'>{_fig_html(fig)}</div>")


def sec_roi(d: dict) -> str:
    roi, bms = d["roi_series"], d["bm_series"]
    if roi.empty:
        return ""
    BM_COLORS = {"Portfolio": "#ffffff", "S&P 500": "#d16969", "Nasdaq 100": "#e8a04e",
                 "MSCI World": "#6a9955", "FTSE All-World": "#4ec9b0",
                 "Euro Stoxx 50": "#9cdcfe", "Emerging Markets": "#c586c0",
                 "Gold": "#d7ba7d", "Bitcoin": "#ce9178", "Fixed Income": "#808080"}
    start = roi.index[0].strftime("%Y-%m-%d")
    fig = go.Figure()
    # benchmarks first (thin), portfolio on top (thick white) so it stands out
    for nm, s in bms.items():
        if not s.empty:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=nm,
                                     line=dict(color=BM_COLORS.get(nm, "#aaa"), width=1.4),
                                     opacity=0.9))
    fig.add_trace(go.Scatter(x=roi.index, y=roi.values, name="Portfolio (you)",
                             line=dict(color=BM_COLORS["Portfolio"], width=3.2)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=485, yaxis=dict(title="Cumulative ROI (%)", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))
    ranked = sorted([("Portfolio (you)", float(roi.iloc[-1]))] +
                    [(n, float(s.iloc[-1])) for n, s in bms.items() if not s.empty],
                    key=lambda x: -x[1])
    rows = "".join(
        f"<tr><td>{'<b>'+n+'</b>' if n.startswith('Portfolio') else n}</td>"
        f"<td class='num'>{_pct(v)}</td></tr>" for n, v in ranked)
    table = f"<table><tr><th>Where the same money went</th><th class='num'>Total ROI</th></tr>{rows}</table>"
    return (f"<h2>Your portfolio vs the same money elsewhere</h2>"
            f"<p class='dim'>Cash-flow matched since your first trade ({start}): every euro you actually "
            "spent on a buy is simultaneously invested — virtually — into each benchmark on that same day. "
            "Whatever you put into Apple on a date, the Bitcoin line gets the same amount into BTC that day, and so on. "
            "Sale proceeds count as returned cash, so selling never looks like a loss. This is the honest "
            "real-money comparison.</p>"
            f"<div class='chart'>{_fig_html(fig)}</div>{table}")


def sec_risk(d: dict) -> str:
    m = d["metrics"]
    if not m:
        return ""
    items = [
        ("Sharpe", f'{m["sharpe"]:.2f}'), ("Sortino", f'{m["sortino"]:.2f}'),
        ("Volatility", _pct(m["volatility"], signed=False)), ("CAGR", _pct(m["cagr"])),
        ("Max drawdown", _pct(m["max_drawdown"])), ("Current drawdown", _pct(m["current_drawdown"])),
        ("VaR 95%", _pct(m["var_95"])), ("CVaR 95%", _pct(m["cvar_95"])),
        ("Win rate", _pct(m["win_rate"], signed=False)),
        ("Beta vs S&P", f'{m["beta"]:.2f}' if m.get("beta") is not None else "—"),
        ("Alpha (ann.)", _pct(m["alpha"]) if m.get("alpha") is not None else "—"),
        ("Best day", _pct(m["best_day"], nd=2)), ("Worst day", _pct(m["worst_day"], nd=2)),
    ]
    return "<h2>Risk &amp; efficiency</h2><div class='cards'>" + \
        "".join(_card(k, v) for k, v in items) + "</div>"


def sec_positions(d: dict, public: bool) -> str:
    rows = []
    for p in d["positions"]:
        t = p["ticker"]
        w = p["position_value"] / d["tot_val"] * 100
        extra = ""
        if not public:
            extra = (f"<td class='num mono'>{p['shares']:.4f}</td>"
                     f"<td class='num mono'>€{p['avg_cost']:.2f}</td>"
                     f"<td class='num mono'>€{p['position_value']:,.0f}</td>"
                     f"<td class='num mono'>€{p['unrealized_pnl']:+,.0f}</td>")
        rows.append(f"<tr><td class='mono'>{t}</td><td>{NAMES.get(t, t)}</td>"
                    f"<td class='num mono'>{w:.1f}%</td>{extra}"
                    f"<td class='num'>{_pct(p['unrealized_pct'])}</td></tr>")
    head = ("<tr><th>Ticker</th><th>Name</th><th class='num'>Weight</th>"
            + ("" if public else "<th class='num'>Shares</th><th class='num'>Avg cost</th>"
               "<th class='num'>Value</th><th class='num'>P&amp;L</th>")
            + "<th class='num'>Return</th></tr>")
    return f"<h2>Positions</h2><table>{head}{''.join(rows)}</table>"


def sec_correlation(d: dict) -> str:
    corr = d["correlation"]
    if not corr:
        return ""
    tk = list(corr.keys())
    z = [[corr[r].get(c) for c in tk] for r in tk]
    text = [[f"{corr[r].get(c, 0):.2f}" for c in tk] for r in tk]
    fig = go.Figure(go.Heatmap(z=z, x=tk, y=tk, text=text, texttemplate="%{text}",
                               colorscale=[[0, "#d16969"], [0.5, theme.BG], [1, "#569cd6"]],
                               zmin=-1, zmax=1, colorbar=dict(thickness=12, title="ρ")))
    fig.update_layout(height=max(320, len(tk) * 40),
                      xaxis=dict(tickangle=-35), margin=dict(l=80, b=60))
    return ("<h2>Return correlation (1Y)</h2>"
            "<p class='dim'>How your positions move together. Lots of deep blue = "
            "concentrated bets; diversification needs low or negative correlation.</p>"
            f"<div class='chart'>{_fig_html(fig)}</div>")


def sec_explainer() -> str:
    return f"""
<h2>How the numbers are computed</h2>
<details open><summary>Why not just use past returns? (the whole point)</summary>
<p>Naive Markowitz feeds the optimizer each asset's <i>average past return</i> as its expected
future return. This is its fatal flaw: a sample mean of returns is almost pure noise and just chases
recent winners. The covariance matrix (how assets move together), by contrast, is statistically
stable and genuinely useful. So this report uses covariance for everything and <b>never uses
trailing returns as a forecast</b>.</p></details>
<details><summary>Covariance &amp; volatility (the trustworthy input)</summary>
<ul>
<li><b>Volatility</b> = standard deviation of daily returns × √252 — how much an asset swings.</li>
<li><b>Covariance</b> = how each pair moves together, estimated from {LOOKBACK_DAYS} days of prices.</li>
<li><b>Portfolio volatility</b> is <i>not</i> the average of asset volatilities — assets that zig
while others zag cancel risk out. Exploiting that is the entire value of diversification.</li>
</ul></details>
<details><summary>Minimum Variance &amp; Risk Parity (no return forecast)</summary>
<p><b>Minimum Variance</b>: scipy (SLSQP) finds the weights with the lowest possible portfolio
volatility (long-only, ≤{MAX_W:.0%} each, sum 100%). Needs only covariance.</p>
<p><b>Risk Parity</b>: weights chosen so every asset contributes an equal share of total risk.
Risk contribution of asset i = wᵢ × (Σw)ᵢ ⁄ portfolio variance. Also covariance-only — this is what
risk-parity and minimum-volatility funds (e.g. the USMV ETF) actually do.</p></details>
<details><summary>Hierarchical Risk Parity — the fix for mega-cap bias (no cap needed)</summary>
<p>Black-Litterman has a built-in tilt: it derives expected returns from market-cap weights
(Π = δ·Σ·w_market), so because US mega-cap tech dominates global market cap, the math hands those names
a proportionally huge implied return and starves smaller, non-US holdings. That's not geographic bias —
it's the cap-weight multiplier. The crude fix is a hardcoded cap ("max 20% per country"), but that just
overrides the math with human guesswork.</p>
<p><b>HRP</b> (Marcos López de Prado) solves it structurally, using <i>only</i> the covariance matrix —
no return forecast, no cap. Three steps: (1) turn correlations into distances
(dᵢⱼ = √(½(1−ρᵢⱼ))) and run <b>hierarchical clustering</b> — an unsupervised-ML algorithm that groups
assets that move together; (2) <b>quasi-diagonalize</b> — reorder so correlated names sit adjacent;
(3) <b>recursive bisection</b> — walk down the cluster tree splitting the risk budget by inverse
variance at each fork. Because Apple/Nvidia/Google land in one cluster, they share a single risk
budget and can't each dominate — you get deep diversification without typing a single
<code>max_weight</code>. It's how real funds get concentration control out of the predictable part of
the data (risk) instead of the unpredictable part (returns).</p></details>
<details><summary>Black-Litterman (a return model that isn't momentum)</summary>
<p>Instead of past returns, reverse the optimization: given the market's cap-weighted mix, what
expected returns would make that mix optimal? <b>Π = δ · Σ · w_market</b> (δ = {BL_DELTA} risk
aversion). These "implied" returns are the market's collective forecast. You can layer your own views
on top (Black-Litterman blends them by confidence); with no views the report uses pure Π. Then the
two BL portfolios run standard max-Sharpe / same-risk optimization on Π instead of on history.</p>
<p class="dim">Cap weights use each stock's market cap. An <b>ETF</b> (e.g. IWDA) has no single-stock
cap, and yfinance reports none — so it's given the <i>median</i> of the fetched stock caps, i.e.
treated as a typical-sized holding rather than crushed to ~0%. A diversified ETF in the prior also
double-counts the stocks it holds — a known limitation of cap-weighted BL on a mixed stock+ETF book.</p></details>
<details><summary>Sharpe ratio &amp; the risk-free rate</summary>
<p>(portfolio return − {RF:.1%}) ÷ volatility = return per unit of risk. The {RF:.1%} cash rate is a
fixed offset — it shifts every Sharpe equally, so it never changes the <i>ranking</i> of portfolios;
it only sets the bar that "doing nothing" (holding cash) clears.</p></details>
<details><summary>Simple ROI vs Money-Weighted Return (IRR)</summary>
<p><b>Simple ROI</b> = Total P&amp;L ÷ gross deposits. One cumulative number; it ignores <i>when</i>
you invested — a euro deployed on day 1 and a euro deployed yesterday count equally.</p>
<p><b>Money-Weighted Return (XIRR)</b> = the single annualised rate r that makes the present value of
every dated cash flow (each buy negative, each sell positive, today's value positive) sum to zero. It
rewards deploying capital early and is the honest "what return did my actual money earn per year."</p>
<p>They answer different questions, so they don't match — and neither is the time-weighted return a
fund reports (TWR strips out your deposit timing to judge the <i>strategy</i>; that needs a clean
daily equity series, see below).</p></details>
<details><summary>The accounting identity (why the numbers tie out)</summary>
<p>Gross deposits − cost basis of shares sold = <b>net cost basis</b> (open). Net cost basis +
unrealized P&amp;L = <b>current value</b>. Realized + unrealized = <b>total P&amp;L</b>. Cash returned
from a sale = that share's cost basis + its realized gain — which is why net invested cash sits below
net cost basis by exactly the realized profit you've banked.</p></details>
<details><summary>Daily equity logging (for exact YTD / calendar returns)</summary>
<p>Today YTD is <i>reconstructed</i>: holdings are repriced on every past business day from yfinance
history. Correct, but recomputed each run and vulnerable to data revisions/splits. The robust upgrade
is an append-only <b>daily equity log</b> (<code>local/equity_log.csv</code>, written every build):
date, value, net cost basis, deposits, withdrawals, realized/unrealized. With it, calendar-year return
becomes a flow-adjusted TWR — chain daily returns rᵢ = (Vᵢ − netflowᵢ)/Vᵢ₋₁ and compound across the
year, so mid-year deposits never masquerade as performance. A daily cron/launchd job appending one row
makes the series authoritative instead of reconstructed.</p></details>
<details><summary>Risk metrics</summary>
<ul>
<li><b>Max drawdown</b> — worst peak-to-trough fall of the value curve.</li>
<li><b>VaR 95%</b> — daily loss not exceeded on 95% of days. <b>CVaR 95%</b> — average loss on the
worst 5% of days.</li>
<li><b>Beta</b> — sensitivity to the S&amp;P 500. <b>Alpha</b> — return beyond what beta predicts.
<b>Sortino</b> — Sharpe counting only downside.</li>
</ul></details>
"""


# ── Assembly ──────────────────────────────────────────────────────────────────

def build(d: dict, public: bool) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "Investment Monitor" + ("" if public else " — private")
    badge = ('<span class="dim">public build — euro amounts hidden</span>'
             if public else '<span class="dim">private build — full data</span>')
    body = "".join([
        f"<h1>{title}</h1><p class='dim'>generated {now} · {badge} · "
        f"<a href='/strategy'>Momentum Strategy →</a></p>",
        sec_summary(d, public),
        sec_positions(d, public),
        sec_roi(d),
        sec_risk(d),
        sec_risk_contrib(d),
        sec_correlation(d),
        sec_frontier(d),
        sec_weights_now(d, public),
        sec_explainer(),
    ])
    return page(title, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the local report in the browser")
    args = ap.parse_args()

    print("gathering data (yfinance)...")
    d = gather()

    local = ROOT / "local/report.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d, public=False))
    print(f"wrote {local}")

    pub = ROOT / "docs/index.html"
    pub.parent.mkdir(exist_ok=True)
    pub.write_text(build(d, public=True))
    print(f"wrote {pub}  (no € amounts)")

    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
