# Investment Monitor

A static-HTML investment dashboard for a Trade Republic portfolio. One Python script
reads your trade history, pulls live data from yfinance, runs a Markowitz mean-variance
optimizer, and writes a self-contained dark-themed HTML report. No server, no API keys.

Two builds per run:

| File | Audience | Content |
|---|---|---|
| `local/report.html` | you (gitignored) | everything, incl. € amounts and € to shift per position |
| `docs/index.html` | GitHub Pages (public) | percentages, weights, ratios only — **no euro amounts, shares, or costs** |

## Setup

```bash
uv venv .venv --python 3.13 && uv pip install -r requirements.txt --python .venv
```

Add your trade history at `input/portfolio.csv` (gitignored):

```
Date,Ticker,Action,Shares,Price,PricePerShare
2024-11-18,ASML.AS,buy,0.25,160.30,641.20
```

## Usage

```bash
bash start.sh                       # build both reports + open the private one
.venv/bin/python build_report.py    # build only
.venv/bin/pytest tests/             # tests
```

Deploy: commit `docs/index.html`, push, serve via **GitHub Pages** (Settings → Pages →
deploy from branch → `/docs`).

## What the report shows

- **Weights to use now** — the actionable core. Your current weights next to two
  optimized alternatives over the same assets:
  - **A · Max Sharpe** — highest return per unit of risk (the textbook optimum)
  - **B · Same-risk max-return** — keeps your current volatility, maximizes expected
    return: a pure upgrade if today's swings are acceptable
- **Efficient frontier** — where your mix sits vs the best possible mixes
- **Every position, side by side** — each holding's EUR value over time from its first
  buy, a sale steps the line down (ends it on a full exit) and rolls proceeds into a
  cash line, bold total on top so selling never reads as a loss (private build only)
- **ROI vs benchmarks** — cash-flow matched (same euros, same dates, into each benchmark)
- **Risk & efficiency** — Sharpe, Sortino, drawdowns, VaR/CVaR, beta, alpha
- **Rolling backtest** — walk-forward monthly re-optimization vs equal-weight + S&P 500,
  no look-ahead, with an explicit selection-bias caveat
- **Correlation heatmap** + positions
- **"How the numbers are computed"** — plain-language explainer of every formula

## Architecture

```
input/portfolio.csv               ← trade history (gitignored)
build_report.py                   ← orchestrates: data → optimizer → HTML
 ├─ tools/portfolio_tools.py      ← CSV → holdings, P&L; live prices
 ├─ tools/portfolio_analytics.py  ← ROI series, per-asset EUR curves, quant metrics, correlation
 ├─ tools/optimizer.py            ← Markowitz: frontier, max-Sharpe, same-risk, backtest
 ├─ tools/portfolio_meta.py       ← sector maps, ETF decomposition
 └─ tools/theme.py                ← VSCode Dark+ palette, plotly template, CSS

build_vol_report.py               ← vol lab: forecast volatility (EWMA/GARCH/HAR), not the mean,
 └─ tools/vol_forecast.py            and vol-target exposure (serve.py route /vol, local-only)
build_edge_report.py              ← edge stack: structural retail edges — capacity, forced flows
 └─ tools/edge_seasonal.py           (tax-loss rebound), horizon, costs (route /edge, local-only)
tools/gates.py + live_state + track ← pre-registered verdict gates, today's-action panel,
                                      live-vs-backtest tracking with kill criteria
```
