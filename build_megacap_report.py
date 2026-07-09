"""Mega-cap PIT screen — size / growth / momentum arms, N-sweep. Debug build writes
the local snapshot only (mirrors build_momentum_report)."""
import argparse
import datetime as dt
import pathlib

import pandas as pd

from tools.megacap import (run_arms, cap_panel, yoy_growth_panel, load_cache,
                           candidate_pool, coverage_report, ARMS, CACHE)
from tools.momentum import (rebalance_dates, to_xetra_calendar, winsorize_prices)
from tools.report_html import page

ROOT = pathlib.Path(__file__).resolve().parent
PRICES_CSV = ROOT / "data" / "universe" / "universe_prices.csv"
META_CSV = ROOT / "data" / "universe" / "universe_meta.csv"
WINSOR_CAP = 0.5
N_SWEEP = (1, 5, 10, 25, 50)
HEADLINE_N = 25
K = 10
CAPITAL = 10_000.0
STAT_KEYS = ("net_return", "sharpe", "max_drawdown")   # keys of tools.pairs_backtest.backtest_stats


def _slip(m) -> int:
    v = m.get("slippage_bps")
    return int(v) if v not in (None, "") else 30


def load_data() -> dict:
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices = winsorize_prices(to_xetra_calendar(prices), cap=WINSOR_CAP)
    meta_df = pd.read_csv(META_CSV)
    slip = {r["ticker"]: _slip(r) for _, r in meta_df.iterrows()
            if r["ticker"] in prices.columns}
    # No cache yet → empty panels: the page still serves, showing 0% coverage and a
    # prompt to run the live fetch (see the plan's runtime note). Once populated,
    # rebuild for real numbers.
    shares, rev = load_cache(CACHE) if CACHE.exists() else ({}, {})
    dates = rebalance_dates(prices.index)
    cap = cap_panel(shares, prices, dates)
    yoy = yoy_growth_panel(rev, dates)
    cover = coverage_report({t: (t in shares) for t in
                             candidate_pool(meta_df, prices.columns)})
    spx = pd.DataFrame(index=prices.index)      # benchmark hook; empty = skipped
    return dict(prices=prices, cap=cap, yoy=yoy, slip=slip, benchmarks=spx,
                capital=CAPITAL, coverage=cover)


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def build_html(data: dict) -> str:
    prices, cap, yoy, slip = data["prices"], data["cap"], data["yoy"], data["slip"]
    cov = data["coverage"]
    rows = []
    for n in N_SWEEP:
        res = run_arms(prices, slip, cap, yoy, n=n, k=K, capital=data["capital"])
        for arm in ARMS:
            st = res[arm]["runs"][1.0]["stats"]   # 1x-cost run
            cells = "".join(f"<td>{_fmt(st.get(k))}</td>" for k in STAT_KEYS)
            hl = " class='headline'" if n == HEADLINE_N else ""
            rows.append(f"<tr{hl}><td>N={n}</td><td>{arm}</td>{cells}</tr>")
    head = "".join(f"<th>{k}</th>" for k in STAT_KEYS)
    empty = ("<p class='prompt'><em>No market-cap data yet — run the live EODHD "
             "fetch (see the plan's runtime note), then rebuild.</em></p>"
             if cov["covered"] == 0 else "")
    body = (
        "<h1>Mega-cap PIT screen — size / growth / momentum</h1>"
        f"<p class='coverage'>Cap coverage: {cov['covered']}/{cov['candidates']} "
        f"candidates ({cov['pct']}%).</p>"
        f"{empty}"
        f"<p>Headline N=25, hold k={K}, monthly, equal-weight, net of slippage. "
        "Fully point-in-time (shares &amp; revenue lagged 75d).</p>"
        f"<table><thead><tr><th>screen</th><th>arm</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return page("Mega-cap PIT screen — size / growth / momentum", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.parse_args()
    html = build_html(load_data())
    (ROOT / "local").mkdir(exist_ok=True)
    (ROOT / "local" / "megacap.html").write_text(html)
    print(f"wrote local/megacap.html ({dt.date.today()})")


if __name__ == "__main__":
    main()
