"""Generate the econophysics research page (LOCAL-ONLY — never a public build).

  python build_econo_report.py            # writes local/econo.html
  ECONO_LEADLAG=0 python build_econo_report.py   # skip a module (env flags)

A research lab on top of the strategy stack: new cross-sectional signals and
overlays (lead-lag networks, correlation structure, phase-transition
indicators, scraped regulatory events) run through the SAME evaluation gate as
the production momentum strategy — walk-forward `run_momentum` with costs and
PIT gates, Monte-Carlo random-book null, DSR/phantom haircut, FF5+WML spanning
alpha. Modules are guarded like the strategy page's FACTORS block: a failed or
disabled module leaves its key None and the section renders "" — the page
never breaks on a half-finished experiment. Killed experiments keep their
section (the negative result is the deliverable).
"""
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from build_momentum_report import (CAPITAL, EXEC_LAG, FEE_EUR, LIQ_MAX,
                                   LOOKBACK, META_CSV, MIN_PRICE, MIN_TURNOVER,
                                   PRICES_CSV, SKIP, START, TRAIN_END,
                                   TURN_CSV, VAL_END, WINSOR_CAP, _slip)
from tools import factors
from tools import significance as sig
from tools import universe_snapshot
from tools.data_buffer import cached_price_history
from tools.leadlag import (leadlag_scores, rank_ic,
                           size_leadlag_baseline_scores)
from tools.momentum import (precompute_eligibility, rebalance_dates,
                            run_momentum, to_xetra_calendar, winsorize_prices)
from tools.momentum_grid import _stats_slice
from tools.report_html import page, pct as _pct
from tools.universe_assemble import death_map, delisting_map
from tools.universe_pit import PITUniverse

ROOT = Path(__file__).parent
OUT = ROOT / "local" / "econo.html"

#: module key → env flag suffix; every gather block and section keys off this.
MODULES = ("leadlag", "corr", "phase", "events", "trials")

# ── Lead-lag experiment grid (every cell is a counted trial) ─────────────────
LL_K = 10                     # book size (production slots)
LL_RECENTS = (5, 21)          # leader recent-return window, days
LL_FREQS = ("M", "Q")
LL_TOP_N = 300                # graph capped at most-liquid names
LL_SEED = 0
#: the strategy page's 64-config grid is inherited history in the DSR trial
#: count; every econo cell (controls included) adds to it and never leaves.
N_INHERITED_TRIALS = 64
PHANTOM_MULTS = (1, 5, 10)    # file-drawer ladder (strategy-page convention)
FACTORS = os.environ.get("FACTORS", "1") != "0"


def _enabled(name: str) -> bool:
    """Module env switch, FACTORS-style: ECONO_<NAME>=0 disables, default on."""
    return os.environ.get(f"ECONO_{name.upper()}", "1") != "0"


# ── Data gathering ────────────────────────────────────────────────────────────

def load_universe() -> dict:
    """The strategy stack's data assembly, factored once for every module:
    winsorized XETRA-calendar prices, slippage, sectors, PIT turnover and the
    membership-gated PITUniverse (same recipe as build_strategy_report)."""
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices = to_xetra_calendar(prices)
    prices = winsorize_prices(prices, cap=WINSOR_CAP)
    meta_df = pd.read_csv(META_CSV)
    meta = {r["ticker"]: dict(r) for _, r in meta_df.iterrows()}
    slip = {t: _slip(m) for t, m in meta.items() if t in prices.columns}
    sectors = {t: m.get("sector") for t, m in meta.items()}
    turnover = (pd.read_csv(TURN_CSV, index_col=0, parse_dates=True)
                if TURN_CSV.exists() else None)
    store = universe_snapshot.load_store()
    ticker_isin = {r["ticker"]: str(r["isin"]) for _, r in meta_df.iterrows()
                   if pd.notna(r.get("isin"))}
    membership = (store, ticker_isin) if len(store) else None
    pit = PITUniverse(prices, delisting_map(meta_df), deaths=death_map(meta_df),
                      membership=membership)
    return dict(prices=prices, slip=slip, sectors=sectors, turnover=turnover,
                pit=pit, meta_df=meta_df)


def _gather_leadlag(u: dict) -> dict:
    """The full M1 experiment: 8 signal cells + size-baseline + placebo, all
    through the production harness, then the gate (MC null, DSR ladder,
    FF5+WML spanning) on the worst-case-robust headline cell."""
    prices, slip, turnover, pit = (u["prices"], u["slip"], u["turnover"],
                                   u["pit"])
    cutoff = pd.Timestamp(START)
    cand = [d for d in rebalance_dates(prices.index, "M")
            if len(prices.loc[:d]) >= LOOKBACK + 1 and d >= cutoff]
    elig = precompute_eligibility(prices, slip, cand, liq_max=LIQ_MAX,
                                  min_obs=LOOKBACK + SKIP, min_price=MIN_PRICE,
                                  pit=pit, turnover=turnover,
                                  turn_floor=MIN_TURNOVER)
    scores, diag_all = {}, {}
    for rec in LL_RECENTS:
        dg: dict = {}
        scores[rec] = leadlag_scores(prices, cand, elig, turnover, recent=rec,
                                     top_n=LL_TOP_N, seed=LL_SEED, diag=dg)
        diag_all[rec] = dg
        # a date missing from the dict would silently fall back to price
        # momentum inside run_momentum — hard-fail instead
        assert set(cand) <= set(scores[rec]), "score precompute must cover all dates"

    te, ve = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)

    def _run(sbd, freq, va, code, kind):
        r = run_momentum(prices, slip, k=LL_K, lookback=LOOKBACK, skip=SKIP,
                         capital=CAPITAL, cost_mults=(1.0,), freq=freq,
                         liq_max=LIQ_MAX, fee_eur=FEE_EUR, min_price=MIN_PRICE,
                         start=START, vol_adjust=va, lazy=True, pit=pit,
                         execute_lag=EXEC_LAG, turnover=turnover,
                         turn_floor=MIN_TURNOVER, elig_by_date=elig,
                         score_by_date=sbd)
        eq, tr = r["runs"][1.0]["equity"], r["runs"][1.0]["trades"]
        return dict(code=code, kind=kind, freq=freq,
                    train=_stats_slice(eq, tr, eq.index[0], te, CAPITAL),
                    val=_stats_slice(eq, tr, te + pd.Timedelta(days=1), ve, CAPITAL),
                    test=_stats_slice(eq, tr, ve + pd.Timedelta(days=1),
                                      eq.index[-1], CAPITAL),
                    full=r["runs"][1.0]["stats"]), r

    cells, runs = [], {}
    for rec in LL_RECENTS:
        for freq in LL_FREQS:
            for va in (False, True):
                code = f"{freq}-r{rec}-{'voladj' if va else 'raw'}"
                cell, r = _run(scores[rec], freq, va, code, "signal")
                cells.append(cell)
                runs[code] = r
    bl = size_leadlag_baseline_scores(prices, cand, elig, turnover, recent=21)
    cell, r = _run(bl, "Q", False, "Q-r21-base", "baseline")
    cells.append(cell)
    pl = leadlag_scores(prices, cand, elig, turnover, recent=21,
                        top_n=LL_TOP_N, seed=LL_SEED, placebo_seed=1)
    cell, r = _run(pl, "Q", False, "Q-r21-plac", "placebo")
    cells.append(cell)

    # headline = worst-case-robust signal cell (pick_ultimate spirit); the
    # selection is itself a trial, which is why every cell is in the ledger
    sigc = [c for c in cells if c["kind"] == "signal"]
    head_cell = max(sigc, key=lambda c: min(c["train"]["sharpe"],
                                            c["val"]["sharpe"]))
    hr = runs[head_cell["code"]]
    hl = hr["holdings_log"]
    rb = [h["date"] for h in hl] + [hl[-1]["next"]]
    pools = sig.period_pools(prices, rb, elig, execute_lag=EXEC_LAG)
    strat_rets = sig.strategy_period_returns(hl)
    ppy = 4.0 if head_cell["freq"] == "Q" else 12.0
    mc = sig.monte_carlo_null(pools, strat_rets, k=LL_K, ppy=ppy,
                              n_trials=1000, seed=0)
    trial_sharpes = [c["full"]["sharpe"] for c in cells]
    n_total = N_INHERITED_TRIALS + len(cells)
    dsr0 = sig.deflated_sharpe_ratio(strat_rets, trial_sharpes, ppy=ppy,
                                     n_trials_effective=n_total)
    dsr_phantom = [dict(mult=m, n=n_total * m,
                        **{k: sig.deflated_sharpe_ratio(
                            strat_rets, trial_sharpes, ppy=ppy,
                            n_trials_effective=n_total * m)[k]
                           for k in ("dsr", "sr_benchmark_annual")})
                   for m in PHANTOM_MULTS]
    t_stat = float(dsr0["sharpe_annual"]
                   * np.sqrt(max(len(strat_rets) / ppy, 1e-9)))

    factor = None
    if FACTORS:
        try:
            fac, fac_src = factors.fetch_factors_daily()
            fx = cached_price_history(["EURUSD=X"],
                                      period="9y")["EURUSD=X"].dropna()
            eq = hr["runs"][1.0]["equity"]
            r_usd = factors.to_usd(eq.pct_change().dropna(), fx)
            excess = (r_usd - fac["RF"].reindex(r_usd.index)).dropna()
            reg = factors.factor_regression(excess, fac)
            if reg:
                key = "FF5+WML" if "FF5+WML" in reg else next(iter(reg))
                factor = dict(model=key, alpha_ann=reg[key]["alpha_ann"],
                              alpha_t=reg[key]["alpha_t"], n=reg[key]["n"],
                              source=fac_src)
        except Exception as e:
            print(f"[econo] factor regression skipped: {e}", file=sys.stderr)

    ic = {}
    for rec in LL_RECENTS:
        t = rank_ic(scores[rec], prices, cand, elig, execute_lag=EXEC_LAG)
        if len(t):
            se = t["ic"].std(ddof=1) / np.sqrt(len(t))
            ic[f"r{rec}"] = dict(mean=float(t["ic"].mean()),
                                 t=float(t["ic"].mean() / se) if se > 0 else 0.0,
                                 n=int(len(t)))

    dv = [v for v in diag_all[LL_RECENTS[-1]].values()]
    diag = (dict(median_kept=float(np.median([v["n_kept"] for v in dv])),
                 expected_false=float(np.median([v["n_expected_false"] for v in dv])),
                 mean_self_lag=float(np.nanmean([v["mean_self_lag"] for v in dv])),
                 median_uni=float(np.median([v["n_uni"] for v in dv])))
            if dv else {})

    return dict(cells=cells,
                headline=dict(code=head_cell["code"], mc=mc,
                              dsr_phantom=dsr_phantom, t_stat=t_stat,
                              factor=factor),
                ic=ic, diag=diag, n_trials=len(cells))


def gather() -> dict:
    """Run every enabled module; a disabled or failed module leaves None.

    Each block is wrapped try/except so one broken experiment (or a missing
    data file) never takes the page down — the section simply doesn't render.
    """
    d: dict = {k: None for k in MODULES}
    modules_needing_data = ("leadlag",)          # corr/phase/events join later
    if not any(_enabled(m) for m in modules_needing_data):
        return d
    try:
        u = load_universe()
    except Exception as e:
        print(f"[econo] universe load failed: {e}", file=sys.stderr)
        return d
    if _enabled("leadlag"):
        try:
            d["leadlag"] = _gather_leadlag(u)
        except Exception as e:
            print(f"[econo] leadlag skipped: {e}", file=sys.stderr)
    return d


# ── Sections (each returns "" when its module produced nothing) ──────────────

def sec_intro() -> str:
    return """<h1>Econophysics lab</h1>
<p class="sub">New signals and overlays, judged by the production gate:
walk-forward with costs and point-in-time universe gates, Monte-Carlo
random-book null, deflated Sharpe with phantom trials, and FF5+WML spanning
alpha (Newey–West t ≥ 2). A killed experiment stays on the page — the
negative result is the deliverable.</p>"""


def _verdict_leadlag(ll: dict) -> tuple[str, str]:
    """(css_class, text). Placebo ≥ headline Sharpe overrides everything —
    that's a leaking pipeline, not alpha."""
    head = ll.get("headline") or {}
    cells = ll.get("cells", [])
    fac = head.get("factor")
    head_full = next((c for c in cells if c["code"] == head.get("code")), None)
    plac = [c for c in cells if c["kind"] == "placebo"]
    if plac and head_full and \
            max(p["full"]["sharpe"] for p in plac) >= head_full["full"]["sharpe"]:
        return "neg", ("Pipeline leak suspected: the shuffled-leader placebo "
                       "matches or beats the signal. Halt — nothing here is "
                       "interpretable until the leak is explained.")
    dsr5 = next((x.get("dsr") for x in head.get("dsr_phantom", [])
                 if x.get("mult") == 5), None)
    mcp = (head.get("mc") or {}).get("p_sharpe")
    alpha_t = fac.get("alpha_t") if fac else None
    if (alpha_t is not None and alpha_t >= 2.0 and mcp is not None
            and mcp < 0.05 and dsr5 is not None and dsr5 > 0.5):
        return "pos", ("Residual alpha candidate: FF5+WML alpha t ≥ 2, beats "
                       "the random-book null, survives the ×5 phantom DSR. "
                       "Keep collecting live out-of-sample data.")
    if fac and fac.get("alpha_ann", 0) > 0:
        return "", ("Positive but not separable from known factors — factor "
                    "exposure, not alpha. Stays observational.")
    return "neg", ("No alpha at monthly/quarterly horizons — killed. The "
                   "negative result stays on the page.")


def sec_leadlag(d: dict, public: bool) -> str:
    ll = d.get("leadlag")
    if ll is None:
        return ""
    rows = []
    for c in ll.get("cells", []):
        rows.append(
            f"<tr><td class='mono'>{c['code']}</td><td>{c['kind']}</td>"
            f"<td class='mono'>{c['train']['sharpe']:.2f}</td>"
            f"<td class='mono'>{c['val']['sharpe']:.2f}</td>"
            f"<td class='mono'>{c['test']['sharpe']:.2f}</td>"
            f"<td>{_pct(c['full']['net_return'] * 100)}</td></tr>")
    head = ll.get("headline") or {}
    fac = head.get("factor")
    mc = head.get("mc") or {}
    ladder = " · ".join(f"×{x['mult']} (N={x['n']}): {x['dsr']:.2f}"
                        for x in head.get("dsr_phantom", []))
    ic_bits = " · ".join(
        f"recent={k[1:]}d: mean {v['mean']:+.3f} (t {v['t']:.1f}, n {v['n']})"
        for k, v in sorted(ll.get("ic", {}).items()))
    dg = ll.get("diag") or {}
    diag_line = (f"graph: median {dg.get('median_uni', 0):.0f} names, "
                 f"{dg.get('median_kept', 0):.0f} edges kept vs "
                 f"{dg.get('expected_false', 0):.0f} expected by chance · "
                 f"mean |self-lag| {dg.get('mean_self_lag', float('nan')):.3f} "
                 f"(residual staleness)") if dg else ""
    fac_line = (f"FF5+WML α {fac['alpha_ann'] * 100:+.1f}%/yr "
                f"(Newey–West t {fac['alpha_t']:.2f}, n {fac['n']})"
                if fac else "factor regression unavailable")
    cls, text = _verdict_leadlag(ll)
    return f"""<h2>Lead-lag network — neighbor momentum</h2>
<p class="sub">Directed lagged-correlation graph on the eligible, actively-
printing universe (top {LL_TOP_N} by PIT turnover); edges must clear a
circular-shift shuffle null; scores are leader-weighted recent returns.
Baseline = the known big→small channel; placebo = shuffled leader identities.</p>
<p class="sub mono">{diag_line}</p>
<table><thead><tr><th>cell</th><th>kind</th><th>train S</th><th>val S</th>
<th>test S</th><th>full net</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Headline <span class="mono">{head.get('code', '—')}</span>:
MC null p(Sharpe) {mc.get('p_sharpe', float('nan')):.3f} ·
Harvey t {head.get('t_stat', float('nan')):.2f} ·
DSR ladder {ladder or '—'} · {fac_line}</p>
<p>Rank IC ({ic_bits or 'n/a'}) — a daily-horizon signal must show decay
here; flat ≈0 IC at both horizons means the graph carries no forecast at
tradeable rebalance frequencies.</p>
<div class="callout"><b>Verdict:</b> <span class="{cls}">{text}</span></div>
<details><summary>Method & references</summary>
<p>Lagged correlation C<sub>ℓ</sub>[j,i] = corr(r<sub>j</sub>(t−ℓ), r<sub>i</sub>(t))
on cross-sectionally demeaned log returns (the demeaning kills an
autocorrelated common factor masquerading as pairwise lead-lag). Edge
threshold ρ* = 99.9th percentile of |entries| under per-column circular
shifts (own autocorrelation preserved, cross-links destroyed). Score:
s<sub>i</sub> = Σ w<sub>j→i</sub> R<sub>j</sub><sup>recent</sup> / Σ|w| —
no in-edges ⇒ no score, never a momentum fallback. Stale-price defenses:
activity gate (≥60% nonzero returns), liquidity cap, t+1 execution with
per-name slippage. References: Lo &amp; MacKinlay (1990, RFS); Curme,
Tumminello, Mantegna, Stanley, Kenett (2015, Quant. Finance); Cohen &amp;
Frazzini (2008, JF); Hou (2007, RFS).</p></details>"""


def sec_corr(d: dict, public: bool) -> str:
    if d.get("corr") is None:
        return ""
    return "<h2>Correlation structure</h2>"


def sec_phase(d: dict, public: bool) -> str:
    if d.get("phase") is None:
        return ""
    return "<h2>Phase-transition indicators</h2>"


def sec_events(d: dict, public: bool) -> str:
    if d.get("events") is None:
        return ""
    return "<h2>Regulatory events</h2>"


def sec_trials(d: dict, public: bool) -> str:
    if d.get("trials") is None:
        return ""
    return "<h2>Trial ledger</h2>"


def sec_footer() -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f'<p class="footnote">Built {ts} · local-only research page</p>'


def build(d: dict, public: bool = False) -> str:
    body = "".join([
        sec_intro(),
        sec_leadlag(d, public),
        sec_corr(d, public),
        sec_phase(d, public),
        sec_events(d, public),
        sec_trials(d, public),
        sec_footer(),
    ])
    return page("Econophysics lab", body)


def main() -> None:
    d = gather()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(d), encoding="utf-8")
    print(f"wrote {OUT}")
    if os.environ.get("ECONO_OPEN"):
        webbrowser.open(OUT.as_uri())


if __name__ == "__main__":
    main()
