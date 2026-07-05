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
from tools.leadlag import (leadlag_scores, network_universe, rank_ic,
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
CORR_N_CLUSTERS = 20          # M2: statistical clusters per rebalance
PH_TARGET_VOL = 0.15          # M3: vol-target baseline (strategy-page RC twin)
PH_TURN_BPS = 25.0            # M3: resize cost charged to BOTH overlays
PHANTOM_MULTS = (1, 5, 10)    # file-drawer ladder (strategy-page convention)
FACTORS = os.environ.get("FACTORS", "1") != "0"


def _enabled(name: str) -> bool:
    """Module env switch, FACTORS-style: ECONO_<NAME>=0 disables, default on."""
    return os.environ.get(f"ECONO_{name.upper()}", "1") != "0"


# ── Data gathering ────────────────────────────────────────────────────────────

def _factor_span(equity: pd.Series, returns: pd.Series | None = None) -> dict | None:
    """FF5+WML spanning alpha of an equity curve (or a raw daily-return
    series) — the page's shared 'real alpha' yardstick. None on any failure
    or when FACTORS=0 (French outage never breaks the page)."""
    if not FACTORS:
        return None
    try:
        fac, fac_src = factors.fetch_factors_daily()
        fx = cached_price_history(["EURUSD=X"], period="9y")["EURUSD=X"].dropna()
        r = returns if returns is not None else equity.pct_change().dropna()
        r_usd = factors.to_usd(r, fx)
        excess = (r_usd - fac["RF"].reindex(r_usd.index)).dropna()
        reg = factors.factor_regression(excess, fac)
        if not reg:
            return None
        key = "FF5+WML" if "FF5+WML" in reg else next(iter(reg))
        return dict(model=key, alpha_ann=reg[key]["alpha_ann"],
                    alpha_t=reg[key]["alpha_t"], n=reg[key]["n"],
                    source=fac_src)
    except Exception as e:
        print(f"[econo] factor regression skipped: {e}", file=sys.stderr)
        return None


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
    # Matched null: random books drawn from the GRAPH MEMBERS, not the full
    # pool. The signal can only ever hold graph members, so the full-pool null
    # conflates the membership tilt (top-N liquidity screen) with selection
    # skill — the matched null isolates selection. Both are reported.
    members = {d: set(network_universe(prices, turnover, d,
                                       elig.get(d, set()), top_n=LL_TOP_N))
               for d in rb}
    pools_m = sig.period_pools(prices, rb, members, execute_lag=EXEC_LAG)
    mc_matched = sig.monte_carlo_null(pools_m, strat_rets, k=LL_K, ppy=ppy,
                                      n_trials=1000, seed=0)

    def _ew(ps):
        x = np.array([float(np.mean(p)) for p in ps if len(p) > 10])
        if len(x) < 2 or x.std(ddof=1) == 0:
            return dict(total=float("nan"), sharpe=float("nan"), n=len(x))
        return dict(total=float(np.prod(1 + x) - 1),
                    sharpe=float(x.mean() / x.std(ddof=1) * np.sqrt(ppy)),
                    n=len(x))

    ew = dict(graph=_ew(pools_m), pool=_ew(pools))
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

    factor = _factor_span(hr["runs"][1.0]["equity"])

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
                              mc_matched=mc_matched, ew=ew,
                              dsr_phantom=dsr_phantom, t_stat=t_stat,
                              factor=factor),
                ic=ic, diag=diag, n_trials=len(cells))


def _gather_corr(u: dict) -> dict:
    """M2: does statistical clustering beat GICS for book construction?
    Same plain-momentum scores three ways — GICS-neutral, cluster-neutral,
    no grouping — judged on val+test Sharpe and PCA effective bets. Plus the
    canonical RMT read of the latest correlation matrix."""
    from tools import quant_grade as qg
    from tools.corr_struct import (cluster_maps_by_date, mp_clip, rand_index)
    from tools.momentum import precompute_scores
    prices, slip, turnover, pit, sectors = (u["prices"], u["slip"],
                                            u["turnover"], u["pit"],
                                            u["sectors"])
    cutoff = pd.Timestamp(START)
    cand = [d for d in rebalance_dates(prices.index, "M")
            if len(prices.loc[:d]) >= LOOKBACK + 1 and d >= cutoff]
    elig = precompute_eligibility(prices, slip, cand, liq_max=LIQ_MAX,
                                  min_obs=LOOKBACK + SKIP, min_price=MIN_PRICE,
                                  pit=pit, turnover=turnover,
                                  turn_floor=MIN_TURNOVER)
    scores = precompute_scores(prices, cand, LOOKBACK, SKIP)
    # trailing-window PIT clusters, capped to the liquid graph universe so the
    # matrix is estimable (T=252 vs N≤300)
    members = {d: set(network_universe(prices, turnover, d,
                                       elig.get(d, set()), top_n=LL_TOP_N))
               for d in cand}
    cl_maps = cluster_maps_by_date(prices, cand, members, window=LOOKBACK,
                                   n_clusters=CORR_N_CLUSTERS)
    te, ve = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)

    def _run(sector_neutral, by_date, code):
        r = run_momentum(prices, slip, k=LL_K, lookback=LOOKBACK, skip=SKIP,
                         capital=CAPITAL, cost_mults=(1.0,), freq="Q",
                         liq_max=LIQ_MAX, fee_eur=FEE_EUR, min_price=MIN_PRICE,
                         start=START, sectors=sectors,
                         sector_neutral=sector_neutral, lazy=True, pit=pit,
                         execute_lag=EXEC_LAG, turnover=turnover,
                         turn_floor=MIN_TURNOVER, elig_by_date=elig,
                         score_by_date=scores, sectors_by_date=by_date)
        eq, tr = r["runs"][1.0]["equity"], r["runs"][1.0]["trades"]
        ebs = []
        for h in r["holdings_log"]:
            picks = [t for t in h["picks"] if t in prices.columns]
            if len(picks) < 2:
                continue
            win = prices[picks].loc[:h["date"]].tail(LOOKBACK) \
                .pct_change().dropna(how="all")
            m = qg.effective_bets(win, np.full(len(picks), 1.0 / len(picks)))
            if m:
                ebs.append(m["n_eff_pca"])
        return dict(code=code,
                    train=_stats_slice(eq, tr, eq.index[0], te, CAPITAL),
                    val=_stats_slice(eq, tr, te + pd.Timedelta(days=1), ve,
                                     CAPITAL),
                    test=_stats_slice(eq, tr, ve + pd.Timedelta(days=1),
                                      eq.index[-1], CAPITAL),
                    full=r["runs"][1.0]["stats"],
                    eff_bets=float(np.mean(ebs)) if ebs else float("nan"))

    variants = [_run(True, None, "GICS-neutral"),
                _run(True, cl_maps, "cluster-neutral"),
                _run(False, None, "no-grouping")]

    # RMT read + GICS agreement on the latest window
    d_last = cand[-1]
    cols = sorted(members[d_last])
    w = prices.loc[:d_last, cols].tail(LOOKBACK + 1)
    r_ = w.pct_change()
    keep = list(r_.columns[r_.notna().sum() >= 126])
    corr = r_[keep].corr()
    _, info = mp_clip(corr.to_numpy(), T=len(r_))
    gics_last = {t: sectors.get(t) for t in keep if sectors.get(t)}
    rand = rand_index(cl_maps.get(d_last, {}), gics_last)
    return dict(rmt=dict(lambda_plus=info["lambda_plus"],
                         n_signal_eigs=info["n_signal_eigs"],
                         market_share=info["var_explained_market"],
                         n_names=len(keep), T=int(len(r_))),
                rand_gics=float(rand), n_clusters=CORR_N_CLUSTERS,
                variants=variants, n_trials=2)


def _gather_phase(u: dict) -> dict:
    """M3: phase-transition stress gauges (observational) + the AR-throttle
    vs vol-target head-to-head on the same plain-momentum book. Promotion
    (and its 4 trials) only if the throttle beats vol targeting on val AND
    test Sharpe with no worse drawdown."""
    from tools import quant_grade as qg
    from tools.phase_transition import (absorption_ratio, ar_throttle,
                                        breadth, susceptibility)
    prices, slip, turnover, pit, sectors = (u["prices"], u["slip"],
                                            u["turnover"], u["pit"],
                                            u["sectors"])
    cutoff = pd.Timestamp(START)
    cand = [d for d in rebalance_dates(prices.index, "M")
            if len(prices.loc[:d]) >= LOOKBACK + 1 and d >= cutoff]
    elig = precompute_eligibility(prices, slip, cand, liq_max=LIQ_MAX,
                                  min_obs=LOOKBACK + SKIP, min_price=MIN_PRICE,
                                  pit=pit, turnover=turnover,
                                  turn_floor=MIN_TURNOVER)
    r = run_momentum(prices, slip, k=LL_K, lookback=LOOKBACK, skip=SKIP,
                     capital=CAPITAL, cost_mults=(1.0,), freq="Q",
                     liq_max=LIQ_MAX, fee_eur=FEE_EUR, min_price=MIN_PRICE,
                     start=START, sectors=sectors, sector_neutral=True,
                     lazy=True, pit=pit, execute_lag=EXEC_LAG,
                     turnover=turnover, turn_floor=MIN_TURNOVER,
                     elig_by_date=elig)
    eq = r["runs"][1.0]["equity"]

    m = breadth(prices)
    chi = susceptibility(m)
    ar_df = absorption_ratio(prices, turnover=turnover, top_n=LL_TOP_N)
    vt = qg.vol_target(eq, target_vol=PH_TARGET_VOL, turn_cost_bps=PH_TURN_BPS)
    art = ar_throttle(eq, ar_df["ar"], turn_cost_bps=PH_TURN_BPS)

    te, ve = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)

    def _row(code, curve, expo):
        val = _stats_slice(curve, [], te + pd.Timedelta(days=1), ve, CAPITAL)
        test = _stats_slice(curve, [], ve + pd.Timedelta(days=1),
                            curve.index[-1], CAPITAL)
        dd = qg.perf_metrics(curve).get("max_dd", float("nan"))
        return dict(code=code, val=val["sharpe"], test=test["sharpe"],
                    dd=dd, expo=expo)

    rows = [_row("raw book", eq, 1.0)]
    if vt:
        rows.append(_row(f"vol-target {PH_TARGET_VOL:.0%}", vt["equity"],
                         vt["avg_exposure"]))
    if art.get("equity") is not None and len(art["equity"]):
        rows.append(_row("AR-throttle", art["equity"], art["avg_exposure"]))
    promoted = (len(rows) == 3
                and rows[2]["val"] > rows[1]["val"]
                and rows[2]["test"] > rows[1]["test"]
                and rows[2]["dd"] >= rows[1]["dd"])
    a = ar_df["ar"].dropna()
    c = chi.dropna()
    return dict(rows=rows, promoted=bool(promoted),
                n_trials=4 if promoted else 0,
                ar_now=float(a.iloc[-1]) if len(a) else float("nan"),
                ar_pct=float((a <= a.iloc[-1]).mean()) if len(a) else float("nan"),
                chi_pct=float((c <= c.iloc[-1]).mean()) if len(c) else float("nan"))


SHORT_STORE = ROOT / "data" / "universe" / "short_positions.csv"
DD_STORE = ROOT / "data" / "universe" / "insider_dealings.csv"
DD_MIN_ROWS = 200            # insider cells wait for the backfill
CROWDED_AVOID = 2.0          # eligibility subtraction: total disclosed short ≥ 2%


def _gather_events(u: dict) -> dict:
    """M4: regulatory-event signals through the production gate + the event
    study as primary evidence. Short-register cells always run (history is
    complete); insider cells and CARs engage once the DD backfill passes
    DD_MIN_ROWS. MC nulls are MATCHED (drawn from each signal's own candidate
    set) — the M1 lesson, baked in."""
    from tools.event_signal import (insider_score_by_date, interact_small,
                                    pit_slice, short_pressure_by_date)
    from tools.event_study import calendar_time_portfolio, car_stats, \
        market_model_ar
    from tools.momentum import precompute_scores
    prices, slip, turnover, pit, sectors = (u["prices"], u["slip"],
                                            u["turnover"], u["pit"],
                                            u["sectors"])
    meta_df = u["meta_df"]
    shorts = pd.read_csv(SHORT_STORE, dtype={"pct": float}) \
        if SHORT_STORE.exists() else None
    if shorts is None or not len(shorts):
        raise RuntimeError("short_positions.csv missing — run tools.short_register")
    dd = pd.read_csv(DD_STORE, dtype=str) if DD_STORE.exists() else None
    insider_ready = dd is not None and len(dd) >= DD_MIN_ROWS
    isin_ticker = {str(r["isin"]): r["ticker"] for _, r in meta_df.iterrows()
                   if pd.notna(r.get("isin"))}

    cutoff = pd.Timestamp(START)
    cand = [d for d in rebalance_dates(prices.index, "M")
            if len(prices.loc[:d]) >= LOOKBACK + 1 and d >= cutoff]
    elig = precompute_eligibility(prices, slip, cand, liq_max=LIQ_MAX,
                                  min_obs=LOOKBACK + SKIP, min_price=MIN_PRICE,
                                  pit=pit, turnover=turnover,
                                  turn_floor=MIN_TURNOVER)
    te, ve = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)

    def _run(sbd, freq, code, elig_override=None):
        r = run_momentum(prices, slip, k=LL_K, lookback=LOOKBACK, skip=SKIP,
                         capital=CAPITAL, cost_mults=(1.0,), freq=freq,
                         liq_max=LIQ_MAX, fee_eur=FEE_EUR, min_price=MIN_PRICE,
                         start=START, lazy=True, pit=pit, execute_lag=EXEC_LAG,
                         turnover=turnover, turn_floor=MIN_TURNOVER,
                         elig_by_date=elig_override or elig, score_by_date=sbd)
        eq, tr = r["runs"][1.0]["equity"], r["runs"][1.0]["trades"]
        return dict(code=code, kind="signal", freq=freq,
                    train=_stats_slice(eq, tr, eq.index[0], te, CAPITAL),
                    val=_stats_slice(eq, tr, te + pd.Timedelta(days=1), ve,
                                     CAPITAL),
                    test=_stats_slice(eq, tr, ve + pd.Timedelta(days=1),
                                      eq.index[-1], CAPITAL),
                    full=r["runs"][1.0]["stats"]), r

    cells, runs = [], {}
    sp = short_pressure_by_date(shorts, cand, isin_ticker, delta_days=21)
    for freq in ("M", "Q"):
        cell, r = _run(sp["covering"], freq, f"cover-{freq}")
        cells.append(cell)
        runs[cell["code"]] = r
    # crowded-avoid: plain momentum, names with heavy disclosed shorts cut
    mom_scores = precompute_scores(prices, cand, LOOKBACK, SKIP)
    elig_avoid = {}
    for d in cand:
        hot = set(sp["crowded"][d]["raw"]
                  [sp["crowded"][d]["raw"] >= CROWDED_AVOID].index)
        elig_avoid[d] = elig[d] - hot
    for freq in ("M", "Q"):
        cell, r = _run(mom_scores, freq, f"avoid-{freq}",
                       elig_override=elig_avoid)
        cells.append(cell)
        runs[cell["code"]] = r
    if insider_ready:
        ins = insider_score_by_date(dd, cand, isin_ticker, window_days=90,
                                    halflife=30)
        ins_small = {d: {"raw": interact_small(ins[d]["raw"], turnover, d),
                         "voladj": interact_small(ins[d]["raw"], turnover, d)}
                     for d in cand}
        for code, sbd in (("insider-Q", ins), ("insider-small-Q", ins_small)):
            cell, r = _run(sbd, "Q", code)
            cells.append(cell)
            runs[cell["code"]] = r

    head_cell = max(cells, key=lambda c: min(c["train"]["sharpe"],
                                             c["val"]["sharpe"]))
    hr = runs[head_cell["code"]]
    hl = hr["holdings_log"]
    headline: dict = dict(code=head_cell["code"])
    if hl:
        rb = [h["date"] for h in hl] + [hl[-1]["next"]]
        sig_scores = (sp["covering"] if head_cell["code"].startswith("cover")
                      else mom_scores)
        matched = {d: set(sig_scores.get(d, {}).get("raw",
                          pd.Series(dtype=float)).index) & elig.get(d, set())
                   for d in rb}
        pools_m = sig.period_pools(prices, rb, matched, execute_lag=EXEC_LAG)
        strat_rets = sig.strategy_period_returns(hl)
        ppy = 4.0 if head_cell["freq"] == "Q" else 12.0
        n_total = N_INHERITED_TRIALS + 10 + 2 + len(cells)
        trial_sharpes = [c["full"]["sharpe"] for c in cells]
        headline.update(
            mc_matched=sig.monte_carlo_null(pools_m, strat_rets, k=LL_K,
                                            ppy=ppy, n_trials=1000, seed=0),
            dsr_phantom=[dict(mult=m, n=n_total * m,
                              dsr=sig.deflated_sharpe_ratio(
                                  strat_rets, trial_sharpes, ppy=ppy,
                                  n_trials_effective=n_total * m)["dsr"])
                         for m in PHANTOM_MULTS],
            factor=_factor_span(hr["runs"][1.0]["equity"]))

    # Event studies — primary evidence. Day 0 = publication.
    spx_like = prices.mean(axis=1)          # EW universe as the market proxy
    exits = shorts[shorts["pct"] < 0.5].copy()
    exits["ticker"] = exits["isin"].map(isin_ticker)
    exits = exits.dropna(subset=["ticker"])
    # day 0 = publication; explicit frame — the stores carry their own
    # event_date column, renaming published_at would duplicate it
    ev_cov = pd.DataFrame({"ticker": exits["ticker"],
                           "event_date": exits["published_at"]})
    car_covering = car_stats(market_model_ar(prices, spx_like, ev_cov))
    out = dict(cells=cells, headline=headline, car_covering=car_covering,
               insider_ready=insider_ready, n_trials=len(cells))
    if insider_ready:
        buys = dd[dd["side"] == "buy"].copy()
        buys["ticker"] = buys["isin"].map(isin_ticker)
        buys = buys.dropna(subset=["ticker"])
        evb = pd.DataFrame({"ticker": buys["ticker"],
                            "event_date": buys["published_at"]})
        out["car_insider"] = car_stats(market_model_ar(prices, spx_like, evb))
        # capacity thesis: small (bottom-turnover-tercile) vs big
        if turnover is not None and len(evb):
            med = turnover.tail(6).median()
            cut_lo = med.quantile(1 / 3)
            small = evb[evb["ticker"].map(med) <= cut_lo]
            big = evb[evb["ticker"].map(med) > cut_lo]
            out["car_insider_small"] = car_stats(
                market_model_ar(prices, spx_like, small))
            out["car_insider_big"] = car_stats(
                market_model_ar(prices, spx_like, big))
        ct = calendar_time_portfolio(prices,
                                     buys[["ticker", "published_at"]],
                                     hold_days=21)
        out["ct_alpha"] = _factor_span(pd.Series(dtype=float), returns=ct)
    return out


def _gather_trials(d: dict) -> dict:
    """Program-level multiple-testing ledger, derived from whatever the
    other modules produced THIS build. N only ever grows across the program;
    killed modules' cells stay counted — that's the file drawer, made
    visible. Pure: no data access."""
    rows = [dict(module="inherited 64-config grid (strategy page)",
                 cells=N_INHERITED_TRIALS, status="baseline search space")]

    def _status(key):
        m = d.get(key)
        if m is None:
            return None, "not run this build"
        if key == "leadlag":
            cls, _ = _verdict_leadlag(m)
            return m.get("n_trials", 0), \
                ("killed — cells stay in N" if cls == "neg" else "candidate")
        if key == "corr":
            return m.get("n_trials", 0), "construction — not promoted"
        if key == "phase":
            return m.get("n_trials", 0), \
                ("promoted (+4 trials)" if m.get("promoted")
                 else "observational (0 trials until promoted)")
        if key == "events":
            n = m.get("n_trials", 0)
            note = "" if m.get("insider_ready") else "; insider cells pending"
            return n, f"running candidate{note}"
        return 0, "?"

    for key, label in (("leadlag", "M1 lead-lag network"),
                       ("corr", "M2 cluster-neutral construction"),
                       ("phase", "M3 AR-throttle"),
                       ("events", "M4 regulatory events")):
        n, status = _status(key)
        rows.append(dict(module=label, cells=n if n is not None else 0,
                         status=status))
    # sibling pages merged from claude/model-ytg40m — their searches count
    # here too (one program, one file drawer), verdicts live on their pages
    rows.append(dict(module="vol lab (build_vol_report, 7 forecasters)",
                     cells=7, status="pre-registered gates on own page"))
    rows.append(dict(module="edge stack (tax-loss sleeve + combiner)",
                     cells=2, status="pre-registered gates on own page"))
    rows.append(dict(module="strategy ensemble selection rule (top-3 Q)",
                     cells=1, status="pre-registered adoption rule"))
    total = sum(r["cells"] for r in rows)
    return dict(rows=rows, total=total, phantom=PHANTOM_MULTS)


def gather() -> dict:
    """Run every enabled module; a disabled or failed module leaves None.

    Each block is wrapped try/except so one broken experiment (or a missing
    data file) never takes the page down — the section simply doesn't render.
    """
    d: dict = {k: None for k in MODULES}
    modules_needing_data = ("leadlag", "corr", "phase", "events")
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
    if _enabled("corr"):
        try:
            d["corr"] = _gather_corr(u)
        except Exception as e:
            print(f"[econo] corr skipped: {e}", file=sys.stderr)
    if _enabled("phase"):
        try:
            d["phase"] = _gather_phase(u)
        except Exception as e:
            print(f"[econo] phase skipped: {e}", file=sys.stderr)
    if _enabled("events"):
        try:
            d["events"] = _gather_events(u)
        except Exception as e:
            print(f"[econo] events skipped: {e}", file=sys.stderr)
    d["trials"] = _gather_trials(d)
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
        mcm_p = (head.get("mc_matched") or {}).get("p_sharpe")
        if mcm_p is not None and mcm_p >= 0.2:
            return "neg", ("Killed — no selection information. The shuffled-"
                           "leader placebo matches the signal and the matched "
                           f"null (random books from the graph members) gives "
                           f"p≈{mcm_p:.2f}: the apparent profit is the "
                           "liquidity-membership screen plus universe beta, "
                           "not lead-lag. The full-pool null credited that "
                           "tilt as significance — lesson recorded.")
        return "neg", ("Pipeline leak suspected: the shuffled-leader placebo "
                       "matches or beats the signal. Halt — nothing here is "
                       "interpretable until the leak is explained.")
    dsr5 = next((x.get("dsr") for x in head.get("dsr_phantom", [])
                 if x.get("mult") == 5), None)
    # judge on the matched null when present — the full-pool null credits the
    # liquidity-membership tilt to the signal
    mcp = ((head.get("mc_matched") or head.get("mc")) or {}).get("p_sharpe")
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
    mcm = head.get("mc_matched") or {}
    ewd = head.get("ew") or {}
    ew_line = ""
    if ewd:
        g, p_ = ewd.get("graph", {}), ewd.get("pool", {})
        ew_line = (f"<p>Membership-tilt check: EW of graph members "
                   f"{g.get('total', float('nan')) * 100:+.1f}% "
                   f"(Sharpe {g.get('sharpe', float('nan')):.2f}, gross) vs EW full pool "
                   f"{p_.get('total', float('nan')) * 100:+.1f}% "
                   f"(Sharpe {p_.get('sharpe', float('nan')):.2f}). If the cells don't beat "
                   f"the members' EW, the 'signal' is the membership screen.</p>")
    return f"""<h2>Lead-lag network — neighbor momentum</h2>
<p class="sub">Directed lagged-correlation graph on the eligible, actively-
printing universe (top {LL_TOP_N} by PIT turnover); edges must clear a
circular-shift shuffle null; scores are leader-weighted recent returns.
Baseline = the known big→small channel; placebo = shuffled leader identities.</p>
<p class="sub mono">{diag_line}</p>
<table><thead><tr><th>cell</th><th>kind</th><th>train S</th><th>val S</th>
<th>test S</th><th>full net</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Headline <span class="mono">{head.get('code', '—')}</span>:
full-pool null p(Sharpe) {mc.get('p_sharpe', float('nan')):.3f} ·
<b>matched null (graph members) p(Sharpe)
{mcm.get('p_sharpe', float('nan')):.3f}</b> ·
Harvey t {head.get('t_stat', float('nan')):.2f} ·
DSR ladder {ladder or '—'} · {fac_line}</p>
{ew_line}
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
    c = d.get("corr")
    if c is None:
        return ""
    rmt = c.get("rmt") or {}
    rows = []
    for v in c.get("variants", []):
        rows.append(
            f"<tr><td class='mono'>{v['code']}</td>"
            f"<td class='mono'>{v['train']['sharpe']:.2f}</td>"
            f"<td class='mono'>{v['val']['sharpe']:.2f}</td>"
            f"<td class='mono'>{v['test']['sharpe']:.2f}</td>"
            f"<td>{_pct(v['full']['net_return'] * 100)}</td>"
            f"<td class='mono'>{v['eff_bets']:.2f}</td></tr>")
    by = {v["code"]: v for v in c.get("variants", [])}
    g, cl = by.get("GICS-neutral"), by.get("cluster-neutral")
    if g and cl:
        better = (cl["val"]["sharpe"] >= g["val"]["sharpe"]
                  and cl["test"]["sharpe"] >= g["test"]["sharpe"]
                  and cl["eff_bets"] > g["eff_bets"])
        vcls, vtext = (("pos", "Cluster-neutral construction beats GICS on "
                        "val AND test with more effective bets — promoted as "
                        "the book-construction default candidate (2 trials "
                        "charged to the ledger).") if better else
                       ("", "Cluster-neutral does not beat GICS-neutral out "
                        "of sample — statistical clustering stays a "
                        "diagnostic/teaching tool, GICS grouping keeps the "
                        "book. (Construction experiment, not alpha; 2 trials "
                        "charged.)"))
    else:
        vcls, vtext = "", "Comparison incomplete."
    return f"""<h2>Correlation structure — RMT + cluster-neutral books</h2>
<p class="sub">Same plain 12-1 momentum scores, three constructions: GICS
round-robin, trailing-correlation cluster round-robin ({c.get('n_clusters')}
clusters, average linkage, PIT windows), no grouping. Construction experiment
— the signal never changes.</p>
<p class="mono sub">RMT (latest window, {rmt.get('n_names', 0)} names ×
T={rmt.get('T', 0)}): λ+ = {rmt.get('lambda_plus', float('nan')):.2f} ·
signal eigenvalues {rmt.get('n_signal_eigs', 0)} ·
market mode {rmt.get('market_share', float('nan')) * 100:.0f}% of variance ·
Rand agreement clusters↔GICS {c.get('rand_gics', float('nan')):.2f}</p>
<table><thead><tr><th>construction</th><th>train S</th><th>val S</th>
<th>test S</th><th>full net</th><th>mean effective bets</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="callout"><b>Verdict:</b> <span class="{vcls}">{vtext}</span></div>
<details><summary>Method &amp; references</summary>
<p>Correlation matrices on T≈252, N≈300 are mostly noise: Marchenko–Pastur
bulk edge λ+ = (1+√(N/T))² separates signal eigenvalues from the random bulk
(Laloux, Cizeau, Bouchaud, Potters 1999; Plerou et al. 2002). Distance
d = √(2(1−ρ)) (Mantegna 1999); average-linkage hierarchical clustering — MST
is single linkage's skeleton and chains, so it stays a viz. Cluster-neutral
selection = the engine's sector round-robin fed per-rebalance statistical
clusters (López de Prado 2016 HRP is the weighted cousin; equal-weight top-k
is this harness's contract). Ledoit–Wolf (2004) shrinkage available in
tools/corr_struct.py for anything downstream needing an invertible Σ.</p>
</details>"""


def sec_phase(d: dict, public: bool) -> str:
    p = d.get("phase")
    if p is None:
        return ""
    rows = "".join(
        f"<tr><td>{r['code']}</td><td class='mono'>{r['val']:.2f}</td>"
        f"<td class='mono'>{r['test']:.2f}</td>"
        f"<td class='mono'>{r['dd'] * 100:.1f}%</td>"
        f"<td class='mono'>{r['expo']:.2f}</td></tr>"
        for r in p.get("rows", []))
    if p.get("promoted"):
        vcls, vtext = "pos", ("AR-throttle beats vol targeting on val AND "
                              "test with no worse drawdown — promoted; its 4 "
                              "threshold trials are charged to the ledger.")
    else:
        vcls, vtext = "", ("AR-throttle does not beat plain vol targeting "
                           "out of sample — stays observational (0 trials "
                           "charged), exactly the expected outcome for a "
                           "coincident stress gauge.")
    return f"""<h2>Phase-transition gauges — breadth, χ, absorption ratio</h2>
<p class="sub">Ising framing: breadth M(t) = magnetization, χ = Var(M) =
susceptibility, absorption ratio = eigenvalue concentration (Kritzman 2011).
Causal trailing windows; observational by construction — same culture pin as
the HMM (never imported by selection).</p>
<p class="mono sub">now: AR {p.get('ar_now', float('nan')):.2f}
(expanding percentile {p.get('ar_pct', float('nan')) * 100:.0f}%) ·
χ percentile {p.get('chi_pct', float('nan')) * 100:.0f}%</p>
<table><thead><tr><th>curve</th><th>val S</th><th>test S</th><th>maxDD</th>
<th>avg exposure</th></tr></thead><tbody>{rows}</tbody></table>
<div class="callout"><b>Verdict:</b> <span class="{vcls}">{vtext}</span></div>
<details><summary>Method &amp; references</summary>
<p>Exposure slides 1→0.3 between the EXPANDING-TRAILING 50th and 90th AR
percentiles (full-sample quantiles are the classic look-ahead in this
literature), applied at t+1 and charged {PH_TURN_BPS:.0f} bps on |Δexposure|
— the identical contract as the vol-target twin, so the comparison is fair.
References: Kritzman, Li, Page, Rigobon (2011, JPM); Preis, Kenett, Stanley,
Helbing, Ben-Jacob (2012, Sci Rep); Sornette (2003) — cited with the honest
caveat that critical-point CRASH PREDICTION has a weak out-of-sample record,
which is why this module is a risk gauge, not a signal.</p></details>"""


def _car_row(label: str, c: dict | None) -> str:
    if not c or not c.get("n"):
        return ""
    car = c.get("car", {})
    return (f"<tr><td>{label}</td>"
            f"<td class='mono'>{car.get(1, float('nan')) * 100:+.2f}%</td>"
            f"<td class='mono'>{car.get(5, float('nan')) * 100:+.2f}%</td>"
            f"<td class='mono'>{car.get(20, float('nan')) * 100:+.2f}%</td>"
            f"<td class='mono'>{c.get('bmp_t', float('nan')):.2f}</td>"
            f"<td class='mono'>{c.get('n', 0)}</td></tr>")


def sec_events(d: dict, public: bool) -> str:
    e = d.get("events")
    if e is None:
        return ""
    car_rows = "".join([
        _car_row("short-covering (exit filings)", e.get("car_covering")),
        _car_row("insider buys — all", e.get("car_insider")),
        _car_row("insider buys — small caps", e.get("car_insider_small")),
        _car_row("insider buys — large caps", e.get("car_insider_big"))])
    cells = "".join(
        f"<tr><td class='mono'>{c['code']}</td>"
        f"<td class='mono'>{c['train']['sharpe']:.2f}</td>"
        f"<td class='mono'>{c['val']['sharpe']:.2f}</td>"
        f"<td class='mono'>{c['test']['sharpe']:.2f}</td>"
        f"<td>{_pct(c['full']['net_return'] * 100)}</td></tr>"
        for c in e.get("cells", []))
    head = e.get("headline") or {}
    mcm = (head.get("mc_matched") or {})
    dsr5 = next((x.get("dsr") for x in head.get("dsr_phantom", [])
                 if x.get("mult") == 5), None)
    fac = head.get("factor")
    ct = e.get("ct_alpha")
    pending = ("" if e.get("insider_ready") else
               "<p class='sub'>Insider cells & CARs pending — the BaFin "
               "backfill is still accruing (store below threshold). "
               "Short-register results below are complete.</p>")
    ins_small = e.get("car_insider_small") or {}
    ins_big = e.get("car_insider_big") or {}
    cap_line = ""
    if ins_small.get("n") and ins_big.get("n"):
        cap_line = (f"<p>Capacity thesis: small-cap insider CAR(+20) "
                    f"{(ins_small['car'].get(20, 0)) * 100:+.2f}% "
                    f"(BMP t {ins_small['bmp_t']:.2f}) vs large-cap "
                    f"{(ins_big['car'].get(20, 0)) * 100:+.2f}% "
                    f"(t {ins_big['bmp_t']:.2f}) — the edge, if any, must "
                    f"live in the small bucket.</p>")
    bmp_ok = any((e.get(k) or {}).get("bmp_t", 0) >= 2.0
                 for k in ("car_covering", "car_insider", "car_insider_small"))
    mcp = mcm.get("p_sharpe")
    if bmp_ok and mcp is not None and mcp < 0.05 and (dsr5 or 0) > 0.5 \
            and fac and fac.get("alpha_t", 0) >= 2.0:
        vcls, vtext = "pos", ("Event evidence AND the strategy gate agree — "
                              "residual alpha candidate; keep collecting "
                              "live history.")
    elif bmp_ok:
        vcls, vtext = "", ("Event study shows a market reaction, but the "
                           "tradeable book doesn't clear the gate (null/DSR/"
                           "spanning) at these rebalance horizons — "
                           "observational; re-judge as live data accrues.")
    else:
        vcls, vtext = "neg", ("No significant post-publication reaction — "
                              "killed at current sample; negative result "
                              "stays on the page.")
    fac_line = (f"FF5+WML α {fac['alpha_ann'] * 100:+.1f}%/yr "
                f"(t {fac['alpha_t']:.2f})" if fac else "spanning n/a")
    ct_line = (f" · calendar-time insider-buy book: α "
               f"{ct['alpha_ann'] * 100:+.1f}%/yr (t {ct['alpha_t']:.2f}, "
               f"n {ct['n']})" if ct else "")
    return f"""<h2>Regulatory events — insider dealings &amp; short register</h2>
<p class="sub">New information, not new math: BaFin Directors' Dealings
(real publication timestamps) and the Bundesanzeiger net-short register
(publication imputed +1bd, flagged). PIT joins strictly published-before;
day 0 of every CAR = publication, entry t+1.</p>
{pending}
<table><thead><tr><th>event study (CAR)</th><th>+1d</th><th>+5d</th>
<th>+20d</th><th>BMP t</th><th>n</th></tr></thead><tbody>{car_rows}</tbody></table>
{cap_line}
<table><thead><tr><th>cell</th><th>train S</th><th>val S</th><th>test S</th>
<th>full net</th></tr></thead><tbody>{cells}</tbody></table>
<p>Headline <span class="mono">{head.get('code', '—')}</span>:
matched-null p(Sharpe) {mcm.get('p_sharpe', float('nan')):.3f} ·
DSR ×5 {dsr5 if dsr5 is not None else float('nan'):.2f} · {fac_line}{ct_line}</p>
<div class="callout"><b>Verdict:</b> <span class="{vcls}">{vtext}</span></div>
<details><summary>Method &amp; references</summary>
<p>Market-model event study (MacKinlay 1997) with strictly pre-event
estimation windows; BMP (1991) standardized t. Calendar-time portfolio
(entry t+1 after publication, 21d hold) regressed on FF5+WML — event
evidence and the strategy gate meet on the same yardstick. Insider signal:
signed half-life-decayed dealing counts (Lakonishok &amp; Lee 2001 — effect
concentrated in small names, hence the turnover-tercile split: the
capacity-edge test). Short register: crowded level (holders' latest pct,
&lt;0.5% exits drop) and covering delta (Jank, Roling, Smajlbegovic 2021 on
this exact register). DD portal discovery retains ~12 months; person pages
recover deeper history for active insiders — deep-backtest coverage bias
disclosed, fresh-window signals unbiased.</p></details>"""


def sec_trials(d: dict, public: bool) -> str:
    t = d.get("trials")
    if t is None:
        return ""
    rows = "".join(f"<tr><td>{r['module']}</td>"
                   f"<td class='mono'>{r['cells']}</td>"
                   f"<td>{r['status']}</td></tr>" for r in t.get("rows", []))
    total = t.get("total", 0)
    ladder = " · ".join(f"×{m}: N={total * m}" for m in t.get("phantom", ()))
    return f"""<h2>Trial ledger — the file drawer, made visible</h2>
<p class="sub">Every cell ever run counts forever, controls and kills
included; N only grows. Deflated-Sharpe verdicts across the program quote
the ×5 phantom row — the honest correction for the search you can't see.</p>
<table><thead><tr><th>module</th><th>cells</th><th>status</th></tr></thead>
<tbody>{rows}</tbody>
<tfoot><tr><td><b>program N</b></td><td class='mono'><b>{total}</b></td>
<td class='mono'>{ladder}</td></tr></tfoot></table>"""


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
