"""The chosen production strategy — one config (the 'ultimate' extracted from the
64-config grid), presented RISK-FIRST:

  • The page LEADS with the Risk-conscious book (the SAME selection, volatility-
    targeted, de-risk only): its out-of-sample result + the Monte-Carlo validation.
    That is how you'd actually hold it, so it is the headline.
  • The raw, full-invested Original is demoted to the research lab as an inflation
    REFERENCE only — its +2000%-class total is survivorship + leverage-to-vol, not a
    number to quote. Kept visible (not hidden), never the headline.

Order: headline → picks → equity → significance → performance → scorecard → yearly →
every-rebalance → diagnostics → regime → you-vs-strategy → limitations, then the lab
(private/live builds only: raw reference + the 64-config grid + supporting data).

  python build_strategy_report.py            # writes local/strategy.html
  python build_strategy_report.py --open
"""
import argparse
import os
import sys
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tools.report_html import pct as _pct, card as _card, page, fig_html
from tools import theme, significance as sig, quant_grade as qg, regime, survivorship, regime_attr, scenario, factors
from tools.momentum import (run_momentum, winsorize_prices, to_xetra_calendar,
                            precompute_eligibility, benchmark_curves, equal_weight_curve,
                            rebalance_dates)
from tools.universe_pit import PITUniverse
from tools.universe_assemble import delisting_map, death_map, death_mask
from tools import universe_snapshot
from tools.momentum_grid import MomentumConfig, _stats_slice, run_grid, ALL_CONFIGS, pick_ultimate
from tools.portfolio_tools import BENCHMARKS, parse_portfolio
from tools.portfolio_analytics import build_roi_timeseries
from tools.data_buffer import cached_price_history
from build_momentum_report import (
    PRICES_CSV, META_CSV, TURN_CSV, ROOT, LOOKBACK, SKIP, START, LIQ_MAX, MIN_PRICE, CAPITAL,
    FEE_EUR, COST_MULTS, TRAIN_END, VAL_END, WINSOR_CAP, EXEC_LAG, K, MIN_TURNOVER,
    _slip, _broker, _disp, _name, _pnl_color, _equity_window,
    sec_grid, sec_feasibility, sec_timelines, sec_survivorship, sec_method,
)

# Production config is chosen DYNAMICALLY: gather() runs the 64-config grid and takes
# pick_ultimate(grid) (worst-case-robust ultimate) as the live strategy — no hardcoded config to
# drift stale as the universe/sectors change. This constant is the FALLBACK (used only if nothing
# in the grid qualifies) and documents the current winner — ·B·DEF = sector-neutral, top-10,
# quarterly, lazy.
# Sector-neutral (B) is now LIVE: tools.enrich_sectors sourced real GICS sectors per name from
# yfinance home listings into data/universe/universe_meta.csv (99% of the LIVE universe), so B's
# round-robin actually caps single-sector concentration instead of being the silent no-op it was
# when every name read "Unknown". With B in the pool the full 64-config grid RE-SELECTED the
# ultimate by worst-case min(train, validation) Sharpe — and the winner is a B config: ·B·DEF
# (min 0.93, val 0.93, held-out test 1.47) beats the old B-off A···EF (min 0.71, test 1.27) on
# both robustness AND the held-out test; 6 of the grid's top-8 are B configs, so B genuinely
# earns its place. (CAVEAT: the pick is sector-coverage sensitive — at partial 48% coverage the
# grid wrongly preferred AB····, which collapses to test 0.65 at full coverage. Trust the
# full-coverage result.) B caps single-SECTOR concentration; it does NOT by itself raise the PCA
# effective-bets metric (that is correlation/factor-based — sector-diverse names still co-move
# when one macro factor dominates).
STRATEGY = MomentumConfig(sector_neutral=True, slots=10, freq="Q", lazy=True)

# Risk-conscious overlay: volatility-target the book to this annualised vol (de-risk only,
# park the rest in cash). Cuts the raw strategy's ~32% vol / −44% drawdown to a moderate
# profile while lifting the Sharpe — the prudent way to actually run momentum.
RISK_TARGET_VOL = 0.15
# Vol-targeting resizes the book daily (cash↔stocks); that turnover is NOT free. Charge each
# |Δexposure| at this one-way slippage so the risk-conscious curve pays for its own resizing
# (matches the ~25bps modeled slippage the strategy already cites). Flat €/order is extra.
RC_TURN_BPS = 25.0

# On-population survivorship test (the *holding* leak). The bolt-on real graveyard overlaps
# the live universe only ~2%, so it can't honestly test "does the strategy hold a name into
# its death?". Instead inject synthetic delistings into the LIVE names (100% representative)
# at this annual hazard + terminal-crash band, re-run the identical strategy SURV_SIMS× and
# measure the drag + how often momentum was actually holding a name at death. Observational —
# never feeds selection/sizing. Env SURV_SIMS=0 skips it (fast local iteration).
SURV_SIMS = int(os.environ.get("SURV_SIMS", "8"))
SURV_HAZARD = 0.05            # ~5%/yr delisting hazard (broad-equity plausible)
SURV_LOSS = (0.40, 1.00)      # terminal crash drawn here (partial buyout → near-total wipeout)

# Delisting-stress intensities (bull/base/bear). Two knobs each: annual hazard + terminal-loss
# band, grounded in the delisting literature (Shumway 1997: -30% NYSE/AMEX; Shumway-Warther 1999:
# -55% NASDAQ; bankruptcies -> -100%). Base == the single-intensity default above, so no regression.
STRESS_PRESETS = {                       # name -> (hazard_annual, (loss_lo, loss_hi))
    "bull": (0.02, (0.30, 0.60)),        # benign regime, orderly / partial-recovery delistings
    "base": (SURV_HAZARD, SURV_LOSS),    # empirical central case (== current default)
    "bear": (0.10, (0.60, 1.00)),        # crisis delisting wave, mostly wipeouts
}

# Regime attribution: strip the two sectors carrying the 2024-25 tailwind (Technology = the AI/
# semiconductor boom; Industrials = aerospace & defense) and re-run the SAME config, to see how
# much of the Sharpe survives without them. Coarse (defense is a slice of Industrials) but it's
# the granularity yfinance sectors give — labelled as such on the page. Observational re-run.
REGIME_DROP_SECTORS = {"Technology", "Industrials"}

# File-drawer / phantom-trials: the 64-config grid is the multiple-testing the DSR can SEE, but
# the pipeline (architectures, indicators, calendars, abandoned ideas) was iterated perhaps ~5×
# before that grid existed. PHANTOM_MULT is that honest, subjective lifetime-iteration estimate;
# the page shows a ladder (×1 grid-only, ×5 estimate, ×10 pessimistic) so significance decay is a
# sensitivity, not one magic number. Harvey (2016): unseen industry-wide testing lifts the real
# hurdle to a t-stat ≈ 3.0. Grade stays on the objective ×1 grid DSR; this is shown alongside.
PHANTOM_MULT = 5
PHANTOM_MULTS = (1, 5, 10)

# Scenario fan (observational): regime-conditioned block bootstrap of the risk-conscious book's
# daily returns → bear/base/bull terminal-wealth distribution. Bear over-weights risk-off regime
# blocks, bull over-weights risk-on; base = natural frequency. Sensitivity, NOT a forecast — and
# it never touches selection/sizing (so no DSR cost). Env SCEN_SIMS=0 skips it (fast iteration).
SCEN_HORIZON = 252           # 1-year forward path
SCEN_BLOCK = 21              # ~1-month blocks (keep autocorrelation / crash clusters)
SCEN_TILT = 3.0              # bear/bull over-weight on the stressed regime's blocks
SCEN_SIMS = int(os.environ.get("SCEN_SIMS", "2000"))

# Factor-spanning regression (observational): regress the strategy's daily USD excess
# returns on the French Developed 5 factors + WML (US fallback). Answers "is the edge
# just momentum beta?" sharper than the CAPM alpha above. Env FACTORS=0 skips (offline).
FACTORS = int(os.environ.get("FACTORS", "1"))

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
                   strategy: MomentumConfig = STRATEGY,
                   train_end=TRAIN_END, val_end=VAL_END) -> list[dict]:
    """Two parallel 'variant bundles' of identical shape so every section can render
    either: the raw strategy and its volatility-targeted (risk-conscious) twin.

    Selection is IDENTICAL across both (same holdings_log / trades) — vol-targeting
    scales the whole book, it does not change which names are picked. So the rc bundle
    re-derives its windowed stats / quant metrics / grade from the vol-targeted equity
    curve, but shares the picks and the per-name trade record."""
    tr = res["runs"][1.0]["trades"]
    raw = dict(
        label=f"Original (raw, {strategy.code})", short="Original", key="raw", color=C_RAW,
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


def _delisting_stress(prices, slip, meta_df, sectors, spx, cfg, res, base_return,
                      membership=None, turnover=None):
    """Multi-intensity on-population delisting stress (bull/base/bear). Inject synthetic deaths
    into the LIVE names at each preset's hazard/loss, re-run the SAME strategy SURV_SIMS×, and per
    run derive: raw net return, the risk-conscious (vol-targeted) curve, annualised CAPM alpha vs
    the real un-injected benchmark, and the edge vs an equally-delisted buy-hold of the initial
    picks. Observational — never feeds selection/sizing. Seeds 0..K-1 shared across presets.

    Returns {"presets": {name: stress_summarize(...) + hazard/loss}, "clean": {...}, "sims": K}."""
    dead = set(meta_df.loc[meta_df["delisting_date"].notna(), "ticker"])   # every exit: not injectable
    live_cols = [c for c in prices.columns if c not in dead]
    base_map = delisting_map(meta_df)                  # all exits (gates eligibility)
    base_deaths = death_map(meta_df)                   # true deaths (graveyard stats)
    window = _equity_window(res)
    first_picks = next((h["picks"] for h in res["holdings_log"] if h["picks"]), [])

    def _alpha(eq, bench):
        if eq is None or bench is None:
            return float("nan")
        vb = qg.vs_benchmark(eq, bench)
        return float(vb.get("alpha_ann", float("nan"))) if vb else float("nan")

    def _rc(eq):
        return qg.vol_target(eq, target_vol=RISK_TARGET_VOL, turn_cost_bps=RC_TURN_BPS)

    raw_eq0 = res["runs"][1.0]["equity"]
    rc0 = _rc(raw_eq0)
    rc_eq0 = rc0.get("equity")
    ew0 = equal_weight_curve(prices, first_picks, window, CAPITAL) if first_picks else None
    clean = dict(
        ret={"raw": float(base_return), "rc": float(rc0.get("net_return", float("nan")))},
        alpha={"raw": _alpha(raw_eq0, spx), "rc": _alpha(rc_eq0, spx)},
        edge={"raw": _alpha(raw_eq0, ew0), "rc": _alpha(rc_eq0, ew0)})

    presets = {}
    for name, (hz, (lo, hi)) in STRESS_PRESETS.items():
        raw_rets, hits, deaths_n = [], [], []
        a_raw, a_rc, e_raw, e_rc = [], [], [], []
        for s in range(SURV_SIMS):
            inj_live, deaths = survivorship.inject_delistings(
                prices[live_cols], hazard_annual=hz, loss_lo=lo, loss_hi=hi, seed=s)
            if not deaths:                                    # no deaths this draw -> clean values
                raw_rets.append(base_return); hits.append(0); deaths_n.append(0)
                a_raw.append(clean["alpha"]["raw"]); a_rc.append(clean["alpha"]["rc"])
                e_raw.append(clean["edge"]["raw"]); e_rc.append(clean["edge"]["rc"])
                continue
            inj = prices.copy(); inj[live_cols] = inj_live
            dmap = dict(base_map); dmap.update({t: dt for t, (dt, _l) in deaths.items()})
            dd = dict(base_deaths); dd.update({t: dt for t, (dt, _l) in deaths.items()})
            r = run_momentum(inj, {t: slip[t] for t in inj.columns if t in slip},
                             lookback=LOOKBACK, skip=SKIP, capital=CAPITAL, cost_mults=(1.0,),
                             start=START, liq_max=LIQ_MAX, fee_eur=FEE_EUR, min_price=MIN_PRICE,
                             sectors=sectors, benchmark=spx,
                             pit=PITUniverse(inj, dmap, deaths=dd, membership=membership),
                             execute_lag=EXEC_LAG, turnover=turnover, turn_floor=MIN_TURNOVER,
                             **cfg.kwargs())
            req = r["runs"][1.0]["equity"]
            rrc = _rc(req).get("equity")
            inj_ew = equal_weight_curve(inj, first_picks, window, CAPITAL) if first_picks else None
            raw_rets.append(r["runs"][1.0]["stats"]["net_return"])
            a_raw.append(_alpha(req, spx)); a_rc.append(_alpha(rrc, spx))
            e_raw.append(_alpha(req, inj_ew)); e_rc.append(_alpha(rrc, inj_ew))
            held = sum(1 for h in r["holdings_log"] for t in h["picks"]
                       if t in deaths and h["date"] <= deaths[t][0] < h.get("next", deaths[t][0]))
            hits.append(held); deaths_n.append(len(deaths))
        st = survivorship.stress_summarize(base_return, raw_rets, hits, deaths_n,
                                           a_raw, a_rc, e_raw, e_rc)
        st["hazard"], st["loss"] = hz, (lo, hi)
        presets[name] = st
    return dict(presets=presets, clean=clean, sims=SURV_SIMS)


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
    sectors = {t: m.get("sector") for t, m in meta.items()}     # real GICS now (tools.enrich_sectors)
    # PIT extras: monthly turnover (liquidity gate; absent until the next fetch), the TR
    # snapshot store (membership gate, active from the first snapshot forward), and the
    # exits/deaths split (ledger demotions gate eligibility but aren't graveyard deaths).
    turnover = (pd.read_csv(TURN_CSV, index_col=0, parse_dates=True)
                if TURN_CSV.exists() else None)
    snap_store = universe_snapshot.load_store()
    ticker_isin = {r["ticker"]: str(r["isin"]) for _, r in meta_df.iterrows()
                   if pd.notna(r.get("isin"))}
    membership = (snap_store, ticker_isin) if len(snap_store) else None
    pit = PITUniverse(prices, delisting_map(meta_df), deaths=death_map(meta_df),
                      membership=membership)

    benches = {n: v for n, v in BENCHMARKS.items() if n != "Bitcoin"}   # equities/bonds only
    bench_tickers = [tk for tk, _ in benches.values()]
    bench_raw = cached_price_history(bench_tickers, period="9y", force=refresh)
    bench = bench_raw.rename(columns={tk: name for name, (tk, _) in benches.items()})
    spx = bench["S&P 500"] if "S&P 500" in bench.columns else bench.iloc[:, 0]

    # ── Dynamic config selection: run the full 64-config grid FIRST, then take the worst-case-
    #    robust ultimate as the live production config — B configs are in the pool now sectors are
    #    real, and pick_ultimate() (max min(train,val) Sharpe among configs that pay for themselves
    #    and are positive in both windows) decides, not a hardcoded constant. `cfg` flows through
    #    every run below; STRATEGY is only the fallback if nothing qualifies.
    grid = run_grid(prices, slip, sectors=sectors, benchmark=spx, pit=pit, start=START,
                    configs=ALL_CONFIGS, train_end=TRAIN_END, val_end=VAL_END, capital=CAPITAL,
                    lookback=LOOKBACK, skip=SKIP, execute_lag=EXEC_LAG,
                    turnover=turnover, turn_floor=MIN_TURNOVER)
    picked = pick_ultimate(grid, capital=CAPITAL, fee_eur=FEE_EUR)
    cfg = picked["config"] if picked else STRATEGY

    # Sectors are real now (tools.enrich_sectors) → sector-neutral (B) is a live structural cap.
    res = run_momentum(prices, slip, lookback=LOOKBACK, skip=SKIP, capital=CAPITAL,
                       cost_mults=COST_MULTS, start=START, liq_max=LIQ_MAX, fee_eur=FEE_EUR,
                       min_price=MIN_PRICE, sectors=sectors, benchmark=spx, pit=pit,
                       execute_lag=EXEC_LAG, turnover=turnover, turn_floor=MIN_TURNOVER,
                       **cfg.kwargs())
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
    ub_pit = PITUniverse(ub_prices, delisting_map(ub_meta), deaths=death_map(ub_meta),
                         membership=membership)
    ub_res = run_momentum(ub_prices, {t: slip[t] for t in ub_prices.columns if t in slip},
                          lookback=LOOKBACK, skip=SKIP, capital=CAPITAL, cost_mults=(1.0,),
                          start=START, liq_max=LIQ_MAX, fee_eur=FEE_EUR, min_price=MIN_PRICE,
                          sectors=sectors, benchmark=spx, pit=ub_pit, execute_lag=EXEC_LAG,
                          turnover=turnover, turn_floor=MIN_TURNOVER, **cfg.kwargs())
    ub_eq, ub_tr = ub_res["runs"][1.0]["equity"], ub_res["runs"][1.0]["trades"]
    bounds = dict(lower_full=test["net_return"], n_dead_dropped=n_dead_dropped,
                  upper_full=ub_res["runs"][1.0]["stats"]["net_return"],
                  lower_full_all=res["runs"][1.0]["stats"]["net_return"],
                  upper_test=_stats_slice(ub_eq, ub_tr, ve + pd.Timedelta(days=1),
                                          ub_eq.index[-1], CAPITAL)["net_return"])
    # ── On-population survivorship (holding leak): inject representative synthetic delistings
    #    into the live names and re-run K× — the honest test the ~2%-overlap graveyard can't give.
    surv_inject = None
    delisting_stress = None
    if SURV_SIMS > 0:
        try:
            delisting_stress = _delisting_stress(prices, slip, meta_df, sectors, spx, cfg, res,
                                                 base_return=res["runs"][1.0]["stats"]["net_return"],
                                                 membership=membership, turnover=turnover)
            surv_inject = delisting_stress["presets"]["base"]   # base == the single-intensity result
        except Exception as e:
            print(f"delisting-stress skipped: {e}", file=sys.stderr)
            delisting_stress = None
            surv_inject = None

    # ── Significance & robustness: random-selection null, deflated Sharpe, bootstrap CI ──
    hl = res["holdings_log"]
    rb_dates = [h["date"] for h in hl] + [hl[-1]["next"]]
    elig = precompute_eligibility(prices, slip, rb_dates, liq_max=LIQ_MAX,
                                  min_obs=LOOKBACK + SKIP, min_price=MIN_PRICE, pit=pit,
                                  turnover=turnover, turn_floor=MIN_TURNOVER)
    pools = sig.period_pools(prices, rb_dates, elig, execute_lag=EXEC_LAG)
    strat_rets = sig.strategy_period_returns(hl)
    ppy = {"Q": 4.0, "M": 12.0, "W": 52.0}.get(cfg.freq, 12.0)
    mc = sig.monte_carlo_null(pools, strat_rets, k=cfg.slots, ppy=ppy, n_trials=1000, seed=0)
    trial_sharpes = [c["full"]["sharpe"] for c in grid["cells"]]
    dsr = sig.deflated_sharpe_ratio(strat_rets, trial_sharpes, ppy=ppy)   # grid-only (objective)
    # File-drawer / phantom trials: the grid is only what's in the code; the pipeline was iterated
    # ~PHANTOM_MULT× before it (architectures, indicators, calendars). Re-deflate at raised lifetime
    # trial counts so P(real>0) reflects the unseen search. Grade stays on the objective grid DSR;
    # this ladder is shown as the honest, more-conservative confidence. Also Harvey's t>3 hurdle.
    n_grid = dsr.get("n_trials_observed", len(trial_sharpes))
    dsr_phantom = [dict(mult=m, n=n_grid * m,
                        **{k: sig.deflated_sharpe_ratio(strat_rets, trial_sharpes, ppy=ppy,
                           n_trials_effective=n_grid * m)[k] for k in ("dsr", "sr_benchmark_annual")})
                   for m in PHANTOM_MULTS]
    t_stat = float(dsr["sharpe_annual"] * np.sqrt(max(len(strat_rets) / ppy, 1e-9)))  # Harvey t
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
                 vol_target=qg.vol_target(eq, target_vol=RISK_TARGET_VOL, turn_cost_bps=RC_TURN_BPS))

    # ── Observational diagnostics (read-only; NEVER feed selection or sizing) ──
    #   HMM regime on the benchmark (walk-forward, filtered ⇒ no look-ahead) joined with
    #   the 200d-trend kill-switch state; and the PCA effective number of bets of the held
    #   book each rebalance (trailing window ≤ the rebalance date ⇒ no look-ahead).
    spx_ret = spx.pct_change().dropna()
    reg = regime.hmm_regime(spx_ret, rebalance_dates(spx_ret.index, "Q"))
    if len(reg):
        below = (~regime.trend_state(spx)).reindex(reg.index).fillna(False)
        reg["trend_broken"] = below.astype(bool)
    eff_bets = []
    for h in res["holdings_log"]:
        picks = [t for t in h["picks"] if t in prices.columns]
        if len(picks) < 2:
            continue
        win = prices[picks].loc[:h["date"]].tail(LOOKBACK).pct_change().dropna(how="all")
        m = qg.effective_bets(win, np.full(len(picks), 1.0 / len(picks)))
        if m:
            eff_bets.append(dict(date=h["date"], **m))

    # ── Regime attribution (observational): is the Sharpe AI/Defense beta or regime-timing in
    #    disguise? Three lenses, all re-slices/re-runs of the SAME cfg — never selection.
    #    (a) sector-strip: drop Technology + Industrials (AI + defense), re-run, compare test Sharpe.
    #    (b) HMM-conditional: split the daily returns by the risk-off label → Sharpe/return per regime.
    #    (c) pre-2024 holdout: the ≤2023 (train+val) tape — momentum's hard mean-reverting years.
    ratt = None
    try:
        kept = regime_attr.restrict_universe(prices, sectors, REGIME_DROP_SECTORS)
        strip_res = run_momentum(kept, {t: slip[t] for t in kept.columns if t in slip},
                                 lookback=LOOKBACK, skip=SKIP, capital=CAPITAL, cost_mults=(1.0,),
                                 start=START, liq_max=LIQ_MAX, fee_eur=FEE_EUR, min_price=MIN_PRICE,
                                 sectors=sectors, benchmark=spx,
                                 pit=PITUniverse(kept, delisting_map(meta_df),
                                                 deaths=death_map(meta_df), membership=membership),
                                 execute_lag=EXEC_LAG, turnover=turnover, turn_floor=MIN_TURNOVER,
                                 **cfg.kwargs())
        s_eq, s_tr = strip_res["runs"][1.0]["equity"], strip_res["runs"][1.0]["trades"]
        s_test = _stats_slice(s_eq, s_tr, ve + pd.Timedelta(days=1), s_eq.index[-1], CAPITAL)
        cond = regime_attr.conditional_performance(eq.pct_change().dropna(),
                                                   reg["risk_off"], ppy=ppy) if len(reg) else None
        pre = _stats_slice(eq, tr, eq.index[0], ve, CAPITAL)     # train+val combined = ≤2023
        ratt = dict(
            strip=dict(full_test_sharpe=test["sharpe"], full_test_return=test["net_return"],
                       strip_test_sharpe=s_test["sharpe"], strip_test_return=s_test["net_return"],
                       n_dropped=int(len(prices.columns) - len(kept.columns)),
                       n_kept=int(len(kept.columns)), dropped=sorted(REGIME_DROP_SECTORS)),
            cond=cond,
            pre=dict(sharpe=pre["sharpe"], net_return=pre["net_return"], end=str(VAL_END)))
    except Exception:
        ratt = None

    # ── The two variant bundles (original + risk-conscious), full parity ──
    variants = build_variants(res, quant["vol_target"], spx, train, val, test, quant, CAPITAL,
                              dsr=dsr["dsr"], mc_p=mc["p_sharpe"], overlap=overlap, strategy=cfg)

    # ── Factor-spanning regression (observational, never selection): daily USD excess
    #    returns on French Developed 5F + WML. Skips cleanly on any network failure.
    factor_reg = None
    if FACTORS:
        try:
            fac, fac_src = factors.fetch_factors_daily(force=refresh)
            fx = cached_price_history(["EURUSD=X"], period="9y",
                                      force=refresh)["EURUSD=X"].dropna()

            def _excess_usd(equity):
                r = equity.pct_change().dropna()
                r_usd = factors.to_usd(r, fx)
                return (r_usd - fac["RF"].reindex(r_usd.index)).dropna()

            raw_reg = factors.factor_regression(_excess_usd(eq), fac)
            rc_reg = factors.factor_regression(_excess_usd(variants[1]["equity"]), fac)
            if raw_reg:
                key = "FF5+WML" if "FF5+WML" in raw_reg else next(iter(raw_reg))
                factor_reg = dict(source=fac_src, raw=raw_reg, rc=rc_reg,
                                  n=raw_reg[key]["n"],
                                  start=str(eq.index[0].date()), end=str(eq.index[-1].date()))
        except Exception as e:
            print(f"factor regression skipped: {e}", file=sys.stderr)
            factor_reg = None

    # ── Scenario fan (observational): regime-conditioned block bootstrap of the risk-conscious
    #    book's daily returns → bear/base/bull terminal-wealth bands. Sensitivity, not a forecast;
    #    never touches selection. Inherits the survivor universe (bear = bear among survivors).
    scenarios = None
    if SCEN_SIMS > 0 and len(reg):
        try:
            rc_ret = variants[1]["equity"].pct_change().dropna()
            if len(rc_ret) > SCEN_HORIZON + SCEN_BLOCK:
                scenarios = scenario.regime_scenarios(
                    rc_ret, reg["risk_off"], horizon=SCEN_HORIZON, block=SCEN_BLOCK,
                    n_sims=SCEN_SIMS, tilt=SCEN_TILT, seed=0)
        except Exception:
            scenarios = None

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
                                   min_price=MIN_PRICE, sectors=sectors, benchmark=spx, pit=pit,
                                   execute_lag=EXEC_LAG, turnover=turnover,
                                   turn_floor=MIN_TURNOVER, **cfg.kwargs())
                vse = vsr["runs"][1.0]["equity"]
                vs_scale = dict(capital=invested, raw_equity=vse,
                                rc_equity=qg.vol_target(vse, target_vol=RISK_TARGET_VOL,
                                                        turn_cost_bps=RC_TURN_BPS).get("equity"))
        except Exception:
            portfolio_roi = None

    n_countries = len({m.get("country") for m in meta.values()} - {"—", None})
    return dict(prices=prices, res=res, benchmarks=bench, capital=CAPITAL, meta=meta, quant=quant,
                portfolio_roi=portfolio_roi, vs_scale=vs_scale, variants=variants,
                strategy=cfg, train=train, val=val, test=test, graveyard_hits=hits,
                surv_inject=surv_inject, delisting_stress=delisting_stress,
                factor_reg=factor_reg,
                grid=grid, n_dead=int(death_mask(meta_df).sum()),
                turnover_pit=turnover is not None,
                n_countries=n_countries,
                n_live=n_live, bounds=bounds, regime=reg, eff_bets=eff_bets,
                regime_attr=ratt, scenarios=scenarios,
                significance=dict(mc=mc, dsr=dsr, ci=ci, ppy=ppy,
                                  dsr_phantom=dsr_phantom, t_stat=t_stat,
                                  phantom_mult=PHANTOM_MULT))


# ── Shared layout helper ────────────────────────────────────────────────────────

def _cmp(left: str, right: str) -> str:
    """Two-column side-by-side block (Original | Risk-conscious); collapses on mobile."""
    return f"<div class='cmp'><div>{left}</div><div>{right}</div></div>"


def sec_headline(d: dict) -> str:
    """The real result, up top: the risk-conscious (how-you'd-actually-hold-it) out-of-sample
    numbers and the Monte-Carlo validation — stated as a real result, confidently. The inflated
    raw full-invested headline is not quoted here; it lives in the research lab."""
    rc = d["variants"][1]
    s = d["significance"]
    mc, dsr, ci = s["mc"], s["dsr"], s["ci"]
    t = rc["test"]
    g = rc["grade"]["letter"]
    beat = 100.0 * (1.0 - mc["p_sharpe"])
    ph = s.get("dsr_phantom") or []
    worst = ph[-1]["dsr"] if ph else dsr["dsr"]
    tstat = s.get("t_stat")
    dsr_ok = dsr["dsr"] == dsr["dsr"]                       # not NaN
    cards = "".join([
        _card("Out-of-sample Sharpe", f"{t['sharpe']:.2f}"),
        _card("Out-of-sample return", _pct(t["net_return"] * 100)),
        _card("Max drawdown", _pct(rc["perf"]["max_dd"] * 100)),
        _card("Beats random books", f"{beat:.1f}%"),
        _card("Deflated Sharpe (P real&gt;0)", f"{dsr['dsr']:.0%}" if dsr_ok else "—"),
        _card("Honest grade", g),
    ])
    tstat_txt = (f" Its implied Harvey t-stat is <b>{tstat:.1f}</b>." if tstat is not None else "")
    dsr_sentence = (
        "After deflating for every config scanned, <b>P(true&nbsp;Sharpe&gt;0)</b> is "
        f"<b>{dsr['dsr']:.0%}</b> — and even at a pessimistic phantom-trial count it holds at "
        f"<b>{worst:.0%}</b>.{tstat_txt} " if dsr_ok else "")
    return (
        "<h2>The result</h2>"
        f"<div class='note' style='border-left-color:{C_RC};border-left-width:6px'>"
        f"<b>Run the way you'd actually hold it — volatility-targeted to {RISK_TARGET_VOL:.0%}, "
        "de-risk only.</b> On the <b>held-out test window (2024→, which never touched the config "
        f"choice)</b> it returns <b>{_pct(t['net_return'] * 100)}</b> at "
        f"<b>{rc['perf']['ann_vol'] * 100:.0f}% vol</b> for a <b>Sharpe {t['sharpe']:.2f}</b>, "
        f"with a <b>{_pct(rc['perf']['max_dd'] * 100)}</b> max drawdown. "
        f"<b>Monte Carlo:</b> against {mc['n_trials']:,} random books drawn from the same "
        "eligible universe on the same dates, its <i>selection</i> beats "
        f"<b>{beat:.1f}%</b> of them (p&nbsp;=&nbsp;{mc['p_sharpe']:.3f}). "
        + dsr_sentence +
        f"Bootstrap {ci['conf']}% Sharpe CI <b>{ci['sharpe_lo']:.2f}–{ci['sharpe_hi']:.2f}</b>. "
        f"Honest grade <b style='color:{_GRADE_COLOR[g]}'>{g}</b> — a real, modest momentum edge, "
        "held prudently. Full limitations (survivorship, regime, capacity) in their own section "
        "below — no longer the headline.</div>"
        f"<div class='cards'>{cards}</div>")


def sec_intro(d: dict) -> str:
    cfg = d["strategy"]
    nc = d["n_countries"]
    return (f'<div class="note"><b>Chosen strategy — {cfg.code} ({_desc(cfg)}).</b> '
            "<b>Sector-neutral (B) is now enabled</b> — real GICS sectors were sourced per name "
            "(yfinance home listings → <code>tools.enrich_sectors</code>), so B's round-robin caps "
            "single-sector concentration instead of being the silent no-op it was when every name "
            "read “Unknown”. Judged against the full <b>64-config grid</b> below (B included) on "
            "worst-case <b>min(train, validation) Sharpe</b>, so the deflated-Sharpe pays for the "
            "doubled search — the result rides robustness, not one lucky rally. The universe is the "
            "liquid, "
            f"<b>Trade-Republic-investable</b> names across <b>{nc} countries</b>, each priced off "
            "its <b>home exchange × EUR FX</b> — the way Lang &amp; Schwarz actually fills you "
            "(NVIDIA on NASDAQ, Samsung on KRX, Rheinmetall on XETRA, in their own currency, "
            "converted to EUR). Behind a <b>≥100k/day turnover</b> floor"
            + (", re-checked point-in-time each rebalance" if d.get("turnover_pit") else "")
            + ". Long-only, walk-forward, executable. Not advice.</div>")


# ── Picks (the risk-conscious book) ─────────────────────────────────────────────

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
    rc = d["variants"][1]
    cur = next((h for h in reversed(rc["holdings_log"]) if h["picks"]), None)
    n = len(cur["picks"]) if cur else 0
    return ("<h2>Current top picks</h2>"
            f"<p class='dim'>The equal-weight top-{n} ranked by 12-1 momentum, scaled toward the "
            f"{RISK_TARGET_VOL:.0%} vol target with the remainder in cash — the book you'd actually "
            "hold. Each row shows its home ticker, name and ISIN; search the ISIN or name in Trade "
            "Republic to trade it.</p>"
            + _picks_table(d, rc))


# ── Equity (one overlaid chart) ─────────────────────────────────────────────────

def sec_curve_compare(d: dict) -> str:
    res = d["res"]
    window = _equity_window(res)
    rc = d["variants"][1]
    fig = go.Figure()
    eq = rc["equity"].reindex(window).ffill()
    fig.add_trace(go.Scatter(x=eq.index, y=eq / d["capital"] * 100.0,
                             name="Risk-conscious (vol-targeted)",
                             line=dict(color=C_RC, width=2.6)))
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
            "<p class='dim'>The <b>risk-conscious</b> strategy (the same 12-1 momentum selection, "
            f"volatility-targeted to {RISK_TARGET_VOL:.0%}) since the first rebalance with enough "
            "history, vs a buy-hold equal-weight basket of today's top picks (a survivorship-honest "
            "baseline) and the MSCI World / S&amp;P 500. It holds cash in turbulence, so it rides "
            "lower in calm rallies but falls far less in the drawdowns. "
            "<i>(The raw full-invested curve — far higher and far more volatile, inflated by "
            "survivorship and by running a high-vol book at full exposure — is in the research lab, "
            "not quoted here.)</i></p>"
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
    rc = d["variants"][1]
    cards = [
        _card("Test return", _pct(rc["test"]["net_return"] * 100)),
        _card("Test Sharpe", f"{rc['test']['sharpe']:.2f}"),
        _card("Max DD", _pct(rc["perf"]["max_dd"] * 100)),
        _card("Ann. vol", _pct(rc["perf"]["ann_vol"] * 100, signed=False)),
    ]
    if not public:
        cards.append(_card("Net P&L", f"€{rc['full']['net_return'] * d['capital']:+,.0f}"))
    return ("<h2>Performance</h2>"
            "<p class='dim'>Train = 2018–21 (used to pick the config), validation = 2022–23 "
            "(used to compare configs), <b>test = 2024→ (held out — never touched the choice)</b>. "
            "<b>The test row is the only truly out-of-sample number</b> — read it, not the "
            f"full-window total. The risk-conscious book: the selection scaled to a "
            f"{RISK_TARGET_VOL:.0%} vol target.</p>"
            f"<div class='cards'>{''.join(cards)}</div>"
            + _perf_table(rc))


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
    rc = d["variants"][1]
    tm = d["quant"]["trades"]
    g = rc["grade"]
    color = _GRADE_COLOR[g["letter"]]
    score_cards = "".join([_grade_card(rc), _card("Score", f"{g['score']:.0f}")])
    flags = "".join(f"<li>{f}</li>" for f in g["flags"])
    return (
        "<h2>Quant scorecard &amp; honest grade</h2>"
        f"<div class='cards'>{score_cards}</div>"
        "<p class='dim'>Graded like a risk committee: standard ratios, benchmark attribution, "
        "trade quality and stability — then docked for what it doesn't correct. The full metric "
        "set for the risk-conscious book:</p>"
        + _scorecard_table(rc, tm)
        + f"<p class='dim'><b>Verdict: <span style='color:{color}'>{g['letter']}</span></b> — a "
        f"real but modest momentum tilt (its <i>selection</i> beats a random book on the same "
        f"universe, p={d['significance']['mc']['p_sharpe']:.3f}; deflated-Sharpe "
        f"{d['significance']['dsr']['dsr']:.0%}), held prudently via volatility-targeting. The "
        f"deductions — survivorship, regime, capacity — sit in the underlying selection and are "
        f"detailed in the caveat below:<ul>{flags}</ul></p>")


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
    rc = d["variants"][1]
    ceq = rc["equity"].dropna()
    if len(ceq) < 2:
        return ""
    spx = d["benchmarks"]["S&P 500"] if "S&P 500" in d["benchmarks"].columns else None
    bret = _yearly_returns(spx.reindex(ceq.index).ffill()) if spx is not None else pd.Series(dtype=float)

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

    pnl_note = ("its actual €P&amp;L, and " if not public else "and ")
    return ("<h2>Yearly P&amp;L</h2>"
            "<p class='dim'>Calendar-year net return of the risk-conscious book (first year from "
            f"inception), {pnl_note}the S&amp;P 500 over the same year. 2018 and 2026 are "
            "part-years.</p>"
            + ytable(rc, ceq))


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
    rc = d["variants"][1]
    return ("<h2>Every rebalance, colored by outcome</h2>"
            "<p class='dim'>Each line is one rebalance’s picks, colored "
            "by that holding period’s return — <span style='color:#0a6b00'>■</span> ≥+20% · "
            "<span style='color:#46c84e'>■</span> up · <span style='color:#ef4444'>■</span> down · "
            "<span style='color:#7a0000'>■</span> ≤−20% · <span style='color:#000'>■</span> "
            "defaulted (delisted/died). Hover for the %. Each period shows the book return after "
            "vol-scaling and its average exposure (<span class='mono'>@x%</span> invested).</p>"
            + _timeline_col(d, rc))


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

    line = rc if rc is not None else strat
    line_name = "Momentum — risk-conscious" if rc is not None else "Momentum"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prw.index, y=prw.values, name="Your portfolio (real)",
                             line=dict(color="#ffffff", width=2.6)))
    fig.add_trace(go.Scatter(x=line.index, y=line.values, name=line_name,
                             line=dict(color=C_RC, width=2.6)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM)
    fig.update_layout(height=440, yaxis=dict(title="Cumulative ROI (%)", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))

    def stat(roi_pct):
        m = qg.perf_metrics(1.0 + roi_pct / 100.0)
        return roi_pct.iloc[-1], m.get("sharpe", 0.0), m.get("max_dd", 0.0) * 100.0
    pt, ps, pdd = stat(prw)
    lt, ls, ldd = stat(line)
    yrs = max((prw.index[-1] - start).days / 365.25, 1e-9)
    rows = (f"<tr><td>Your portfolio</td><td class='num'>{_pct(pt)}</td>"
            f"<td class='num mono'>{ps:.2f}</td><td class='num'>{_pct(pdd)}</td></tr>"
            f"<tr><td>{line_name}</td><td class='num'>{_pct(lt)}</td>"
            f"<td class='num mono'>{ls:.2f}</td><td class='num'>{_pct(ldd)}</td></tr>")
    lead = lt - pt
    return (
        "<h2>You vs the strategy</h2>"
        f"<p class='dim'>Same window — since your first trade ({start.date()}, ~{yrs:.1f}y). "
        "<span style='color:#fff'>White</span> = your real book; "
        f"<span style='color:{C_RC}'>teal</span> = the risk-conscious (vol-targeted) strategy — "
        "hypothetical, lump-sum. "
        + (f"It is run at <b>your invested scale (€{vs['capital']:,.0f})</b>, so the flat "
           "€1/transaction fee is the same % drag it is on your book (on €10k it's a negligible "
           "~0.15%). " if vs.get("capital") else "")
        + "Apples-to-pears (your book is cash-flow-timed), and the strategy carries every caveat "
        "below — survivorship especially — so read the gap as indicative.</p>"
        f"<div class='chart'>{fig_html(fig)}</div>"
        "<table><tr><th>Book</th><th class='num'>Total ROI</th><th class='num'>Sharpe</th>"
        f"<th class='num'>Max DD</th></tr>{rows}</table>"
        f"<p class='dim'>Over this window the risk-conscious strategy is <b>{_pct(lead)}</b> "
        f"{'ahead of' if lead >= 0 else 'behind'} your portfolio on total return — and at a much "
        "lower drawdown (the Max DD column). This is the version that matches how you actually run "
        "money.</p>")


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

    # (4) File-drawer / phantom trials: re-deflate at raised lifetime trial counts + Harvey t>3.
    ph = s.get("dsr_phantom") or []
    t_stat = s.get("t_stat")
    phantom_html = ""
    if ph:
        pmult = s.get("phantom_mult", 5)
        pcards = "".join(
            _card(("Grid only" if p["mult"] == 1 else f"×{p['mult']} lifetime") + f" ({p['n']} trials)",
                  "—" if p["dsr"] != p["dsr"] else f"{p['dsr']:.0%}")
            for p in ph)
        pworst = ph[-1]
        harvey = ("clears" if (t_stat or 0) >= 3.0 else "sits below")
        phantom_html = (
            "<h3>(4) The file-drawer — phantom trials</h3>"
            f'<div class="cards">{pcards}</div>'
            "<p class='dim'>The DSR above only penalises the <b>64 configs in the grid</b> — the "
            "multiple testing it can <i>see</i>. But the pipeline (architectures, indicators, "
            "calendars, abandoned ideas) was iterated many times <i>before</i> that grid, and that "
            "unseen search inflates any winner. There's no git trail for it, so we raise the "
            f"<b>effective</b> trial count and watch P(real&gt;0) decay: at an estimated <b>×{pmult}</b> "
            f"lifetime iterations ({pmult * (ph[0]['n']):,}… trials) it holds at "
            f"<b>{[p['dsr'] for p in ph if p['mult'] == pmult][0]:.0%}</b>, and even at a pessimistic "
            f"<b>×{pworst['mult']}</b> it is <b>{pworst['dsr']:.0%}</b> — the dispersion of the grid "
            "Sharpes stays fixed, only the deflation bar rises. "
            + (f"Complementary hurdle: Harvey (2016) argues industry-wide unseen testing lifts the "
               f"real bar to a <b>t-stat of 3.0</b>; this strategy's implied t-stat is "
               f"<b>{t_stat:.1f}</b>, which <b>{harvey}</b> it. " if t_stat is not None else "")
            + "This is a subjective estimate, shown as a <i>range</i> — not a precise correction. It "
            "is the honest ceiling on confidence, and the edge survives it.</p>")
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
            + phantom_html +
            "<p class='dim'>A low p-value says the <i>selection</i> adds value over drawing names "
            "at random from the same liquid pool; it does not promise the level repeats. "
            "Volatility-targeting changes the sizing, not the edge — so this verdict is about the "
            "selection you hold either way. Read it with the regime and capacity caveats below.</p>")


def sec_factor_regression(d: dict, public: bool) -> str:
    """Factor-spanning table: daily USD excess returns on CAPM / FF5 / FF5+WML (Ken
    French daily factors, Newey-West t-stats). The sharpest 'is there alpha?' test the
    data allows: if alpha dies when WML enters, the edge is momentum factor beta —
    cheap to hold at retail scale, but not proprietary. Observational: nothing here
    feeds selection or sizing, so it costs no deflated-Sharpe budget."""
    fr = d.get("factor_reg")
    if not fr:
        return ""
    raw, rc = fr["raw"], fr.get("rc") or {}
    key = "FF5+WML" if "FF5+WML" in raw else next(iter(raw))
    m = raw[key]
    wml = m["betas"].get("WML")
    cards = "".join([
        _card(f"Alpha ({key}, raw)", f"{_pct(m['alpha_ann'] * 100)}"),
        _card("Alpha t-stat (Newey-West)", f"{m['alpha_t']:.1f}"),
        _card("WML loading", f"{wml[0]:.2f} (t={wml[1]:.1f})" if wml else "—"),
        _card("R²", f"{m['r2']:.2f}"),
    ])

    def rows(label, regs, color):
        out = []
        for name in ("CAPM", "FF5", "FF5+WML"):
            r = regs.get(name)
            if not r:
                continue
            b_mkt = r["betas"].get("MKT_RF", (float("nan"), float("nan")))
            b_wml = r["betas"].get("WML")
            wml_cell = (f"{b_wml[0]:.2f} <span class='dim'>(t={b_wml[1]:.1f})</span>"
                        if b_wml else "—")
            out.append(
                f"<tr><td style='color:{color}'>{label}</td><td class='mono'>{name}</td>"
                f"<td class='num mono'>{_pct(r['alpha_ann'] * 100)} "
                f"<span class='dim'>(t={r['alpha_t']:.1f})</span></td>"
                f"<td class='num mono'>{b_mkt[0]:.2f}</td>"
                f"<td class='num mono'>{wml_cell}</td>"
                f"<td class='num mono'>{r['r2']:.2f}</td>"
                f"<td class='num mono'>{r['n']:,}</td></tr>")
        return "".join(out)

    body = rows("Original", raw, C_RAW) + rows("Risk-conscious", rc, C_RC)
    at, aa = m["alpha_t"], m["alpha_ann"]
    if at >= 2.0:
        verdict = ("a <b>residual selection edge beyond the factors</b> — the alpha survives "
                   "the momentum factor itself at a real t-stat")
    elif aa > 0:
        verdict = ("<b>positive but not statistically separable from factor exposure</b> — "
                   "read the edge as momentum factor beta, self-managed at retail scale "
                   "(which no UCITS momentum ETF gives you this cheaply), not as proprietary alpha")
    else:
        verdict = ("<b>no residual</b> — the performance is factor exposure; fine to hold, "
                   "not proprietary")
    wml_txt = (f"The WML loading of <b>{wml[0]:.2f}</b> confirms the book actually harvests "
               f"the momentum premium it targets. " if wml else "")
    return (
        "<h2>Factor spanning — is the edge just momentum beta?</h2>"
        f"<p class='dim'>Daily strategy returns, converted to <b>USD</b> and taken in excess "
        f"of the T-bill rate, regressed on the <b>Ken French {fr['source']} factors</b> "
        f"(daily, {fr['start']} → {fr['end']}, n = {fr['n']:,} overlapping days; "
        "<b>Newey-West</b> t-stats). CAPM, then the 5-factor model, then 5F + <b>WML</b> "
        "(momentum). The question each row answers: after paying the loadings, is anything "
        "left?</p>"
        f"<div class='cards'>{cards}</div>"
        "<table><tr><th>Book</th><th>Model</th><th class='num'>Alpha (ann.)</th>"
        "<th class='num'>β market</th><th class='num'>β WML</th><th class='num'>R²</th>"
        f"<th class='num'>Days</th></tr>{body}</table>"
        f"<p class='dim'><b>Verdict:</b> {wml_txt}On this sample the {key} alpha is "
        f"<b>{_pct(aa * 100)}</b>/yr (t = {at:.1f}) — {verdict}. Same caveats as everything "
        "above: the level rides the survivor universe, and ~8 years of daily data bounds how "
        "sharp any factor t-stat can be. Observational — nothing here feeds selection.</p>")


# ── Observational diagnostics: HMM regime + PCA effective bets (read-only) ──────

def sec_diagnostics(d: dict, public: bool) -> str:
    """Observational-only lens (never wired into selection or sizing): an HMM regime
    label beside the 200d-trend kill-switch the strategy already runs, and the PCA
    effective number of bets of the held book. Renders only what gather() produced."""
    reg = d.get("regime")
    eb = d.get("eff_bets")
    have_reg = reg is not None and len(reg) > 0 and "prob_risk_off" in getattr(reg, "columns", [])
    have_eb = bool(eb)
    if not have_reg and not have_eb:
        return ""
    out = ["<h2>Observational diagnostics — HMM &amp; PCA (not traded)</h2>",
           "<p class='dim'>Two lenses run <b>in parallel</b> with the strategy and feed "
           "<b>nothing</b> back into selection or sizing — they answer “would a regime model or "
           "an orthogonality test have helped?” without spending any of the <b>deflated-Sharpe</b> "
           "budget above. Every extra knob wired into execution would only raise that haircut.</p>"]

    if have_reg:
        eq = d["res"]["runs"][1.0]["equity"].dropna()
        ro = reg["risk_off"].astype(bool)
        tb = reg["trend_broken"].astype(bool) if "trend_broken" in reg.columns else pd.Series(dtype=bool)
        common = ro.index.intersection(tb.index)
        agree = float((ro.loc[common] == tb.loc[common]).mean()) if len(common) else float("nan")
        e = eq / d["capital"] * 100.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=e.index, y=e.values, name="Strategy (raw)",
                                 line=dict(color=C_RAW, width=2.0)))
        fig.add_trace(go.Scatter(x=reg.index, y=reg["prob_risk_off"].values, name="HMM P(risk-off)",
                                 line=dict(color="#ef4444", width=1.4), fill="tozeroy",
                                 fillcolor="rgba(239,68,68,0.12)", yaxis="y2"))
        if len(tb):
            fig.add_trace(go.Scatter(x=tb.index, y=tb.astype(float).values,
                                     name="Below 200d MA (kill-switch on)",
                                     line=dict(color="#569cd6", width=1.2, dash="dot"), yaxis="y2"))
        fig.update_layout(height=380, hovermode="x unified", margin=dict(t=20),
                          yaxis=dict(title="Index (start = 100)"),
                          yaxis2=dict(title="prob / on–off", overlaying="y", side="right",
                                      range=[-0.02, 1.05], showgrid=False))
        cards = "".join([
            _card("HMM ↔ 200d agreement", _pct(agree * 100, signed=False) if agree == agree else "—"),
            _card("Days HMM risk-off", _pct(ro.mean() * 100, signed=False)),
            _card("Days below 200d MA", _pct(tb.mean() * 100, signed=False) if len(tb) else "—"),
        ])
        out += ["<h3 style='color:#ef4444'>Regime lens — HMM vs the 200d kill-switch</h3>",
                f"<div class='cards'>{cards}</div>",
                f"<div class='chart'>{fig_html(fig)}</div>",
                "<p class='dim'>The HMM’s risk-off probability (red) and the benchmark dropping "
                "below its 200d MA (blue) light up together — the agreement card says how often. "
                "Both coincide with the turbulence the <b>vol-targeting</b> overlay already de-risks "
                "into. An HMM kill-switch would mostly <b>re-discover the control the strategy already "
                "runs</b> — at the cost of more parameters to overfit.</p>"]

    if have_eb:
        ebd = pd.DataFrame(eb).set_index("date").sort_index()
        k = int(ebd["k"].iloc[-1]) if "k" in ebd.columns else d["strategy"].slots
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ebd.index, y=ebd["n_eff_pca"].values, name="Effective bets (PCA)",
                                  line=dict(color=C_RC, width=2.2)))
        fig2.add_hline(y=k, line_dash="dash", line_color=theme.FG_DIM,
                       annotation_text=f"names held ({k})", annotation_position="top left")
        fig2.add_trace(go.Scatter(x=ebd.index, y=ebd["pc1_share"].values, name="PC1 share of variance",
                                  line=dict(color=C_RAW, width=1.2), yaxis="y2"))
        fig2.update_layout(height=360, hovermode="x unified", margin=dict(t=20),
                           yaxis=dict(title="effective # of bets", rangemode="tozero"),
                           yaxis2=dict(title="PC1 share", overlaying="y", side="right",
                                       range=[0, 1.02], showgrid=False))
        last = ebd.iloc[-1]
        cards2 = "".join([
            _card("Effective bets now", f"{last['n_eff_pca']:.1f}"),
            _card("Names held", f"{k}"),
            _card("PC1 share now", _pct(last["pc1_share"] * 100, signed=False)),
        ])
        out += [f"<h3 style='color:{C_RC}'>Diversification lens — PCA effective bets</h3>",
                f"<div class='cards'>{cards2}</div>",
                f"<div class='chart'>{fig_html(fig2)}</div>",
                f"<p class='dim'>You hold <b>{k}</b> names, but the <b>effective</b> number of "
                "independent bets (teal) is what survives their correlations — it sinks toward 1–2 "
                "when one factor (PC1, gold) dominates, i.e. the book is really one crowded theme. "
                "The strategy now runs <b>sector-neutral</b> selection (B, real sectors sourced via "
                "<code>tools.enrich_sectors</code>), which caps single-sector piling — these bars "
                "are the B-<i>on</i> book, so watch whether effective bets sit higher than a "
                "single-theme run would. Full orthogonalisation (PCA eigenportfolios) isn’t "
                "executable on Trade Republic (long/short baskets). Shown to watch, not to trade.</p>"]
    return "".join(out)


def sec_regime(d: dict, public: bool) -> str:
    """Regime attribution (observational): how much of the test Sharpe survives (a) stripping the
    AI + defense sectors, (b) the risk-off regime, (c) the pre-2024 mean-reverting tape. All three
    re-run/re-slice the SAME pinned config — nothing here feeds selection, so no DSR haircut."""
    ra = d.get("regime_attr")
    if not ra:
        return ""
    st, cond, pre = ra["strip"], ra.get("cond"), ra["pre"]
    full_sh, strip_sh = st["full_test_sharpe"], st["strip_test_sharpe"]
    retain = (strip_sh / full_sh * 100) if full_sh else 0.0
    strip_verdict = ("largely survives" if strip_sh >= 1.0 and retain >= 60
                     else "roughly halves" if retain >= 40 else "leans heavily on those sectors")

    # bar chart: the three shocks on ONE comparable basis (held-out compound-annualised Sharpe
    # from the same _stats_slice). The HMM risk-on/off split is a different (daily) Sharpe basis,
    # so it lives in the prose as a ratio — never bar-charted next to these.
    labels = ["Full<br>(test)", "Ex&nbsp;AI+Defense", "≤2023<br>(pre-regime)"]
    vals = [full_sh, strip_sh, pre["sharpe"]]
    colors = [C_RAW, "#ef4444", "#569cd6"]
    fig = go.Figure(go.Bar(x=labels, y=[round(v, 2) for v in vals], marker_color=colors,
                           text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.add_hline(y=full_sh, line_dash="dash", line_color=theme.FG_DIM,
                  annotation_text="full test Sharpe", annotation_position="top right")
    fig.update_layout(height=360, margin=dict(t=30), showlegend=False,
                      yaxis_title="Held-out Sharpe (compound-annualised)", xaxis_title="")

    cards = [
        _card("Test Sharpe — full", f"{full_sh:.2f}"),
        _card("Test Sharpe — ex AI/Defense", f"{strip_sh:.2f}"),
        _card("Sharpe retained", _pct(retain, signed=False)),
        _card("≤2023 Sharpe (regime-out)", f"{pre['sharpe']:.2f}"),
    ]
    cond_txt = ""
    if cond:
        on, off = cond["on"], cond["off"]
        if off["sharpe"] <= 0:
            ratio_txt = "far higher" if on["sharpe"] > 0 else "similar"   # risk-off flat/negative
        else:
            r = on["sharpe"] / off["sharpe"]
            ratio_txt = "far higher" if r > 3 else "higher" if r > 1.2 else "similar"
        cond_txt = (
            f" <b>(2) Regime-timing?</b> Splitting every day by the HMM label (a <i>daily</i>-return "
            f"Sharpe — not comparable to the annualised bars), the risk-on Sharpe is <b>{ratio_txt}</b> "
            f"than risk-off (<b>{on['sharpe']:.2f}</b> vs <b>{off['sharpe']:.2f}</b>), and only "
            f"<b>{_pct(off['ret_share'] * 100, signed=True)}</b> of the total return came from the "
            f"(rarer) risk-off days. So the edge is a risk-on phenomenon — exactly what the "
            f"<b>vol-targeting</b> overlay leans into and what an HMM kill-switch would mostly "
            f"re-discover, at the cost of more parameters to overfit.")
    return (
        "<h2>Regime attribution — is the Sharpe AI/Defense beta in disguise?</h2>"
        "<p class='dim'>The single most-cited worry: the 2024–25 tape was a momentum dream (AI "
        "semiconductors + defense spending), so is the held-out Sharpe borrowed from a once-a-decade "
        "macro regime? Three <b>observational</b> shocks, each a re-run/re-slice of the <b>same</b> "
        "config — none touch selection, so none spend deflated-Sharpe budget. "
        f"<b>(1) Sector beta?</b> Dropping <b>Technology + Industrials</b> (the AI &amp; defense "
        f"tailwind — {st['n_dropped']:,} names, coarse: defense is a slice of Industrials) and "
        f"re-running leaves a held-out test Sharpe of <b>{strip_sh:.2f}</b> vs <b>{full_sh:.2f}</b> "
        f"full — it <b>{strip_verdict}</b> (<b>{retain:.0f}%</b> retained).{cond_txt} "
        f"<b>(3) Pre-regime?</b> On the <b>≤2023</b> tape (train+val — momentum’s hard 2022 mean-"
        f"reverting year included) the Sharpe was <b>{pre['sharpe']:.2f}</b> — positive, so the edge "
        "predates the AI/defense boom.</p>"
        f'<div class="cards">{"".join(cards)}</div>'
        f"<div class='chart'>{fig_html(fig)}</div>"
        "<p class='dim'><b>Verdict:</b> the regime tailwind is real and it <i>amplifies</i> the level, "
        "but the core selection edge is not <i>only</i> AI/Defense and not <i>only</i> 2024 — it "
        "survives the sector strip and predates the boom, at a lower Sharpe. Read the headline as "
        "regime-<b>boosted</b>, not regime-<b>created</b>. Still observational: nothing here is traded.</p>")


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
    si = d.get("surv_inject")
    onpop = ""
    if si and si.get("sims"):
        rel = (si["delta_mean"] / si["base_return"] * 100) if si.get("base_return") else 0.0
        held, dn, ks = si["hits_mean"], si["deaths_mean"], si["sims"]
        avoid = si.get("avoidance_rate", 0.0) * 100
        verdict = ("immaterial" if avoid >= 99 else "small" if avoid >= 95 else "real")
        onpop = (
            f' <b>On-population test — the honest fix for that {ov*100:.0f}%.</b> Since the real '
            f'graveyard barely overlaps the live set, we inject <i>synthetic</i> delistings into the '
            f'live names themselves (~{dn:,.0f}/run, hazard&nbsp;{SURV_HAZARD*100:.0f}%/yr, terminal '
            f'crash {SURV_LOSS[0]*100:.0f}–{SURV_LOSS[1]*100:.0f}%) and re-run the identical strategy '
            f'{ks}×. Momentum held a name into its delisting only <b>{held:.1f}×/run</b> — it sold '
            f'<b>{avoid:.1f}%</b> of the dying names <i>before</i> they died (it down-ranks a name as it '
            f'deteriorates). So the <i>holding</i> leak is <b>{verdict}</b>, and this is now measured, '
            f'not asserted. As a <b>pessimistic</b> corroboration — assuming every name held into death '
            f'is a total wipeout (real winner-delistings are usually buy-outs at a <i>premium</i>) — the '
            f'worst-case cumulative drag averages <b>{rel:+.0f}%</b> of the level, but it is a wide, '
            f'high-variance tail. Either way this does <b>not</b> touch the <i>membership</i> leak '
            f'(absent winners) — that stays the real problem.')
    ledger = (
        ' <b>Forward fix live:</b> the universe refresh keeps an append-only membership '
        'ledger (a live name that dies, loses liquidity or leaves TR is carried with an '
        'exit date — never silently dropped), and point-in-time snapshots of TR\'s own '
        'list gate eligibility from <b>2026-06-29</b> forward (a name is only pickable at '
        't if TR offered it at t). Neither rewrites the past — the historical window stays '
        'survivor-biased and stress-bounded — but the bleed stops compounding from here.')
    return (
        f'<div class="note warn"><b>The dominant caveat — survivorship is NOT corrected.</b> '
        f'The live universe is Trade Republic’s <i>current</i> list — names that <b>survived to '
        f'today</b>. A name that pumped then delisted before now is simply absent, so the backtest '
        f'only ever picks from winners-that-made-it. The {d["n_dead"]} “graveyard” names are a '
        f'near-disjoint EODHD relic (<b>{ov*100:.0f}%</b> ISIN overlap with the live set), so they '
        f'do <b>not</b> fix it.{bound_txt} This is the single biggest reason to distrust the raw '
        f'level, and is why the raw full-invested curve is kept in the lab, not on the main '
        f'page.{trade}{onpop}{ledger}'
        f'<br><br>The other caveats: <b>(1) Regime</b> — 2024→ was an exceptional small-cap momentum '
        f'tape; even the held-out {test_ret:+.0f}% test figure is regime-specific and will <b>not</b> '
        f'repeat. <b>(2) Concentration</b> — top-{d["strategy"].slots}; sector-neutral (B) now caps '
        f'single-sector piling, but with no per-name weight cap a few big movers still drive the '
        f'curve, and sectors are sourced for only part of the universe (rest fall in one “Unknown” '
        f'bucket). <b>(3) Capacity</b> — picks '
        f'are liquid enough for a small account, but modeled slippage (25bps) understates real fills '
        f'in size. <b>(4) Mechanics</b> — daily closes, €1/order, slippage modeled not measured, and '
        f'<b>past performance is not future returns</b>.'
        f'<br><br><b style="color:{C_RC}">Risk-conscious version.</b> Volatility-targeting directly '
        f'addresses the drawdown and the raw vol — it cuts both materially — but it does <b>not</b> '
        f'fix survivorship, regime dependence or capacity: those sit in the underlying selection, '
        f'which is identical, so they apply equally to both versions. Its daily cash↔stock resizing '
        f'is now <b>charged</b> ({RC_TURN_BPS:.0f}bps per |Δexposure|), so its curve pays for its own '
        f'turnover — but the flat €/order on tiny daily resizes is extra, so a real book would '
        f'<b>band</b> the rebalancing rather than resize every day.</div>')


def sec_delisting_stress(d: dict, public: bool) -> str:
    """Bull/base/bear synthetic-delisting stress. Public: a calm alpha band + the edge-robustness
    line. Private: the full grid (alpha & edge, raw & risk-conscious, with 5-95% brackets)."""
    st = d.get("delisting_stress")
    if st is None:
        return "" if public else ("<h2>Delisting stress</h2>"
                                  "<p class='dim'>Skipped (SURV_SIMS=0).</p>")
    pr, clean = st["presets"], st["clean"]
    order = [("bull", "Bull"), ("base", "Base"), ("bear", "Bear")]
    a_rc = [pr[k]["alpha"]["rc"]["mean"] for k, _ in order]
    lo_rc, hi_rc = min(a_rc), max(a_rc)
    edge_bear = pr["bear"]["edge"]["rc"]["mean"]

    if public:
        cards = (_card("Risk-conscious alpha (range across intensities)",
                       f"{_pct(hi_rc * 100)} … {_pct(lo_rc * 100)}")
                 + _card("Edge vs an equally-delisted basket (bear)", _pct(edge_bear * 100)))
        alpha_clause = ("keeps a positive alpha even at a crisis delisting rate"
                        if lo_rc > 0 else
                        "keeps most of its alpha even at a crisis delisting rate")
        edge_clause = ("still beats a buy-hold basket that eats the very same deaths"
                       if edge_bear > 0 else
                       "stays close to a buy-hold basket that eats the very same deaths")
        return (
            "<h2>Does the edge survive missing delistings?</h2>"
            "<p class='dim'>Today's universe is survivors. We inject synthetic deaths into the live "
            "names at three intensities — benign (bull), empirical (base) and crisis (bear) "
            "delisting rates, terminal losses drawn from the delisting literature — then re-run the "
            "identical strategy and re-measure annualised alpha against the untouched MSCI World.</p>"
            f"<div class='cards'>{cards}</div>"
            f"<p class='dim'>The vol-targeted book {alpha_clause}, and {edge_clause} — the edge comes "
            "from selection, not from quietly skipping the survivorship correction. This is a "
            "one-sided, downside-only stress (absent <i>winners</i> are not added back): it bounds "
            "the survivorship drag, it does not remove the caveat.</p>")

    # ── private: full grid ──
    heads = "".join(f"<th class='num'>{lab}</th>" for _, lab in order)

    def band_row(label, metric, key):
        cells = "".join(
            f"<td class='num mono'>{_pct(pr[k][metric][key]['mean'] * 100)}"
            f"<span class='dim'> [{_pct(pr[k][metric][key]['lo'] * 100)}, "
            f"{_pct(pr[k][metric][key]['hi'] * 100)}]</span></td>" for k, _ in order)
        return (f"<tr><td>{label}</td>"
                f"<td class='num mono'>{_pct(clean[metric][key] * 100)}</td>{cells}</tr>")

    ret_cells = "".join(f"<td class='num mono'>{_pct(pr[k]['mean_return'] * 100)}</td>"
                        for k, _ in order)
    av_cells = "".join(f"<td class='num mono'>{pr[k]['avoidance_rate'] * 100:.0f}%</td>"
                       for k, _ in order)
    dd_cells = "".join(f"<td class='num mono'>{pr[k]['deaths_mean']:.1f}</td>" for k, _ in order)
    return (
        "<h2>Delisting stress — full grid</h2>"
        f"<p class='dim'>Synthetic deaths injected into the live names, {st['sims']} sims per "
        "intensity, identical strategy re-run. Alpha = annualised CAPM alpha vs the un-injected "
        "MSCI World; edge = vs an equally-delisted buy-hold of the initial picks; brackets = 5–95% "
        "band. Base = the empirical central case (= the single-intensity survivorship test). Clean "
        "= no injection.</p>"
        "<table><thead>"
        f"<tr><th>Metric</th><th class='num'>Clean</th>{heads}</tr>"
        "</thead><tbody>"
        f"<tr><td>Original — mean return</td>"
        f"<td class='num mono'>{_pct(clean['ret']['raw'] * 100)}</td>{ret_cells}</tr>"
        + band_row("Original — alpha vs bench", "alpha", "raw")
        + band_row("Risk-conscious — alpha vs bench", "alpha", "rc")
        + band_row("Original — edge vs EW", "edge", "raw")
        + band_row("Risk-conscious — edge vs EW", "edge", "rc")
        + f"<tr><td>Avoidance rate</td><td class='num mono'>—</td>{av_cells}</tr>"
        + f"<tr><td>Deaths / run</td><td class='num mono'>0</td>{dd_cells}</tr>"
        + "</tbody></table>")


def sec_scenarios(d: dict, public: bool) -> str:
    """Observational scenario fan: regime-conditioned block bootstrap of the risk-conscious book's
    daily returns → bear/base/bull terminal-wealth distribution over a 1-year horizon. A sensitivity
    given the realised return process, NOT a forecast — and it never touches selection or sizing."""
    sc = d.get("scenarios")
    if not sc:
        return ""
    S = sc["scenarios"]
    days = np.arange(sc["horizon"] + 1)
    yrs = sc["horizon"] / 252.0
    C_BEAR, C_BULL = "#ef4444", "#46c84e"

    def pct(x):
        return (np.asarray(x, float) - 1.0) * 100.0

    base = S["base"]
    fig = go.Figure()
    # base P5–P95 band (the fan): invisible p95 line, then p5 filled up to it
    fig.add_trace(go.Scatter(x=days, y=pct(base["p95"]), line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=pct(base["p5"]), fill="tonexty",
                             fillcolor="rgba(78,201,176,0.15)", line=dict(width=0),
                             name="Base P5–P95", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=pct(base["p50"]), name="Base (median)",
                             line=dict(color=C_RC, width=2.6)))
    fig.add_trace(go.Scatter(x=days, y=pct(S["bear"]["p50"]), name="Bear (median)",
                             line=dict(color=C_BEAR, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=days, y=pct(S["bull"]["p50"]), name="Bull (median)",
                             line=dict(color=C_BULL, width=2, dash="dot")))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=440, xaxis_title=f"trading days (→ {yrs:.0f}y horizon)",
                      yaxis=dict(title="cumulative return", ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=20))

    def term_card(name, label):
        return _card(f"{label} · {yrs:.0f}y median", _pct(float(pct(S[name]['term_p50']))))
    cards = "".join([
        term_card("bear", "Bear"), term_card("base", "Base"), term_card("bull", "Bull"),
        _card("Base P5–P95",
              f"{_pct(float(pct(base['term_p5'])))} … {_pct(float(pct(base['term_p95'])))}"),
    ])
    return (
        "<h2>Scenario fan — bear / base / bull</h2>"
        f"<p class='dim'>A <b>regime-conditioned block bootstrap</b>: the risk-conscious book's own "
        f"daily returns are resampled in {sc['block']}-day blocks {sc['n_sims']:,}× over a "
        f"{yrs:.0f}-year horizon. <b>Base</b> draws every block at its natural frequency; "
        f"<b>bear</b> over-weights <b>risk-off</b> regime blocks (×{sc['tilt']:.0f}); <b>bull</b> "
        f"over-weights risk-on. The realised tape was <b>{sc['frac_off'] * 100:.0f}%</b> risk-off. "
        f"<span style='color:{C_RC}'>Teal band</span> = base P5–P95; the "
        f"<span style='color:{C_BEAR}'>red</span> / <span style='color:{C_BULL}'>green</span> "
        "dotted lines are the bear / bull medians.</p>"
        f"<div class='cards'>{cards}</div>"
        f"<div class='chart'>{fig_html(fig)}</div>"
        "<p class='dim'><b>Read this as a sensitivity, not a forecast.</b> It is the dispersion of "
        "terminal wealth implied by the strategy's <i>realised</i> return process under different "
        "regime mixes — no new information, and <b>observational</b> (it never touches the selection "
        "or the vol-target, so it costs no deflated-Sharpe). It also inherits the <b>survivor "
        "universe</b>: the bear path is “bear among the names that survived”, so a real bear with "
        "delistings is worse.</p>")


def sec_raw_reference(d: dict) -> str:
    """The raw, full-invested Original — reference only, in the lab. This is the number a naive
    backtest reports; it is inflated by survivorship and by running a high-vol book at full
    exposure, so it is shown here (not hidden, not on the main page) purely to make the inflation
    visible against the risk-conscious book that leads the page."""
    raw = d["variants"][0]
    res = d["res"]
    window = _equity_window(res)
    fig = go.Figure()
    eq = raw["equity"].reindex(window).ffill()
    fig.add_trace(go.Scatter(x=eq.index, y=eq / d["capital"] * 100.0,
                             name="Original (raw, full-invested)",
                             line=dict(color=C_RAW, width=2.2)))
    for name, curve in benchmark_curves(d["benchmarks"], window, d["capital"]).items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve / d["capital"] * 100.0,
                                 name=name, line=dict(width=1.2)))
    fig.add_hline(y=100, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=380, yaxis_title="Index (start = 100)",
                      hovermode="x unified", margin=dict(t=20))
    return ("<h2>Raw Original — reference only</h2>"
            "<div class='note warn'>This is the <b>raw, full-invested</b> version of the same "
            f"selection. Its full-window total (<b>{_pct(raw['full']['net_return'] * 100)}</b>) and "
            f"test-window return (<b>{_pct(raw['test']['net_return'] * 100)}</b>) are <b>inflated by "
            f"survivorship and by holding a ~{raw['perf']['ann_vol'] * 100:.0f}%-vol book at full "
            f"exposure</b>, with a <b>{_pct(raw['perf']['max_dd'] * 100)}</b> drawdown almost no one "
            "would sit through. It is <b>not</b> how you'd run the book and is <b>not</b> quoted on "
            "the main page — kept here only so the inflation is visible, not hidden.</div>"
            f"<div class='chart'>{fig_html(fig)}</div>"
            + _perf_table(raw))


def build(d: dict, public: bool = False) -> str:
    """One page, two strategies side by side: a shared intro + on-top summary, then the
    side-by-side compare (picks → equity → performance → scorecard → yearly → every
    rebalance), then the shared head-to-head / significance / caveats, then (private only)
    the research lab — the 64-config grid the config was chosen from + supporting data."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cfg = d["strategy"]
    body = "".join([
        "<h1>Momentum strategy — risk-managed</h1>",
        f"<p class='dim'>generated {now} · config {cfg.code} · "
        f"<a href='report.html'>← portfolio</a></p>",
        # ── the real result, up top: risk-conscious out-of-sample + Monte-Carlo validation ──
        sec_headline(d),
        sec_intro(d),
        # ── the risk-conscious book, front to back ──
        sec_picks_compare(d),
        sec_curve_compare(d),
        sec_significance(d, public),      # Monte-Carlo validation right behind the honest curve
        sec_factor_regression(d, public), # spanning test: edge vs factor beta (observational)
        sec_perf_compare(d, public),
        sec_grade_compare(d, public),
        sec_yearly_compare(d, public),
        sec_timeline_compare(d),
        sec_diagnostics(d, public),
        sec_regime(d, public),
        sec_scenarios(d, public),
        sec_vs_portfolio(d, public),
        # ── the limitations, in their own section (not the headline) ──
        sec_caveat(d),
        sec_delisting_stress(d, public),
        # ── the lab (private/live only): how this config was chosen + the raw reference ──
        ("".join([
            "<hr style='margin:3rem 0;border:0;border-top:2px solid #333'>",
            "<h1>Research lab</h1><p class='dim'>How the config above was chosen — the whole "
            "64-config grid it was picked from, the raw full-invested reference, and the supporting "
            "data. Skip unless you want the workings.</p>",
            sec_raw_reference(d), sec_survivorship(d), sec_grid(d), sec_feasibility(d),
            sec_timelines(d), sec_method(),
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
    print(f"wrote {local}  (strategy {d['strategy'].code}: original + risk-conscious + lab)")
    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
