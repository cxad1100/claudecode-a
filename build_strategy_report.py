"""Strategy page — DATA layer.

This module assembles everything the Strategy tab needs (engines → numbers →
registry records) and delegates ALL rendering to strategy_ui.render(). The split is
deliberate: pre-registered selection/adoption math and data assembly live here;
presentation (bands, dossiers, folds, charts) lives in strategy_ui.py and can be
reworked without touching a single number.

  python build_strategy_report.py            # writes local/strategy.html
  python build_strategy_report.py --open

Framework contract: a new strategy enters the page as ONE make_record() call in
build_registry() below (it renders a generic dossier + registry row + chart trace
automatically); ledger-only entries go to tools/strategy_registry.STATIC_RECORDS.
"""
import argparse
import os
import sys
import webbrowser

import numpy as np
import pandas as pd

from tools import theme, significance as sig, quant_grade as qg, regime, survivorship, regime_attr, scenario, factors
from tools.momentum import (run_momentum, winsorize_prices, to_xetra_calendar,
                            precompute_eligibility, benchmark_curves, equal_weight_curve,
                            rebalance_dates)
from tools.universe_pit import PITUniverse
from tools.universe_assemble import delisting_map, death_map, death_mask
from tools import universe_snapshot
from tools.momentum_grid import (MomentumConfig, _stats_slice, run_grid,
                                 ALL_CONFIGS, pick_ultimate, pick_top_n)
from tools.portfolio_tools import BENCHMARKS, parse_portfolio
from tools.portfolio_analytics import build_roi_timeseries
from tools.data_buffer import cached_price_history
from tools import strategy_registry as sreg
from build_momentum_report import (
    PRICES_CSV, META_CSV, TURN_CSV, ROOT, LOOKBACK, SKIP, START, LIQ_MAX, MIN_PRICE, CAPITAL,
    FEE_EUR, COST_MULTS, TRAIN_END, VAL_END, WINSOR_CAP, EXEC_LAG, K, MIN_TURNOVER,
    _slip, _equity_window,
)
import strategy_ui as ui
from strategy_ui import C_RAW, C_RC     # entity colors live with the UI design system

# Production config is chosen DYNAMICALLY: gather() runs the 64-config grid and takes
# pick_ultimate(grid) as the live strategy. This constant is the FALLBACK (used only
# if nothing in the grid qualifies). The pick is sector-coverage sensitive — trust
# full-coverage grids only; the lab grid shows the current cells (recomputed every
# build, never quoted here).
STRATEGY = MomentumConfig(sector_neutral=True, slots=10, freq="Q", lazy=True)

# Risk-conscious overlay: volatility-target the book to this annualised vol (de-risk
# only, park the rest in cash).
RISK_TARGET_VOL = 0.15
# Charge the overlay's daily |Δexposure| resizing at this one-way slippage so the
# risk-conscious curve pays for its own turnover. Flat €/order is extra.
RC_TURN_BPS = 25.0

# On-population survivorship test (the *holding* leak): inject synthetic delistings
# into the LIVE names at this hazard + terminal-crash band, re-run SURV_SIMS×.
# Observational — never feeds selection/sizing. Env SURV_SIMS=0 skips (fast iteration).
SURV_SIMS = int(os.environ.get("SURV_SIMS", "8"))
SURV_HAZARD = 0.05            # ~5%/yr delisting hazard (broad-equity plausible)
SURV_LOSS = (0.40, 1.00)      # terminal crash drawn here

# Delisting-stress intensities (bull/base/bear), grounded in the delisting literature
# (Shumway 1997: -30% NYSE/AMEX; Shumway-Warther 1999: -55% NASDAQ; bankruptcies -100%).
STRESS_PRESETS = {                       # name -> (hazard_annual, (loss_lo, loss_hi))
    "bull": (0.02, (0.30, 0.60)),
    "base": (SURV_HAZARD, SURV_LOSS),    # == the single-intensity default
    "bear": (0.10, (0.60, 1.00)),
}

# Regime attribution: strip the two sectors carrying the 2024-25 tailwind and re-run
# the SAME config. Observational re-run.
REGIME_DROP_SECTORS = {"Technology", "Industrials"}

# File-drawer / phantom-trials: the 64-config grid is the multiple testing the DSR can
# SEE; PHANTOM_MULT is the honest lifetime-iteration estimate. Grade stays on the
# objective ×1 grid DSR; the page shows the ladder as a sensitivity.
PHANTOM_MULT = 5
PHANTOM_MULTS = (1, 5, 10)

# Scenario fan (observational): regime-conditioned block bootstrap of the
# risk-conscious book's daily returns. Env SCEN_SIMS=0 skips (fast iteration).
SCEN_HORIZON = 252
SCEN_BLOCK = 21
SCEN_TILT = 3.0
SCEN_SIMS = int(os.environ.get("SCEN_SIMS", "2000"))

# Factor-spanning regression (observational). Env FACTORS=0 skips (offline).
FACTORS = int(os.environ.get("FACTORS", "1"))


def build_variants(res: dict, vt: dict, spx, train: dict, val: dict, test: dict,
                   quant: dict, capital: float, *, dsr: float, mc_p: float, overlap: float,
                   strategy: MomentumConfig = STRATEGY,
                   train_end=TRAIN_END, val_end=VAL_END) -> list[dict]:
    """Two parallel 'variant bundles' of identical shape: the raw strategy and its
    volatility-targeted (risk-conscious) twin. Selection is IDENTICAL across both
    (same holdings_log / trades).

    Each bundle carries TWO window sets: `train/val/test/full` on the pre-registered
    selection basis (backtest_stats — what the grid/adoption rules read) and `windows`
    on the canonical display basis (qg.window_metrics — what every comparison surface
    renders). net_return/max drawdown agree across bases; only the Sharpe convention
    differs. The grade is display-layer, so it reads the canonical test Sharpe."""
    tr = res["runs"][1.0]["trades"]
    raw_eq = res["runs"][1.0]["equity"]

    def _canon(eq):
        return {k: qg.window_metrics(eq, lo, hi)
                for k, (lo, hi) in sreg.window_bounds(eq, train_end, val_end).items()}

    raw_windows = _canon(raw_eq)
    raw = dict(
        label=f"Original (raw, {strategy.code})", short="Original", key="raw", color=C_RAW,
        equity=raw_eq, holdings_log=res["holdings_log"], trades=tr,
        train=train, val=val, test=test, full=res["runs"][1.0]["stats"],
        windows=raw_windows,
        perf=quant["perf"], bench=quant["bench"], roll=quant["roll"],
        grade=qg.grade(raw_windows.get("test", {}).get("sharpe", 0.0), dsr, mc_p, overlap),
        exposure=None, exposure_latest=1.0)

    rc_eq = vt["equity"]
    te, ve = pd.Timestamp(train_end), pd.Timestamp(val_end)
    one = pd.Timedelta(days=1)
    rc_test = _stats_slice(rc_eq, tr, ve + one, rc_eq.index[-1], capital)
    rc_windows = _canon(rc_eq)
    rc = dict(
        label=f"Risk-conscious (vol-target {RISK_TARGET_VOL:.0%})", short="Risk-conscious",
        key="rc", color=C_RC,
        equity=rc_eq, holdings_log=res["holdings_log"], trades=tr,
        train=_stats_slice(rc_eq, tr, rc_eq.index[0], te, capital),
        val=_stats_slice(rc_eq, tr, te + one, ve, capital),
        test=rc_test,
        full=_stats_slice(rc_eq, tr, rc_eq.index[0], rc_eq.index[-1], capital),
        windows=rc_windows,
        perf=vt,
        bench=qg.vs_benchmark(rc_eq, spx) if spx is not None else {},
        roll=qg.rolling_sharpe(rc_eq),
        grade=qg.grade(rc_windows.get("test", {}).get("sharpe", 0.0), dsr, mc_p, overlap),
        exposure=vt.get("exposure"),
        exposure_latest=vt.get("exposure_latest", vt.get("avg_exposure", 1.0)))
    return [raw, rc]


MEGACAP_INDEX = "Nasdaq 100"          # per-interval yardstick for the holdings table
DIP_WINDOW = 21                       # buy-the-dip lookback: 1-month reversal horizon (~21 sessions)


def _arm_holdings_table(hl_log, prices, index_series, name_map, giants) -> tuple:
    """Per rebalance period: each held name's % return over the hold, the equal-weight
    basket total, and the index's return over the SAME interval — so every interval is
    scored against the yardstick. Returns a tuple of period dicts (newest last)."""
    from build_megacap_report import _period_returns
    idx = index_series.dropna() if index_series is not None else None
    out = []
    for h in hl_log:
        if not h.get("picks"):
            continue
        d0, d1 = pd.Timestamp(h["date"]), pd.Timestamp(h["next"])
        rets = _period_returns(prices, h["picks"], d0, d1)
        if not rets:
            continue
        names = [dict(t=t, name=name_map.get(t, t), ret=float(rets[t]),
                      giant=t in giants)
                 for t in sorted(h["picks"], key=lambda x: -rets.get(x, -9))
                 if t in rets]
        basket = sum(rets.values()) / len(rets)
        iret = float("nan")
        if idx is not None:
            p0, p1 = idx.loc[:d0], idx.loc[:d1]
            if len(p0) and len(p1) and float(p0.iloc[-1]) > 0:
                iret = float(p1.iloc[-1]) / float(p0.iloc[-1]) - 1.0
        out.append(dict(start=str(d0.date()), end=str(d1.date()), names=names,
                        basket=basket, index=iret))
    return tuple(out)


def _megacap_arms():
    """Run the mega-cap headline-N arms (size / growth / momentum) on the live EDGAR
    fundamentals cache → ({arm: equity}, covered-name count, {arm: holdings table}).
    (None, 0, {}) when no cache / empty coverage. Cheap (~seconds); the cap/revenue
    panels are strictly point-in-time (filing-dated), so no look-ahead leaks."""
    try:
        import pandas as _pd
        from build_megacap_report import (load_data, HEADLINE_N, K as MC_K,
                                           GLOBAL_GIANTS, META_CSV)
        from tools.megacap import run_arms, run_value_arms, ARMS
        from tools.dipbuy import run_dipbuy
        _IDXN = MEGACAP_INDEX
        data = load_data()
        if data["coverage"]["covered"] == 0:
            return None, 0, {}
        res = run_arms(data["prices"], data["slip"], data["cap"], data["yoy"],
                       n=HEADLINE_N, k=MC_K, capital=CAPITAL)
        dip_res = run_dipbuy(data["prices"], data["slip"], data["cap"],
                             n=HEADLINE_N, k=MC_K, window=DIP_WINDOW, capital=CAPITAL)
        val_res = run_value_arms(data["prices"], data["slip"], data["cap"],
                                 data["rev"], data["yoy"], n=HEADLINE_N, k=MC_K,
                                 capital=CAPITAL)
        arms = {arm: res[arm]["runs"][1.0].get("equity") for arm in ARMS}
        arms["dip"] = dip_res["runs"][1.0].get("equity")
        arms["value"] = val_res["value"]["runs"][1.0].get("equity")
        arms["garp"] = val_res["garp"]["runs"][1.0].get("equity")
        # Constructed books: 50/50 momentum+value blend (offense+defense) and vol-managed value.
        arms["combo"] = qg.blend_equity({"momentum": arms["momentum"], "value": arms["value"]},
                                        {"momentum": 0.5, "value": 0.5}, CAPITAL)
        _vt = qg.vol_target(arms["value"], target_vol=RISK_TARGET_VOL,
                            turn_cost_bps=RC_TURN_BPS) if arms["value"] is not None else {}
        arms["value_vt"] = _vt.get("equity")
        # holdings tables: name map (EDGAR universe + giants) + the Nasdaq interval index
        try:
            nm = dict(_pd.read_csv(META_CSV)[["ticker", "name"]].values)
        except Exception:
            nm = {}
        nm.update({tk: disp for disp, tk in GLOBAL_GIANTS.items()})
        giants = set(GLOBAL_GIANTS.values())
        bench = data.get("benchmarks")
        idx = (bench[_IDXN] if bench is not None and _IDXN in getattr(bench, "columns", [])
               else None)
        holdings = {arm: _arm_holdings_table(res[arm]["holdings_log"], data["prices"],
                                             idx, nm, giants) for arm in ARMS}
        holdings["dip"] = _arm_holdings_table(dip_res["holdings_log"], data["prices"],
                                              idx, nm, giants)
        for kind in ("value", "garp"):
            holdings[kind] = _arm_holdings_table(val_res[kind]["holdings_log"],
                                                 data["prices"], idx, nm, giants)
        return arms, int(data["coverage"]["covered"]), holdings
    except Exception:
        return None, 0, {}


# Mega-cap arm → (record id, display name). Names put the DISTINCTIVE word first so
# the left-menu label (r.name pre-em-dash head) is unique under the shared "Mega-cap"
# family header; the em-dash tail carries the method.
_MEGACAP_ARM_META = {
    "size": ("megacap_size", "Largest-cap — top-25 PIT market-cap"),
    "growth": ("megacap_growth", "Revenue growth — fastest trailing YoY"),
    "momentum": ("megacap_momentum", "12-1 momentum — within the mega-caps"),
}

# Value arms systematize the user's own thesis — "value at a fair price". Only revenue +
# shares are available (no earnings), so valuation = price-to-sales, US filers only (one
# revenue currency; see megacap.ps_panel). id, menu name, verdict phrase.
_VALUE_ARM_META = {
    "value": ("megacap_value", "Value — cheapest price-to-sales (US mega-caps)",
              "buying the cheapest-on-sales names"),
    "garp": ("megacap_garp", "GARP — growth at a reasonable price (US mega-caps)",
             "buying cheap-relative-to-growth names"),
}


def build_registry(variants: list, *, ensemble=None, vol_core=None, vol_core_eq=None,
                   bench=None, ew_eq=None, portfolio_roi=None,
                   train_end=TRAIN_END, val_end=VAL_END,
                   megacap_names: int = 0, megacap_arms=None,
                   megacap_holdings=None) -> list:
    """Normalize everything gather() computed into StrategyRecords — the ONE list every
    comparison surface (leaderboard, parallel chart, dossiers) renders. Adding a future
    strategy = one make_record() call here (curve-bearing) or one STATIC_RECORDS entry
    (ledger-only). No engine calls — curves come in, records go out.

    Statuses derive from the same ensemble["adopt"] branch that decides what
    local/strategy_track.csv tracks, so the ★ live row always agrees with the tracker."""
    raw, rc = variants
    ens_adopt = bool(ensemble and ensemble.get("adopt"))
    recs = []
    if ensemble and ensemble.get("equity") is not None:
        codes = " + ".join(ensemble["codes"])
        recs.append(sreg.make_record(
            "mom_ens", f"Momentum ensemble — top-{ensemble['n']} quarterly ({codes})",
            "momentum", "adopted" if ens_adopt else "candidate",
            equity=ensemble["equity"], train_end=train_end, val_end=val_end,
            adopted="2026-07-06" if ens_adopt else None,
            cost_model=f"slip + €{FEE_EUR:.0f}/order per sleeve",
            avg_exposure=1.0,
            gate=(f"pre-reg rule: min(train,val) {ensemble['ens_min']:.2f} ≥ "
                  f"{ensemble['single_min']:.2f} − 0.05 → "
                  + ("ADOPTED" if ens_adopt else "bench")),
            live=ens_adopt, color="#c586c0"))
    recs.append(sreg.make_record(
        "mom_rc", rc["label"], "momentum",
        "variant" if ens_adopt else "adopted",
        variant_of="mom_ens" if ens_adopt else None,
        equity=rc["equity"], windows=rc.get("windows"),
        train_end=train_end, val_end=val_end,
        cost_model=f"slip + €{FEE_EUR:.0f} + {RC_TURN_BPS:.0f}bp resize",
        avg_exposure=rc["perf"].get("avg_exposure"),
        gate=None if ens_adopt else "single pick_ultimate book (ensemble on the bench)",
        live=not ens_adopt, color=C_RC))
    recs.append(sreg.make_record(
        "mom_raw", "Momentum — raw (unmanaged)", "momentum", "reference",
        variant_of="mom_rc",
        equity=raw["equity"], windows=raw.get("windows"),
        train_end=train_end, val_end=val_end,
        cost_model=f"slip + €{FEE_EUR:.0f} · full-invested",
        avg_exposure=1.0,
        verdict="reference — inflated (survivorship + full exposure on a high-vol book); "
                "curve in the lab only",
        flags=("inflated",), color=C_RAW))
    if vol_core_eq is not None and vol_core:
        recs.append(sreg.make_record(
            "vol_core", "GARCH vol-managed IWDA core", "vol-managed core", "adopted",
            equity=vol_core_eq, train_end=train_end, val_end=val_end,
            adopted="2026-07-05",
            cost_model="5bp band-rebalance + €1",
            avg_exposure=vol_core.get("managed", {}).get("avg_exposure"),
            gate="PASS (pre-registered, vol lab)", href="vol.html",
            color="#569cd6"))
    if bench is not None:
        start = pd.Timestamp(START)
        for rid, col in (("bench_msci", "MSCI World"), ("bench_spx", "S&P 500")):
            if col in getattr(bench, "columns", []):
                recs.append(sreg.make_record(
                    rid, f"{col} (buy-hold)", "benchmark", "benchmark",
                    equity=bench[col].dropna().loc[start:],
                    train_end=train_end, val_end=val_end,
                    cost_model="—", avg_exposure=1.0))
    if ew_eq is not None:
        recs.append(sreg.make_record(
            "ew_baseline", "Equal-weight initial picks (buy-hold)", "benchmark",
            "benchmark", equity=ew_eq, train_end=train_end, val_end=val_end,
            cost_model="—", avg_exposure=1.0, color=theme.FG_DIM))
    if portfolio_roi is not None and not getattr(portfolio_roi, "empty", True):
        curve = 1.0 + portfolio_roi / 100.0
        recs.append(sreg.make_record(
            "portfolio", "Your portfolio (real, cash-flow-timed)", "your book",
            "portfolio", since=str(portfolio_roi.index[0].date()),
            windows=dict(full=qg.window_metrics(curve, curve.index[0], curve.index[-1])),
            cost_model="real fills",
            verdict="money-weighted real book over its own window — head-to-head below",
            color="#ffffff"))
    if megacap_arms:
        # Live results: one curve-bearing record per arm (uniform pane, registry row,
        # chart trace). Survivor-biased universe like the momentum family, so the
        # verdict leads with the internal-comparison caveat — real numbers, honestly
        # framed, no adoption gate yet.
        n_txt = f"{megacap_names} EDGAR names" if megacap_names else "live EDGAR cache"
        # The Largest-cap arm is the control every other arm is scored against: does a tilt
        # (reversal, value, GARP) beat simply owning the biggest names? Verdicts state it.
        def _full_ret(e):
            s = e.dropna() if e is not None and not getattr(e, "empty", True) else pd.Series(dtype=float)
            return float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) > 1 else float("nan")
        size_r = _full_ret(megacap_arms.get("size"))
        def _win_ret(e, key):
            s = e.dropna() if e is not None and not getattr(e, "empty", True) else pd.Series(dtype=float)
            if len(s) < 2:
                return float("nan")
            lo, hi = sreg.window_bounds(s, train_end, val_end)[key]
            return qg.window_metrics(s, lo, hi)["net_return"]
        size_val = _win_ret(megacap_arms.get("size"), "val")
        for arm, (rid, nm) in _MEGACAP_ARM_META.items():
            eq = megacap_arms.get(arm)
            if eq is None or getattr(eq, "empty", True):
                continue
            recs.append(sreg.make_record(
                rid, nm, "mega-cap", "candidate",
                equity=eq, train_end=train_end, val_end=val_end,
                href="megacap.html", cost_model="slippage (half-spread)",
                holdings=(megacap_holdings or {}).get(arm, ()),
                gate=f"incubating — top-25 PIT cap screen ({n_txt}); no pre-registered "
                     "adoption gate yet",
                verdict="survivor-biased universe → internal comparison only, not an "
                        "achievable return; full N-sweep on megacap.html"))
        # Buy-the-dip reversal book — a 4th mega-cap arm, but a distinct thesis (short-term
        # reversal, not size/growth/trend), so it gets its OWN verdict rather than the shared
        # internal-comparison line. The verdict is DATA-DRIVEN: it compares the reversal book
        # to the Largest-cap arm (its control — same universe, no dip tilt) and states plainly
        # whether the overreaction premium survives, so it can't silently rot into a false
        # win. Registered only when its curve is present.
        dip_eq = megacap_arms.get("dip")
        if dip_eq is not None and not getattr(dip_eq, "empty", True):
            dip_r = _full_ret(dip_eq)
            have_cmp = size_r == size_r and dip_r == dip_r
            beats = have_cmp and dip_r > size_r
            if have_cmp:
                finding = (
                    f"The overreaction premium does NOT survive on mega-caps: reversal "
                    f"returned {dip_r*100:+.0f}% full-window vs the Largest-cap arm's "
                    f"{size_r*100:+.0f}% — buying the biggest 1-month losers underperforms "
                    f"simply holding the biggest names. "
                    if not beats else
                    f"Reversal returned {dip_r*100:+.0f}% full-window vs the Largest-cap "
                    f"arm's {size_r*100:+.0f}%, edging its control — but on a survivor-biased "
                    f"universe, so treat it as internal comparison, not achievable alpha. ")
            else:
                finding = ("Hold the k biggest 1-month losers among the top-25 mega-caps — "
                           "read against the Largest-cap arm below. ")
            recs.append(sreg.make_record(
                "megacap_dip",
                "Dip-buy — 1-month reversal on the top-25 mega-caps", "mega-cap",
                "candidate", equity=dip_eq, train_end=train_end, val_end=val_end,
                href="megacap.html", cost_model="slippage (half-spread), t+1 fill",
                holdings=(megacap_holdings or {}).get("dip", ()),
                gate=f"incubating — hold the k biggest 1-month losers among the top-25 PIT "
                     f"mega-caps ({n_txt}); {'LAGS' if not beats else 'vs'} the Largest-cap "
                     "arm. No adoption gate pre-registered.",
                verdict="Short-term reversal on the most-efficient names — the hardest case "
                        "for the effect. " + finding + "Survivor-biased universe → an "
                        "internal comparison against the arms below, not an absolute return."))
        # Value + GARP — the user's own "value at a fair price" thesis, systematized on
        # price-to-sales (US filers only). Each carries the same data-driven verdict vs the
        # Largest-cap control.
        for kind, (rid, nm, phrase) in _VALUE_ARM_META.items():
            eq = megacap_arms.get(kind)
            if eq is None or getattr(eq, "empty", True):
                continue
            r = _full_ret(eq)
            beats = r == r and size_r == size_r and r > size_r
            if r == r and size_r == size_r:
                vfind = (f"Returned {r*100:+.0f}% full-window vs the Largest-cap arm's "
                         f"{size_r*100:+.0f}% — {phrase} "
                         f"{'beats' if beats else 'LAGS on total return'} simply holding the "
                         "biggest names. ")
            else:
                vfind = phrase.capitalize() + ". "
            # The real edge of a value tilt is downside, not upside: if it held up through
            # the validation drawdown while the Largest-cap arm bled, say so — that is where
            # 'value at a fair price, no huge bets' actually pays.
            arm_val = _win_ret(eq, "val")
            if arm_val == arm_val and size_val == size_val and arm_val > 0 > size_val:
                vfind += (f"But defensive where it counts: {arm_val*100:+.0f}% through the "
                          f"validation drawdown while the Largest-cap arm lost "
                          f"{abs(size_val)*100:.0f}% — the trade-off is upside in the bull run "
                          "for resilience in the fall. ")
            recs.append(sreg.make_record(
                rid, nm, "mega-cap", "candidate",
                equity=eq, train_end=train_end, val_end=val_end,
                href="megacap.html", cost_model="slippage (half-spread)",
                holdings=(megacap_holdings or {}).get(kind, ()),
                gate=f"incubating — top-25 PIT cap screen, US filers only (P/S needs one "
                     f"revenue currency); {n_txt}. No adoption gate pre-registered.",
                verdict="Systematizing 'value at a fair price' on mega-caps. " + vfind +
                        "US filers only (P/S currency-safety); survivor-biased universe → an "
                        "internal comparison against the arms below, not an absolute return."))
        # Constructed books from the arms above — the "can a blend beat just owning the
        # biggest?" test. combo = 50/50 momentum(offense)+value(defense); value_vt =
        # vol-managed value. Derived curves (no own holdings). The honest win condition is
        # RISK-ADJUSTED (Sharpe) + drawdown, not raw return — data-driven verdict says which.
        size_eq = megacap_arms.get("size")
        size_m = (qg.perf_metrics(size_eq.dropna())
                  if size_eq is not None and not getattr(size_eq, "empty", True) else {})
        for kind, rid, nm, desc in (
            ("combo", "megacap_combo", "Combo — 50/50 momentum + value (offense + defense)",
             "a 50/50 daily-rebalanced blend of the momentum (offense) and value (defense) arms"),
            ("value_vt", "megacap_value_vt", "Value, vol-managed — value arm at 15% target vol",
             "the value arm scaled to a 15% volatility target (de-risk only, rest in cash)")):
            eq = megacap_arms.get(kind)
            if eq is None or getattr(eq, "empty", True):
                continue
            m = qg.perf_metrics(eq.dropna())
            wc = ""
            if size_m:
                better = (m["sharpe"] > size_m["sharpe"]) or (abs(m["max_dd"]) < abs(size_m["max_dd"]))
                wc = (f"Sharpe {m['sharpe']:.2f} vs the Largest-cap arm's {size_m['sharpe']:.2f}, "
                      f"max drawdown {m['max_dd']*100:.0f}% vs {size_m['max_dd']*100:.0f}% — "
                      f"{'improves' if better else 'does not improve'} the risk-adjusted profile. ")
            recs.append(sreg.make_record(
                rid, nm, "mega-cap", "candidate",
                equity=eq, train_end=train_end, val_end=val_end,
                cost_model="slippage (half-spread)"
                           + (f" + {RC_TURN_BPS:.0f}bp resize" if kind == "value_vt" else ""),
                gate=f"incubating — constructed from the mega-cap arms ({n_txt}); the win "
                     "condition is risk-adjusted, not raw return. No adoption gate pre-registered.",
                verdict=f"Built from your own arms: {desc}. " + wc +
                        "Survivor-biased universe → an internal comparison against the arms "
                        "below, not an absolute return."))
    else:
        if megacap_names:
            mc_gate = (f"incubating — EDGAR PIT fundamentals live for {megacap_names} names; "
                       "adoption gate not yet pre-registered")
            mc_verdict = ("PIT market-cap screen → size / YoY-revenue-growth / 12-1 momentum "
                          f"arms, running on SEC-EDGAR filing-dated shares+revenue "
                          f"({megacap_names} names) — full N-sweep on megacap.html.")
            mc_flags = ()
        else:
            mc_gate = "awaiting cap data — run the EDGAR fundamentals fetch"
            mc_verdict = ("PIT market-cap screen → size / YoY-revenue-growth / 12-1 momentum "
                          "arms; no fundamentals data yet.")
            mc_flags = ("awaiting_data",)
        recs.append(sreg.make_record(
            "megacap", "Mega-cap PIT screen — size / growth / momentum", "mega-cap",
            "research", href="megacap.html",
            gate=mc_gate, verdict=mc_verdict, flags=mc_flags))
    recs.extend(sreg.STATIC_RECORDS)
    sreg.assign_colors(recs)
    return recs


def _delisting_stress(prices, slip, meta_df, sectors, spx, cfg, res, base_return,
                      membership=None, turnover=None):
    """Multi-intensity on-population delisting stress (bull/base/bear). Inject synthetic
    deaths into the LIVE names at each preset's hazard/loss, re-run the SAME strategy
    SURV_SIMS×. Observational — never feeds selection/sizing. Seeds 0..K-1 shared.

    Returns {"presets": {name: stress_summarize(...) + hazard/loss}, "clean": {...},
    "sims": K}."""
    dead = set(meta_df.loc[meta_df["delisting_date"].notna(), "ticker"])
    live_cols = [c for c in prices.columns if c not in dead]
    base_map = delisting_map(meta_df)
    base_deaths = death_map(meta_df)
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
    prices = to_xetra_calendar(prices)                         # L&S/XETRA sessions only
    prices = winsorize_prices(prices, cap=WINSOR_CAP)          # de-glitch the raw feed
    meta_df = pd.read_csv(META_CSV)
    n_live = int(meta_df["delisting_date"].isna().sum())
    meta = {r["ticker"]: dict(r) for _, r in meta_df.iterrows()}
    slip = {t: _slip(m) for t, m in meta.items() if t in prices.columns}
    sectors = {t: m.get("sector") for t, m in meta.items()}     # real GICS (enrich_sectors)
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

    # ── Dynamic config selection: full 64-config grid, then pick_ultimate ──
    grid = run_grid(prices, slip, sectors=sectors, benchmark=spx, pit=pit, start=START,
                    configs=ALL_CONFIGS, train_end=TRAIN_END, val_end=VAL_END, capital=CAPITAL,
                    lookback=LOOKBACK, skip=SKIP, execute_lag=EXEC_LAG,
                    turnover=turnover, turn_floor=MIN_TURNOVER)
    picked = pick_ultimate(grid, capital=CAPITAL, fee_eur=FEE_EUR)
    cfg = picked["config"] if picked else STRATEGY

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
    # Equal-weight buy-hold of the first rebalance's picks — the survivorship-honest
    # baseline. Computed here (not at render time) so the registry can carry it.
    window = _equity_window(res)
    first_picks = next((h["picks"] for h in res["holdings_log"] if h["picks"]), [])
    ew_eq = equal_weight_curve(prices, first_picks, window, CAPITAL) if first_picks else None

    # ── Upper bound: drop dead names that were never TR-tradeable ──
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
    # ── On-population survivorship (holding leak) ──
    surv_inject = None
    delisting_stress = None
    if SURV_SIMS > 0:
        try:
            delisting_stress = _delisting_stress(prices, slip, meta_df, sectors, spx, cfg, res,
                                                 base_return=res["runs"][1.0]["stats"]["net_return"],
                                                 membership=membership, turnover=turnover)
            surv_inject = delisting_stress["presets"]["base"]   # base == single-intensity
        except Exception as e:
            print(f"delisting-stress skipped: {e}", file=sys.stderr)
            delisting_stress = None
            surv_inject = None

    # ── Significance & robustness: random-selection null, deflated Sharpe, bootstrap ──
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
    test_canon = qg.window_metrics(eq, ve + pd.Timedelta(days=1), eq.index[-1])
    quant = dict(perf=qg.perf_metrics(eq), bench=qg.vs_benchmark(eq, spx),
                 trades=qg.trade_metrics(tr, CAPITAL, years), roll=qg.rolling_sharpe(eq),
                 grade=qg.grade(test_canon.get("sharpe", 0.0), dsr["dsr"], mc["p_sharpe"], overlap),
                 isin_overlap=overlap,
                 vol_target=qg.vol_target(eq, target_vol=RISK_TARGET_VOL, turn_cost_bps=RC_TURN_BPS))

    # ── Observational diagnostics (read-only; NEVER feed selection or sizing) ──
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

    # ── Regime attribution (observational): three lenses, same cfg ──
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
        pre = _stats_slice(eq, tr, eq.index[0], ve, CAPITAL)     # train+val combined
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

    # ── Factor-spanning regression (observational, never selection) ──
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

    # ── Adopted overlay: GARCH vol-managed MSCI World core (pre-registered gates) ──
    vol_core = None
    vol_core_eq = None
    try:
        from tools import vol_forecast as vf
        core_px = cached_price_history(["IWDA.AS"], period="9y",
                                       force=refresh)["IWDA.AS"].dropna()
        core_r = core_px.pct_change().dropna()
        fc = vf.garch11_vol(core_r)
        vm = vf.vol_managed(core_r, fc, target_vol=RISK_TARGET_VOL,
                            band=0.10, cost_bps=5.0, fee_eur=1.0,
                            capital=CAPITAL)
        bh = qg.perf_metrics((1.0 + core_r).cumprod() * CAPITAL)
        w_now = float(min(RISK_TARGET_VOL / fc.iloc[-1], 1.0)) if len(fc) else None
        vol_core_eq = vm.get("equity")               # the registry's comparable curve
        vol_core = dict(etf="MSCI World (IWDA.AS)", bh=bh,
                        managed={k: vm[k] for k in ("sharpe", "ann_return",
                                                    "max_dd", "avg_exposure",
                                                    "n_trades_per_year")},
                        fc_now=float(fc.iloc[-1]) if len(fc) else None,
                        w_now=w_now, asof=str(core_r.index[-1].date()))
    except Exception as e:
        print(f"vol-core overlay skipped: {e}", file=sys.stderr)

    # ── Ensemble candidate (pre-registered method upgrade, +1 ledger trial):
    #    equal-capital average of the top-3 QUARTERLY configs by min(train,val).
    #    ADOPTION RULE (fixed ex ante): adopt over the single pick iff
    #    ensemble min(train,val) ≥ single's min(train,val) − 0.05.
    ensemble = None
    try:
        top3 = pick_top_n(grid, n=3, freq="Q", capital=CAPITAL, fee_eur=FEE_EUR)
        if picked is not None and len(top3) >= 2:
            sleeves = []
            sleeve_books = []          # per-sleeve holdings for the buy-now panel + history
            for c in top3:
                rr = run_momentum(prices, slip, k=c["config"].slots,
                                  lookback=LOOKBACK, skip=SKIP,
                                  capital=CAPITAL / len(top3),
                                  cost_mults=(1.0,), freq="Q",
                                  liq_max=LIQ_MAX, fee_eur=FEE_EUR,
                                  min_price=MIN_PRICE, start=START,
                                  sectors=sectors, benchmark=spx, pit=pit,
                                  execute_lag=EXEC_LAG, turnover=turnover,
                                  turn_floor=MIN_TURNOVER,
                                  vol_adjust=c["config"].vol_adjust,
                                  sector_neutral=c["config"].sector_neutral,
                                  trend_filter=c["config"].trend_filter,
                                  lazy=c["config"].lazy)
                sleeves.append(rr["runs"][1.0])
                sleeve_books.append(dict(code=c["code"], slots=c["config"].slots,
                                         holdings_log=rr["holdings_log"]))
            idx = sleeves[0]["equity"].index
            for sl in sleeves[1:]:
                idx = idx.union(sl["equity"].index)
            parts = [sl["equity"].reindex(idx).ffill()
                     .fillna(CAPITAL / len(top3)) for sl in sleeves]
            ens_eq = sum(parts)
            e_tr = _stats_slice(ens_eq, [], ens_eq.index[0], te, CAPITAL)
            e_va = _stats_slice(ens_eq, [], te + pd.Timedelta(days=1), ve, CAPITAL)
            e_te = _stats_slice(ens_eq, [], ve + pd.Timedelta(days=1),
                                ens_eq.index[-1], CAPITAL)
            ens_min = min(e_tr["sharpe"], e_va["sharpe"])
            single_min = min(picked["train"]["sharpe"], picked["val"]["sharpe"])
            qd = [dd for dd in rebalance_dates(ens_eq.index, "Q")]
            pr = ens_eq.reindex(qd).dropna().pct_change().dropna()
            dsr5 = sig.deflated_sharpe_ratio(
                pr, trial_sharpes, ppy=4.0,
                n_trials_effective=(n_grid + 1) * PHANTOM_MULT)["dsr"] \
                if len(pr) > 8 else None
            ens_alpha = None
            if factor_reg is not None:
                try:
                    areg = factors.factor_regression(_excess_usd(ens_eq), fac)
                    if areg:
                        akey = "FF5+WML" if "FF5+WML" in areg else next(iter(areg))
                        ens_alpha = dict(model=akey,
                                         alpha_ann=areg[akey]["alpha_ann"],
                                         alpha_t=areg[akey]["alpha_t"],
                                         n=areg[akey]["n"])
                except Exception:
                    ens_alpha = None
            ensemble = dict(codes=[c["code"] for c in top3], n=len(top3),
                            sleeves=sleeve_books,
                            train=e_tr, val=e_va, test=e_te,
                            max_dd=qg.perf_metrics(ens_eq).get("max_dd",
                                                               float("nan")),
                            trades_per_year=float(sum(c["trades_per_year"]
                                                      for c in top3)),
                            ens_min=float(ens_min), single_min=float(single_min),
                            single_code=picked["code"],
                            adopt=bool(ens_min >= single_min - 0.05),
                            dsr5=dsr5, alpha=ens_alpha, equity=ens_eq)
    except Exception as e:
        print(f"ensemble skipped: {e}", file=sys.stderr)

    # ── Live tracking (pre-registered kill criteria, tools/track.py) ──
    track_d = None
    try:
        from tools import track as _track
        live_eq = (ensemble["equity"] if ensemble and ensemble["adopt"]
                   else variants[1]["equity"])
        r_last = float(live_eq.pct_change().dropna().iloc[-1])
        tpath = ROOT / "local" / "strategy_track.csv"
        _track.append_snapshot(str(live_eq.index[-1].date()),
                               dict(r_live=r_last,
                                    equity=float(live_eq.iloc[-1]),
                                    pick=(f"ens[{','.join(ensemble['codes'])}]"
                                          if ensemble and ensemble["adopt"]
                                          else cfg.code)),
                               path=tpath)
        trk = _track.read_track(tpath)
        lv = _track.live_vs_backtest(trk, sharpe_lo=ci.get("sharpe_lo", 0.0),
                                     backtest_max_dd=quant["perf"].get(
                                         "max_dd", -0.3))
        track_d = dict(n=lv.get("n", 0), needed=lv.get("needed", 63),
                       kill=lv.get("kill", False),
                       reasons=lv.get("reasons", []),
                       path="local/strategy_track.csv")
    except Exception as e:
        print(f"tracking skipped: {e}", file=sys.stderr)

    # ── Venture instrumentation (charter M-A): LIVE path only ──
    venture = None
    ritual = None
    try:
        from tools import track as _tr2
        from tools import venture as vt
        trk2 = _tr2.read_track(ROOT / "local" / "strategy_track.csv")
        cf_path = ROOT / "local" / "venture_cashflows.csv"
        if trk2 is not None and "equity" in trk2.columns \
                and len(trk2.dropna(subset=["equity"])) >= 2:
            live_book = trk2["equity"].dropna()
            if not cf_path.exists():
                cf_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([dict(date=str(live_book.index[0].date()),
                                   amount=float(CAPITAL),
                                   note="seed (paper)")]).to_csv(cf_path,
                                                                 index=False)
            cf = pd.read_csv(cf_path, parse_dates=["date"])
            iwda = cached_price_history(["IWDA.AS"], period="9y",
                                        force=refresh)["IWDA.AS"].dropna()
            vs = vt.venture_summary(cf, live_book, iwda)
            venture = dict(**vs, live=True, satellite=dict(live=0, cap=3),
                           deposits=int(len(cf)))
        else:
            n_rows = 0 if trk2 is None else int(len(trk2))
            venture = dict(live=False, n_rows=n_rows,
                           satellite=dict(live=0, cap=3))
    except Exception as e:
        print(f"venture section skipped: {e}", file=sys.stderr)
    try:
        items = []

        def _age(name, last_iso):
            if not last_iso:
                items.append(dict(name=name, last="never", age_days=9999,
                                  alarm=True))
                return
            age = (pd.Timestamp.now() - pd.Timestamp(last_iso)).days
            items.append(dict(name=name, last=str(last_iso)[:10],
                              age_days=int(age), alarm=bool(age > 35)))

        snap = universe_snapshot.load_store()
        _age("TR snapshot", snap["snapshot_date"].max() if len(snap) else None)
        for name, path, col in (
                ("Short-register fetch",
                 ROOT / "data/universe/short_positions.csv", "fetched_at"),
                ("BaFin dealings fetch",
                 ROOT / "data/universe/insider_dealings.csv", "fetched_at")):
            last = None
            if path.exists():
                df_ = pd.read_csv(path, usecols=[col])
                last = df_[col].max()
            _age(name, last)
        _age("Universe price fetch",
             pd.Timestamp(PRICES_CSV.stat().st_mtime, unit="s").isoformat()
             if PRICES_CSV.exists() else None)
        ritual = dict(items=items)
    except Exception as e:
        print(f"ritual section skipped: {e}", file=sys.stderr)

    # ── Scenario fan (observational) ──
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

    # ── Your real portfolio's ROI + a same-scale strategy run, for the head-to-head ──
    portfolio_roi, vs_scale, portfolio_bench = None, None, {}
    pf_csv = ROOT / "input" / "portfolio.csv"
    if pf_csv.exists():
        txns = None
        try:
            txns = parse_portfolio(pf_csv)["transactions"]
            pr, pf_bench = build_roi_timeseries(txns)   # keep the cash-flow-matched benchmarks
            if pr is not None and not pr.empty:
                portfolio_roi = pr
                portfolio_bench = pf_bench or {}        # the /portfolio report's own market view
        except Exception:
            portfolio_roi = None
        try:                                    # vs_scale is separable — a failure here
            invested = sum(float(t["price"])    # must NOT wipe portfolio_roi parsed above
                           for t in (txns or []) if t["action"] == "buy")
            if invested > 0:
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
            vs_scale = None

    n_countries = len({m.get("country") for m in meta.values()} - {"—", None})
    mc_arms, mc_names, mc_holdings = _megacap_arms()   # live screen → curves + holdings tables
    registry = build_registry(variants, ensemble=ensemble, vol_core=vol_core,
                              vol_core_eq=vol_core_eq, bench=bench, ew_eq=ew_eq,
                              portfolio_roi=portfolio_roi, megacap_names=mc_names,
                              megacap_arms=mc_arms, megacap_holdings=mc_holdings)
    return dict(prices=prices, res=res, benchmarks=bench, capital=CAPITAL, meta=meta, quant=quant,
                portfolio_roi=portfolio_roi, portfolio_bench=portfolio_bench,
                vs_scale=vs_scale, variants=variants,
                registry=registry, ew_eq=ew_eq,
                strategy=cfg, raw_windows=dict(train=train, val=val, test=test),
                graveyard_hits=hits,
                surv_inject=surv_inject, delisting_stress=delisting_stress,
                factor_reg=factor_reg, vol_core=vol_core,
                ensemble={k: v for k, v in (ensemble or {}).items()
                          if k != "equity"} or None,
                track=track_d, venture=venture, ritual=ritual,
                grid=grid, n_dead=int(death_mask(meta_df).sum()),
                turnover_pit=turnover is not None,
                n_countries=n_countries,
                n_live=n_live, bounds=bounds, regime=reg, eff_bets=eff_bets,
                regime_attr=ratt, scenarios=scenarios,
                significance=dict(mc=mc, dsr=dsr, ci=ci, ppy=ppy,
                                  dsr_phantom=dsr_phantom, t_stat=t_stat,
                                  phantom_mult=PHANTOM_MULT))


def build(d: dict, public: bool = False) -> str:
    """Render the page — all presentation lives in strategy_ui."""
    return ui.render(d, public)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    d = gather(refresh=args.refresh)
    local = ROOT / "local/strategy.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d))                          # live/local only — no docs/ export
    print(f"wrote {local}  (strategy {d['strategy'].code}: registry of "
          f"{len(d.get('registry') or [])} records)")
    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
