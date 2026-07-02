import re

import numpy as np
import pandas as pd

import build_strategy_report as bs
from tools import quant_grade as qg
from tools.momentum import run_momentum
from tools.momentum_grid import _stats_slice


def _fake_d():
    idx = pd.bdate_range("2018-01-01", periods=500)
    rng = np.random.default_rng(0)
    px = pd.DataFrame({f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 500)))
                       for i in range(15)}, index=idx)
    slip = {t: 10 for t in px.columns}
    res = run_momentum(px, slip, k=5, lookback=200, skip=10, cost_mults=(1.0,))
    eq, tr = res["runs"][1.0]["equity"], res["runs"][1.0]["trades"]
    te, ve = pd.Timestamp("2019-06-30"), pd.Timestamp("2019-09-30")
    train = _stats_slice(eq, tr, eq.index[0], te, 10_000.0)
    val = _stats_slice(eq, tr, te + pd.Timedelta(days=1), ve, 10_000.0)
    test = _stats_slice(eq, tr, ve + pd.Timedelta(days=1), eq.index[-1], 10_000.0)
    # a synthetic benchmark so the curve / yearly / vs-benchmark rows have something to chew on
    spx = pd.Series(3000 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, 500))), index=idx, name="S&P 500")
    benchmarks = spx.to_frame()
    quant = dict(
        perf=qg.perf_metrics(eq), bench=qg.vs_benchmark(eq, spx),
        trades=qg.trade_metrics(tr, 10_000.0, years=2.0), roll=qg.rolling_sharpe(eq),
        grade=qg.grade(test["sharpe"], 0.78, 0.04, 0.02), isin_overlap=0.02,
        vol_target=qg.vol_target(eq, target_vol=bs.RISK_TARGET_VOL))
    variants = bs.build_variants(res, quant["vol_target"], spx, train, val, test, quant, 10_000.0,
                                 dsr=0.78, mc_p=0.04, overlap=0.02,
                                 train_end="2019-06-30", val_end="2019-09-30")
    # observational diagnostics (synthetic, small) — read-only HMM regime + PCA effective bets
    ridx = idx[200:]
    ramp = np.linspace(0.2, 0.8, len(ridx))
    reg = pd.DataFrame({"prob_risk_off": ramp, "risk_off": ramp > 0.5,
                        "trend_broken": ramp > 0.5}, index=ridx)
    eff_bets = [dict(date=h["date"], n_eff_pca=3.0 + 0.1 * i, pc1_share=0.4,
                     n_eff_weight=5.0, k=5)
                for i, h in enumerate(res["holdings_log"]) if h["picks"]]
    return dict(prices=px, res=res, benchmarks=benchmarks, capital=10_000.0,
                meta={t: dict(name=t, local_id="000", country="X", sector="Y") for t in px.columns},
                strategy=bs.STRATEGY, quant=quant, variants=variants, portfolio_roi=None,
                train=train, val=val, test=test, graveyard_hits=0,
                surv_inject=dict(base_return=1.2, sims=15, mean_return=1.19, delta_mean=-0.01,
                                 delta_lo=-0.05, delta_hi=0.02, hits_mean=0.3, deaths_mean=12.0,
                                 avoidance_rate=0.975),
                regime_attr=dict(
                    strip=dict(full_test_sharpe=1.47, full_test_return=2.45,
                               strip_test_sharpe=1.05, strip_test_return=1.1,
                               n_dropped=1193, n_kept=2000, dropped=["Industrials", "Technology"]),
                    cond=dict(on=dict(n_days=900, sharpe=1.8, cum_return=2.0, ret_share=1.1),
                              off=dict(n_days=300, sharpe=-0.4, cum_return=-0.1, ret_share=-0.1),
                              total_return=2.3),
                    pre=dict(sharpe=1.02, net_return=3.9, end="2023-12-31")),
                scenarios=dict(horizon=252, block=21, n_sims=2000, tilt=3.0, frac_off=0.30,
                               scenarios={n: dict(
                                   p5=np.linspace(1.0, 1.05 + 0.05 * i, 253),
                                   p50=np.linspace(1.0, 1.10 + 0.05 * i, 253),
                                   p95=np.linspace(1.0, 1.20 + 0.05 * i, 253),
                                   term_p5=1.05 + 0.05 * i, term_p50=1.10 + 0.05 * i,
                                   term_p95=1.20 + 0.05 * i)
                                   for i, n in enumerate(("bear", "base", "bull"))}),
                regime=reg, eff_bets=eff_bets,
                n_dead=42, n_countries=1, n_live=10,
                significance=dict(
                    mc=dict(null_sharpe=np.array([0.1, 0.2, 0.3, 0.25]), strat_sharpe=0.6,
                            null_sharpe_median=0.22, p_sharpe=0.04, p_total=0.05, n_trials=1000),
                    dsr=dict(dsr=0.78, n_trials=32, n_trials_observed=32, T=20,
                             sr_benchmark_annual=1.1, sharpe_annual=1.4),
                    dsr_phantom=[dict(mult=1, n=32, dsr=0.78, sr_benchmark_annual=1.1),
                                 dict(mult=5, n=160, dsr=0.71, sr_benchmark_annual=1.4),
                                 dict(mult=10, n=320, dsr=0.66, sr_benchmark_annual=1.6)],
                    t_stat=2.6, phantom_mult=5,
                    ci=dict(conf=95, sharpe=1.2, sharpe_lo=0.4, sharpe_hi=1.9,
                            cagr=0.3, cagr_lo=0.1, cagr_hi=0.5),
                    ppy=4.0))


def test_strategy_page_builds():
    html = bs.build(_fake_d(), public=False)
    assert "<html" in html.lower() and "strategy" in html.lower()
    assert "Validation" in html and bs.STRATEGY.code in html
    # the page now leads with the risk-conscious result, not a two-way "billed equally" split
    assert "Risk-conscious" in html


def test_two_variant_bundles():
    d = _fake_d()
    raw, rc = d["variants"]
    assert raw["key"] == "raw" and rc["key"] == "rc"
    # selection is identical across versions (same picks log, same trades)
    assert raw["holdings_log"] is rc["holdings_log"]
    assert raw["trades"] is rc["trades"]
    # vol-targeting only ever de-risks → exposure in (0, 1]
    assert 0.0 < rc["exposure_latest"] <= 1.0
    assert rc["perf"]["ann_vol"] < raw["perf"]["ann_vol"] + 1e-9   # de-risked vs raw


# ── the reframe: real results lead, raw vanity gone from the top ────────────────

def test_headline_leads_with_real_results():
    d = _fake_d()
    html = bs.build(d, public=False)
    # the old alarmist "the headline is inflated" banner is gone
    assert "Read this first" not in html
    h = bs.sec_headline(d)
    assert "The result" in h                                      # confident, real-results lead
    assert "Monte Carlo" in h                                     # the validation up front
    assert "held-out" in h.lower() or "out-of-sample" in h.lower()
    assert "Deflated Sharpe" in h or "P(true" in h
    # it leads the page, above the picks and the equity chart
    assert html.index("The result") < html.index("Current top picks")
    assert html.index("The result") < html.index("Walk-forward equity")


def test_headline_no_euro_amounts():
    assert "€" not in bs.sec_headline(_fake_d())                  # percentages / Sharpe only


def test_raw_vanity_is_off_the_main_page():
    html = bs.build(_fake_d(), public=False)
    main = html.split("Research lab")[0]                          # everything above the lab
    # the raw full-invested strategy (block + equity trace) is NOT on the main page anymore
    assert "Raw Original" not in main
    assert "Original (raw, full-invested)" not in main           # no raw rocket trace up top
    assert "Two ways to run it" not in main                       # the side-by-side framing is gone
    assert "Risk-conscious" in main                               # the risk-conscious book is the focus


def test_raw_reference_lives_only_in_the_lab():
    d = _fake_d()
    html = bs.build(d, public=False)
    # raw isn't deleted — it's demoted to the lab as an inflation reference
    marker = "Raw Original — reference only"
    assert marker in html and html.index(marker) > html.index("Research lab")
    h = bs.sec_raw_reference(d)
    assert "reference" in h.lower() and "inflated" in h.lower()   # framed as inflated, not a headline
    # public build drops the whole lab (and the raw with it)
    assert "Research lab" not in bs.build(d, public=True)


def test_curve_is_risk_conscious_only():
    h = bs.sec_curve_compare(_fake_d())
    assert "Walk-forward equity" in h
    assert "Original" not in h                                    # no raw rocket line on the top chart
    assert "risk-conscious" in h.lower()


def test_picks_shows_the_risk_conscious_book():
    d = _fake_d()
    h = bs.sec_picks_compare(d)
    assert "Current top picks" in h and "Risk-conscious" in h
    picks = next(hh["picks"] for hh in reversed(d["res"]["holdings_log"]) if hh["picks"])
    for t in picks:                                              # single column now — each name once
        disp = str(d["meta"][t].get("home") or t).split(".")[0]
        assert f">{disp}<" in h


def test_grade_section_has_no_scare_box():
    h = bs.sec_grade_compare(_fake_d(), public=False)
    assert "Quant scorecard" in h
    assert "not clean alpha" not in h.lower()                    # the mid-page red box is gone
    # the grade + honest deductions still render, just calmly
    assert "Verdict" in h


def test_significance_section_rendered_once():
    html = bs.build(_fake_d(), public=False)
    assert html.count("Significance &amp; robustness") == 1
    assert "Monte" in html or "random books" in html            # the validation is present


def test_phantom_trials_block_shows_decay_and_harvey():
    html = bs.sec_significance(_fake_d(), public=False)
    assert "file-drawer" in html.lower() and "phantom trials" in html.lower()
    # the ladder shows the grid-only and the raised lifetime counts
    assert "Grid only" in html and "×5 lifetime" in html and "×10 lifetime" in html
    assert "66%" in html                                       # the pessimistic ×10 P(real>0)
    assert "t-stat" in html and "3.0" in html                 # Harvey hurdle framing
    assert "subjective estimate" in html                      # honesty: it's a range, not precise


def test_no_info_dropped_from_page():
    html = bs.build(_fake_d(), public=False)
    for phrase in ["Current top picks", "Walk-forward equity", "Performance",
                   "Quant scorecard", "Deflated Sharpe",
                   "Yearly P&amp;L", "Every rebalance", "survivorship is NOT corrected",
                   "Regime", "Concentration", "Capacity", "Research lab"]:
        assert phrase in html, f"dropped: {phrase!r}"


def test_strategy_public_no_euro_amounts():
    html = bs.build(_fake_d(), public=True)
    euros = re.findall(r"€[0-9][0-9.,]*", html)
    assert all(e == "€1" for e in euros), euros


def test_diagnostics_section_renders_both_lenses():
    html = bs.sec_diagnostics(_fake_d(), public=False)
    assert "HMM" in html and "Effective bets" in html
    assert "not traded" in html                                   # observational framing, explicit
    assert "agreement" in html.lower() and "sector-neutral" in html  # the candid takeaways


def test_diagnostics_sits_before_caveats_in_page():
    html = bs.build(_fake_d(), public=False)
    assert "Observational diagnostics" in html
    assert html.index("Observational diagnostics") < html.index("survivorship is NOT corrected")


def test_caveat_shows_onpopulation_survivorship_result():
    d = _fake_d()
    html = bs.sec_caveat(d)
    # the on-population injection answers the ~2%-overlap complaint with a number
    assert "On-population" in html
    assert "held a name into its delisting" in html
    assert "sold" in html and "%" in html                        # leads with the stable avoidance rate
    assert "pessimistic" in html                                 # frames the return drag as worst-case
    # still names the membership leak as the UNfixed, real problem
    assert "membership" in html and "absent winners" in html


def test_caveat_graceful_without_injection():
    d = _fake_d()
    d.pop("surv_inject")
    html = bs.sec_caveat(d)                                       # gather() may have skipped it
    assert "On-population" not in html                            # no half-rendered study
    assert "survivorship is NOT corrected" in html.lower() or "NOT corrected" in html


def test_diagnostics_absent_renders_nothing():
    d = _fake_d()
    d.pop("regime")
    d.pop("eff_bets")
    assert bs.sec_diagnostics(d, public=False) == ""             # graceful when gather skipped them


def test_regime_attribution_renders_three_lenses():
    d = _fake_d()
    html = bs.sec_regime(d, public=False)
    assert "Regime attribution" in html
    assert "Technology" in html and "Industrials" in html        # (1) sector strip named
    assert "risk-on" in html and "risk-off" in html              # (2) HMM-conditional split
    assert "2023" in html                                        # (3) pre-2024 holdout
    # leads with the honest retained-Sharpe framing, not a scary single number
    assert "retained" in html and "regime-" in html


def test_regime_attribution_in_full_page_before_caveat():
    html = bs.build(_fake_d(), public=False)
    assert "Regime attribution" in html
    assert html.index("Regime attribution") < html.index("survivorship is NOT corrected")


def test_regime_attribution_absent_renders_nothing():
    d = _fake_d()
    d.pop("regime_attr")
    assert bs.sec_regime(d, public=False) == ""                  # graceful when gather skipped it


def test_regime_attribution_public_no_euro():
    d = _fake_d()
    assert "€" not in bs.sec_regime(d, public=True)              # ratios/percentages only


def test_scenario_fan_renders_bear_base_bull():
    d = _fake_d()
    h = bs.sec_scenarios(d, public=False)
    assert "Scenario fan" in h
    assert "Bear" in h and "Base" in h and "Bull" in h
    assert "block bootstrap" in h.lower()
    # framed honestly: a sensitivity, observational, inherits the survivor universe
    assert "sensitivity" in h.lower() and "not a forecast" in h.lower()
    assert "survivor" in h.lower()


def test_scenario_absent_renders_nothing():
    d = _fake_d()
    d.pop("scenarios")
    assert bs.sec_scenarios(d, public=False) == ""              # graceful when gather skipped it


def test_scenario_public_no_euro():
    assert "€" not in bs.sec_scenarios(_fake_d(), public=True)  # multiples / percentages only


def test_scenario_sits_after_regime_before_caveat():
    html = bs.build(_fake_d(), public=False)
    assert "Scenario fan" in html
    assert (html.index("Regime attribution") < html.index("Scenario fan")
            < html.index("survivorship is NOT corrected"))
