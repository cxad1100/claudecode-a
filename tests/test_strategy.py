import re

import numpy as np
import pandas as pd

import build_strategy_report as bs
import strategy_ui as ui
from tools import quant_grade as qg
from tools import strategy_registry as sreg
from tools.momentum import run_momentum
from tools.momentum_grid import _stats_slice, MomentumConfig

TE, VE = "2019-06-30", "2019-09-30"          # fixture window boundaries


def _fake_d():
    idx = pd.bdate_range("2018-01-01", periods=500)
    rng = np.random.default_rng(0)
    px = pd.DataFrame({f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 500)))
                       for i in range(15)}, index=idx)
    slip = {t: 10 for t in px.columns}
    res = run_momentum(px, slip, k=5, lookback=200, skip=10, cost_mults=(1.0,))
    eq, tr = res["runs"][1.0]["equity"], res["runs"][1.0]["trades"]
    te, ve = pd.Timestamp(TE), pd.Timestamp(VE)
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
                                 train_end=TE, val_end=VE)
    # ensemble + vol-core curves for the registry (synthetic but full-shape)
    ens_full = dict(
        codes=["···DEF", "A···EF", "A···E·"], n=3,
        sleeves=[dict(code=c, slots=5, holdings_log=res["holdings_log"])
                 for c in ("···DEF", "A···EF", "A···E·")],
        train=dict(sharpe=0.78, net_return=2.1), val=dict(sharpe=0.95, net_return=0.4),
        test=dict(sharpe=1.05, net_return=0.35),
        max_dd=-0.31, trades_per_year=24.0,
        ens_min=0.86, single_code="A····F", single_min=0.85,
        adopt=True, dsr5=0.57,
        alpha=dict(model="FF5+WML", alpha_ann=0.17, alpha_t=1.85, n=2100),
        equity=eq * 1.01)
    vol_core_eq = pd.Series(10_000 * np.exp(np.cumsum(rng.normal(0.0004, 0.006, 500))), index=idx)
    vol_core = dict(
        etf="MSCI World (IWDA.AS)",
        bh=dict(sharpe=0.81, ann_return=0.11, max_dd=-0.274),
        managed=dict(sharpe=0.89, ann_return=0.10, max_dd=-0.191,
                     avg_exposure=0.78, n_trades_per_year=9.0),
        fc_now=0.11, w_now=1.0, asof="2026-07-04")
    ew_eq = eq * 0.98
    portfolio_roi = pd.Series(np.linspace(0.0, 20.0, 300), index=idx[200:])
    registry = bs.build_registry(variants, ensemble=ens_full, vol_core=vol_core,
                                 vol_core_eq=vol_core_eq, bench=benchmarks, ew_eq=ew_eq,
                                 portfolio_roi=portfolio_roi,
                                 train_end=TE, val_end=VE)
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
                strategy=bs.STRATEGY, quant=quant, variants=variants,
                registry=registry, ew_eq=ew_eq, portfolio_roi=portfolio_roi, vs_scale=None,
                vol_core=vol_core,
                ensemble={k: v for k, v in ens_full.items() if k != "equity"},
                raw_windows=dict(train=train, val=val, test=test), graveyard_hits=0,
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
    # both bundles carry the canonical display windows next to the selection windows
    for v in (raw, rc):
        assert set(v["windows"]) == {"train", "val", "test", "full"}
        assert "net_return" in v["windows"]["full"]


# ── the registry framework: one list, one basis, one chart ──────────────────────

def test_registry_schema_and_single_live_row():
    d = _fake_d()
    recs = d["registry"]
    assert recs, "registry empty"
    for r in recs:
        assert r.status in sreg.STATUSES, r.status
        assert r.id and r.name and r.family
    live = [r for r in recs if r.live]
    assert len(live) == 1                        # exactly one tracked book
    # the live flag derives from the same branch as the tracker: ensemble adopted here
    assert live[0].id == "mom_ens" and d["ensemble"]["adopt"]
    # the single-config book becomes a variant of the ensemble when the ensemble is live
    rc_rec = next(r for r in recs if r.id == "mom_rc")
    assert rc_rec.status == "variant" and rc_rec.variant_of == "mom_ens"
    # the raw reference is flagged, never live
    raw_rec = next(r for r in recs if r.id == "mom_raw")
    assert raw_rec.status == "reference" and "inflated" in raw_rec.flags and not raw_rec.live


def test_registry_live_row_follows_adoption_rule():
    d = _fake_d()
    ens = dict(d["ensemble"], adopt=False, equity=d["variants"][0]["equity"] * 1.01)
    recs = bs.build_registry(d["variants"], ensemble=ens, vol_core=None, vol_core_eq=None,
                             bench=None, ew_eq=None, portfolio_roi=None)
    live = [r for r in recs if r.live]
    assert len(live) == 1 and live[0].id == "mom_rc"        # bench → single book is live
    ens_rec = next(r for r in recs if r.id == "mom_ens")
    assert ens_rec.status == "candidate"


def test_registry_leaderboard_renders_rows_and_ledger():
    d = _fake_d()
    h = ui.sec_registry(d, public=False)
    assert "Strategy registry" in h
    assert "★" in h                                          # live row marked
    assert "reference, inflated" in h                        # raw row flagged
    assert "Raw Original" not in h                           # lab marker stays unique to the lab
    assert "GARCH vol-managed IWDA core" in h
    assert "Your portfolio" in h                             # private build shows the real book row
    # killed/cut strategies live ONLY inside the collapsed ledger
    assert "Registry ledger" in h
    for name in ("NN event", "Lead-lag", "Tax-loss"):
        assert h.index(name.split()[0]) > h.index("<details>")
    # status badges present
    assert "class='badge'" in h and "adopted" in h and "killed" in h


def test_registry_public_hides_portfolio_row_and_euro():
    d = _fake_d()
    h = ui.sec_registry(d, public=True)
    assert "Your portfolio" not in h
    euros = re.findall(r"€[0-9][0-9.,]*", h)
    assert all(e == "€1" for e in euros), euros


def test_registry_metrics_are_uniform_across_surfaces():
    """The SAME canonical number (same function, same window) for the ★ LIVE book
    appears in the headline, the leaderboard row and the performance matrix — no
    more different numbers under the same label."""
    d = _fake_d()
    live = next(r for r in d["registry"] if r.live)
    s_txt = f"{live.windows['test']['sharpe']:.2f}"
    assert s_txt in ui.sec_headline(d)
    assert s_txt in ui.sec_registry(d, public=False)
    assert s_txt in ui.sec_perf_compare(d, public=False)
    # and the headline names the live book instead of quoting a different variant
    assert live.name.split(" — ")[0] in ui.sec_headline(d)


def test_family_ordering_is_contiguous():
    d = _fake_d()
    recs = sreg.family_ordered(d["registry"])
    fams = [r.family for r in recs if r.status in ("adopted", "candidate", "variant",
                                                   "reference")]
    # momentum block unbroken: ens, rc, raw together, never split by vol core
    first_mom, last_mom = fams.index("momentum"), len(fams) - 1 - fams[::-1].index("momentum")
    assert all(f == "momentum" for f in fams[first_mom:last_mom + 1])
    h = ui.sec_registry(d, public=False)
    assert (h.index("Momentum ensemble") < h.index("Risk-conscious")
            < h.index("raw (unmanaged)") < h.index("GARCH vol-managed IWDA core"))


def test_window_labels_derive_from_constants():
    lbl = sreg.window_labels("2018-01-01", "2021-12-31", "2023-12-31")
    assert lbl == dict(train="Train 2018–21", val="Validation 2022–23",
                       test="Test 2024→", full="Full 2018→")
    lbl2 = sreg.window_labels("2019-01-01", "2022-12-31", "2024-12-31")
    assert lbl2["test"] == "Test 2025→" and lbl2["train"] == "Train 2019–22"


def test_future_strategy_is_one_record():
    """The framework contract: appending ONE record adds a leaderboard row and a
    chart trace — nothing else to edit."""
    d = _fake_d()
    eq = d["variants"][1]["equity"] * 1.02
    rec = sreg.make_record("new_strat", "Shiny new thing", "test-family", "candidate",
                           equity=eq, train_end=TE, val_end=VE,
                           cost_model="slip + €1")
    d2 = dict(d, registry=list(d["registry"]) + [rec])
    assert "Shiny new thing" in ui.sec_registry(d2, public=False)
    assert "Shiny new thing" in ui.sec_parallel_curves(d2, public=False)
    assert rec.since == str(eq.dropna().index[0].date())
    assert set(rec.windows) == {"train", "val", "test", "full"}


def test_vol_core_windows_share_boundaries_with_momentum():
    d = _fake_d()
    mom_eq = d["variants"][1]["equity"]
    vc = next(r for r in d["registry"] if r.id == "vol_core")
    b_mom = sreg.window_bounds(mom_eq, bs.TRAIN_END, bs.VAL_END)
    b_vc = sreg.window_bounds(vc.equity, bs.TRAIN_END, bs.VAL_END)
    assert b_mom["test"][0] == b_vc["test"][0]               # same held-out start
    assert b_mom["train"][1] == b_vc["train"][1]             # same train end


def test_raw_windows_alias_gone():
    d = _fake_d()
    assert "test" not in d                                   # no ambiguous top-level raw alias
    assert "net_return" in d["raw_windows"]["test"]


# ── the reframe: real results lead, raw vanity gone from the top ────────────────

def test_headline_leads_with_real_results():
    d = _fake_d()
    html = bs.build(d, public=False)
    # the old alarmist "the headline is inflated" banner is gone
    assert "Read this first" not in html
    h = ui.sec_headline(d)
    assert "The result" in h                                      # confident, real-results lead
    assert "Monte Carlo" in h                                     # the validation up front
    assert "held-out" in h.lower() or "out-of-sample" in h.lower()
    assert "Deflated Sharpe" in h or "P(true" in h
    assert "live tracked book" in h                               # names what the tracker runs
    # it leads the page: overview band (registry, chart), then the dossiers
    assert (html.index("The result") < html.index("Strategy registry")
            < html.index("Walk-forward equity") < html.index("Strategy dossiers"))


def test_headline_no_euro_amounts():
    assert "€" not in ui.sec_headline(_fake_d())                  # percentages / Sharpe only


def test_raw_vanity_is_off_the_main_page():
    html = bs.build(_fake_d(), public=False)
    main = html.split("Research lab")[0]                          # everything above the lab
    # the raw full-invested strategy appears above the lab ONLY as the flagged registry row
    assert "Raw Original" not in main
    assert "Original (raw, full-invested)" not in main           # no raw rocket trace up top
    assert "Two ways to run it" not in main                       # the side-by-side framing is gone
    assert "Risk-conscious" in main                               # the risk-conscious book is the focus
    assert "reference, inflated" in main                          # the registry row is honest about it


def test_raw_reference_lives_only_in_the_lab():
    d = _fake_d()
    html = bs.build(d, public=False)
    # raw isn't deleted — it's demoted to the lab as an inflation reference
    marker = "Raw Original — reference only"
    assert marker in html and html.index(marker) > html.index("Research lab")
    h = ui.sec_raw_reference(d)
    assert "reference" in h.lower() and "inflated" in h.lower()   # framed as inflated, not a headline
    # public build drops the whole lab (and the raw with it)
    assert "Research lab" not in bs.build(d, public=True)


def test_parallel_chart_traces():
    d = _fake_d()
    h = ui.sec_parallel_curves(d, public=False)
    assert "Walk-forward equity" in h
    assert "Momentum ensemble" in h                              # the live book (★)
    assert "★" in h or "\\u2605" in h                            # Plotly JSON-escapes the star
    assert "Risk-conscious" in h                                 # the variant, in parallel
    assert "IWDA core" in h                                      # the other adopted strategy
    assert "Equal-weight" in h                                   # survivorship-honest baseline
    assert "S&amp;P 500" in h or "S&P 500" in h                  # benchmarks overlaid
    # exclusions: the raw rocket and the cash-flow-timed real book
    assert "unmanaged" not in h
    assert "Your portfolio" not in h


def test_dossiers_show_every_strategy_in_parallel():
    """One dossier card per strategy — ensemble (sleeve order sheets), the single
    book, the vol core's ETF action — each clearly bounded and tradeable."""
    d = _fake_d()
    h = ui.sec_dossiers(d, public=False)
    assert "Strategy dossiers" in h and "what each strategy holds now" in h
    # ensemble · single book · family evidence · vol core — each its own card
    assert h.count("class='dossier'") == 4
    # the shared evidence is explicitly family-scoped, and closes the momentum family
    assert "Momentum family — evidence" in h and "family-scoped, not page-global" in h
    assert (h.index("Momentum single book") < h.index("Momentum family — evidence")
            < h.index("GARCH vol-managed IWDA core"))
    # per-book history folds live inside the dossiers
    assert "Every rebalance — sleeve" in h and "Every rebalance — risk-conscious" in h
    # every ensemble sleeve gets its own order sheet (the live book here)
    for code in d["ensemble"]["codes"]:
        assert code in h
    assert "Sleeve 1/3" in h and "★ live book" in h
    # the single risk-conscious book is its own dossier, not the page's only focus
    assert "Momentum single book" in h
    # the vol core is actionable too: the ETF row + today's target exposure
    assert "IWDA" in h and "IE00B4L5Y983" in h and "today's order" in h
    # tradeable rows for the momentum picks
    picks = next(hh["picks"] for hh in reversed(d["res"]["holdings_log"]) if hh["picks"])
    for t in picks:
        disp = str(d["meta"][t].get("home") or t).split(".")[0]
        assert f">{disp}<" in h
    # method folded inside each dossier, parallel panel layout inside
    assert "class='par'" in h and h.count("details class='ev'") >= 3


def test_command_strip_reads_the_stack_state():
    d = _fake_d()
    h = ui.sec_command(d, public=False)
    assert "live book" in h and "★" in h
    assert "ens[" in h                               # ensemble adopted in the fixture
    assert "core exposure" in h and "IWDA" in h
    assert "grade" in h
    assert "€" not in ui.sec_command(d, public=True)


def test_evidence_band_folds_with_visible_verdicts():
    html = bs.build(_fake_d(), public=False)
    assert "evidence &amp; risk" in html
    # verdict summaries visible, workings folded
    assert "selection beats" in html and "random books (p" in html
    assert "details class='ev'" in html
    # the survivorship caveat is NOT folded away as a whole
    assert "The dominant caveat" in html


def test_perf_matrix_has_one_column_per_strategy():
    d = _fake_d()
    h = ui.sec_perf_compare(d, public=False)
    assert "all strategies, same windows" in h
    assert "Momentum ensemble" in h and "Risk-conscious" in h
    assert "GARCH vol-managed IWDA core" in h
    assert "S&amp;P 500 (buy-hold)" in h or "S&P 500 (buy-hold)" in h
    for lbl in ("Train", "Validation", "Test", "Full"):
        assert lbl in h


def test_yearly_matrix_has_one_column_per_strategy():
    d = _fake_d()
    h = ui.sec_yearly_compare(d, public=False)
    assert "Yearly P&amp;L — all strategies" in h
    assert "Momentum ensemble" in h and "Risk-conscious" in h
    assert "S&amp;P 500" in h
    assert "★ P&amp;L" in h                       # the live book's € column (private)
    assert "€" not in ui.sec_yearly_compare(d, public=True)


def test_history_folds_live_inside_dossiers():
    d = _fake_d()
    h = ui.sec_dossiers(d, public=False)
    # one history fold per momentum book: 3 sleeves + the single rc book
    assert h.count("Every rebalance — sleeve") == 3
    assert h.count("Every rebalance — risk-conscious") == 1


def test_grade_section_has_no_scare_box():
    h = ui.sec_grade_compare(_fake_d(), public=False)
    assert "Quant scorecard" in h
    assert "not clean alpha" not in h.lower()                    # the mid-page red box is gone
    # the grade + honest deductions still render, just calmly
    assert "Verdict" in h


def test_significance_section_rendered_once():
    html = bs.build(_fake_d(), public=False)
    assert html.count("Significance &amp; robustness") == 1
    assert "Monte" in html or "random books" in html            # the validation is present
    # the beat/DSR stat cards live in the headline ONLY — no duplicated card row
    assert html.count("Beats random books") == 1


def test_phantom_trials_block_shows_decay_and_harvey():
    html = ui.sec_significance(_fake_d(), public=False)
    assert "file-drawer" in html.lower() and "phantom trials" in html.lower()
    # the ladder shows the grid-only and the raised lifetime counts
    assert "Grid only" in html and "×5 lifetime" in html and "×10 lifetime" in html
    assert "66%" in html                                       # the pessimistic ×10 P(real>0)
    assert "t-stat" in html and "3.0" in html                 # Harvey hurdle framing
    assert "subjective estimate" in html                      # honesty: it's a range, not precise


def test_no_info_dropped_from_page():
    html = bs.build(_fake_d(), public=False)
    for phrase in ["Strategy registry", "Registry ledger", "Strategy dossiers",
                   "what each strategy holds now",
                   "Walk-forward equity", "Performance — all strategies",
                   "Quant scorecard", "Deflated Sharpe",
                   "Yearly P&amp;L", "Every rebalance", "survivorship is NOT corrected",
                   "Regime", "Concentration", "Capacity", "Research lab"]:
        assert phrase in html, f"dropped: {phrase!r}"


def test_strategy_public_no_euro_amounts():
    html = bs.build(_fake_d(), public=True)
    euros = re.findall(r"€[0-9][0-9.,]*", html)
    assert all(e == "€1" for e in euros), euros


def test_diagnostics_section_renders_both_lenses():
    html = ui.sec_diagnostics(_fake_d(), public=False)
    assert "HMM" in html and "Effective bets" in html
    assert "not traded" in html                                   # observational framing, explicit
    assert "agreement" in html.lower() and "sector-neutral" in html  # the candid takeaways


def test_diagnostics_sits_before_caveats_in_page():
    html = bs.build(_fake_d(), public=False)
    assert "Observational diagnostics" in html
    assert html.index("Observational diagnostics") < html.index("survivorship is NOT corrected")


def test_caveat_shows_onpopulation_survivorship_result():
    d = _fake_d()
    html = ui.sec_caveat(d)
    # the on-population injection answers the ~2%-overlap complaint with a number
    assert "On-population" in html
    assert "held a name into its delisting" in html
    assert "sold" in html and "%" in html                        # leads with the stable avoidance rate
    assert "pessimistic" in html                                 # frames the return drag as worst-case
    # still names the membership leak as the UNfixed, real problem
    assert "membership" in html and "absent winners" in html


def test_caveat_quotes_the_headline_test_number():
    d = _fake_d()
    html = ui.sec_caveat(d)
    rc_ret = d["variants"][1]["windows"]["test"]["net_return"] * 100
    assert f"{rc_ret:+.0f}%" in html                             # risk-conscious, not the raw figure
    assert "risk-conscious book" in html


def test_caveat_graceful_without_injection():
    d = _fake_d()
    d.pop("surv_inject")
    html = ui.sec_caveat(d)                                       # gather() may have skipped it
    assert "On-population" not in html                            # no half-rendered study
    assert "survivorship is NOT corrected" in html.lower() or "NOT corrected" in html


def test_diagnostics_absent_renders_nothing():
    d = _fake_d()
    d.pop("regime")
    d.pop("eff_bets")
    assert ui.sec_diagnostics(d, public=False) == ""             # graceful when gather skipped them


def test_regime_attribution_renders_three_lenses():
    d = _fake_d()
    html = ui.sec_regime(d, public=False)
    assert "Regime attribution" in html
    assert "Technology" in html and "Industrials" in html        # (1) sector strip named
    assert "risk-on" in html and "risk-off" in html              # (2) HMM-conditional split
    assert "2023" in html                                        # (3) pre-test holdout, derived
    # leads with the honest retained-Sharpe framing, not a scary single number
    assert "retained" in html and "regime-" in html
    # the bars are the raw pre-overlay selection basis, and say so
    assert "pre-overlay" in html and "Selection" in html


def test_regime_attribution_in_full_page_before_caveat():
    html = bs.build(_fake_d(), public=False)
    assert "Regime attribution" in html
    assert html.index("Regime attribution") < html.index("survivorship is NOT corrected")


def test_regime_attribution_absent_renders_nothing():
    d = _fake_d()
    d.pop("regime_attr")
    assert ui.sec_regime(d, public=False) == ""                  # graceful when gather skipped it


def test_regime_attribution_public_no_euro():
    d = _fake_d()
    assert "€" not in ui.sec_regime(d, public=True)              # ratios/percentages only


def test_scenario_fan_renders_bear_base_bull():
    d = _fake_d()
    h = ui.sec_scenarios(d, public=False)
    assert "Scenario fan" in h
    assert "Bear" in h and "Base" in h and "Bull" in h
    assert "block bootstrap" in h.lower()
    # framed honestly: a sensitivity, observational, inherits the survivor universe
    assert "sensitivity" in h.lower() and "not a forecast" in h.lower()
    assert "survivor" in h.lower()


def test_scenario_absent_renders_nothing():
    d = _fake_d()
    d.pop("scenarios")
    assert ui.sec_scenarios(d, public=False) == ""              # graceful when gather skipped it


def test_scenario_public_no_euro():
    assert "€" not in ui.sec_scenarios(_fake_d(), public=True)  # multiples / percentages only


def test_scenario_sits_after_regime_before_caveat():
    html = bs.build(_fake_d(), public=False)
    assert "Scenario fan" in html
    assert (html.index("Regime attribution") < html.index("Scenario fan")
            < html.index("survivorship is NOT corrected"))


# ── Delisting-stress (Task 3) ───────────────────────────────────────────────────

def _stress_fixture(k=24, n=520, seed=0):
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(seed)
    prices = pd.DataFrame(
        {f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))) for i in range(k)},
        index=idx)
    meta_df = pd.DataFrame({"ticker": list(prices.columns), "delisting_date": [pd.NaT] * k})
    sectors = {t: ("Tech" if i % 2 else "Fin") for i, t in enumerate(prices.columns)}
    spx = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n))), index=idx)
    slip = {t: 5.0 for t in prices.columns}
    return prices, meta_df, sectors, spx, slip


def test_base_preset_matches_default():
    assert bs.STRESS_PRESETS["base"] == (bs.SURV_HAZARD, bs.SURV_LOSS)


def test_delisting_stress_grid_shape(monkeypatch):
    monkeypatch.setattr(bs, "SURV_SIMS", 2)
    prices, meta_df, sectors, spx, slip = _stress_fixture()
    cfg = MomentumConfig()
    res = bs.run_momentum(prices, slip, lookback=bs.LOOKBACK, skip=bs.SKIP, capital=bs.CAPITAL,
                          cost_mults=(1.0,), start=bs.START, liq_max=bs.LIQ_MAX, fee_eur=bs.FEE_EUR,
                          min_price=bs.MIN_PRICE, sectors=sectors, benchmark=spx,
                          pit=bs.PITUniverse(prices, {}), execute_lag=bs.EXEC_LAG, **cfg.kwargs())
    base = res["runs"][1.0]["stats"]["net_return"]
    out = bs._delisting_stress(prices, slip, meta_df, sectors, spx, cfg, res, base_return=base)
    assert set(out["presets"]) == {"bull", "base", "bear"}
    assert out["sims"] == 2
    for name, st in out["presets"].items():
        assert set(st["alpha"]) == {"raw", "rc"} and set(st["edge"]) == {"raw", "rc"}
        assert "avoidance_rate" in st and st["sims"] == 2
        assert st["hazard"] == bs.STRESS_PRESETS[name][0]
    assert set(out["clean"]) == {"ret", "alpha", "edge"}


def test_surv_inject_backcompat_keys(monkeypatch):
    """The shared sec_survivorship reads d['surv_inject'] — base preset must carry its keys."""
    monkeypatch.setattr(bs, "SURV_SIMS", 2)
    prices, meta_df, sectors, spx, slip = _stress_fixture()
    cfg = MomentumConfig()
    res = bs.run_momentum(prices, slip, lookback=bs.LOOKBACK, skip=bs.SKIP, capital=bs.CAPITAL,
                          cost_mults=(1.0,), start=bs.START, liq_max=bs.LIQ_MAX, fee_eur=bs.FEE_EUR,
                          min_price=bs.MIN_PRICE, sectors=sectors, benchmark=spx,
                          pit=bs.PITUniverse(prices, {}), execute_lag=bs.EXEC_LAG, **cfg.kwargs())
    base = res["runs"][1.0]["stats"]["net_return"]
    out = bs._delisting_stress(prices, slip, meta_df, sectors, spx, cfg, res, base_return=base)
    base_preset = out["presets"]["base"]
    for key in ("base_return", "sims", "mean_return", "delta_mean", "delta_lo", "delta_hi",
                "hits_mean", "deaths_mean", "avoidance_rate"):
        assert key in base_preset


# ── Delisting-stress rendering (Task 4) ───────────────────────────────────────────

def _fake_stress():
    def bd(m, l, h):
        return dict(mean=m, lo=l, hi=h)

    def preset(hz, a_raw, a_rc, e_raw, e_rc, av):
        return dict(base_return=2.0, sims=8, mean_return=1.7, delta_mean=-0.3,
                    delta_lo=-0.6, delta_hi=0.0, hits_mean=0.4, deaths_mean=5.0,
                    avoidance_rate=av, hazard=hz, loss=(0.4, 1.0),
                    alpha={"raw": bd(*a_raw), "rc": bd(*a_rc)},
                    edge={"raw": bd(*e_raw), "rc": bd(*e_rc)})
    return dict(sims=8,
                clean=dict(ret={"raw": 2.0, "rc": 1.2}, alpha={"raw": 0.14, "rc": 0.10},
                           edge={"raw": 0.05, "rc": 0.04}),
                presets=dict(
                    bull=preset(0.02, (0.12, 0.08, 0.15), (0.09, 0.06, 0.11),
                                (0.05, 0.03, 0.06), (0.04, 0.02, 0.05), 0.95),
                    base=preset(0.05, (0.10, 0.05, 0.14), (0.08, 0.04, 0.10),
                                (0.045, 0.02, 0.06), (0.035, 0.01, 0.05), 0.92),
                    bear=preset(0.10, (0.06, -0.02, 0.12), (0.05, 0.00, 0.09),
                                (0.04, 0.00, 0.06), (0.03, 0.00, 0.05), 0.88)))


def test_delisting_band_public_percentages_only():
    html = ui.sec_delisting_stress({"delisting_stress": _fake_stress()}, public=True)
    assert "%" in html
    assert "€" not in html                          # public invariant
    assert "delisting" in html.lower()


def test_delisting_table_private_has_all_presets():
    html = ui.sec_delisting_stress({"delisting_stress": _fake_stress()}, public=False)
    for token in ("Bull", "Base", "Bear", "Original", "Risk-conscious"):
        assert token in html
    assert "€" not in html
    assert "%" in html                      # private table renders percentages
    assert "Avoidance rate" in html         # row-level spot check


def test_delisting_stub_when_absent():
    assert ui.sec_delisting_stress({"delisting_stress": None}, public=True) == ""
    stub = ui.sec_delisting_stress({"delisting_stress": None}, public=False)
    assert "SURV_SIMS" in stub


def _fake_factor_reg():
    def _m(alpha, t, cols, r2):
        return dict(alpha_ann=alpha, alpha_t=t,
                    betas={c: ((0.6, 8.0) if c == "WML" else (0.5, 5.0)) for c in cols},
                    r2=r2, n=2000)
    f5 = ("MKT_RF", "SMB", "HML", "RMW", "CMA")
    return dict(
        source="Developed", start="2018-04-02", end="2026-07-02", n=2000,
        raw={"CAPM": _m(0.21, 2.4, ("MKT_RF",), 0.55),
             "FF5": _m(0.18, 2.1, f5, 0.58),
             "FF5+WML": _m(0.12, 1.4, f5 + ("WML",), 0.61)},
        rc={"CAPM": _m(0.15, 2.2, ("MKT_RF",), 0.50),
            "FF5": _m(0.12, 1.9, f5, 0.53),
            "FF5+WML": _m(0.08, 1.2, f5 + ("WML",), 0.56)})


def test_factor_section_hidden_without_data():
    assert ui.sec_factor_regression(_fake_d(), public=False) == ""


def test_factor_section_renders_models_and_verdict():
    d = _fake_d()
    d["factor_reg"] = _fake_factor_reg()
    html = ui.sec_factor_regression(d, public=False)
    assert "Factor spanning" in html and "Newey-West" in html and "Developed" in html
    assert html.count("CAPM") >= 2 and html.count("FF5+WML") >= 2   # both books, all models
    assert "0.60" in html or "0.6" in html                          # WML loading shown
    assert "not statistically separable" in html                    # t=1.4 verdict branch
    assert "€" not in html                                          # public-safe by content
    full = bs.build(d, public=True)
    assert "Factor spanning" in full                                # rendered on public build


def test_factor_section_residual_alpha_verdict():
    d = _fake_d()
    fr = _fake_factor_reg()
    fr["raw"]["FF5+WML"]["alpha_t"] = 2.6
    d["factor_reg"] = fr
    assert "residual selection edge" in ui.sec_factor_regression(d, public=False)


def test_sec_vol_core_renders_adopted_overlay_and_hides_on_none():
    import strategy_ui as st
    d = {"vol_core": dict(
        etf="MSCI World (IWDA.AS)",
        bh=dict(sharpe=0.81, ann_return=0.11, max_dd=-0.274),
        managed=dict(sharpe=0.89, ann_return=0.10, max_dd=-0.191,
                     avg_exposure=0.78, n_trades_per_year=9.0),
        fc_now=0.11, w_now=1.0, asof="2026-07-04")}
    html = st.sec_vol_core(d, public=True)
    assert "GARCH" in html and "IWDA" in html
    assert "0.89" in html and "0.81" in html
    assert "pre-registered" in html.lower()
    assert "€" not in html                        # public-safe by construction
    assert st.sec_vol_core({}, public=True) == ""


def test_sec_ensemble_renders_adoption_rule_and_codes():
    import strategy_ui as st
    d = {"ensemble": dict(
        codes=["···DEF", "A···EF", "····EF"], n=3,
        train=dict(sharpe=0.78, net_return=2.1),
        val=dict(sharpe=0.95, net_return=0.4),
        test=dict(sharpe=1.05, net_return=0.35),
        max_dd=-0.31, trades_per_year=24.0,
        ens_min=0.78, single_code="A····F", single_min=0.75,
        adopt=True, dsr5=0.61,
        alpha=dict(model="FF5+WML", alpha_ann=0.12, alpha_t=1.9, n=2100))}
    html = st.sec_ensemble(d, public=True)
    assert "···DEF" in html and "A···EF" in html
    assert "pre-registered" in html.lower()
    assert "adopt" in html.lower()
    assert "selection basis" in html                 # the rule's basis is named explicitly
    assert "€" not in html
    assert st.sec_ensemble({}, public=True) == ""


def test_sec_track_renders_kill_status():
    import strategy_ui as st
    d = {"track": dict(n=12, needed=63, kill=False, reasons=[],
                       path="local/strategy_track.csv")}
    html = st.sec_track(d, public=False)
    assert "12" in html and "63" in html
    assert "kill" in html.lower()
    assert st.sec_track({}, public=False) == ""


def test_sec_venture_renders_shadow_comparison_and_ladder():
    import strategy_ui as st
    d = {"venture": dict(live=True, book_xirr=0.14, shadow_xirr=0.09,
                         excess=0.05, months=6.2, dd=-0.12,
                         dd_state="normal", satellite=dict(live=0, cap=3),
                         deposits=3)}
    html = st.sec_venture(d, public=True)
    assert "shadow" in html.lower()
    assert "ladder" in html.lower() or "half-vol" in html.lower()
    assert "pre-registered" in html.lower()
    assert "€" not in html
    assert st.sec_venture({}, public=True) == ""
    accruing = st.sec_venture({"venture": dict(live=False, n_rows=1,
                                               satellite=dict(live=0, cap=3))},
                              public=True)
    assert "arms at 2+" in accruing and "XIRR" in accruing


def test_sec_ritual_freshness_alarms():
    import strategy_ui as st
    d = {"ritual": dict(items=[
        dict(name="TR snapshot", last="2026-06-29", age_days=8, alarm=False),
        dict(name="BaFin dealings fetch", last="2026-05-01", age_days=67,
             alarm=True)])}
    html = st.sec_ritual(d, public=False)
    assert "TR snapshot" in html and "BaFin" in html
    assert "alarm" in html.lower() or "overdue" in html.lower()
    assert st.sec_ritual(d, public=True) == ""    # private only
    assert st.sec_ritual({}, public=False) == ""
