"""Econophysics research page — scaffold invariants.

The page is a research lab: every module section must tolerate an absent
result (None / missing key → renders ""), the page must never write into
docs/ (it is local-only, unlike the other reports), and module env flags
must actually disable their gather blocks.
"""
import inspect

import build_econo_report as er


def test_build_renders_with_all_modules_none():
    html = er.build({})                       # no module produced anything
    assert html.startswith("<!DOCTYPE html>")
    assert "Econophysics" in html
    assert "plotly" in html                   # standard report chrome present


def test_module_sections_render_empty_on_none():
    for sec in (er.sec_leadlag, er.sec_corr, er.sec_phase, er.sec_events,
                er.sec_trials):
        assert sec({}, False) == ""
        assert sec({"leadlag": None, "corr": None, "phase": None,
                    "events": None, "trials": None}, False) == ""


def test_page_is_local_only():
    src = inspect.getsource(er)
    assert "docs/" not in src and 'docs"' not in src


def test_env_flag_disables_module(monkeypatch):
    monkeypatch.delenv("ECONO_LEADLAG", raising=False)
    assert er._enabled("leadlag") is True     # default on
    monkeypatch.setenv("ECONO_LEADLAG", "0")
    assert er._enabled("leadlag") is False
    monkeypatch.setenv("ECONO_LEADLAG", "1")
    assert er._enabled("leadlag") is True


def _fake_stats(sh=1.0, net=0.5):
    return dict(net_return=net, sharpe=sh, max_drawdown=-0.2, n_trades=40,
                win_rate=0.6, avg_days=90, total_costs=100.0)


def test_sec_leadlag_renders_cells_and_verdict():
    cell = dict(code="Q-r21-raw", kind="signal", train=_fake_stats(),
                val=_fake_stats(0.8), test=_fake_stats(0.6), full=_fake_stats())
    base = dict(cell, code="Q-r21-base", kind="baseline")
    plac = dict(cell, code="Q-r21-plac", kind="placebo", full=_fake_stats(-0.2))
    d = {"leadlag": dict(
        cells=[cell, base, plac],
        headline=dict(code="Q-r21-raw", mc=dict(p_total=0.03, p_sharpe=0.04),
                      dsr_phantom=[dict(mult=1, n=74, dsr=0.7),
                                   dict(mult=5, n=370, dsr=0.55),
                                   dict(mult=10, n=740, dsr=0.4)],
                      t_stat=2.4,
                      factor={"model": "FF5+WML", "alpha_ann": 0.06,
                              "alpha_t": 2.2, "n": 1500, "source": "dev"}),
        ic=dict(r5=dict(mean=0.02, t=1.1, n=90), r21=dict(mean=0.01, t=0.6, n=90)),
        diag=dict(median_kept=140, expected_false=90, mean_self_lag=0.04,
                  median_uni=300),
        n_trials=10)}
    html = er.sec_leadlag(d, False)
    assert "Q-r21-raw" in html and "placebo" in html.lower()
    assert "baseline" in html.lower()
    assert "DSR" in html
    assert "verdict" in html.lower()


def test_sec_leadlag_placebo_beats_headline_forces_leak_warning():
    cell = dict(code="Q-r21-raw", kind="signal", train=_fake_stats(),
                val=_fake_stats(), test=_fake_stats(), full=_fake_stats(sh=0.5))
    plac = dict(cell, code="Q-r21-plac", kind="placebo", full=_fake_stats(sh=1.5))
    d = {"leadlag": dict(cells=[cell, plac],
                         headline=dict(code="Q-r21-raw", mc=dict(p_sharpe=0.01),
                                       dsr_phantom=[dict(mult=5, n=370, dsr=0.9)],
                                       t_stat=3.0,
                                       factor={"model": "FF5+WML", "alpha_ann": 0.1,
                                               "alpha_t": 3.0, "n": 1500,
                                               "source": "dev"}),
                         ic={}, diag={}, n_trials=10)}
    html = er.sec_leadlag(d, False)
    assert "leak" in html.lower()


def test_sec_corr_renders_variants_and_rmt():
    d = {"corr": dict(
        rmt=dict(lambda_plus=1.42, n_signal_eigs=12, market_share=0.31,
                 n_names=300, T=252),
        rand_gics=0.71, n_clusters=20,
        variants=[dict(code="GICS-neutral", train=_fake_stats(),
                       val=_fake_stats(0.7), test=_fake_stats(0.9),
                       full=_fake_stats(), eff_bets=2.6),
                  dict(code="cluster-neutral", train=_fake_stats(),
                       val=_fake_stats(0.8), test=_fake_stats(1.0),
                       full=_fake_stats(), eff_bets=3.4),
                  dict(code="no-grouping", train=_fake_stats(),
                       val=_fake_stats(0.5), test=_fake_stats(0.8),
                       full=_fake_stats(), eff_bets=2.1)],
        n_trials=2)}
    html = er.sec_corr(d, False)
    assert "cluster-neutral" in html and "GICS" in html
    assert "λ" in html or "lambda" in html.lower()
    assert "effective bets" in html.lower()
    assert "verdict" in html.lower()
    assert er.sec_corr({}, False) == ""


def test_sec_phase_renders_overlays_and_verdict():
    d = {"phase": dict(
        ar_now=0.71, ar_pct=0.93, chi_pct=0.55,
        rows=[dict(code="raw book", val=1.0, test=0.9, dd=-0.25, expo=1.0),
              dict(code="vol-target 15%", val=1.1, test=0.95, dd=-0.18, expo=0.8),
              dict(code="AR-throttle", val=1.05, test=0.97, dd=-0.17, expo=0.85)],
        promoted=False, n_trials=0)}
    html = er.sec_phase(d, False)
    assert "absorption" in html.lower()
    assert "vol-target" in html and "AR-throttle" in html
    assert "verdict" in html.lower()
    assert "observational" in html.lower()


def test_sec_events_renders_car_cells_and_verdict():
    d = {"events": dict(
        car_insider=dict(car={1: 0.01, 5: 0.02, 20: 0.03}, bmp_t=2.8, n=140),
        car_insider_small=dict(car={20: 0.06}, bmp_t=3.1, n=45),
        car_insider_big=dict(car={20: 0.005}, bmp_t=0.6, n=95),
        car_covering=dict(car={1: 0.004, 5: 0.01, 20: 0.015}, bmp_t=2.1, n=300),
        ct_alpha=dict(model="FF5+WML", alpha_ann=0.09, alpha_t=2.4, n=500),
        cells=[dict(code="cover-Q", kind="signal", train=_fake_stats(),
                    val=_fake_stats(0.6), test=_fake_stats(0.7),
                    full=_fake_stats())],
        headline=dict(code="cover-Q", mc_matched=dict(p_sharpe=0.03),
                      dsr_phantom=[dict(mult=5, n=410, dsr=0.6)]),
        insider_ready=True, n_trials=6)}
    html = er.sec_events(d, False)
    assert "cover-Q" in html and "CAR" in html
    assert "small" in html.lower() and "verdict" in html.lower()
    assert er.sec_events({}, False) == ""


def test_sec_events_insider_pending_note():
    d = {"events": dict(car_covering=dict(car={20: 0.01}, bmp_t=1.0, n=50),
                        cells=[], headline={}, insider_ready=False,
                        n_trials=4)}
    html = er.sec_events(d, False)
    assert "backfill" in html.lower() or "pending" in html.lower()


def test_sec_trials_ledger_counts_and_statuses():
    d = {"leadlag": dict(n_trials=10, cells=[], headline={}),
         "corr": dict(n_trials=2, variants=[]),
         "phase": dict(n_trials=0, promoted=False, rows=[]),
         "events": dict(n_trials=4, cells=[], headline={},
                        insider_ready=False)}
    d["trials"] = er._gather_trials(d)
    html = er.sec_trials(d, False)
    assert "64" in html                       # inherited grid
    assert "89" in html                       # 64+10+2+0+4 +7 vol +2 edge
    assert "lead-lag" in html.lower()
    assert "file drawer" in html.lower() or "killed" in html.lower()
    # phase not promoted → its 0 trials shown, marked observational
    assert "observational" in html.lower()


def test_sec_trials_handles_missing_modules():
    d = {"leadlag": None, "corr": None, "phase": None, "events": None}
    d["trials"] = er._gather_trials(d)
    html = er.sec_trials(d, False)
    assert "64" in html                       # inherited always counted
    assert "not run" in html.lower()
