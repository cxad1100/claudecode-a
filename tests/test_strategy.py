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
    return dict(prices=px, res=res, benchmarks=benchmarks, capital=10_000.0,
                meta={t: dict(name=t, local_id="000", country="X", sector="Y") for t in px.columns},
                strategy=bs.STRATEGY, quant=quant, variants=variants, portfolio_roi=None,
                train=train, val=val, test=test, graveyard_hits=0,
                n_dead=42, n_countries=1, n_live=10,
                significance=dict(
                    mc=dict(null_sharpe=np.array([0.1, 0.2, 0.3, 0.25]), strat_sharpe=0.6,
                            null_sharpe_median=0.22, p_sharpe=0.04, p_total=0.05, n_trials=1000),
                    dsr=dict(dsr=0.78, n_trials=32, T=20, sr_benchmark_annual=1.1,
                             sharpe_annual=1.4),
                    ci=dict(conf=95, sharpe=1.2, sharpe_lo=0.4, sharpe_hi=1.9,
                            cagr=0.3, cagr_lo=0.1, cagr_hi=0.5),
                    ppy=4.0))


def test_strategy_page_builds():
    html = bs.build(_fake_d(), public=False)
    assert "<html" in html.lower() and "strategy" in html.lower()
    assert "Validation" in html and bs.STRATEGY.code in html
    # both versions present, billed equally
    assert "Original" in html and "Risk-conscious" in html


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


def test_picks_compare_same_names_both_columns():
    d = _fake_d()
    html = bs.sec_picks_compare(d)
    picks = next(h["picks"] for h in reversed(d["res"]["holdings_log"]) if h["picks"])
    for t in picks:
        disp = str(d["meta"][t].get("home") or t).split(".")[0]
        assert html.count(f">{disp}<") >= 2, f"{disp} should appear in both columns"
    assert "Identical selection" in html


def test_significance_section_rendered_once():
    html = bs.build(_fake_d(), public=False)
    # selection is shared, so significance must NOT be duplicated per version
    assert html.count("Significance &amp; robustness") == 1
    assert "applies to both" in html


def test_no_info_dropped_from_original_page():
    html = bs.build(_fake_d(), public=False)
    for phrase in ["Current top picks", "Walk-forward equity", "Performance",
                   "Quant scorecard", "Bias audit", "Deflated Sharpe",
                   "Yearly P&amp;L", "Every rebalance", "survivorship is NOT corrected",
                   "Regime", "Concentration", "Capacity", "Research lab"]:
        assert phrase in html, f"dropped: {phrase!r}"


def test_strategy_public_no_euro_amounts():
    html = bs.build(_fake_d(), public=True)
    euros = re.findall(r"€[0-9][0-9.,]*", html)
    assert all(e == "€1" for e in euros), euros
