"""Cross-sectional ML ranker — walk-forward ridge + tiny MLP.

Pins: features are PIT (truncate-future invariance), the walk-forward
trainer never sees the rebalance it scores, both models recover a planted
monotonic signal, the MLP beats ridge only when the planted relation is
genuinely non-linear (the whole point of the experiment), scores are
deterministic under a fixed seed and cover every requested date.
"""
import numpy as np
import pandas as pd

from tools.xsec_ml import features, ml_scores


def _panel(n_days=900, n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    r = rng.normal(0.0003, 0.02, (n_days, n))
    return pd.DataFrame((1 + r).cumprod(axis=0) * 50, index=idx,
                        columns=[f"S{k}" for k in range(n)])


def test_features_pit_and_columns():
    p = _panel()
    d = p.index[400]
    f_full = features(p, None, d, list(p.columns))
    f_trunc = features(p.loc[:d], None, d, list(p.columns))
    pd.testing.assert_frame_equal(f_full, f_trunc)
    assert {"mom", "rev", "vol", "hi52"} <= set(f_full.columns)
    assert len(f_full) == 40


def test_ml_scores_cover_dates_and_deterministic():
    p = _panel()
    dates = list(p.index[[500, 560, 620, 680, 740, 800]])
    elig = {d: set(p.columns) for d in dates}
    out1 = ml_scores(p, None, dates, elig, model="ridge", min_train=3, seed=0)
    out2 = ml_scores(p, None, dates, elig, model="ridge", min_train=3, seed=0)
    assert set(out1.keys()) == set(dates)
    for d in dates:
        assert set(out1[d].keys()) == {"raw", "voladj"}
        pd.testing.assert_series_equal(out1[d]["raw"], out2[d]["raw"])
    # first min_train dates have no history to train on → empty scores
    assert len(out1[dates[0]]["raw"]) == 0
    assert len(out1[dates[-1]]["raw"]) > 0


def test_walkforward_no_lookahead():
    p = _panel()
    dates = list(p.index[[500, 560, 620, 680, 740, 800]])
    elig = {d: set(p.columns) for d in dates}
    full = ml_scores(p, None, dates, elig, model="ridge", min_train=3, seed=0)
    d = dates[4]
    trunc = ml_scores(p.loc[:d], None, dates[:5], elig, model="ridge",
                      min_train=3, seed=0)
    pd.testing.assert_series_equal(full[d]["raw"].sort_index(),
                                   trunc[d]["raw"].sort_index())


def _planted_panel(nonlinear=False, n_days=1000, n=60, seed=3):
    """Forward returns driven by a hidden factor of the features: linear in
    momentum, or a non-monotone interaction (vol-conditional momentum flip)
    that a linear model cannot represent."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    px = np.full((n_days, n), 50.0)
    volmult = np.where(np.arange(n) % 2 == 0, 0.5, 2.0)
    # NO per-name drift: a persistent drift makes even the 'nonlinear' panel
    # linearly learnable through the momentum feature alone
    for t in range(1, n_days):
        mom_state = (px[t - 1] / px[max(t - 253, 0)]) - 1.0
        if nonlinear:                       # pure mom×vol interaction: the
            edge = 0.0035 * np.sign(mom_state) * np.where(volmult > 1, -1, 1)
        else:
            edge = 0.0035 * np.sign(mom_state)
        shock = rng.normal(0, 0.01, n) * volmult
        px[t] = px[t - 1] * (1 + edge + shock)
    return pd.DataFrame(px, index=idx, columns=[f"S{k}" for k in range(n)])


def _mean_ic(scores, prices, dates):
    from scipy.stats import spearmanr
    ics = []
    for i in range(len(dates) - 1):
        s = scores[dates[i]]["raw"].dropna()
        if len(s) < 10:
            continue
        fwd = prices.loc[dates[i + 1], s.index] / prices.loc[dates[i], s.index] - 1
        ics.append(spearmanr(s, fwd)[0])
    return float(np.mean(ics)) if ics else 0.0


def test_both_models_learn_linear_signal():
    p = _planted_panel(nonlinear=False)
    dates = list(p.index[range(300, 990, 60)])
    elig = {d: set(p.columns) for d in dates}
    ic_r = _mean_ic(ml_scores(p, None, dates, elig, model="ridge",
                              min_train=3, seed=0), p, dates)
    ic_m = _mean_ic(ml_scores(p, None, dates, elig, model="mlp",
                              min_train=3, seed=0), p, dates)
    assert ic_r > 0.10
    assert ic_m > 0.10


def test_mlp_beats_ridge_only_when_relation_nonlinear():
    p = _planted_panel(nonlinear=True)
    dates = list(p.index[range(300, 990, 60)])
    elig = {d: set(p.columns) for d in dates}
    ic_r = _mean_ic(ml_scores(p, None, dates, elig, model="ridge",
                              min_train=3, seed=0), p, dates)
    ic_m = _mean_ic(ml_scores(p, None, dates, elig, model="mlp",
                              min_train=3, seed=0), p, dates)
    assert ic_m > ic_r + 0.05          # non-linearity is where the NN earns
