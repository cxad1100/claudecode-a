"""Cross-sectional ML ranker — walk-forward ridge and a tiny MLP.

The experiment this module exists for: does a neural network CHOOSE better
than the linear null model on the features we actually have? Gu, Kelly & Xiu
(2020, RFS) — the benchmark study — find NN gains come mostly from feature
interactions among ~94 fundamental characteristics over 60 years; on
price-only features with a few dozen rebalances of training data the honest
prior is that ridge ties the net. So the net must BEAT ridge on validation,
through the same gate, before anyone calls it an improvement.

Capacity matched to the data on purpose: one hidden layer, a handful of tanh
units, L2, full-batch L-BFGS — hand-rolled numpy/scipy (no torch dependency),
deterministic under seed. Labels are cross-sectional RANKS of next-period
returns (uniformised — rank IC is the objective that matches how the book is
built: top-k selection cares about ordering, not magnitudes).

Walk-forward contract (same as every score precompute in this repo): the
model scored at rebalance d is trained ONLY on (features, forward-rank)
pairs from rebalances strictly before d; features at d use prices.loc[:d];
output {date: {"raw","voladj"}} covers every requested date (empty Series
until min_train prior rebalances exist — never a silent fallback).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = ["features", "ml_scores"]

_FEATS = ("mom", "rev", "vol", "hi52", "turn")


def features(prices: pd.DataFrame, turnover: pd.DataFrame | None, asof,
             names: list, *, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Per-name PIT feature row at `asof`: 12-1 momentum, 1-month reversal,
    63d vol, 52-week-high proximity, trailing turnover rank (0.5 when no
    turnover data). All from prices.loc[:asof] only."""
    w = prices.loc[:asof]
    cols = [t for t in names if t in w.columns]
    tail = w[cols].tail(lookback + skip + 1)
    out = pd.DataFrame(index=cols, dtype=float)
    px = tail.ffill()
    last = px.iloc[-1]
    out["mom"] = px.iloc[-skip - 1] / px.iloc[0] - 1.0
    out["rev"] = last / px.iloc[-skip - 1] - 1.0
    out["vol"] = px.pct_change().tail(63).std() * np.sqrt(252.0)
    out["hi52"] = last / px.tail(252).max() - 1.0
    if turnover is not None:
        med = turnover.loc[:asof].tail(6).reindex(columns=cols).median()
        out["turn"] = med.rank(pct=True)
    else:
        out["turn"] = 0.5
    return out.replace([np.inf, -np.inf], np.nan)


def _zx(f: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per feature (rank-robust: winsor at ±3)."""
    z = (f - f.mean()) / f.std(ddof=1).replace(0.0, np.nan)
    return z.clip(-3, 3).fillna(0.0)


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    n_feat = x.shape[1]
    a = x.T @ x + l2 * np.eye(n_feat)
    return np.linalg.solve(a, x.T @ y)


class _MLP:
    """One-hidden-layer tanh MLP fit by full-batch L-BFGS with L2 — the
    smallest thing that can represent feature interactions."""

    def __init__(self, n_in: int, hidden: int = 8, l2: float = 1e-3,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.h = hidden
        self.n_in = n_in
        self.l2 = l2
        scale = 1.0 / np.sqrt(n_in)
        self.w0 = np.concatenate([
            rng.normal(0, scale, n_in * hidden), np.zeros(hidden),
            rng.normal(0, 1.0 / np.sqrt(hidden), hidden), [0.0]])

    def _unpack(self, w):
        i = self.n_in * self.h
        w1 = w[:i].reshape(self.n_in, self.h)
        b1 = w[i:i + self.h]
        w2 = w[i + self.h:i + 2 * self.h]
        b2 = w[-1]
        return w1, b1, w2, b2

    def _forward(self, w, x):
        w1, b1, w2, b2 = self._unpack(w)
        return np.tanh(x @ w1 + b1) @ w2 + b2

    def fit(self, x: np.ndarray, y: np.ndarray):
        n = len(y)

        def loss_grad(w):
            w1, b1, w2, b2 = self._unpack(w)
            h = np.tanh(x @ w1 + b1)                  # n × hidden
            pred = h @ w2 + b2
            err = pred - y
            loss = float(np.mean(err ** 2) + self.l2 * np.sum(w * w))
            # analytic gradient — numerical differentiation over the weight
            # vector made each fit ~50x slower than the math requires
            g_pred = 2.0 * err / n                    # dL/dpred
            g_w2 = h.T @ g_pred
            g_b2 = float(g_pred.sum())
            g_h = np.outer(g_pred, w2) * (1.0 - h ** 2)
            g_w1 = x.T @ g_h
            g_b1 = g_h.sum(axis=0)
            grad = np.concatenate([g_w1.ravel(), g_b1, g_w2, [g_b2]]) \
                + 2.0 * self.l2 * w
            return loss, grad

        res = minimize(loss_grad, self.w0, method="L-BFGS-B", jac=True,
                       options=dict(maxiter=300))
        self.w = res.x
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(self.w, x)


def ml_scores(prices: pd.DataFrame, turnover: pd.DataFrame | None, dates,
              elig_by_date: dict, *, model: str = "ridge", min_train: int = 8,
              hidden: int = 8, l2_mlp: float = 1e-3, l2_ridge: float = 1.0,
              lookback: int = 252, skip: int = 21, seed: int = 0) -> dict:
    """run_momentum-ready walk-forward scores. At rebalance d: train on all
    prior rebalances' (z-scored features → forward cross-sectional return
    rank in [-0.5, 0.5]) and score d's eligible names. Empty Series until
    min_train prior rebalances exist."""
    dates = list(dates)
    feats, fwd_rank = {}, {}
    for i, d in enumerate(dates):
        names = sorted(elig_by_date.get(d, set()))
        f = features(prices, turnover, d, names, lookback=lookback, skip=skip)
        feats[d] = _zx(f.dropna(how="all"))
        if i + 1 < len(dates):
            nxt = dates[i + 1]
            cols = [t for t in feats[d].index if t in prices.columns]
            fr = (prices.loc[:nxt].iloc[-1][cols]
                  / prices.loc[:d].iloc[-1][cols] - 1.0).dropna()
            fwd_rank[d] = fr.rank(pct=True) - 0.5
    out = {}
    for i, d in enumerate(dates):
        train = [dd for dd in dates[:i] if dd in fwd_rank]
        raw = pd.Series(dtype=float)
        if len(train) >= min_train:
            xs, ys = [], []
            for dd in train:
                common = feats[dd].index.intersection(fwd_rank[dd].index)
                xs.append(feats[dd].loc[common, list(_FEATS)].to_numpy())
                ys.append(fwd_rank[dd].loc[common].to_numpy())
            x = np.vstack(xs)
            y = np.concatenate(ys)
            xd = feats[d].loc[:, list(_FEATS)].to_numpy()
            if model == "ridge":
                beta = _fit_ridge(x, y, l2=l2_ridge)
                pred = xd @ beta
            elif model == "mlp":
                net = _MLP(x.shape[1], hidden=hidden, l2=l2_mlp,
                           seed=seed).fit(x, y)
                pred = net.predict(xd)
            else:
                raise ValueError(f"unknown model {model!r}")
            raw = pd.Series(pred, index=feats[d].index).dropna()
        out[d] = {"raw": raw, "voladj": raw.copy()}
    return out
