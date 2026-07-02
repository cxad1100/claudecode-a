"""Regime-conditioned block bootstrap — bear / base / bull outcome distributions.

Observational scenario layer for the momentum strategy. It resamples the strategy's OWN
realised daily returns in fixed-length blocks (so short-run autocorrelation and the
crash-cluster structure survive), tilting the sampling probability toward risk-off blocks
(bear) or risk-on blocks (bull). Base samples every block at its natural frequency.

This is a *sensitivity*, not a forecast: it shows the dispersion of terminal wealth implied
by the realised return process under different regime mixes. It uses common random numbers
across the three scenarios (same seed reset each time) so bear/base/bull differ only by the
sampling tilt, and it never touches selection or sizing — pure numbers in, distribution out.

Caveats it does NOT remove: the returns come from the survivor universe, so the bear path is
'bear among survivors' — a real bear with delistings is worse.
"""
import numpy as np
import pandas as pd


def _pct_paths(wealth: np.ndarray) -> dict:
    """P5 / P50 / P95 across sims at each time step + the terminal percentiles.
    `wealth` is (n_sims, horizon+1), each row a cumulative-wealth path starting at 1.0."""
    p5, p50, p95 = np.percentile(wealth, [5, 50, 95], axis=0)
    return dict(p5=p5, p50=p50, p95=p95,
                term_p5=float(p5[-1]), term_p50=float(p50[-1]), term_p95=float(p95[-1]))


def regime_scenarios(returns: pd.Series, risk_off: pd.Series, *, horizon: int = 252,
                     block: int = 21, n_sims: int = 2000, tilt: float = 3.0,
                     seed: int = 0) -> dict | None:
    """Block-bootstrap `returns` into bear/base/bull terminal-wealth distributions.

    returns  : daily strategy returns (e.g. the risk-conscious equity curve's pct_change).
    risk_off : boolean Series (True = HMM risk-off day); reindexed onto `returns`.
    horizon  : simulated path length in trading days (252 ≈ 1y).
    block    : bootstrap block length in days (keeps autocorrelation/crash clusters intact).
    tilt     : how hard bear up-weights risk-off blocks / bull up-weights risk-on blocks.

    A block's regime = the majority risk-off flag over its `block` days (circular). Returns
    None if the series is too short to bootstrap. Each scenario resets the RNG to `seed`
    (common random numbers) so they differ only by the sampling weights; a degenerate mask
    (no risk-off, or no risk-on) makes the affected scenario identical to base."""
    r = np.asarray(returns, float)
    T = r.size
    if T < max(2 * block, 50):
        return None
    ro = risk_off.reindex(returns.index).fillna(False).to_numpy(bool)

    # Circular block matrix: B[i] = the `block` returns starting at i (wrapping past the end),
    # and its regime flag = majority risk-off over those same days.
    wrap = (np.arange(T)[:, None] + np.arange(block)[None, :]) % T   # (T, block) indices
    B = r[wrap]                                                      # (T, block) block returns
    block_off = ro[wrap].mean(axis=1) >= 0.5                         # (T,) majority risk-off

    n_blk = int(np.ceil(horizon / block))
    weights = dict(
        base=np.ones(T),
        bear=np.where(block_off, tilt, 1.0),
        bull=np.where(block_off, 1.0, tilt),
    )
    scenarios = {}
    for name, w in weights.items():
        rng = np.random.default_rng(seed)                           # common random numbers
        p = w / w.sum()
        idx = rng.choice(T, size=(n_sims, n_blk), p=p)              # block starts per sim
        paths = B[idx].reshape(n_sims, n_blk * block)[:, :horizon]  # (n_sims, horizon) returns
        wealth = np.empty((n_sims, horizon + 1))
        wealth[:, 0] = 1.0
        wealth[:, 1:] = np.cumprod(1.0 + paths, axis=1)             # start each path at 1.0
        scenarios[name] = _pct_paths(wealth)

    return dict(horizon=horizon, block=block, n_sims=n_sims, tilt=tilt,
                frac_off=float(ro.mean()), scenarios=scenarios)
