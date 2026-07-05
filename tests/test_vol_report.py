"""Smoke test for build_vol_report: full gather() → build() on synthetic OHLC data
(no network, no universe panel — the momentum leg degrades to a note)."""
import numpy as np
import pandas as pd
import pytest

import build_vol_report as V


def _ohlc(start: str, n: int, seed: int) -> pd.DataFrame:
    """Vol-clustered synthetic price path with plausible H/L range."""
    rng = np.random.default_rng(seed)
    sig = 0.008 + 0.012 * (np.sin(np.arange(n) / 120.0) ** 2)   # slow vol cycles
    r = rng.normal(0.0003, 1.0, n) * sig
    close = 100 * np.cumprod(1 + r)
    rng_frac = np.abs(rng.normal(0, sig)) + 0.002
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Close": close, "High": close * (1 + rng_frac),
                         "Low": close * (1 - rng_frac)}, index=idx)


@pytest.fixture()
def synthetic_data(monkeypatch):
    frames = {V.ETF: _ohlc("2009-09-01", 4200, seed=1),
              V.PROXY: _ohlc("1995-01-01", 7800, seed=2)}
    monkeypatch.setattr(V, "cached_ohlc", lambda t, force=False, **kw: frames[t])
    monkeypatch.setattr(V, "_momentum_returns",
                        lambda refresh=False, panel=None: None)


def test_gather_and_build(synthetic_data):
    d = V.gather()
    assert d["etf"] and d["spx"] and d["mom"] is None
    assert set(d["etf"]["variants"]) >= set(V.METHODS) | {"trend"}
    assert d["etf"]["winner"] in V.METHODS
    for m in V.METHODS:                                   # exposure always capped
        assert d["etf"]["variants"][m]["exposure"].max() <= 1.0 + 1e-12
    assert len(d["grid"]) == 12
    assert np.isfinite(d["significance"]["ci"]["sharpe"])
    assert "^GSPC" not in d["significance"]["best"]       # research proxy never certified

    html = V.build(d)
    for needle in ("Vol lab", "Is volatility actually predictable", "mean is not",
                   "Applied to the tradeable ETF", "Applied to the momentum strategy",
                   "Significance", "Cost", "Moreira", "universe_prices.csv"):
        assert needle in html


def test_compute_underlying_requires_history():
    rng = np.random.default_rng(0)
    short = pd.Series(rng.normal(0, 0.01, 400),
                      index=pd.bdate_range("2020-01-01", periods=400))
    assert V.compute_underlying(short, "too-short") == {}
