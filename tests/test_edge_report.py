"""Smoke test for build_edge_report: gather() → build() on a synthetic universe
(no network, no real price panel)."""
import numpy as np
import pandas as pd
import pytest

import build_edge_report as E
from tests.test_edge_seasonal import _universe, _slip


@pytest.fixture()
def synthetic(monkeypatch):
    prices = _universe(plant_rebound=True, seed=7, years=(2015, 2023))
    monkeypatch.setattr(E, "load_universe_panel", lambda: dict(
        prices=prices, slip=_slip(prices), pit=None, meta={},
        fee_eur=1.0, liq_max=30, min_price=1.0))
    rng = np.random.default_rng(0)
    core = pd.Series(rng.normal(0.0008, 0.015, len(prices.index)), index=prices.index)
    monkeypatch.setattr(E, "momentum_net_returns",
                        lambda refresh=False, panel=None: core)


def test_gather_and_build(synthetic):
    d = E.gather()
    assert d["seasonal"] and d["mc"] and d["stack"]
    assert d["mc"]["p_sharpe"] < 0.05                     # the planted rebound is found
    assert d["stack"]["overlay"]["exposure"].max() <= 1.0 + 1e-12
    html = E.build(d)
    for needle in ("Edge Stack", "structural edge", "tax-loss rebound",
                   "Monte-Carlo", "Honesty box", "Wachtel"):
        assert needle in html


def test_build_degrades_without_data(monkeypatch):
    monkeypatch.setattr(E, "load_universe_panel", lambda: None)
    d = E.gather()
    assert d["seasonal"] is None and d["stack"] is None
    html = E.build(d)
    assert "universe_prices.csv" in html                  # graceful note, no crash


def test_mc_null_matches_sleeve_breadth(synthetic):
    d = E.gather()
    picked = [len(y["picks"]) for y in d["seasonal"]["years"] if y["picks"]]
    assert d["mc"]["k_eff"] == max(1, int(round(np.mean(picked))))
