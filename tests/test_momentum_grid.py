from tools.momentum_grid import MomentumConfig, ALL_CONFIGS


def test_all_configs_is_64_unique():
    assert len(ALL_CONFIGS) == 64
    assert len({c.code for c in ALL_CONFIGS}) == 64


def test_baseline_config_code_and_kwargs():
    base = MomentumConfig()
    assert base.code == "······"                          # all upgrades off
    kw = base.kwargs()
    assert kw["k"] == 15 and kw["freq"] == "M"
    assert kw["vol_adjust"] is False and kw["lazy"] is False
    full = MomentumConfig(True, True, True, 10, "Q", True)
    assert full.code == "ABCDEF"
    assert full.kwargs()["k"] == 10 and full.kwargs()["freq"] == "Q"


import numpy as np
import pandas as pd
from tools.momentum_grid import run_grid


def _grid_px(n=900, ncols=12, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    cols = {f"T{i}": 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
            for i in range(ncols)}
    return pd.DataFrame(cols, idx)


def test_run_grid_splits_train_val_and_covers_configs():
    px = _grid_px()
    slip = {t: 10 for t in px.columns}
    configs = [MomentumConfig(), MomentumConfig(slots=10)]      # 2 of the 64
    res = run_grid(px, slip, sectors={t: "X" for t in px.columns},
                   configs=configs, train_end="2020-06-30",
                   lookback=200, skip=10)
    assert {c["code"] for c in res["cells"]} == {"······", "···D··"}
    cell = res["cells"][0]
    for part in ("train", "val", "full"):
        assert "sharpe" in cell[part] and "net_return" in cell[part]
    assert cell["trades_per_year"] >= 0


from tools.momentum_grid import feasibility


def test_feasibility_fee_drag_arithmetic():
    cell = {"code": "······", "trades_per_year": 40.0, "full": {"net_return": 0.50}}
    f = feasibility(cell, capital=10_000.0, fee_eur=1.0)
    assert abs(f["annual_fee_eur"] - 40.0) < 1e-9          # 40 trades × €1
    assert abs(f["fee_drag_pct"] - 0.40) < 1e-9            # €40 / €10k = 0.40%
    assert f["pays_for_itself"] is True                    # net 50% >> drag


from tools.momentum_grid import pick_ultimate


def test_pick_ultimate_rewards_worst_case_robustness():
    cells = [
        {"code": "······", "train": {"sharpe": 1.0, "net_return": 2.0},
         "val": {"sharpe": 1.5, "net_return": 5.0}, "trades_per_year": 40, "full": {"net_return": 5.0}},
        {"code": "·B····", "train": {"sharpe": 1.2, "net_return": 1.3},
         "val": {"sharpe": 2.2, "net_return": 5.0}, "trades_per_year": 50, "full": {"net_return": 5.0}},
        {"code": "A·····", "train": {"sharpe": 0.2, "net_return": 4.0},   # val-lucky, weak train
         "val": {"sharpe": 3.0, "net_return": 4.0}, "trades_per_year": 50, "full": {"net_return": 4.0}},
    ]
    u = pick_ultimate({"cells": cells})
    assert u["code"] == "·B····"            # min(train,val)=1.2 beats baseline 1.0 and A's 0.2


def test_pick_top_n_quarterly_by_worst_case_robustness():
    from tools.momentum_grid import pick_top_n

    def cell(code, freq, tr, va, trades=8.0, net=1.0):
        return dict(code=code, config=None, trades_per_year=trades,
                    train=dict(sharpe=tr, net_return=net),
                    val=dict(sharpe=va, net_return=net),
                    full=dict(sharpe=1.0, net_return=net))

    grid = dict(cells=[
        cell("AAAAAA", "M", 0.9, 0.9),      # monthly — excluded (freq filter)
        cell("BBBBBB", "Q", 0.9, 0.7),      # min 0.7 → 1st
        cell("CCCCCC", "Q", 0.6, 0.65),     # min 0.6 → 2nd
        cell("DDDDDD", "Q", 0.5, 0.55),     # min 0.5 → 3rd
        cell("EEEEEE", "Q", 0.4, 0.45),     # min 0.4 → cut at n=3
        cell("FFFFFF", "Q", 0.9, -0.1),     # negative val → filtered out
        cell("GGGGGG", "Q", 0.9, 0.8, net=-1.0),   # negative return → out
    ])
    # mark freq on the fake configs via code position: use explicit key
    for c in grid["cells"]:
        c["freq"] = "M" if c["code"] == "AAAAAA" else "Q"
    top = pick_top_n(grid, n=3, freq="Q")
    assert [c["code"] for c in top] == ["BBBBBB", "CCCCCC", "DDDDDD"]


import math

from tools.momentum_grid import grid_distribution, grid_percentile


def _dist_grid():
    """Five configs whose TEST-window sharpe/return fan out evenly — a synthetic grid
    for the distribution helpers (the 'show all configs, not just the best' summary)."""
    def cell(code, ts, tr):
        return dict(code=code, train=dict(sharpe=1.0, net_return=1.0),
                    val=dict(sharpe=1.0, net_return=1.0),
                    test=dict(sharpe=ts, net_return=tr),
                    full=dict(sharpe=ts, net_return=tr), trades_per_year=10.0)
    return dict(cells=[cell("A", 0.0, 0.00), cell("B", 0.5, 0.10),
                       cell("C", 1.0, 0.20), cell("D", 1.5, 0.30),
                       cell("E", 2.0, 0.40)])


def test_grid_distribution_summarizes_all_configs():
    d = grid_distribution(_dist_grid(), window="test")
    assert d["n"] == 5
    assert abs(d["sharpe"]["median"] - 1.0) < 1e-9          # median of 0,.5,1,1.5,2
    assert d["sharpe"]["min"] == 0.0 and d["sharpe"]["max"] == 2.0
    assert abs(d["ret"]["median"] - 0.20) < 1e-9


def test_grid_distribution_skips_missing_window_and_nan():
    g = _dist_grid()
    g["cells"].append(dict(code="X", train={}, val={}, full={}))            # no 'test' key
    g["cells"].append(dict(code="Y", test=dict(sharpe=float("nan"),
                                               net_return=float("nan"))))   # NaN
    d = grid_distribution(g, window="test")
    assert d["sharpe"]["n"] == 5                            # X and Y both skipped


def test_grid_percentile_locates_the_live_book():
    g = _dist_grid()
    # sharpe 1.5 is >= four of the five (0,.5,1,1.5) → 80th percentile
    assert abs(grid_percentile(g, 1.5, window="test", metric="sharpe") - 80.0) < 1e-9
    assert math.isnan(grid_percentile({"cells": []}, 1.0))                  # empty → NaN
