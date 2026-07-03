"""The 64-permutation momentum matrix: the six blueprint upgrades (A–F) as a
config, run walk-forward over the survivorship-corrected universe, split into
train/validation, with a small-account feasibility calc. Pure (data in, numbers
out); the build script supplies prices/sectors/benchmark/pit.
"""
from dataclasses import dataclass
from itertools import product

import pandas as pd

from tools.momentum import (run_momentum, precompute_eligibility, precompute_scores,
                            rebalance_dates)
from tools.pairs_backtest import backtest_stats


@dataclass(frozen=True)
class MomentumConfig:
    vol_adjust: bool = False        # A
    sector_neutral: bool = False    # B
    trend_filter: bool = False      # C
    slots: int = 15                 # D: 15 off / 10 on
    freq: str = "M"                 # E: "M" off / "Q" on
    lazy: bool = False              # F

    @property
    def code(self) -> str:
        flags = [self.vol_adjust, self.sector_neutral, self.trend_filter,
                 self.slots == 10, self.freq == "Q", self.lazy]
        return "".join(c if on else "·" for c, on in zip("ABCDEF", flags))

    def kwargs(self) -> dict:
        return dict(vol_adjust=self.vol_adjust, sector_neutral=self.sector_neutral,
                    trend_filter=self.trend_filter, k=self.slots, freq=self.freq,
                    lazy=self.lazy)


ALL_CONFIGS = [MomentumConfig(va, sn, tf, slots, freq, lz)
               for va, sn, tf, slots, freq, lz in product(
                   (False, True), (False, True), (False, True),
                   (15, 10), ("M", "Q"), (False, True))]


def _stats_slice(equity: pd.Series, trades: list, lo, hi, capital: float) -> dict:
    """backtest_stats over the [lo, hi] window, re-based to `capital` at the slice
    start so net_return/sharpe/drawdown describe that sub-period in isolation."""
    eq = equity.loc[lo:hi]
    tr = [t for t in trades if lo <= pd.Timestamp(t["entry"]) <= hi]
    if len(eq) < 2:
        return dict(net_return=0.0, sharpe=0.0, max_drawdown=0.0, win_rate=None,
                    total_costs=0.0)
    return backtest_stats(eq / eq.iloc[0] * capital, tr, capital)


def run_grid(prices, slippage_bps, *, sectors=None, benchmark=None, pit=None,
             configs=None, start="2018-01-01", train_end="2022-12-31", val_end=None,
             capital=10_000.0, lookback=252, skip=21, execute_lag=0,
             turnover=None, turn_floor=100_000.0, turn_window=6) -> dict:
    """Run each config walk-forward over [start, end]; partition each equity curve +
    trades into train (≤ train_end), validation (train_end < t ≤ val_end) and — when
    `val_end` is given — a held-out test (> val_end). `test` is the honest check: the
    config is *chosen* on validation, so val is no longer untouched; test never informs
    the pick. Without `val_end`, validation runs to the end (two-way split, back-compat).

    Returns {"cells": [{code, config, train, val, [test], full, trades_per_year}], ...}.
    One run per config — picks use only past data, so the slices are genuine OOS.
    """
    configs = configs or ALL_CONFIGS
    te = pd.Timestamp(train_end)
    ve = pd.Timestamp(val_end) if val_end else None
    # Eligibility is config-independent → compute it once over all candidate monthly
    # rebalance dates (a superset of the quarterly ones) and share across every config.
    cutoff = pd.Timestamp(start)
    cand_dates = [d for d in rebalance_dates(prices.index, "M")
                  if len(prices.loc[:d]) >= lookback + 1 and d >= cutoff]
    elig_by_date = precompute_eligibility(prices, slippage_bps, cand_dates,
                                          min_obs=lookback + skip, pit=pit,
                                          turnover=turnover, turn_floor=turn_floor,
                                          turn_window=turn_window)
    score_by_date = precompute_scores(prices, cand_dates, lookback, skip)
    cells = []
    for cfg in configs:
        r = run_momentum(prices, slippage_bps, capital=capital, cost_mults=(1.0,),
                         lookback=lookback, skip=skip, start=start, sectors=sectors,
                         benchmark=benchmark, pit=pit, execute_lag=execute_lag,
                         elig_by_date=elig_by_date, score_by_date=score_by_date, **cfg.kwargs())
        eq = r["runs"][1.0]["equity"]
        tr = r["runs"][1.0]["trades"]
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9) if len(eq) else 1.0
        val_hi = ve if ve is not None else eq.index[-1]
        cell = dict(
            code=cfg.code, config=cfg,
            train=_stats_slice(eq, tr, eq.index[0], te, capital),
            val=_stats_slice(eq, tr, te + pd.Timedelta(days=1), val_hi, capital),
            full=r["runs"][1.0]["stats"],
            trades_per_year=len(tr) / years,
            timeline=[dict(date=str(h["date"].date()), ret=h["ret"], dead=list(h["dead"]))
                      for h in r["holdings_log"]])
        if ve is not None:
            cell["test"] = _stats_slice(eq, tr, ve + pd.Timedelta(days=1), eq.index[-1], capital)
        cells.append(cell)
    return dict(cells=cells, train_end=train_end, val_end=val_end)


def feasibility(cell: dict, *, capital: float = 10_000.0, fee_eur: float = 1.0) -> dict:
    """Phase-6: fixed-fee drag on a €`capital` account. annual_fee = trades/yr × fee;
    fee_drag_pct = that as % of capital; pays_for_itself = net return clears the drag."""
    annual_fee = cell["trades_per_year"] * fee_eur
    drag_pct = annual_fee / capital * 100.0
    return dict(annual_fee_eur=annual_fee, fee_drag_pct=drag_pct,
                pays_for_itself=cell["full"]["net_return"] * 100.0 > drag_pct)


def pick_ultimate(grid: dict, *, capital: float = 10_000.0, fee_eur: float = 1.0):
    """The 'ultimate' config: among configs that pay for themselves and are positive
    in BOTH train and validation, the one maximising min(train_Sharpe, val_Sharpe) —
    worst-case robustness, so a val-lucky or train-overfit cell can't win. None if
    nothing qualifies. Turnover (fewer trades) breaks ties."""
    cands = []
    for c in grid["cells"]:
        if not feasibility(c, capital=capital, fee_eur=fee_eur)["pays_for_itself"]:
            continue
        if c["train"]["net_return"] <= 0 or c["val"]["net_return"] <= 0:
            continue
        robust = min(c["train"]["sharpe"], c["val"]["sharpe"])
        cands.append((robust, -c["trades_per_year"], c))
    if not cands:
        return None
    return max(cands, key=lambda x: (x[0], x[1]))[2]
