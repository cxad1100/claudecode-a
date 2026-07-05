"""Shared universe-panel preparation + the production momentum net-returns helper.

Both research pages (/vol and /edge) need the same two inputs: the prepared universe
price panel (winsorized, tradeable-calendar-aligned, with slippage map and PIT) and the
production momentum strategy's net daily returns. This module is their single home so
the panel is read and prepared ONCE per page build and no report imports another
report's privates. Imports of build_momentum_report/build_strategy_report constants are
lazy to avoid import cycles at module load.
"""
import pandas as pd


def load_universe_panel() -> dict | None:
    """Universe panel + costs, prepared exactly like the strategy page (winsorized,
    tradeable calendar, PIT). None when the price panel isn't on disk; any other
    failure (malformed CSV, bad meta) propagates — a corrupt panel should be loud."""
    from build_momentum_report import PRICES_CSV, META_CSV, LIQ_MAX, MIN_PRICE, FEE_EUR, \
        WINSOR_CAP, _slip
    if not PRICES_CSV.exists():
        return None
    from tools.momentum import winsorize_prices, to_xetra_calendar
    from tools.universe_pit import PITUniverse
    from tools.universe_assemble import delisting_map
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices = winsorize_prices(to_xetra_calendar(prices), cap=WINSOR_CAP)
    meta_df = pd.read_csv(META_CSV)
    meta = {r["ticker"]: dict(r) for _, r in meta_df.iterrows()}
    slip = {t: _slip(m) for t, m in meta.items() if t in prices.columns}
    pit = PITUniverse(prices, delisting_map(meta_df))
    return dict(prices=prices, slip=slip, pit=pit, meta=meta,
                fee_eur=FEE_EUR, liq_max=LIQ_MAX, min_price=MIN_PRICE)


def momentum_net_returns(refresh: bool = False, panel: dict | None = None) -> pd.Series | None:
    """Net daily returns of the production momentum strategy (one run_momentum call,
    no grid/MC). Pass a prepared `panel` (from load_universe_panel) to avoid reading
    the universe twice; None when the panel isn't available."""
    from build_momentum_report import LOOKBACK, SKIP, START, LIQ_MAX, MIN_PRICE, \
        CAPITAL as MCAP, FEE_EUR as MFEE, EXEC_LAG
    from build_strategy_report import STRATEGY
    from tools.momentum import run_momentum
    if panel is None:
        panel = load_universe_panel()
    if panel is None:
        return None
    res = run_momentum(panel["prices"], panel["slip"], lookback=LOOKBACK, skip=SKIP,
                       capital=MCAP, cost_mults=(1.0,), start=START, liq_max=LIQ_MAX,
                       fee_eur=MFEE, min_price=MIN_PRICE, sectors=None, benchmark=None,
                       pit=panel["pit"], execute_lag=EXEC_LAG, **STRATEGY.kwargs())
    eq = res["runs"][1.0]["equity"]
    return eq.dropna().pct_change().dropna()
