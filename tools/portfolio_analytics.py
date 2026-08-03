"""
Portfolio quant analytics: ROI time series, risk metrics, technicals.
Cash-flow matched benchmarks (same EUR amount on same dates).
All returns in EUR terms with FX conversion for USD-denominated instruments.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from tools.portfolio_tools import TICKER_MAP, BENCHMARKS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _norm(h: pd.Series) -> pd.Series:
    """Strip timezone, normalize to midnight."""
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    h.index = h.index.normalize()
    return h.sort_index()


def _price_on(h: pd.Series, date: pd.Timestamp) -> float | None:
    future = h[h.index >= date]
    if not future.empty:
        #return float(future.iloc[0])
        return float(future.values.flatten()[0])
    return float(h.iloc[-1]) if not h.empty else None


# ── Core time-series builder ──────────────────────────────────────────────────

def build_roi_timeseries(transactions: list[dict]) -> tuple[pd.Series, dict, dict]:
    """
    Returns (portfolio_roi_series, {benchmark_name: roi_series}, {key: eur_series}).

    The first two are pd.Series with DatetimeIndex, values = cumulative ROI %.
    The third maps each ticker to its EUR position value per business day (NaN when
    not held), plus "__cash__" (running sell proceeds, NaN before the first sell,
    absent if there are none) and "__total__" (holdings + cash — the same figure that
    feeds the ROI numerator, so it reconciles to the sum of the other keys).

    It also carries "__roi__" -> {ticker: ROI % series}, each position's own
    (value + its sale proceeds) / its own buys - 1, i.e. the portfolio formula applied
    per position, plus "__total__" for the portfolio line itself. Same index and same
    NaN gaps as the EUR series, so the two views are swappable on one chart.
    """
    buys = [t for t in transactions if t["action"] == "buy"]
    if not buys:
        return pd.Series(dtype=float), {}, {}

    start_date = min(t["date"] for t in buys)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # --- Batch download: portfolio tickers + benchmarks + EUR/USD in 2 calls ---
        port_tickers = list(set(t["ticker"] for t in transactions))
        yf_port_tickers = [TICKER_MAP.get(tk, tk) for tk in port_tickers]
        bm_tickers = [ticker for ticker, _ in BENCHMARKS.values()] + ["EURUSD=X"]

        def _batch_download(tickers: list[str], start: str) -> dict[str, pd.Series]:
            """Download multiple tickers at once, return {ticker: Close series}."""
            result = {}
            try:
                raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
                close = raw["Close"] if "Close" in raw else raw
                if len(tickers) == 1:
                    s = close.dropna()
                    if not s.empty:
                        result[tickers[0]] = _norm(s)
                else:
                    for tk in tickers:
                        if tk in close.columns:
                            s = close[tk].dropna()
                            if not s.empty:
                                result[tk] = _norm(s)
            except Exception:
                pass
            return result

        port_data = _batch_download(yf_port_tickers, start_date)
        bm_data   = _batch_download(bm_tickers, start_date)

        # Map back: orig portfolio ticker → series
        port_hist: dict[str, pd.Series] = {}
        for orig, yft in zip(port_tickers, yf_port_tickers):
            if yft in port_data:
                port_hist[orig] = port_data[yft]

        eurusd = bm_data.get("EURUSD=X", pd.Series(dtype=float))

        bm_hists: dict[str, tuple[pd.Series, str]] = {}
        for name, (ticker, currency) in BENCHMARKS.items():
            if ticker in bm_data:
                bm_hists[name] = (bm_data[ticker], currency)

    biz_days = pd.bdate_range(start=start_date, end=datetime.today())

    # Chronological queue — consumed with a `<=` pointer so weekend/holiday-dated
    # transactions (e.g. Tradegate Sunday trades) apply on the next business day
    # instead of being silently skipped.
    txn_queue = sorted(transactions, key=lambda x: (x["date"], x["action"]))
    txn_idx = 0

    # ── Portfolio series (cash-flow matched, same formula as benchmarks) ─────────
    # Return = (current_value / total_invested - 1), matching the benchmark formula
    # at line 204. This makes pp-delta comparisons meaningful.
    holdings: dict[str, float] = {}
    avg_cost: dict[str, float] = {}
    total_invested = 0.0
    cash_from_sells = 0.0   # sale proceeds stay in the return calc (otherwise sells look like losses)
    port_vals: dict[str, float] = {}
    # Per-asset EUR value on each business day. Filled inside the SAME day-walk that
    # produces the ROI line, from the same prices and the same `<=` txn pointer, so
    # the two charts can never disagree.
    asset_vals: dict[str, dict[str, float]] = {}
    cash_vals: dict[str, float] = {}
    total_vals: dict[str, float] = {}
    seen_sell = False
    # Per-asset ROI %, on the SAME formula as the portfolio line: a position's own
    # (value + its sale proceeds) / its own buys - 1. Rebasing the EUR curve instead
    # (value / first value) would score buying more shares as a gain.
    invested_tk: dict[str, float] = {}
    cash_tk: dict[str, float] = {}
    roi_vals: dict[str, dict[str, float]] = {}

    def _position_values(dt: pd.Timestamp) -> dict[str, float]:
        vals: dict[str, float] = {}
        for tk, sh in holdings.items():
            if sh <= 0:
                continue
            p = _price_on(port_hist[tk], dt) if tk in port_hist else avg_cost.get(tk)
            if p is None:
                p = avg_cost.get(tk, 0.0)
            vals[tk] = sh * p
        return vals

    for date in biz_days:
        ds = str(date.date())
        while txn_idx < len(txn_queue) and txn_queue[txn_idx]["date"] <= ds:
            txn = txn_queue[txn_idx]
            txn_idx += 1
            tk = txn["ticker"]
            if txn["action"] == "buy":
                sh, pps = float(txn["shares"]), float(txn["pps"])
                prev = holdings.get(tk, 0.0)
                new  = prev + sh
                avg_cost[tk] = (prev * avg_cost.get(tk, pps) + sh * pps) / new if new else pps
                holdings[tk] = new
                total_invested += float(txn["price"])
                invested_tk[tk] = invested_tk.get(tk, 0.0) + float(txn["price"])
            elif txn["action"] == "sell":
                holdings[tk] = max(0.0, holdings.get(tk, 0.0) - float(txn["shares"]))
                cash_from_sells += float(txn["price"])
                cash_tk[tk] = cash_tk.get(tk, 0.0) + float(txn["price"])
                seen_sell = True

        if total_invested == 0:
            continue

        pos_vals = _position_values(date)
        value = sum(pos_vals.values()) + cash_from_sells
        port_vals[ds] = round((value / total_invested - 1) * 100, 4)

        for tk, v in pos_vals.items():
            asset_vals.setdefault(tk, {})[ds] = v
            # Only while held, so the % line starts and ends exactly where the € one does.
            roi_vals.setdefault(tk, {})[ds] = (v + cash_tk.get(tk, 0.0)) / invested_tk[tk] * 100 - 100
        if seen_sell:
            cash_vals[ds] = cash_from_sells      # NaN before the first sell
        total_vals[ds] = value
        roi_vals.setdefault("__total__", {})[ds] = port_vals[ds]

    portfolio_series = pd.Series(port_vals)
    portfolio_series.index = pd.to_datetime(portfolio_series.index)

    def _daily(vals: dict[str, float]) -> pd.Series:
        """Dict of 'YYYY-MM-DD' -> value, reindexed on every business day.

        Reindexing leaves NaN where the position wasn't held, so plotly breaks the
        line instead of drawing it along the axis — and a re-buy after a full exit
        renders as two segments rather than one phantom stroke.
        """
        s = pd.Series(vals, dtype=float)
        if not s.empty:
            s.index = pd.to_datetime(s.index)
        return s.reindex(biz_days)

    asset_values: dict[str, pd.Series] = {tk: _daily(v) for tk, v in asset_vals.items()}
    if cash_vals:
        asset_values["__cash__"] = _daily(cash_vals)
    asset_values["__total__"] = _daily(total_vals)
    # Nested under a reserved key: callers iterate the top level for tickers, and
    # every consumer already filters keys starting with "__".
    asset_values["__roi__"] = {tk: _daily(v) for tk, v in roi_vals.items()}

    # ── Benchmark series ──────────────────────────────────────────────────────
    buy_events = sorted([(t["date"], float(t["price"])) for t in buys], key=lambda x: x[0])
    benchmark_series: dict[str, pd.Series] = {}

    for name, (bm_hist, currency) in bm_hists.items():
        bm_shares = 0.0
        bm_invested = 0.0
        buy_idx = 0
        bm_vals: dict[str, float] = {}

        for date in biz_days:
            ds = str(date.date())

            while buy_idx < len(buy_events) and buy_events[buy_idx][0] <= ds:
                bdate, eur_amt = buy_events[buy_idx]
                bts = pd.Timestamp(bdate)
                bm_px = _price_on(bm_hist, bts)
                if bm_px:
                    if currency == "USD" and not eurusd.empty:
                        fx = _price_on(eurusd, bts) or 1.0
                        bm_shares += (eur_amt * fx) / bm_px
                    else:
                        bm_shares += eur_amt / bm_px
                    bm_invested += eur_amt
                buy_idx += 1

            if bm_invested == 0:
                continue

            bm_px_now = _price_on(bm_hist, date)
            if bm_px_now is None:
                continue

            native_val = bm_shares * bm_px_now
            if currency == "USD" and not eurusd.empty:
                fx_now = _price_on(eurusd, date) or float(eurusd.iloc[-1])
                eur_val = native_val / fx_now
            else:
                eur_val = native_val

            bm_vals[ds] = round((eur_val / bm_invested - 1) * 100, 4)

        s = pd.Series(bm_vals)
        s.index = pd.to_datetime(s.index)
        benchmark_series[name] = s

    return portfolio_series, benchmark_series, asset_values


# ── Quant metrics ─────────────────────────────────────────────────────────────

def compute_quant_metrics(
    portfolio_series: pd.Series,
    sp500_series: pd.Series | None = None,
    rf_annual_pct: float = 4.5,
) -> dict:
    """
    Compute standard quant risk/return metrics.
    portfolio_series: cumulative ROI % (e.g. 35.9 means +35.9%).
    """
    if len(portfolio_series) < 10:
        return {}

    # Index level: starts at 1.0
    lvl = (portfolio_series / 100 + 1).dropna()
    daily_ret = lvl.pct_change().dropna()

    if len(daily_ret) < 5:
        return {}

    n_days  = len(daily_ret)
    n_years = n_days / 252
    rf_d    = rf_annual_pct / 252 / 100

    total_roi  = float(portfolio_series.iloc[-1])
    cagr       = ((1 + total_roi / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    vol        = float(daily_ret.std()) * (252 ** 0.5) * 100

    excess = daily_ret - rf_d
    sharpe = float(excess.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0

    down = daily_ret[daily_ret < rf_d]
    sortino_denom = float(down.std()) * (252 ** 0.5) if len(down) > 0 and down.std() > 0 else 1e-9
    sortino = float((daily_ret.mean() - rf_d) * 252 / sortino_denom)

    # Max drawdown
    cum = (1 + daily_ret).cumprod()
    roll_max = cum.cummax()
    dd_series = (cum - roll_max) / roll_max
    max_dd = float(dd_series.min()) * 100
    current_dd = float(dd_series.iloc[-1]) * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # VaR / CVaR 95%
    var_95  = float(daily_ret.quantile(0.05)) * 100
    cvar_95 = float(daily_ret[daily_ret <= daily_ret.quantile(0.05)].mean()) * 100

    win_rate = float((daily_ret > 0).mean() * 100)

    # Beta / Alpha vs S&P 500
    beta = alpha = tracking_error = info_ratio = None
    if sp500_series is not None and len(sp500_series) > 10:
        sp_lvl   = (sp500_series / 100 + 1).dropna()
        sp_ret   = sp_lvl.pct_change().dropna()
        common   = daily_ret.index.intersection(sp_ret.index)
        if len(common) > 10:
            p = daily_ret[common].values
            s = sp_ret[common].values
            cov_mat = np.cov(p, s)
            beta    = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] > 0 else None
            if beta is not None:
                alpha_d = float(p.mean() - beta * s.mean())
                alpha   = round(alpha_d * 252 * 100, 2)
                te      = float((p - s).std()) * (252 ** 0.5) * 100
                ir      = float((p - s).mean() * 252 * 100) / (te + 1e-9)
                tracking_error = round(te, 2)
                info_ratio     = round(ir, 2)

    return {
        "total_roi":       round(total_roi, 2),
        "cagr":            round(cagr, 2),
        "volatility":      round(vol, 2),
        "sharpe":          round(sharpe, 2),
        "sortino":         round(sortino, 2),
        "max_drawdown":    round(max_dd, 2),
        "current_drawdown": round(current_dd, 2),
        "calmar":          round(calmar, 2),
        "var_95":          round(var_95, 2),
        "cvar_95":         round(cvar_95, 2),
        "win_rate":        round(win_rate, 1),
        "beta":            round(beta, 2) if beta is not None else None,
        "alpha":           round(alpha, 2) if alpha is not None else None,
        "tracking_error":  tracking_error,
        "info_ratio":      info_ratio,
        "best_day":        round(float(daily_ret.max()) * 100, 2),
        "worst_day":       round(float(daily_ret.min()) * 100, 2),
        "n_trading_days":  n_days,
    }


# ── Technical indicators per position ────────────────────────────────────────

def compute_position_technicals(holdings: dict, ticker_map: dict) -> dict[str, dict]:
    """
    For each open position compute: RSI-14, MA50/MA200 signal, momentum 1M/3M/6M.
    Batch downloads all tickers in one yf.download call.
    """
    results = {}

    active = {tk: ticker_map.get(tk, tk) for tk, h in holdings.items() if h.get("shares", 0) > 0}
    if not active:
        return results

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        yf_tickers = list(set(active.values()))
        try:
            raw = yf.download(yf_tickers, period="1y", auto_adjust=True, progress=False)
            close_raw = raw["Close"] if "Close" in raw else raw
            if len(yf_tickers) == 1:
                batch = {yf_tickers[0]: close_raw.dropna()}
            else:
                batch = {t: close_raw[t].dropna() for t in yf_tickers if t in close_raw.columns}
        except Exception:
            batch = {}

        for tk, yft in active.items():
            hist = batch.get(yft)
            if hist is None or len(hist) < 20:
                continue
            try:
                c = hist.values.astype(float)

                # RSI-14
                delta = np.diff(c)
                gain  = np.where(delta > 0, delta, 0.0)
                loss  = np.where(delta < 0, -delta, 0.0)
                avg_g = np.convolve(gain, np.ones(14)/14, mode='valid')[-1]
                avg_l = np.convolve(loss, np.ones(14)/14, mode='valid')[-1]
                rsi   = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 50.0

                # Moving averages
                ma50  = float(np.mean(c[-50:])) if len(c) >= 50 else None
                ma200 = float(np.mean(c[-200:])) if len(c) >= 200 else None
                price = float(c[-1])

                if ma50 and ma200:
                    if price > ma50 > ma200:
                        ma_signal = "BULLISH"
                    elif price < ma50 < ma200:
                        ma_signal = "BEARISH"
                    elif ma50 > ma200:
                        ma_signal = "MIXED"
                    else:
                        ma_signal = "MIXED"
                elif ma50:
                    ma_signal = "ABOVE MA50" if price > ma50 else "BELOW MA50"
                else:
                    ma_signal = "N/A"

                # Momentum
                def ret_n(n):
                    return round((c[-1] / c[-min(n, len(c))] - 1) * 100, 2) if len(c) >= n else None

                # 52W high/low
                high_52w = float(np.max(c))
                low_52w  = float(np.min(c))
                pct_from_high = round((price / high_52w - 1) * 100, 1)
                pct_from_low  = round((price / low_52w  - 1) * 100, 1)

                results[tk] = {
                    "rsi":           round(rsi, 1),
                    "rsi_signal":    "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else "NEUTRAL"),
                    "ma50":          round(ma50, 2) if ma50 else None,
                    "ma200":         round(ma200, 2) if ma200 else None,
                    "ma_signal":     ma_signal,
                    "mom_1m":        ret_n(21),
                    "mom_3m":        ret_n(63),
                    "mom_6m":        ret_n(126),
                    "pct_from_52w_high": pct_from_high,
                    "pct_from_52w_low":  pct_from_low,
                }
            except Exception:
                pass

    return results


# ── Portfolio-level stats ────────────────────────────────────────────────────

def compute_concentration_metrics(positions: list[dict]) -> dict:
    """HHI concentration, top-N weight, sector breakdown."""
    total_val = sum(p["position_value"] for p in positions if p["position_value"] > 0)
    if total_val == 0:
        return {}

    weights = [p["position_value"] / total_val for p in positions if p["position_value"] > 0]
    hhi = sum(w ** 2 for w in weights)  # 0 = max diversification, 1 = full concentration
    top1_w = max(weights) * 100
    top3_w = sum(sorted(weights, reverse=True)[:3]) * 100

    return {
        "hhi":           round(hhi, 4),
        "concentration": "HIGH" if hhi > 0.25 else ("MODERATE" if hhi > 0.15 else "DIVERSIFIED"),
        "top1_weight":   round(top1_w, 1),
        "top3_weight":   round(top3_w, 1),
        "n_positions":   len([p for p in positions if p["position_value"] > 0]),
    }


def compute_correlation_matrix(holdings: dict, ticker_map: dict) -> dict:
    """
    Pairwise 1-year daily return correlation for all open positions.
    Returns {ticker: {ticker: corr}} using original portfolio tickers as labels.
    """
    active = {tk: ticker_map.get(tk, tk)
              for tk, h in holdings.items() if h.get("shares", 0) > 0}
    if len(active) < 2:
        return {}

    yf_tickers = list(set(active.values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            raw   = yf.download(yf_tickers, period="1y", auto_adjust=True, progress=False)
            close = raw["Close"] if "Close" in raw else raw
            if len(yf_tickers) == 1:
                rets = pd.DataFrame({yf_tickers[0]: close.pct_change().dropna()})
            else:
                rets = close.pct_change().dropna()
            # Map yfinance ticker → original portfolio ticker (keep first if dupe)
            yft_to_orig = {}
            for orig, yft in active.items():
                yft_to_orig.setdefault(yft, orig)
            rets = rets[[c for c in yf_tickers if c in rets.columns]]
            rets = rets.rename(columns=yft_to_orig)
            corr = rets.corr().round(3)
            return corr.to_dict()
        except Exception:
            return {}


def xirr(flows: list[tuple], bracket=(-0.9999, 10.0)) -> float | None:
    """
    Money-weighted (internal) rate of return, annualised, from dated cash flows.

    flows: list of (datetime.date, amount). Convention: money OUT of pocket is
    negative (buys), money IN is positive (sells + today's portfolio value).
    Solves Sum cf_i / (1+r)^(days_i/365) = 0 for r. None if no sign change.
    """
    from scipy.optimize import brentq
    if len(flows) < 2:
        return None
    t0 = min(d for d, _ in flows)
    years = [(d - t0).days / 365.0 for d, _ in flows]
    amts = [float(a) for _, a in flows]
    if not (min(amts) < 0 < max(amts)):
        return None

    def npv(r):
        return sum(a / (1 + r) ** y for a, y in zip(amts, years))

    try:
        return float(brentq(npv, *bracket, maxiter=200))
    except Exception:
        return None
