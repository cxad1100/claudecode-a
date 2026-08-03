"""Portfolio parser and P&L calculator for Trade Republic holdings."""

import warnings
import csv
from pathlib import Path

import yfinance as yf

# Map TR tickers → yfinance price ticker
# EUNL.F (Frankfurt) has stale yfinance data — use IWDA.AS (same ISIN IE00B4L5Y983) for price
TICKER_MAP = {
    "NVD.F":  "NVD.F",
    "AMZ.F":  "AMZ.F",
    "ABEA.F": "ABEA.F",
    "TSFA.F": "TSFA.F",
    "LHL.F":  "LHL.F",
    "TCO0.F": "TCO0.F",
    "ASME.F": "ASME.F",
    "IES.F":  "IES.F",
    "CRIN.F": "CRIN.F",
    "EUNL.F": "IWDA.AS",  # same ISIN, IWDA.AS has live yfinance data; EUNL.F is stale
    "IPJ1.F": "IPJ1.F",
    "MSF.F":  "MSF.F",
}

# Tickers priced in non-EUR
TICKER_CURRENCY: dict[str, str] = {}

# Map portfolio ticker → fast_scores key (yf_ticker in universe)
RATING_LOOKUP = {
    "NVD.F":  "NVD.F",
    "AMZ.F":  "AMZ.F",
    "ABEA.F": "ABEA.F",
    "TSFA.F": "TSFA.F",
    "ASME.F": "ASME.F",
    "IES.F":  "IES.F",
    "CRIN.F": "CRIN.F",
    "EUNL.F": "EUNL.F",
    "WBD.MI": "WBD.MI",
    "APC.F":  "AAPL",
}

# Canonical display names
COMPANY_NAMES = {
    "NVD.F":  "NVIDIA Corporation",
    "AMZ.F":  "Amazon.com Inc.",
    "ABEA.F": "Alphabet Inc. (Google)",
    "TSFA.F": "Taiwan Semiconductor (TSMC)",
    "LHL.F":  "Lenovo Group",
    "TCO0.F": "Tesco PLC",
    "ASME.F": "ASML Holding",
    "IES.F":  "Intesa Sanpaolo",
    "CRIN.F": "UniCredit",
    "EUNL.F": "iShares Core MSCI World ETF",
    "WBD.MI": "Webuild S.p.A.",
    "APC.F":  "Apple Inc.",
    "MSF.F":  "Microsoft Corporation",
}


def parse_portfolio(csv_path: str | Path) -> dict:
    """
    Parse trade history CSV into current holdings with avg cost and realized P&L.

    Returns:
      holdings: {ticker: {shares, avg_cost_eur, total_invested, first_buy, last_activity}}
      realized: {ticker: {pnl_eur, shares_sold, proceeds}}
      transactions: list of all rows
    """
    holdings: dict[str, dict] = {}
    realized: dict[str, dict] = {}
    transactions = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker   = row["Ticker"].strip()
            action   = row["Action"].strip().lower()
            shares   = float(row["Shares"])
            price    = float(row["Price"])          # total EUR
            pps      = float(row["PricePerShare"])  # EUR per share
            date     = row["Date"].strip()

            transactions.append({
                "date": date, "ticker": ticker, "action": action,
                "shares": shares, "price": price, "pps": pps,
            })

            if action == "buy":
                if ticker not in holdings:
                    holdings[ticker] = {
                        "shares": 0.0, "avg_cost": 0.0,
                        "total_invested": 0.0, "first_buy": date,
                    }
                h = holdings[ticker]
                new_shares = h["shares"] + shares
                # Weighted average cost
                h["avg_cost"] = (h["shares"] * h["avg_cost"] + shares * pps) / new_shares if new_shares > 0 else pps
                h["shares"] = new_shares
                h["total_invested"] = h["total_invested"] + price
                h["last_activity"] = date

            elif action == "sell":
                if ticker not in realized:
                    realized[ticker] = {"pnl_eur": 0.0, "shares_sold": 0.0, "proceeds": 0.0}
                r = realized[ticker]
                avg_cost = holdings.get(ticker, {}).get("avg_cost", pps)
                r["pnl_eur"]    += (pps - avg_cost) * shares
                r["shares_sold"] += shares
                r["proceeds"]   += price

                if ticker in holdings:
                    holdings[ticker]["shares"] -= shares
                    holdings[ticker]["last_activity"] = date
                    if holdings[ticker]["shares"] <= 0.001:
                        del holdings[ticker]   # fully exited

    return {
        "holdings": holdings,
        "realized": realized,
        "transactions": transactions,
    }


BENCHMARKS = {
    "S&P 500":          ("CSPX.AS", "EUR"),  # iShares Core S&P 500 UCITS ETF Acc — EUR-listed, no FX noise
    "Nasdaq 100":       ("CNDX.AS", "EUR"),  # iShares Nasdaq 100 UCITS — EUR-listed
    "MSCI World":       ("IWDA.AS", "EUR"),  # iShares Core MSCI World — EUR-listed
    "FTSE All-World":   ("VWCE.DE", "EUR"),  # Vanguard FTSE All-World Acc — developed + EM
    "Euro Stoxx 50":    ("EXW1.DE", "EUR"),  # iShares Core EURO STOXX 50
    "Emerging Markets": ("EUNM.F",  "EUR"),  # iShares MSCI EM UCITS ETF Acc — EUR-listed Frankfurt
    "Gold":             ("GLD",     "USD"),
    "Bitcoin":          ("BTC-USD", "USD"),
    "Fixed Income":     ("BND",     "USD"),
}


def fetch_benchmark_returns(transactions: list[dict]) -> dict[str, dict]:
    """
    Cash-flow-matched benchmark comparison.

    For every buy in the portfolio, invest the same EUR amount into each benchmark
    at the price on that day (or next available trading day). Compare total
    hypothetical value today to total invested. Same logic the portfolio uses.
    """
    import pandas as pd

    buys = [t for t in transactions if t["action"] == "buy"]
    if not buys:
        return {}

    start_date = min(t["date"] for t in buys)
    total_invested_eur = sum(float(t["price"]) for t in buys)

    def _normalize(series):
        """Strip timezone from index, normalize to date-level."""
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        series.index = series.index.normalize()
        return series

    def _price_on(series, date_str: str) -> float:
        """Return first available price on or after date_str."""
        ts = pd.Timestamp(date_str)
        future = series[series.index >= ts]
        return float((future if not future.empty else series).iloc[0])

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Fetch EUR/USD for FX conversion of USD-denominated benchmarks
        try:
            eurusd = _normalize(yf.Ticker("EURUSD=X").history(start=start_date)["Close"].dropna())
        except Exception:
            eurusd = pd.Series(dtype=float)

        for name, (ticker, currency) in BENCHMARKS.items():
            try:
                hist = _normalize(yf.Ticker(ticker).history(start=start_date)["Close"].dropna())
                if hist.empty:
                    results[name] = {"return_pct": None, "ticker": ticker}
                    continue

                accumulated_shares = 0.0

                for txn in buys:
                    eur_spent = float(txn["price"])
                    date_str  = txn["date"]

                    bm_price_native = _price_on(hist, date_str)

                    if currency == "USD" and not eurusd.empty:
                        # Convert EUR → USD on transaction date, then buy benchmark
                        fx = _price_on(eurusd, date_str)  # EUR/USD rate
                        usd_spent = eur_spent * fx
                        accumulated_shares += usd_spent / bm_price_native
                    else:
                        # EUR-denominated — invest directly
                        accumulated_shares += eur_spent / bm_price_native

                current_price_native = float(hist.iloc[-1])
                total_native_value   = accumulated_shares * current_price_native

                if currency == "USD" and not eurusd.empty:
                    # Convert USD value back to EUR at today's rate
                    current_fx    = float(eurusd.iloc[-1])
                    current_value_eur = total_native_value / current_fx
                else:
                    current_value_eur = total_native_value

                ret_pct = (current_value_eur / total_invested_eur - 1) * 100

                results[name] = {
                    "return_pct":    round(ret_pct, 2),
                    "current_value": round(current_value_eur, 2),
                    "ticker":        ticker,
                }
            except Exception:
                results[name] = {"return_pct": None, "ticker": ticker}

    return results


def fetch_current_prices(holdings: dict) -> dict[str, float | None]:
    """Fetch current EUR prices for all held tickers. Converts USD/GBP/HKD via live FX."""
    import time as _time

    def _fetch_price_raw(yf_ticker: str, attempts: int = 3) -> float | None:
        for attempt in range(attempts):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    info = yf.Ticker(yf_ticker).fast_info
                    price = getattr(info, "last_price", None)
                    if price and float(price) > 0:
                        return round(float(price), 4)
                    hist = yf.Ticker(yf_ticker).history(period="5d")
                    if not hist.empty:
                        return round(float(hist["Close"].iloc[-1]), 4)
            except Exception:
                pass
            if attempt < attempts - 1:
                _time.sleep(20)
        return None

    # Fetch GBP/EUR only if TSCO.L is in holdings
    gbpeur = None
    if any(TICKER_MAP.get(t, t) == "TSCO.L" for t in holdings):
        gbpeur = _fetch_price_raw("GBPEUR=X")

    prices = {}
    for ticker in holdings:
        yf_ticker = TICKER_MAP.get(ticker, ticker)
        raw = _fetch_price_raw(yf_ticker)
        if raw is None:
            prices[ticker] = None
            continue
        if TICKER_CURRENCY.get(yf_ticker) == "GBP" and gbpeur:
            prices[ticker] = round(raw * gbpeur, 4)
        else:
            prices[ticker] = raw
    return prices


def compute_portfolio_summary(portfolio: dict, current_prices: dict) -> dict:
    """
    Compute full P&L summary.

    Returns:
      positions: list of position dicts (sorted by value desc)
      totals: {total_invested, current_value, unrealized_pnl, unrealized_pct,
               realized_pnl, total_pnl}
    """
    holdings = portfolio["holdings"]
    realized = portfolio["realized"]

    positions = []
    total_invested   = 0.0
    current_value    = 0.0
    unrealized_pnl   = 0.0

    for ticker, h in holdings.items():
        shares    = h["shares"]
        avg_cost  = h["avg_cost"]
        cur_price = current_prices.get(ticker)

        if cur_price:
            pos_value   = shares * cur_price
            pos_pnl     = (cur_price - avg_cost) * shares
            pos_pnl_pct = (cur_price / avg_cost - 1) * 100 if avg_cost > 0 else 0.0
        else:
            pos_value   = shares * avg_cost  # fallback: cost basis
            pos_pnl     = 0.0
            pos_pnl_pct = 0.0

        positions.append({
            "ticker":       ticker,
            "shares":       round(shares, 6),
            "avg_cost":     round(avg_cost, 4),
            "current_price": cur_price,
            "position_value": round(pos_value, 2),
            "cost_basis":   round(shares * avg_cost, 2),
            "unrealized_pnl": round(pos_pnl, 2),
            "unrealized_pct": round(pos_pnl_pct, 2),
            "first_buy":    h.get("first_buy", ""),
            "last_activity": h.get("last_activity", ""),
        })

        total_invested += shares * avg_cost
        current_value  += pos_value
        unrealized_pnl += pos_pnl

    # Realized P&L across all closed positions
    total_realized = sum(r["pnl_eur"] for r in realized.values())

    positions.sort(key=lambda x: -x["position_value"])

    return {
        "positions": positions,
        "totals": {
            "total_invested":   round(total_invested, 2),
            "current_value":    round(current_value, 2),
            "unrealized_pnl":   round(unrealized_pnl, 2),
            "unrealized_pct":   round((current_value / total_invested - 1) * 100, 2) if total_invested > 0 else 0,
            "realized_pnl":     round(total_realized, 2),
            "total_pnl":        round(unrealized_pnl + total_realized, 2),
        },
        "realized_detail": realized,
    }
