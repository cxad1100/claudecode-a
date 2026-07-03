"""Long-only cross-sectional momentum engine (pure functions, no I/O).

Strategy: at each monthly rebalance, rank every eligible name by 12-1 momentum
(trailing ~12 months skipping the most recent ~1 month), hold an equal-weight
global top-k until the next rebalance. Walk-forward, no look-ahead: ranks use
only data with index <= the rebalance date; returns accrue strictly after.

Holdings are computed once and are independent of trading costs; each cost
multiple re-prices the identical schedule (the cost-sensitivity table).
"""

import numpy as np
import pandas as pd

from tools.pairs_backtest import backtest_stats


def rebalance_dates(index, freq: str = "M") -> list[pd.Timestamp]:
    """Last trading day present in the index for each period (default month)."""
    idx = pd.DatetimeIndex(index)
    # Map deprecated period aliases to the period-END forms pandas now requires.
    actual_freq = {"M": "ME", "Q": "QE", "Y": "YE"}.get(freq, freq)
    last = pd.Series(idx, index=idx).resample(actual_freq).last().dropna()
    return list(last)


def momentum_scores(prices: pd.DataFrame, asof, lookback: int = 252,
                    skip: int = 21, vol_adjust: bool = False) -> pd.Series:
    """12-1 momentum per ticker: price(asof-skip) / price(asof-lookback) - 1.

    Uses only rows with index <= asof (no look-ahead). Returns an empty Series
    when there is not yet `lookback`+1 rows of history. inf/NaN dropped. When
    `vol_adjust` (upgrade A), divide each raw score by the name's annualised daily
    volatility over the lookback window — penalising choppy names, rewarding steady
    climbers.
    """
    hist = prices.loc[:asof]
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    recent = hist.iloc[-(skip + 1)]              # ~skip days before asof
    base = hist.iloc[-(lookback + 1)]            # ~lookback days before asof
    scores = recent / base - 1.0
    if vol_adjust:
        rets = hist.iloc[-(lookback + 1):].pct_change().iloc[1:]
        vol = rets.std() * np.sqrt(252)
        scores = scores / vol.replace(0.0, np.nan)
    return scores.replace([np.inf, -np.inf], np.nan).dropna()


def eligible(prices: pd.DataFrame, asof, slippage_bps: dict,
             liq_max: int = 30, min_obs: int = 273, min_price: float = 1.0,
             turnover: pd.DataFrame | None = None, turn_floor: float = 100_000.0,
             turn_window: int = 6) -> set[str]:
    """Tradeable names at `asof`: tight half-spread, enough history, and a last
    price >= min_price. The price floor drops sub-EUR1 penny listings whose 12-1
    momentum is dominated by tick/illiquidity noise rather than real return.

    When a monthly `turnover` frame is given (month-end index, per-ticker median daily
    EUR turnover), a covered name must ALSO clear `turn_floor` on the median of its last
    `turn_window` monthly values at/before `asof` — point-in-time liquidity, replacing
    the build-time full-window gate (a mild look-ahead) for covered names. Uncovered
    names (graveyard, pre-refresh data) keep the static build-time gate."""
    hist = prices.loc[:asof]
    last = hist.ffill().iloc[-1]                  # last valid price per ticker (vectorized)
    count = hist.notna().sum()                    # observations per ticker (vectorized)
    tw = (turnover.loc[:asof].tail(turn_window)
          if turnover is not None and len(turnover) else None)
    out: set[str] = set()
    for t in prices.columns:
        if slippage_bps.get(t, 10**9) > liq_max:
            continue
        if count[t] < min_obs or last[t] < min_price:
            continue
        if tw is not None and t in tw.columns:
            s = tw[t].dropna()
            if len(s) and float(s.median()) < turn_floor:
                continue
        out.add(t)
    return out


def precompute_eligibility(prices: pd.DataFrame, slippage_bps: dict, dates, *,
                           liq_max: int = 30, min_obs: int = 273,
                           min_price: float = 1.0, pit=None,
                           turnover: pd.DataFrame | None = None,
                           turn_floor: float = 100_000.0, turn_window: int = 6) -> dict:
    """{date: eligible ∩ listed} per date — config-independent, so the grid computes
    it once and shares it across all 64 configs (otherwise the dominant cost)."""
    out = {}
    for d in dates:
        e = eligible(prices, d, slippage_bps, liq_max, min_obs, min_price,
                     turnover=turnover, turn_floor=turn_floor, turn_window=turn_window)
        if pit is not None:
            e = {t for t in e if pit.listed(t, d)}
        out[d] = e
    return out


def precompute_scores(prices: pd.DataFrame, dates, lookback: int = 252,
                      skip: int = 21) -> dict:
    """{date: {"raw": Series, "voladj": Series}} per date — scores are config-
    independent (only the A toggle picks the variant), so the grid computes the
    expensive vol-adjusted scores once per date and shares them across configs."""
    return {d: {"raw": momentum_scores(prices, d, lookback, skip),
                "voladj": momentum_scores(prices, d, lookback, skip, vol_adjust=True)}
            for d in dates}


def select_topk(scores: pd.Series, eligible_set: set[str], k: int,
                sectors: dict | None = None) -> list[str]:
    """Top-k tickers by score, restricted to the eligible set, highest first. When
    `sectors` is given (upgrade B), fill k by round-robin over distinct sectors —
    the best remaining name per sector each pass — structurally capping single-sector
    concentration. 'Unknown' is one bucket like any other (≤ one per pass)."""
    s = scores[[t for t in scores.index if t in eligible_set]].sort_values(ascending=False)
    if sectors is None:
        return list(s.head(k).index)
    by_sec: dict[str, list[str]] = {}
    for t in s.index:                                  # already score-desc
        by_sec.setdefault(sectors.get(t, "Unknown"), []).append(t)
    order = sorted(by_sec, key=lambda sec: s[by_sec[sec][0]], reverse=True)
    picks: list[str] = []
    while len(picks) < k and any(by_sec.values()):
        for sec in order:
            if by_sec[sec]:
                picks.append(by_sec[sec].pop(0))
                if len(picks) >= k:
                    break
    return picks


def trend_ok(benchmark: pd.Series, asof, ma: int = 200) -> bool:
    """True when the benchmark closes at/above its `ma`-day moving average at asof
    (risk-on). Too little history → True (don't gate). Upgrade C kill-switch."""
    s = benchmark.loc[:asof].dropna()
    if len(s) < ma:
        return True
    return float(s.iloc[-1]) >= float(s.iloc[-ma:].mean())


def to_xetra_calendar(prices: pd.DataFrame, min_prints: int = 20) -> pd.DataFrame:
    """Restrict the price frame to the XETRA (Lang & Schwarz) trading calendar.

    Trade Republic fills via L&S, which trades on XETRA sessions: Mon–Fri minus German
    holidays. The global price frame, by contrast, carries every exchange's calendar —
    Israeli/Gulf venues trade Sun–Thu, and foreign markets stay open on German holidays.
    Those non-XETRA sessions are days you *can't actually trade*, and they're thinly
    populated, so an iloc lookback offset (or a month-end rebalance) landing on one
    collapses the scored set to the 3–4 names that happened to print — the report showed
    rebalances of just 'QLTU TMRP MLSR ALTF' (Tel-Aviv names, untradeable here).

    The XETRA calendar is derived from the universe's own XETRA-listed names: a German
    XETRA stock (yfinance `.DE`, e.g. SAP.DE; or the legacy EODHD `.XETRA`) prints only when
    XETRA is open, so days where a quorum (`min_prints`) of them trade are exactly XETRA
    sessions. Aligning here means every rebalance and every lookback lands on a real,
    tradeable European session, and the 252-row window is ≈ 12 true months (no Sundays)."""
    xc = [c for c in prices.columns if c.endswith(".DE") or c.endswith(".XETRA")]
    if len(xc) < min_prints:
        return prices
    return prices.loc[prices[xc].notna().sum(axis=1) >= min_prints]


def winsorize_prices(prices: pd.DataFrame, cap: float = 0.5) -> pd.DataFrame:
    """Rebuild a price frame with daily returns clipped to ±`cap`, recompounded from
    each column's first valid price. Kills split-adjustment glitches — a Frankfurt
    cross-listing feed printing a +1500% one-day jump (Orkla, Seadrill) that momentum
    would otherwise chase into a phantom return. Clean columns are untouched; NaN tails
    (dead names) are preserved so the graveyard still liquidates them. A capped glitch
    spike now reverses the next bar, so the name looks *choppy* to momentum, not
    explosive — exactly the de-selection we want. Cap is a judgment call: ±50% kills
    glitches while keeping all but the rarest real one-day moves."""
    r = prices.pct_change().clip(lower=-cap, upper=cap)
    out = {}
    for c in prices.columns:
        s = prices[c]
        first = s.first_valid_index()
        if first is None:
            continue
        rc = r[c].loc[first:].copy()
        rc.iloc[0] = 0.0
        lvl = float(s.loc[first]) * (1.0 + rc.fillna(0.0)).cumprod()
        lvl[s.loc[first:].isna()] = float("nan")          # keep dead/gap tails as NaN
        out[c] = lvl
    return pd.DataFrame(out).reindex(prices.index)


def _exec_date(index: pd.DatetimeIndex, d, lag: int = 0):
    """Execution bar for a signal at date `d`: `lag`=0 trades at the signal-day close
    (baseline), `lag`=1 at the next bar (t+1 — you can't trade on the same close you
    used to rank). Clamps at the end of the index."""
    if lag <= 0:
        return d
    pos = index.searchsorted(pd.Timestamp(d), side="right")   # first bar strictly after d
    return index[min(pos + lag - 1, len(index) - 1)]


def run_momentum(prices: pd.DataFrame, slippage_bps: dict, *, k: int = 15,
                 lookback: int = 252, skip: int = 21, capital: float = 10_000.0,
                 cost_mults: tuple = (0.0, 1.0, 2.0), freq: str = "M",
                 liq_max: int = 30, fee_eur: float = 1.0,
                 min_price: float = 1.0, start: str | None = None,
                 vol_adjust: bool = False, sectors: dict | None = None,
                 sector_neutral: bool = False, benchmark=None,
                 trend_filter: bool = False, lazy: bool = False, pit=None,
                 execute_lag: int = 0,
                 turnover: pd.DataFrame | None = None, turn_floor: float = 100_000.0,
                 turn_window: int = 6,
                 elig_by_date: dict | None = None,
                 score_by_date: dict | None = None) -> dict:
    """Walk-forward momentum backtest.

    Returns {"runs": {mult: {equity, trades, stats}}, "holdings_log": [...],
             "start": iso}. The schedule (holdings_log) is cost-independent;
    each cost multiple compounds the same equal-weight daily returns minus a
    rebalance cost drag. `start` clips the first rebalance to that date (scores
    still use the full prior history, so no look-ahead is introduced).

    Upgrade toggles (all default off → the baseline): `vol_adjust` (A),
    `sector_neutral` + `sectors` (B), `trend_filter` + `benchmark` (C), `lazy` (F).
    A `pit` (PITUniverse) activates point-in-time eligibility (already-dead names are
    never picked) and the graveyard — a name dying mid-hold is forward-filled to its
    last traded price (liquidated to cash) so the backtest eats the real loss.
    `execute_lag=1` fills one bar after the signal (t+1) instead of at the signal-day
    close, removing the same-bar look-ahead; scores still use only data through `d`.
    """
    dates = [d for d in rebalance_dates(prices.index, freq)
             if len(prices.loc[:d]) >= lookback + 1]
    if start is not None:
        cutoff = pd.Timestamp(start)
        dates = [d for d in dates if d >= cutoff]
    holdings_log = []
    for i in range(len(dates) - 1):
        d = dates[i]
        if score_by_date is not None and d in score_by_date:  # precomputed (grid: shared across configs)
            sc = score_by_date[d]
            scores = sc["voladj"] if vol_adjust else sc["raw"]
        else:
            scores = momentum_scores(prices, d, lookback, skip, vol_adjust=vol_adjust)
        if elig_by_date is not None:                          # precomputed (grid: shared across configs)
            elig = elig_by_date.get(d, set())
        else:
            elig = eligible(prices, d, slippage_bps, liq_max, lookback + skip, min_price,
                            turnover=turnover, turn_floor=turn_floor, turn_window=turn_window)
            if pit is not None:
                elig = {t for t in elig if pit.listed(t, d)}  # drop already-dead names
        if trend_filter and benchmark is not None and not trend_ok(benchmark, d):
            picks = []                                         # kill-switch → cash
        else:
            picks = select_topk(scores, elig, k,
                                 sectors=sectors if sector_neutral else None)
        died = pit.died_between(d, dates[i + 1]) if pit is not None else set()
        dead = {t for t in picks if t in died}
        holdings_log.append(dict(date=d, next=dates[i + 1], picks=picks,
                                 scores={t: float(scores[t]) for t in picks},
                                 ret={}, dead=dead))

    runs = {}
    for mult in cost_mults:
        equity_val = capital
        eq_points = [(holdings_log[0]["date"], capital)] if holdings_log \
            else [(prices.index[0], capital)]
        trades = []
        prev: set[str] = set()
        for h in holdings_log:
            d, nxt, picks = h["date"], h["next"], h["picks"]
            ed, enxt = _exec_date(prices.index, d, execute_lag), _exec_date(prices.index, nxt, execute_lag)
            if not picks:
                if prev:                                # liquidate to cash: charge exit
                    equity_val -= sum(fee_eur + slippage_bps[t] / 1e4 * (equity_val / len(prev))
                                      for t in prev) * mult
                prev = set()
                # Hold cash FLAT across the period: the account keeps its value and earns 0.
                # Record a point each trading day so the equity curve plateaus — without this
                # the kill-switch gap is left empty and the plot draws a straight diagonal
                # across it (looks like data loss; a 2022-bear cash spell spanned ~6 months).
                for day in prices.loc[ed:enxt].index:
                    eq_points.append((day, equity_val))
                continue
            w = equity_val / len(picks)                 # equal notional per name
            traded = (set(picks) ^ prev)                # enters + exits
            cost = sum(fee_eur + slippage_bps[t] / 1e4 * w
                       for t in traded) * mult
            equity_val -= cost
            seg = prices.loc[ed:enxt, picks].ffill().bfill()  # ffill: dead leg held at last price;
            #   bfill: if the rebalance date is a non-trading day (e.g. a Dec-31 holiday that
            #   only sits in the index because another name printed), the leading NaN can't be
            #   forward-filled — use the first tradeable price on/after d so the period return
            #   isn't NaN. Only leading NaNs are touched (mid-series gaps are ffilled first).
            if lazy:                                     # weights drift, no daily re-equal-weight
                basket = (seg / seg.iloc[0]).mean(axis=1)
                for day in seg.index[1:]:
                    eq_points.append((day, equity_val * float(basket[day])))
                equity_val = equity_val * float(basket.iloc[-1])
            else:
                rets = seg.pct_change().iloc[1:].fillna(0.0)
                port_ret = rets.mean(axis=1)            # equal-weight daily return
                for day, r in port_ret.items():
                    equity_val *= (1.0 + r)
                    eq_points.append((day, equity_val))
            for t in picks:
                name_ret = float(seg[t].iloc[-1] / seg[t].iloc[0] - 1.0)
                h["ret"][t] = name_ret                  # per-pick period return (gross, for the timeline)
                c = (fee_eur + slippage_bps[t] / 1e4 * w) * mult \
                    if t in traded else 0.0
                trades.append(dict(pair=t, entry=d, exit=nxt, days=len(seg) - 1,
                                   gross=w * name_ret, costs=c,
                                   net=w * name_ret - c, capital=w))
            prev = set(picks)
        equity = pd.Series(dict(eq_points)).sort_index()
        runs[mult] = dict(equity=equity, trades=trades,
                       stats=backtest_stats(equity, trades, capital))
    return {"runs": runs, "holdings_log": holdings_log,
            "start": str(dates[0].date()) if dates else None}


def benchmark_curves(bench_prices: pd.DataFrame, window, capital: float) -> dict:
    """Buy-hold equity per benchmark over `window`, each normalized to `capital`
    at the window start. Missing benchmarks (all-NaN) are skipped."""
    out = {}
    for name in bench_prices.columns:
        s = bench_prices[name].reindex(window).ffill().dropna()
        if s.empty or float(s.iloc[0]) <= 0:
            continue
        out[name] = capital * s / float(s.iloc[0])
    return out


def equal_weight_curve(prices: pd.DataFrame, tickers: list[str], window,
                       capital: float) -> pd.Series:
    """Equity of an equal-weight buy-hold basket of `tickers` over `window`,
    normalized to `capital` at the start — the 'hold everything equally'
    honesty benchmark. Empty `tickers` → empty Series (matches momentum_scores)."""
    if not tickers:
        return pd.Series(dtype=float)
    sub = prices.reindex(window)[tickers].ffill().dropna(how="all")
    norm = sub / sub.iloc[0]                          # each name -> 1.0 at start
    port = norm.mean(axis=1)                          # equal weight
    return capital * port
