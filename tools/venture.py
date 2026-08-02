"""Venture instrumentation — the charter's north-star measurement layer.

Money-weighted (XIRR) comparison of the live book against a SAME-CASHFLOW
IWDA shadow: every deposit buys the benchmark on the same date, so the
comparison is cashflow-fair (index TWR vs a contribution-driven IRR is
mechanically biased — the red-team finding this module answers).

Also the pre-registered book drawdown ladder, operational and exempt from
year-boundary review timing:
    normal            dd > −25%
    half-vol          −35% < dd ≤ −25%   → halve the vol target
    derisk            dd ≤ −35%          → de-risk to IWDA/cash + review

Pure functions; the report supplies data and owns the cashflow ledger file.
"""
import numpy as np
import pandas as pd

__all__ = ["xirr", "shadow_curve", "dd_state", "venture_summary"]

_LADDER = ((-0.35, "derisk"), (-0.25, "half-vol"))


def xirr(cashflows: list, terminal_value: float, terminal_date) -> float:
    """Annualized money-weighted return. `cashflows` = [(date, amount)] with
    deposits NEGATIVE (money out of pocket); terminal_value closes the
    position. Bisection on NPV — robust, no scipy dependency needed."""
    t1 = pd.Timestamp(terminal_date)
    flows = [(pd.Timestamp(d), float(a)) for d, a in cashflows]
    flows.append((t1, float(terminal_value)))
    t0 = min(d for d, _ in flows)
    years = np.array([(d - t0).days / 365.25 for d, _ in flows])
    amts = np.array([a for _, a in flows])

    def npv(rate):
        return float(np.sum(amts / (1.0 + rate) ** years))

    lo, hi = -0.9999, 100.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def shadow_curve(cashflows: pd.DataFrame, bench: pd.Series) -> pd.Series:
    """Equity of investing each deposit (`date`, `amount` > 0) into the
    benchmark at that date's price — deposits landing on non-trading days
    execute at the next session. The same-cashflow IWDA shadow."""
    px = bench.dropna()
    units = pd.Series(0.0, index=px.index)
    for _, row in cashflows.iterrows():
        pos = int(px.index.searchsorted(pd.Timestamp(row["date"])))
        if pos >= len(px):
            continue
        units.iloc[pos:] += float(row["amount"]) / float(px.iloc[pos])
    eq = (units * px)
    return eq[eq > 0]


def dd_state(book_equity: pd.Series) -> dict:
    """Trailing-peak drawdown mapped to the pre-registered ladder."""
    eq = book_equity.dropna()
    if len(eq) < 2:
        return dict(state="normal", dd=0.0)
    dd = float((eq / eq.cummax() - 1.0).iloc[-1])
    for bar, name in _LADDER:
        if dd <= bar:
            return dict(state=name, dd=dd)
    return dict(state="normal", dd=dd)


def venture_summary(cashflows: pd.DataFrame, book_equity: pd.Series,
                    bench: pd.Series) -> dict:
    """The `sec_venture` payload: book XIRR vs same-cashflow shadow XIRR,
    excess, months elapsed, drawdown-ladder state."""
    book = book_equity.dropna()
    shadow = shadow_curve(cashflows, bench)
    flows = [(row["date"], -float(row["amount"]))
             for _, row in cashflows.iterrows()]
    end = book.index[-1]
    book_r = xirr(flows, float(book.iloc[-1]), end)
    sh_end = shadow.index[shadow.index.searchsorted(end)
                          - (0 if end in shadow.index else 1)] \
        if len(shadow) else end
    shadow_r = xirr(flows, float(shadow.loc[:end].iloc[-1]), sh_end) \
        if len(shadow) else float("nan")
    months = (book.index[-1] - book.index[0]).days / 30.44
    s = dd_state(book)
    return dict(book_xirr=float(book_r), shadow_xirr=float(shadow_r),
                excess=float(book_r - shadow_r), months=float(months),
                dd=s["dd"], dd_state=s["state"])
