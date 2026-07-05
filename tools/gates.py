"""Pre-registered verdict gates — written BEFORE the first real-data run.

The point of this module is chronological honesty: the pass/fail thresholds below were
committed while every backtest in the repo still ran on synthetic data, so the real-data
verdicts are binding decisions, not post-hoc rationalisation. The reports render these
verdicts at the top of the page; the configuration they output IS the strategy.

Pure functions. Each check is a dict(name, value, threshold, passed) so the report can
show the measured number next to the bar it had to clear.
"""
import numpy as np

SLEEVE_P_MAX = 0.05        # MC selection test must beat this
SLEEVE_MIN_WINDOWS = 6     # fewer yearly observations than this is not evidence
SLEEVE_DEATH_FRAC = 0.20   # a window is "death-tainted" above this fraction of picks
SLEEVE_MAX_TAINTED = 1     # more tainted windows than this → survivorship-dominated


def _check(name: str, value, threshold: str, passed: bool) -> dict:
    return dict(name=name, value=str(value), threshold=threshold, passed=bool(passed))


def sleeve_verdict(mc: dict | None, years: list[dict]) -> dict:
    """KEEP the tax-loss sleeve only if the pre-registered evidence bar is met;
    otherwise CUT (sleeve weight 0 in the stack)."""
    checks = []
    p = mc.get("p_sharpe", float("nan")) if mc else float("nan")
    checks.append(_check("MC p-value vs random picks", f"{p:.3f}" if p == p else "—",
                         f"< {SLEEVE_P_MAX}", p == p and p < SLEEVE_P_MAX))
    n = len(years)
    checks.append(_check("Yearly windows", n, f">= {SLEEVE_MIN_WINDOWS}",
                         n >= SLEEVE_MIN_WINDOWS))
    nets = [y["net_ret"] for y in years if y.get("picks")]
    med = float(np.median(nets)) if nets else float("nan")
    checks.append(_check("Median window net return", f"{med:+.2%}" if med == med else "—",
                         "> 0", med == med and med > 0.0))
    tainted = sum(1 for y in years if y.get("picks")
                  and len(y.get("dead", set())) > SLEEVE_DEATH_FRAC * len(y["picks"]))
    checks.append(_check("Death-tainted windows (>20% of picks died)", tainted,
                         f"<= {SLEEVE_MAX_TAINTED}", tainted <= SLEEVE_MAX_TAINTED))
    verdict = "KEEP" if all(c["passed"] for c in checks) else "CUT"
    return dict(verdict=verdict, checks=checks)


def overlay_verdict(stats: dict, variants: dict, incumbent: str = "rolling") -> dict:
    """ADOPT the QLIKE-h21 winner only if it beats the incumbent trailing-vol overlay on
    forecast quality AND is no worse as a strategy (Sharpe and drawdown). Ties and
    failures keep the incumbent — deliberate status-quo bias: change requires evidence."""
    winner = min(stats, key=lambda m: stats[m]["qlike_h21"])
    if winner == incumbent:
        return dict(verdict="KEEP incumbent", method=incumbent, winner=winner,
                    checks=[_check("QLIKE winner", incumbent,
                                   "a challenger must out-forecast the incumbent", False)])
    checks = [
        _check(f"QLIKE h21: {winner} vs {incumbent}",
               f"{stats[winner]['qlike_h21']:.4f} vs {stats[incumbent]['qlike_h21']:.4f}",
               "winner lower",
               stats[winner]["qlike_h21"] < stats[incumbent]["qlike_h21"]),
        _check("Variant Sharpe no worse",
               f"{variants[winner].get('sharpe', 0):.2f} vs "
               f"{variants[incumbent].get('sharpe', 0):.2f}", ">=",
               variants[winner].get("sharpe", -9) >= variants[incumbent].get("sharpe", -9)),
        _check("Variant max drawdown no worse",
               f"{variants[winner].get('max_dd', 0):.1%} vs "
               f"{variants[incumbent].get('max_dd', 0):.1%}", ">=",
               variants[winner].get("max_dd", -9) >= variants[incumbent].get("max_dd", -9)),
    ]
    adopt = all(c["passed"] for c in checks)
    return dict(verdict=f"ADOPT {winner}" if adopt else "KEEP incumbent",
                method=winner if adopt else incumbent, winner=winner, checks=checks)


def stack_verdict(sleeve_v: dict, overlay_method: str | None = None,
                  w_sleeve: float = 0.20, target_vol: float = 0.15) -> dict:
    """Compose the final runnable configuration from the gate outcomes."""
    w = w_sleeve if sleeve_v["verdict"] == "KEEP" else 0.0
    method = overlay_method or "per /vol verdict"
    statement = (f"Run: momentum core {1 - w:.0%}"
                 + (f" + tax-loss sleeve {w:.0%}" if w > 0 else " (sleeve CUT)")
                 + f", vol-managed at {target_vol:.0%} target using the {method} forecast.")
    return dict(w_sleeve=w, method=method, statement=statement)
