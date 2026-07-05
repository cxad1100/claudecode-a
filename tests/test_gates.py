"""Gate-boundary tests — verdicts must flip exactly at the pre-registered thresholds."""
import numpy as np

from tools import gates as G


def _years(n=8, net=0.03, dead_frac=0.0, k=10):
    out = []
    for i in range(n):
        picks = [f"T{j}" for j in range(k)]
        dead = set(picks[:int(round(dead_frac * k))])
        out.append(dict(year=2016 + i, picks=picks, net_ret=net, dead=dead, ret={}))
    return out


def test_sleeve_keep_when_all_pass():
    v = G.sleeve_verdict(dict(p_sharpe=0.049), _years())
    assert v["verdict"] == "KEEP" and all(c["passed"] for c in v["checks"])


def test_sleeve_p_boundary():
    assert G.sleeve_verdict(dict(p_sharpe=0.049), _years())["verdict"] == "KEEP"
    assert G.sleeve_verdict(dict(p_sharpe=0.051), _years())["verdict"] == "CUT"
    assert G.sleeve_verdict(None, _years())["verdict"] == "CUT"        # no MC → CUT


def test_sleeve_window_count_boundary():
    assert G.sleeve_verdict(dict(p_sharpe=0.01), _years(n=6))["verdict"] == "KEEP"
    assert G.sleeve_verdict(dict(p_sharpe=0.01), _years(n=5))["verdict"] == "CUT"


def test_sleeve_median_boundary():
    assert G.sleeve_verdict(dict(p_sharpe=0.01), _years(net=0.0))["verdict"] == "CUT"
    assert G.sleeve_verdict(dict(p_sharpe=0.01), _years(net=1e-6))["verdict"] == "KEEP"


def test_sleeve_survivorship_flag():
    ok = _years(n=8, dead_frac=0.0)
    ok[0]["dead"] = set(ok[0]["picks"][:3])                # one tainted window → fine
    assert G.sleeve_verdict(dict(p_sharpe=0.01), ok)["verdict"] == "KEEP"
    bad = _years(n=8, dead_frac=0.3)                       # every window tainted
    v = G.sleeve_verdict(dict(p_sharpe=0.01), bad)
    assert v["verdict"] == "CUT"
    assert any("Death-tainted" in c["name"] and not c["passed"] for c in v["checks"])


def _stats(q):
    return {m: dict(qlike_h21=q[m]) for m in q}


def _variants(s, dd):
    return {m: dict(sharpe=s[m], max_dd=dd[m]) for m in s}


def test_overlay_adopts_strictly_better_challenger():
    v = G.overlay_verdict(_stats(dict(rolling=-1.98, garch=-2.04)),
                          _variants(dict(rolling=0.60, garch=0.70),
                                    dict(rolling=-0.30, garch=-0.25)))
    assert v["verdict"] == "ADOPT garch" and v["method"] == "garch"


def test_overlay_keeps_incumbent_when_challenger_worse_as_strategy():
    v = G.overlay_verdict(_stats(dict(rolling=-1.98, garch=-2.04)),
                          _variants(dict(rolling=0.70, garch=0.60),      # worse Sharpe
                                    dict(rolling=-0.30, garch=-0.25)))
    assert v["verdict"] == "KEEP incumbent" and v["method"] == "rolling"
    v2 = G.overlay_verdict(_stats(dict(rolling=-1.98, garch=-2.04)),
                           _variants(dict(rolling=0.60, garch=0.70),
                                     dict(rolling=-0.25, garch=-0.30)))  # worse DD
    assert v2["method"] == "rolling"


def test_overlay_winner_is_incumbent():
    v = G.overlay_verdict(_stats(dict(rolling=-2.10, garch=-2.04)),
                          _variants(dict(rolling=0.6, garch=0.6),
                                    dict(rolling=-0.3, garch=-0.3)))
    assert v["verdict"] == "KEEP incumbent" and v["method"] == "rolling"


def test_stack_composition():
    keep = G.stack_verdict(dict(verdict="KEEP"), overlay_method="garch")
    cut = G.stack_verdict(dict(verdict="CUT"), overlay_method="garch")
    assert keep["w_sleeve"] == 0.20 and "sleeve 20%" in keep["statement"]
    assert cut["w_sleeve"] == 0.0 and "sleeve CUT" in cut["statement"]
    assert "garch" in keep["statement"]
