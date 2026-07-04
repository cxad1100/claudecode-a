"""Econophysics research page — scaffold invariants.

The page is a research lab: every module section must tolerate an absent
result (None / missing key → renders ""), the page must never write into
docs/ (it is local-only, unlike the other reports), and module env flags
must actually disable their gather blocks.
"""
import inspect

import build_econo_report as er


def test_build_renders_with_all_modules_none():
    html = er.build({})                       # no module produced anything
    assert html.startswith("<!DOCTYPE html>")
    assert "Econophysics" in html
    assert "plotly" in html                   # standard report chrome present


def test_module_sections_render_empty_on_none():
    for sec in (er.sec_leadlag, er.sec_corr, er.sec_phase, er.sec_events,
                er.sec_trials):
        assert sec({}, False) == ""
        assert sec({"leadlag": None, "corr": None, "phase": None,
                    "events": None, "trials": None}, False) == ""


def test_page_is_local_only():
    src = inspect.getsource(er)
    assert "docs/" not in src and 'docs"' not in src


def test_env_flag_disables_module(monkeypatch):
    monkeypatch.delenv("ECONO_LEADLAG", raising=False)
    assert er._enabled("leadlag") is True     # default on
    monkeypatch.setenv("ECONO_LEADLAG", "0")
    assert er._enabled("leadlag") is False
    monkeypatch.setenv("ECONO_LEADLAG", "1")
    assert er._enabled("leadlag") is True
