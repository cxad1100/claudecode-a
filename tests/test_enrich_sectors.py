"""Sanity checks for the pure merge in the offline sector enricher (tools.enrich_sectors)."""
import pandas as pd

from tools.enrich_sectors import merge_sectors


def test_merge_sectors_fills_only_fetched():
    meta = pd.DataFrame({"ticker": ["A", "B", "C"],
                         "sector": ["Unknown", "Unknown", "Unknown"]})
    cache = {"A": "Technology", "B": None}              # B tried→no sector; C never fetched
    out = merge_sectors(meta, cache)
    assert list(out["sector"]) == ["Technology", "Unknown", "Unknown"]


def test_merge_sectors_keeps_existing_when_no_fetch():
    meta = pd.DataFrame({"ticker": ["A"], "sector": ["Energy"]})
    out = merge_sectors(meta, {})                       # empty cache → nothing overwritten
    assert out["sector"].iloc[0] == "Energy"
