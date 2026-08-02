"""Factor-spanning machinery: French CSV parsing, USD conversion, HAC regression."""
import numpy as np
import pandas as pd
import pytest

from tools.factors import (parse_french_csv, to_usd, factor_regression,
                           fetch_factors_daily, MODELS)

FRENCH_SAMPLE = """This file was created by CMPT_ME_BEME_RETS using the 202506 CRSP database.
Missing data are indicated by -99.99 or -999.

,Mkt-RF,SMB,HML,RMW,CMA,RF
20240102,  1.00, 0.50,-0.25, 0.10, 0.05, 0.02
20240103, -0.50, 0.20, 0.10,-99.99, 0.00, 0.02
20240104,  0.30,-0.10, 0.05, 0.15,-0.05, 0.02

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
2024, 12.00, 3.00, 1.00, 2.00, 0.50, 4.80
"""


def test_parse_french_csv_daily_rows_only_decimals_and_sentinels():
    df = parse_french_csv(FRENCH_SAMPLE)
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
                              pd.Timestamp("2024-01-04")]          # annual block ignored
    assert df.loc["2024-01-02", "MKT_RF"] == pytest.approx(0.01)   # percent → decimal
    assert np.isnan(df.loc["2024-01-03", "RMW"])                   # -99.99 → NaN
    assert set(df.columns) == {"MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"}


def test_to_usd_flat_fx_is_identity_and_appreciation_compounds():
    idx = pd.bdate_range("2024-01-02", periods=4)
    r = pd.Series([0.01, -0.02, 0.005, 0.0], index=idx)
    flat = pd.Series(1.10, index=idx)
    assert np.allclose(to_usd(r, flat).values, r.values)
    fx = pd.Series([1.00, 1.01, 1.01, 1.01], index=idx)            # +1% EUR appreciation day 2
    out = to_usd(r, fx)
    assert out.iloc[1] == pytest.approx((1 - 0.02) * 1.01 - 1)


def test_factor_regression_recovers_alpha_and_betas():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=1500)
    f = pd.DataFrame(rng.normal(0, 0.01, (1500, 6)),
                     columns=["MKT_RF", "SMB", "HML", "RMW", "CMA", "WML"], index=idx)
    alpha, b_mkt, b_wml = 0.0004, 1.1, 0.5
    r = alpha + b_mkt * f["MKT_RF"] + b_wml * f["WML"] \
        + pd.Series(rng.normal(0, 0.002, 1500), index=idx)
    reg = factor_regression(r, f)
    m = reg["FF5+WML"]
    assert m["alpha_ann"] == pytest.approx(alpha * 252, rel=0.35)  # noisy but near
    assert m["betas"]["MKT_RF"][0] == pytest.approx(b_mkt, abs=0.05)
    assert m["betas"]["WML"][0] == pytest.approx(b_wml, abs=0.05)
    assert m["alpha_t"] > 2.0 and m["r2"] > 0.8 and m["n"] == 1500
    assert set(reg) == set(MODELS)                                  # all three models ran
    assert reg["CAPM"]["betas"].keys() == {"MKT_RF"}


def test_factor_regression_skips_model_with_missing_column():
    idx = pd.bdate_range("2020-01-01", periods=300)
    f = pd.DataFrame({"MKT_RF": np.random.default_rng(1).normal(0, 0.01, 300)}, index=idx)
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 300), index=idx)
    reg = factor_regression(r, f)
    assert "CAPM" in reg and "FF5" not in reg and "FF5+WML" not in reg


def test_fetch_factors_daily_uses_cache_via_injected_fetcher(tmp_path):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return FRENCH_SAMPLE

    df1, src1 = fetch_factors_daily(cache_dir=tmp_path, _get_text=fake_fetch)
    n_first = len(calls)
    df2, src2 = fetch_factors_daily(cache_dir=tmp_path, _get_text=fake_fetch)
    assert n_first >= 1 and len(calls) == n_first          # second call served from cache
    assert (df1.index == df2.index).all() and src1 == src2
