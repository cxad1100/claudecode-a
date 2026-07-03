"""Factor-spanning test data + regression (Ken French daily factors).

Answers the sharpest version of "is there alpha?": regress the strategy's daily excess
returns (USD) on the French Developed 5 factors + WML momentum (US 2x3 fallback when the
Developed files are unavailable). If alpha dies once WML enters, the edge is momentum
factor BETA — still worth holding at retail scale, but not proprietary; if a positive
alpha survives with a real t-stat, there is residual selection edge beyond the factors.

Pure pieces (parser, USD conversion, HAC OLS) are unit-tested; the fetch is a thin
cached download (7-day TTL pickle in local/buffer) that the build wraps in try/except —
a French-site outage must never break the report.

Conventions: French CSVs quote percent → /100 here; sentinels -99.99/-999 → NaN; the
momentum column ("Mom") is normalized to WML; everything is USD, so the caller converts
EUR strategy returns via to_usd() and subtracts the French RF before regressing.
"""
import io
import pathlib
import pickle
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "local" / "buffer"
CACHE_TTL_S = 7 * 24 * 3600            # factor history barely moves — weekly is plenty

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_URLS = {                              # source label -> (5-factor zip, momentum zip)
    "Developed": (_BASE + "Developed_5_Factors_Daily_CSV.zip",
                  _BASE + "Developed_Mom_Factor_Daily_CSV.zip"),
    "US": (_BASE + "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
           _BASE + "F-F_Momentum_Factor_daily_CSV.zip"),
}

MODELS = {
    "CAPM": ["MKT_RF"],
    "FF5": ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
    "FF5+WML": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "WML"],
}


def _norm_col(c: str) -> str:
    c = str(c).strip().upper().replace("-", "_").replace(" ", "")
    return "WML" if c in ("MOM", "UMD", "WML") else c


def parse_french_csv(text: str) -> pd.DataFrame:
    """French library CSV → daily decimal frame. Keeps only rows whose first field is an
    8-digit date (the daily block); header junk and the trailing annual table drop out.
    Percent → decimal; -99.99/-999 sentinels → NaN."""
    header, rows, dates = None, [], []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        if parts[0] == "" and header is None and any(parts[1:]):
            header = [_norm_col(c) for c in parts[1:]]
            continue
        if len(parts[0]) == 8 and parts[0].isdigit():
            dates.append(pd.Timestamp(parts[0]))
            rows.append([float(v) if v else np.nan for v in parts[1:]])
    df = pd.DataFrame(rows, index=dates, columns=header[:len(rows[0])] if rows else header)
    df = df.mask(df <= -99.0) / 100.0
    return df


def _get_zip_csv_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return z.read(name).decode("utf-8", errors="replace")


def fetch_factors_daily(force: bool = False, cache_dir=CACHE_DIR,
                        _get_text=_get_zip_csv_text) -> tuple:
    """(factors_df, source_label). Developed 5F+WML preferred, US fallback; inner-joined
    on date. Cached (pickle, 7d TTL); `_get_text` is injectable for tests."""
    cache_dir = pathlib.Path(cache_dir)
    path = cache_dir / "french_factors.pkl"
    if not force and path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL_S:
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            pass
    last_err = None
    for source, (u5, umom) in _URLS.items():
        try:
            f5 = parse_french_csv(_get_text(u5))
            mom = parse_french_csv(_get_text(umom))
            wml = [c for c in mom.columns if c == "WML"]
            df = f5.join(mom[wml], how="inner") if wml else f5
            out = (df.sort_index(), source)
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps(out))
            except Exception:
                pass
            return out
        except Exception as e:                       # site hiccup / format drift → next source
            last_err = e
    raise RuntimeError(f"French factor download failed: {last_err}")


def to_usd(ret_eur: pd.Series, eurusd: pd.Series) -> pd.Series:
    """EUR daily returns → USD daily returns via the EURUSD (USD per EUR) level series:
    r_usd = (1+r_eur)·(fx_t/fx_{t-1}) − 1. FX is ffilled onto the return dates; the
    first bar's unknown FX move is treated as 0 (ratio 1)."""
    fx = eurusd.reindex(ret_eur.index).ffill()
    ratio = (fx / fx.shift(1)).fillna(1.0)
    return (1.0 + ret_eur) * ratio - 1.0


def factor_regression(ret_excess: pd.Series, factors: pd.DataFrame,
                      models: dict | None = None, hac_lags: int = 5) -> dict:
    """OLS of daily excess returns on each factor model, Newey-West (HAC) t-stats.

    Returns {model: {alpha_ann, alpha_t, betas: {col: (coef, t)}, r2, n}}. A model whose
    columns aren't all present is skipped (e.g. WML missing from a degraded fetch).
    Caller supplies EXCESS returns (already RF-subtracted) on the same currency basis
    as the factors."""
    out = {}
    for name, cols in (models or MODELS).items():
        if any(c not in factors.columns for c in cols):
            continue
        df = pd.concat([ret_excess.rename("y"), factors[cols]], axis=1, join="inner").dropna()
        if len(df) < 60:                             # too short to say anything
            continue
        X = sm.add_constant(df[cols])
        fit = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        out[name] = dict(
            alpha_ann=float(fit.params["const"]) * 252.0,
            alpha_t=float(fit.tvalues["const"]),
            betas={c: (float(fit.params[c]), float(fit.tvalues[c])) for c in cols},
            r2=float(fit.rsquared),
            n=int(fit.nobs))
    return out
