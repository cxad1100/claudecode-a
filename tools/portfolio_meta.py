"""Shared portfolio metadata: sector map, ETF decomposition, colors.

Imported by both the Portfolio page (app.py) and the Optimizer page so neither
page has to import the other.
"""

# ── ETF sector decomposition ─────────────────────────────────────────────────
_MSCI_WORLD_WEIGHTS = {   # iShares MSCI World — approximate weights (rebalanced quarterly)
    "Information Technology": 0.240,
    "Financials":             0.155,
    "Healthcare":             0.120,
    "Industrials":            0.110,
    "Consumer Discretionary": 0.105,
    "Communication Services": 0.080,
    "Consumer Staples":       0.060,
    "Energy":                 0.050,
    "Materials":              0.040,
    "Real Estate":            0.025,
    "Utilities":              0.015,
}
ETF_SECTOR_WEIGHTS = {
    "IWDA.AS": _MSCI_WORLD_WEIGHTS,
    "EUNL.F":  _MSCI_WORLD_WEIGHTS,   # EUNL.F = IWDA Frankfurt listing, same ISIN
}
PORTFOLIO_ETFS = set(ETF_SECTOR_WEIGHTS)

# ── Sector overrides for portfolio stocks (no sector feed) ────────────────────
PORTFOLIO_SECTOR_MAP: dict[str, str] = {
    "NVD.F":  "Information Technology",
    "AMZ.F":  "Consumer Discretionary",
    "ABEA.F": "Communication Services",
    "TSFA.F": "Information Technology",
    "LHL.F":  "Information Technology",
    "TCO0.F": "Consumer Staples",
    "ASME.F": "Information Technology",
    "IES.F":  "Financials",
    "CRIN.F": "Financials",
    "IPJ1.F": "Industrials",
    "APC.F":  "Information Technology",
    "MSF.F":  "Information Technology",
}

# VSCode Dark+ token palette — coherent with tools/theme.py
SECTOR_COLORS = {
    "Information Technology": "#569cd6",
    "Communication Services": "#9cdcfe",
    "Consumer Discretionary": "#ce9178",
    "Financials":             "#4ec9b0",
    "Industrials":            "#808080",
    "Energy":                 "#d16969",
    "Healthcare":             "#c586c0",
    "Consumer Staples":       "#b5cea8",
    "Real Estate":            "#d7ba7d",
    "Materials":              "#6a9955",
    "Utilities":              "#dcdcaa",
    "Unknown":                "#5a5a5a",
}

ALL_SECTORS = sorted(set(PORTFOLIO_SECTOR_MAP.values()) | set(_MSCI_WORLD_WEIGHTS))


def sector_exposure_matrix(tickers: list[str]) -> tuple[list[str], list[list[float]]]:
    """
    Build a (sector × asset) exposure matrix S where S[i][j] = fraction of asset j
    attributed to sector i. Stocks load 1.0 onto their PORTFOLIO_SECTOR_MAP sector;
    ETFs are decomposed via ETF_SECTOR_WEIGHTS (normalized to sum 1).

    Returns (sectors, matrix). Only sectors with non-zero exposure are included.
    """
    exposure: dict[str, dict[str, float]] = {}  # sector -> {ticker: frac}
    for tk in tickers:
        if tk in ETF_SECTOR_WEIGHTS:
            w = ETF_SECTOR_WEIGHTS[tk]
            tot = sum(w.values()) or 1.0
            for sec, sw in w.items():
                exposure.setdefault(sec, {})[tk] = sw / tot
        else:
            sec = PORTFOLIO_SECTOR_MAP.get(tk, "Unknown")
            exposure.setdefault(sec, {})[tk] = 1.0

    sectors = sorted(exposure)
    matrix = [[exposure[sec].get(tk, 0.0) for tk in tickers] for sec in sectors]
    return sectors, matrix
