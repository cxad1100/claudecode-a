# data/ — what's here and who uses it

Two independent strategy stacks keep their data side by side.

## Momentum / Strategy → `data/universe/`
The Trade-Republic-tradeable universe, the single source of truth for the
Momentum + Strategy pages. Built by
`tools.tr_tradeable --enumerate` → `tools.build_tr_universe`.

| file | tracked | what |
|------|---------|------|
| `universe_meta.csv`   | ✅ | one row per name: ticker, name, country, isin, currency, slippage_bps, `delisting_date` (blank = live), `exit_reason` (`delisted`/`removed`/`demoted`, blank = live — the membership ledger, `tools.universe_reconcile`; only `delisted` counts as a graveyard death), `med_turnover`. |
| `snapshots.csv`       | ✅ | append-only PIT snapshots of TR's enumerated list (`tools.universe_snapshot`). Gates backtest eligibility from the first snapshot (2026-06-29) forward via `PITUniverse(membership=…)`. |
| `universe_prices.csv` | ❌ gitignored (large) | EUR-converted daily close per ticker, 2017→. Regenerate with `build_tr_universe fetch`. |
| `universe_turnover.csv` | ❌ gitignored (regenerable) | month-end median daily EUR turnover per name, written by `fetch` — the point-in-time liquidity gate's source. Absent → the engine falls back to the static build-time gate. |
| `tr_universe.csv`     | ❌ gitignored (TR-account-derived) | raw TR enumeration: isin, name, country. |
| `tr_ticker_map.json`  | ❌ gitignored | isin → yfinance ticker cache (resolution). |

## Pairs Lab → top-level `data/`
The broker-scraped pairs universe (Lang & Schwarz), used by `tools.pairs_universe`
and built/verified by `tools.build_universe` / `tools.verify_universe`.

| file | what |
|------|------|
| `universe_meta.csv`        | pairs universe: ticker, local_id (WKN), name, country, sector, currency, slippage_bps. |
| `universe.csv` / `universe_raw.csv` | canonical + raw broker lists feeding the pairs build. |
| `universe_price_flags.csv` | price-sanity flags from `verify_universe`. |
| `pairs_prices.csv`         | cached daily closes for the pairs universe. |
| `wkn_ticker_map.json`      | WKN → yfinance ticker cache (`tools.wkn_resolve`). |
| `eur_delisted_seed.csv`    | hand-seeded dead EUR names (`tools.dead_stocks`). |

Gitignored, never committed: anything large (`*_prices.csv`) or account-derived
(`tr_universe.csv`, `tr_ticker_map.json`). See `.gitignore`.
