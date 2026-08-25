# Data sources

The five processed inputs in `data/` are the market-data bundle used by this
release. The SPX 5×5 implied-volatility surface comes from the data release
accompanying [Chen et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4628457).
The surface was constructed from OptionMetrics IvyDB US data accessed through
WRDS. The [upstream release](https://github.com/rotmanfinhub/vol-surface-vae-pub)
includes the surface data and its processing code.

The remaining market panel combines the following FRED and Yahoo Finance series:

| Source | Series |
| --- | --- |
| Yahoo Finance | S&P 500 (`^GSPC`), Nikkei 225 (`^N225`), U.S. Dollar Index (`DX-Y.NYB`), copper (`HG=F`), wheat (`ZW=F`), and VIX (`^VIX`) |
| FRED | USD/CAD (`DEXCAUS`), USD/JPY (`DEXJPUS`), gold (`GOLDAMGBD228NLBM`), WTI crude oil (`DCOILWTICO`), 2-year and 10-year Treasury yields (`DGS2`, `DGS10`), and AAA and BBB corporate spreads (`BAMLC0A1CAAA`, `BAMLC0A4CBBB`) |

| File | Contents |
| --- | --- |
| `multi_factor_data.npz` | Factor dates, levels, returns, and column names. |
| `multi_factor_levels.parquet` | Processed factor-level panel. |
| `multi_factor_returns.parquet` | Processed factor-return panel. |
| `spx_vol_surface_history_full_data_fixed.parquet` | Processed SPX implied-volatility surface history. |
| `vol_surface_with_ret.npz` | Surface grid and local market series. |

`data/CHECKSUMS.tsv` records the frozen byte size and SHA-256 digest for every
file. The tests verify the data schemas, date coverage, and the 39-variable
date-aligned panel contract.
