# Data sources

The five processed inputs in `data/` are the frozen market-data bundle used by
this release. Their provenance is Rotman FinHub and WRDS, including
OptionMetrics options data.

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
