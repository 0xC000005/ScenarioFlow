# FMTS multi-seed benchmark aggregate

Threshold passing is evaluated for each run before any seed averaging.
Training, rollout, and chronological-window variation are reported separately.

## Pass-all profile

| Candidate | Runs passing | Pass rate | Training seeds passing every rollout |
|---|---:|---:|---:|
| Joint-increment bootstrap | 0/5 | 0.000 | n/a |
| DeepVAR | 0/15 | 0.000 | 0 |
| DeepVAR + ScenarioFlow support | 15/15 | 1.000 | 3 |
| Filtered historical simulation | 0/5 | 0.000 | n/a |
| GARCH-CCC | 0/5 | 0.000 | n/a |
| State-matched historical simulation | 0/5 | 0.000 | n/a |
| CSDI | 0/15 | 0.000 | 0 |
| PCA-VAR | 0/5 | 0.000 | n/a |
| Random walk | 0/5 | 0.000 | n/a |
| ScenarioFlow | 10/15 | 0.667 | 2 |
| ScenarioFlow + graph | 5/15 | 0.333 | 1 |
| TimeGrad | 0/15 | 0.000 | 0 |

## Paired controls

### deepvar_vs_deepvar_support

Deltas are right minus left; negative proper-score deltas favor the right candidate.

| Proper score | Mean run delta | 95% moving-block interval |
|---|---:|---:|
| CRPS | -0.0352506 | [-0.17031, 0.0906085] |
| Energy Score | 0.17619 | [0.10568, 0.283402] |
| Variogram | 0.00974147 | [-0.00380157, 0.0238212] |

### scenarioflow_vs_graph

Deltas are right minus left; negative proper-score deltas favor the right candidate.

| Proper score | Mean run delta | 95% moving-block interval |
|---|---:|---:|
| CRPS | 0.0294898 | [0.0196013, 0.0394941] |
| Energy Score | 0.0259219 | [0.0146214, 0.0429257] |
| Variogram | 0.00404246 | [0.00193636, 0.00721997] |
