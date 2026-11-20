# Target-disjoint rerun (causal-v1)

These files back the target-disjoint rerun reported in Appendix C.3 of the
camera-ready paper. Every method was retrained once and scored on the same 441
validation windows as the main benchmark, under a stricter data-handling
contract:

- **No target overlap.** Training windows start at indices 168–3980 and
  validation windows at 4010–4450. The 29 training windows whose 30-day targets
  overlap the validation targets are dropped. The last training target is
  2016-01-27 and the first validation target is 2016-01-28.
- **Forward-only filling.** Missing values are filled forward only. Leading
  missing values are not back-filled; the first usable window is index 168.
- **Fixed variable maps.** Each variable's state map is fixed by its variable
  class before any data is read.

The screens and thresholds are the same as in the main benchmark. The
evaluation uses one training run and one sampling seed per method (64 scenarios
per window), so it does not measure seed variability. It reuses the validation
block that was used for model selection and is not an independent test.

The table as printed in the paper is
[`paper_assets/tables/golden/fmts-causal-v1.tex`](../../paper_assets/tables/golden/fmts-causal-v1.tex).

| File | Contents |
| --- | --- |
| `aggregate.json` | Metrics, screen decisions, and paired comparisons for all 12 methods. |
| `scenarioflow-results.json` | Full ScenarioFlow evaluation, including the data-contract metadata. |
| `verification.json` | Independent recomputation of the saved per-window scores and screen decisions for the 11 completed evaluations. |
| `summary.md` | Completion status per method. |
| `CHECKSUMS.tsv` | Byte size and SHA-256 of each file above. |

Eleven of the twelve evaluations completed. The fine-tuned Toto evaluation
stopped before scoring because one generated VIX path overflowed when it was
converted back to market units; no path was clipped or removed, and its missing
scores are not zeros.

These are frozen results. The training and evaluation code in this repository
implements the main benchmark's data handling; the causal-v1 split and its
checkpoints are not part of this release.
