# Third-party models and data

The MIT License in this repository covers the ScenarioFlow code. The items
below are derived from other parties' work and keep their original terms.

| Item | Where | Derived from | Terms |
| --- | --- | --- | --- |
| Fine-tuned Moirai checkpoint, manifest, and sample arrays (`foundation.moirai-1-1.*`) | Release files listed in `artifacts/manifest-v1.tsv` | [Salesforce/moirai-1.1-R-small](https://huggingface.co/Salesforce/moirai-1.1-R-small) | CC BY-NC 4.0: non-commercial use only |
| Fine-tuned Toto checkpoint, manifest, and sample arrays (`foundation.toto-1-0.*`) | Release files listed in `artifacts/manifest-v1.tsv` | [Datadog/Toto-Open-Base-1.0](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | Apache License 2.0 |
| Market data in `data/` | This repository | Chen et al. data release (OptionMetrics IvyDB US through WRDS), Yahoo Finance, FRED | Providers' terms; see [DATA_SOURCES.md](DATA_SOURCES.md) |

The two fine-tuned checkpoints are comparison baselines. ScenarioFlow itself
does not use them.

The paper, poster, and explainer video under `web/public/` are copyright their
authors. The University of Toronto, Ontario Teachers' Pension Plan, and NeurIPS
logos drawn by `web/public/story/logos_white.js` are trademarks of their
respective owners. None of these files is covered by the MIT License.
