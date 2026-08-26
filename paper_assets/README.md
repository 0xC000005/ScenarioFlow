# Paper assets

This directory contains the released tables and figures for the ICAIF and
Foundation Models for Temporal Systems papers. `ASSETS.tsv` records each file's
venue, size, SHA-256, and source.

## Verify the released assets

```bash
uv sync --extra dev --extra paper
uv run pytest -q tests/test_frozen_evidence.py tests/test_paper_release_boundary.py
uv run python scripts/check_release.py --root . --all
```

## Frozen paper-facing views

The workshop and ICAIF tables under `tables/golden/`, together with the figures
in `figures/`, are the released paper-facing assets. They are preserved as
evidence for the published results; the historical experiment tree and its
paper-rendering launchers are not shipped in this source archive.

The dollar-liquidity tables are frozen ScenarioFlow-NL outputs. They are not part
of the numerical no-OpenAI inference route.

## Illustrations

Files under `figures/illustrations/` are the final conceptual illustrations used
by the papers. They are preserved, not recreated by the numerical pipeline. The
LaTeX workspaces are not included in this repository.
