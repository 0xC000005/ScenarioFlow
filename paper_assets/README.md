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

## Regenerate the numerical results

The full benchmark uses the archived command schedule and preserved original
experiment sources:

```bash
./scripts/reproduce.sh
./scripts/reproduce_paper.sh tables
```

The first command trains and evaluates the released candidates across the full
seed grid. The second calls the original aggregator and table generators. It
writes its outputs under `build/paper/` and leaves the released files unchanged.
Fresh CUDA training can change learned-model values slightly. The archived
tables retain the values used in the papers; a fresh table reports the new full
rerun.

The workshop and ICAIF tables under `tables/golden/` are the frozen
paper-facing views. The original multi-seed generator is preserved at
`reproduction/source/paper/paper/scripts/generate_fmts_multiseed_appendix.py`;
the original ICAIF table generator is beside it as `generate_paper_tables.py`.
The ICAIF generator produces the upstream comparison tables. The final ICAIF
table is preserved in its camera-ready form.

## Regenerate the numerical plots

Fetch the Stage 2 checkpoint, then run the original plotting scripts:

```bash
uv run scenarioflow fetch --artifact core.stage2 \
  --bundle-dir /path/to/release-bundle
./scripts/reproduce_paper.sh plots
```

The original plotting files are preserved under
`reproduction/source/scenarioflow/experiments/backfill/block_ar/`. Generated
plots are written under `build/paper/`.

The dollar-liquidity tables are frozen ScenarioFlow-NL outputs. They are not part
of the numerical no-OpenAI reproduction route.

## Illustrations

Files under `figures/illustrations/` are the final conceptual illustrations used
by the papers. They are preserved, not recreated by the numerical pipeline. The
LaTeX workspaces are not included in this repository.
