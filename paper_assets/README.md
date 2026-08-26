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

`ASSETS.tsv` classifies every file as `generated`, `verified-from-evidence`, or
`preserved-only`. `generated` means byte-identical reproduction.
`verified-from-evidence` means that the released arrays reproduce the same
figure content, while PNG encoding, metadata, and rasterization can differ
across Matplotlib versions. `preserved-only` means no exact surviving generator
is claimed.

## Regenerate the numerical assets

Fetch the `evidence` artifact profile, then choose a caller-owned output
directory:

```bash
./scripts/setup.sh --profile evidence
./scripts/reproduce_paper.sh all \
  --artifact-root artifacts/cache \
  --output-dir build/paper-reproduction
```

Use `tables` or `plots` in place of `all` to render only one group. These
commands consume the checksum-pinned released JSON/NPZ evidence and require
neither an OpenAI API key nor Codex. They never overwrite `paper_assets/` or
`results/frozen/`. `--output-dir` is required: the command never chooses a
generated destination for the caller.

The byte-identical generated outputs are:

- the FMTS multi-seed table from `results/frozen/multiseed/summary.json`;
- the three dollar-liquidity table views from the frozen reports and released
  narrative/start-only arrays;
- the joint-factor fan-selection JSON.

The ICAIF nine-panel marginal, cross-cell sensitivity, and joint-factor fan
figures are deterministic within one installed environment and are
`verified-from-evidence`. The launcher compares every regenerated panel with
its golden at native resolution and on a 192 x 192 luminance tile. Each panel
must pass normalized RGB MAE at most 0.12, luminance correlation at least 0.42,
and a Sobel edge-density ratio from 0.75 through 1.55; dimensions permit at most
one pixel of renderer-version drift per axis. Golden PNG byte hashes remain
pinned in `ASSETS.tsv`; regenerated PNG
byte hashes are not promised across Matplotlib/Pillow versions.

The ICAIF nine-panel marginal is
`figures/generated/iv-marginal-daily-changes.png`. The separate
`iv-marginal-daily-changes-all25.png` remains the README display and is not
claimed as an output of the nine-panel recipe.

## Frozen paper-facing views

The final compact FMTS benchmark tables and the final ICAIF benchmark table do
not have an exact surviving generator. They remain `preserved-only`: the paper
command verifies their exact bytes and checks the surviving ScenarioFlow row
values against checksum-pinned frozen benchmark results. The relationship to
the named camera-ready/selected-source file is declared provenance, not a
mechanically verified source transformation. The command does not copy these
views into the output directory or claim to regenerate their final rows. A
fresh benchmark aggregate under `scenarioflow.benchmark` is a new result, not
a byte-exact reconstruction of those final camera-ready views.

No historical experiment tree, numbered experiment route, LaTeX workspace, or
checkpoint-driven plot launcher is restored by this workflow.

## Illustrations

Files under `figures/illustrations/` are the final conceptual illustrations used
by the papers. They are `preserved-only`, have no generator command, and are not
recreated by the numerical pipeline. The LaTeX workspaces are not included in
this repository.
