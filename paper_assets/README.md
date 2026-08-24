# Paper assets

This directory contains the final tables and figures used by the ICAIF and
Foundation Models for Temporal Systems papers. `paper_assets/ASSETS.tsv` records
the venue, source authority, byte size, SHA-256, and reproduction mode of every
asset.

## Verify the frozen evidence

The numerical authorities live under `results/frozen/` and are indexed by
`results/FROZEN_AUTHORITY.tsv`. From the repository root, run:

```bash
uv sync --extra dev --extra paper
uv run pytest -q tests/test_frozen_evidence.py tests/test_paper_figures.py
uv run python scripts/check_release.py --root . --all
```

## Export the benchmark tables

This validates the frozen authorities before writing reader-facing JSON, CSV,
and LaTeX fragments to a separate build directory:

```bash
uv run scenarioflow paper export-tables \
  --authority results/FROZEN_AUTHORITY.tsv \
  --output build/paper/tables
```

The dollar-liquidity table also uses two checksum-pinned numerical arrays. Fetch
them from a publisher-provided bundle, then export all three paper views:

```bash
uv run scenarioflow fetch \
  --artifact evidence.nl.dollar-liquidity-cohesive \
  --artifact evidence.nl.dollar-liquidity-start-only \
  --bundle-dir RELEASE_BUNDLE \
  --artifact-root artifacts/cache

uv run scenarioflow paper export-dollar-liquidity \
  --artifact-root artifacts/cache \
  --output build/paper/dollar-liquidity
```

## Regenerate the numerical plots

Fetch the two plot checkpoints once:

```bash
uv run scenarioflow fetch \
  --artifact core.iv-only \
  --artifact core.stage2 \
  --bundle-dir RELEASE_BUNDLE \
  --artifact-root artifacts/cache
```

Then run any numerical recipe:

```bash
uv run scenarioflow paper plot iv-marginal \
  --artifact-root artifacts/cache \
  --output build/paper/iv-marginal-daily-changes.png

uv run scenarioflow paper plot iv-cross-cell \
  --artifact-root artifacts/cache \
  --output build/paper/iv-cross-cell-sensitivity.png

uv run scenarioflow paper plot joint-factor-fans \
  --artifact-root artifacts/cache \
  --output build/paper/joint-factor-fans.png
```

The committed PNGs remain the paper authorities. A fresh raster may differ in
font rendering or image metadata across Matplotlib environments. The joint fan
recipe also writes its history-selection JSON for an exact comparison.

## Preserved illustrations

The files under `figures/illustrations/` are final conceptual illustrations.
They are preserved byte-for-byte and have no regeneration command. The release
does not include either LaTeX paper workspace or an illustration authoring
toolchain.
