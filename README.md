# ScenarioFlow

ScenarioFlow is a reproducible research release for conditional market-scenario generation.

The release includes the portable Stage 1 flow-matching model, Stage 2 free-rollout
objective, adaptive-graph variant, and the frozen processed inputs.

## Repository contents

- `data/`: five checksum-pinned processed market-data files.
- `src/scenarioflow/`: state/data preparation, model, losses, training, inference,
  evaluation, and artifact APIs.
- `artifacts/manifest-v1.tsv`: checksums and release paths for large external files.
- `paper_assets/`: final ICAIF/FMTS tables and figures, with reproduction modes.
- `tests/`: data and public-release contract tests.
- `scripts/check_release.py`: secret, host-path, and file-size release checks.

## Verify the release contract

```bash
pytest -q tests/test_release_contract.py
python scripts/check_release.py --all
```

## Numerical APIs and CLIs

Install the locked environment and inspect the parameterized workflows:

```bash
uv sync --extra dev
uv run scenarioflow setup --check
uv run scenarioflow fetch --profile core --bundle-dir /path/to/release-bundle
uv run scenarioflow train --help
uv run scenarioflow infer --help
uv run scenarioflow reproduce --help
uv run scenarioflow demo doctor
```

`scenarioflow train` consumes a portable NPZ batch and either a JSON model
configuration or an existing checkpoint. `scenarioflow infer` consumes a
checkpoint plus a portable NPZ conditioning batch and writes deterministic sampled
normalized-innovation paths for a supplied seed. The reusable Python surfaces are
`stage1_training_step`, `stage2_training_step`, `load_model`, and
`sample_normalized_innovations`.

Large checkpoints and evidence files are not stored in Git. Fetching accepts a
publisher-provided URL or a local release bundle and installs a file only after its
size and SHA-256 match the tracked manifest. Missing OpenAI or Codex requirements
affect only the corresponding optional demo; numerical workflows remain available.

See [DATA_SOURCES.md](DATA_SOURCES.md) for the processed-data provenance and
[Paper assets](paper_assets/README.md) for table and figure reproduction, and
[LICENSE](LICENSE) for the repository license.

## Papers

- ICAIF paper: link will be added after publication.
- FMTS workshop paper: link will be added after publication.
