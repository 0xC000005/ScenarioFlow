# ScenarioFlow

ScenarioFlow is a reproducible research release for conditional market-scenario generation.

The release includes the portable Stage 1 flow-matching model, Stage 2 free-rollout
objective, adaptive-graph variant, and the frozen processed inputs.

## Repository contents

- `data/`: five checksum-pinned processed market-data files.
- `src/scenarioflow/`: state/data preparation, model, losses, training, and inference APIs.
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
uv run python -m scenarioflow.training --help
uv run python -m scenarioflow.inference --help
```

`scenarioflow.training` consumes a portable NPZ batch and either a JSON model
configuration or an existing checkpoint. `scenarioflow.inference` consumes a
checkpoint plus a portable NPZ conditioning batch and writes deterministic sampled
normalized-innovation paths for a supplied seed. The reusable Python surfaces are
`stage1_training_step`, `stage2_training_step`, `load_model`, and
`sample_normalized_innovations`.

See [DATA_SOURCES.md](DATA_SOURCES.md) for the processed-data provenance and
[LICENSE](LICENSE) for the repository license.
