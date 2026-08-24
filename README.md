# ScenarioFlow

ScenarioFlow is a reproducible research release for conditional market-scenario generation.

This initial release contract provides the frozen processed inputs, their verification
tests, and a minimal Python package layout. Numerical models, evaluation workflows,
and demos are migrated in subsequent release steps.

## Repository contents

- `data/`: five checksum-pinned processed market-data files.
- `src/scenarioflow/`: the Python package namespace.
- `tests/`: data and public-release contract tests.
- `scripts/check_release.py`: secret, host-path, and file-size release checks.

## Verify the release contract

```bash
pytest -q tests/test_release_contract.py
python scripts/check_release.py --all
```

See [DATA_SOURCES.md](DATA_SOURCES.md) for the processed-data provenance and
[LICENSE](LICENSE) for the repository license.
