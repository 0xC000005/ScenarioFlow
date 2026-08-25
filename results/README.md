# Frozen results

This directory contains the numerical results used to build the ScenarioFlow
paper tables. `FROZEN_AUTHORITY.tsv` records each file's release path, size,
checksum, and packaging transformation.

Verify the complete release from the repository root:

```bash
uv run python -c "from pathlib import Path; from scenarioflow.paper import verify_release_authorities; verify_release_authorities(Path('.'), Path('results/FROZEN_AUTHORITY.tsv'))"
```

Large checkpoints and generated arrays are resolved separately through the
artifact manifest. The five processed model inputs are committed under
`data/`.
