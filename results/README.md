# Frozen results

This directory contains the result and model-metadata records used by the
ScenarioFlow papers. `FROZEN_AUTHORITY.tsv` maps each public file to its
original frozen path, byte count, and SHA-256 digest.

Of the 57 authorities, 51 are copied byte for byte. Six JSON files contained
machine-local paths, so their paths were replaced with stable provenance URIs.
Their records, numerical values, model revisions, seeds, hashes, and protocol
fields are unchanged. These six rows are marked `portable-json-v1` in the
manifest; all other rows are marked `identity`.

Some frozen provenance maps used paths as JSON object keys. The portable
projection therefore normalizes recognized paths in both keys and string
values. It preserves every record and rejects any key collision or unknown
host path.

Verify the complete release from the repository root:

```bash
uv run python -c "from pathlib import Path; from scenarioflow.paper import verify_release_authorities; verify_release_authorities(Path('.'), Path('results/FROZEN_AUTHORITY.tsv'))"
```

Large checkpoints and generated arrays are resolved separately through the
artifact manifest. The five processed model inputs are committed under
`data/`.
