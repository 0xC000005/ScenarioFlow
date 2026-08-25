# Release artifacts

The five processed datasets live in Git and are never downloaded by the
artifact command. `manifest-v1.tsv` covers the 43 large checkpoints, runtime
assets, and evidence files distributed separately from the repository.

Fetch a named profile from the release URL:

```bash
scenarioflow fetch --profile core --base-url "$RELEASE_BASE_URL"
```

Or use a local release bundle with the same member paths as `url_path`:

```bash
scenarioflow fetch --profile core --bundle-dir /path/to/scenarioflow-v1
```

Available profiles are `core`, `baselines`, `multiseed`, `foundation`,
`nl-runtime`, `evidence`, and `all`. The optional IV-only checkpoint belongs
only to `all`; `core` is the three 39-state Stage 1, Stage 2, and adaptive-graph
checkpoints. Set `SCENARIOFLOW_ARTIFACT_BASE_URL` instead of `--base-url` when
a permanent release URL becomes available.

The `evidence` profile includes the complete sampled arrays used to regenerate
the numerical paper figures. These arrays preserve the published sample decks;
the plotting command does not draw a smaller or different set of paths.

`github-authority-v1.tsv` records the checksums of the three Gaussian-CCC replay
files stored in this repository.
