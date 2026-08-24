# Release artifacts

The five processed datasets live in Git and are never downloaded by the
artifact command. `manifest-v1.tsv` covers the 41 large checkpoints, runtime
assets, and evidence files distributed separately from the repository.

Fetch a named profile from a publisher-provided URL:

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

`github-authority-v1.tsv` pins three Gaussian-CCC replay files that Task 5 adds
to Git. They are provenance records, not download entries.
