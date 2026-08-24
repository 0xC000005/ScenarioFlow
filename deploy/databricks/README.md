# Databricks job

The optional Asset Bundle builds the ScenarioFlow wheel and runs the same
`scenarioflow reproduce` command used locally and in Docker. It does not contain
a separate training or evaluation implementation.

Place the processed data, fetched checkpoints, and fetched evidence artifacts in
Unity Catalog volumes. The defaults use `/Volumes/main/default/scenarioflow`; use
bundle variables if your catalog or schema differs.

```bash
databricks bundle validate -t dev --var="cluster_id=<existing-cluster-id>"
databricks bundle deploy -t dev --var="cluster_id=<existing-cluster-id>"
databricks bundle run -t dev reproduce
```

The job parameters are defined in `resources.yml`. The full three-training-seed,
five-rollout-seed schedule is the default and the output directory is resumable.
The numerical reproduction job has no secret references. OpenAI and Codex are
used only by the optional local demos and are not part of this bundle.
