# Databricks deployment

Databricks uses a separate numerical image. The local demo image and Compose
stack are not part of this deployment.

## Prerequisites

Use an authenticated, actively supported Databricks CLI.
Choose the current latest release or the extended-support v0.299.x line. The
bundle enforces a minimum of 0.299.0. Build the image from a checkout that
includes the root `scripts/reproduce.sh` benchmark launcher. The target
workspace also needs Databricks Container Services and a GPU node type
compatible with Databricks Runtime 18.3 LTS.

## Build and push the image

Choose a registry location and immutable tag:

```bash
export IMAGE_URL="<registry>/scenarioflow-databricks:<tag>"
docker build -f deploy/databricks/Dockerfile -t "$IMAGE_URL" .
docker push "$IMAGE_URL"
```

The image contains the processed data and ScenarioFlow numerical code. It does
not contain the released weights. Copy the artifact bundle to a Databricks
Volume so that the artifact root contains
`models/core/stage2/best_model.pt`.

## Configure and deploy the job

Set the image and a GPU node type available in your workspace. The default
artifact and output locations are under
`/Volumes/main/default/scenarioflow/`; override them when needed.

```bash
export BUNDLE_VAR_image_url="$IMAGE_URL"
export BUNDLE_VAR_gpu_node_type="<gpu-node-type>"
export BUNDLE_VAR_artifact_root="/Volumes/main/default/scenarioflow/artifacts"
export BUNDLE_VAR_output_root="/Volumes/main/default/scenarioflow/results"

databricks bundle validate
databricks bundle deploy
```

The job uses dedicated single-node GPU compute and Databricks Runtime 18.3 LTS,
matching the base image.

## Run the numerical workflows

Run numerical inference with the released Stage 2 weights:

```bash
databricks bundle run scenarioflow --params operation=inference,seed=2001
```

Train Stage 1 and Stage 2 for one seed:

```bash
databricks bundle run scenarioflow --params operation=train,seed=1001
```

Run the complete multi-seed reproduction:

```bash
databricks bundle run scenarioflow --params operation=reproduce
```

The reproduction route invokes the root launcher with `--resume-all`, runs its
complete multi-seed grid, and does not use the single-seed job parameter. A
successful full run writes its semantic rerun aggregate below the configured
result root at `aggregate/` and names that path in the completion JSON. All
three routes invoke the same scripts used by the native release. Checkpoints
and results are written to the configured output Volume.
