# Databricks deployment

Databricks uses a separate numerical image. The local demo image and Compose
stack are not part of this deployment.

## Build and push the image

Choose a registry location and immutable tag:

```bash
export IMAGE_URL="<registry>/scenarioflow-databricks:<tag>"
docker build -f deploy/databricks/Dockerfile -t "$IMAGE_URL" .
docker push "$IMAGE_URL"
```

The image contains the processed data, preserved experiment source, and the
existing numerical scripts. It does not contain the released weights. Copy the
artifact bundle to a Databricks Volume so that the artifact root contains
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

The workspace must have Databricks Container Services enabled. The job uses
dedicated single-node GPU compute and Databricks Runtime 18.3 LTS, matching the
base image.

## Run the numerical workflows

Run numerical inference with the released Stage 2 weights:

```bash
databricks bundle run scenarioflow --params operation=inference
```

Train Stage 1 and Stage 2 for one seed:

```bash
databricks bundle run scenarioflow --params operation=train,seed=1001
```

Run the complete multi-seed reproduction:

```bash
databricks bundle run scenarioflow --params operation=reproduce
```

All three routes invoke the same scripts used by the native release. Checkpoints
and results are written to the configured output Volume.
