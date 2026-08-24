# ScenarioFlow release candidate

Date: 2026-08-24

## Status

The local release candidate passes its publication gate. The repository contains
the ScenarioFlow, Stage 2, adaptive-graph, baseline, paper-asset, and demo code
selected for the ICAIF and FMTS release. The inherited VAE branches, research
logs, paper workspaces, credentials, and old Git history are not included.

Remote publication has not been performed. The repository has no remote, and no
artifact bundle, paper link, Hugging Face release, DOI, or hosted-demo change was
made during this gate.

## Verification

- Python: 231 tests passed with all 41 external artifacts available.
- Web: 19 tests passed; lint and production build passed.
- Artifacts: 41 files and 1,186,583,031 bytes matched the tracked manifest.
- Processed data: five files and 4,392,783 bytes matched `data/CHECKSUMS.tsv`.
- Reproduction: the ScenarioFlow and adaptive-graph smoke DAG trained, evaluated,
  resumed without rerunning completed nodes, and reproduced the same summary hash.
- Paper assets: all frozen authorities passed; the full benchmark and
  dollar-liquidity tables regenerated; all three numerical plotting paths rendered
  a small verification deck from the released checkpoints.
- Docker: the three images built; native and container inference matched exactly;
  both cached demo APIs and the standalone frontend passed smoke checks.
- Databricks: the bundle and job resource validated against the current Databricks
  JSON schema; the job calls the same `scenarioflow reproduce` wheel entry point.
- Package: the wheel built with the MIT license and console entry point included.
- Release scans: the working tree and full fresh history had no credential-shaped
  values or machine-specific home paths. No tracked file exceeds 100 MiB.
- Repository size: 219 tracked files, 32,665,580 tracked bytes, and 15.71 MiB of
  loose Git objects. The largest tracked file is 6,614,391 bytes.
- Documentation: all local Markdown links resolve.

The committed multiseed tables and numerical authorities were verified. The full
three-training-seed, five-rollout-seed training schedule was not recomputed during
this packaging gate; `scenarioflow reproduce` and the Databricks job retain that
full schedule as their default.

## External publication steps

1. Create the public Git remote and push the fresh history.
2. Publish the checksum-pinned artifact bundle and set its release URL.
3. Replace the two paper-link placeholders after the papers have public URLs.
4. Add the hosted demo URL to the repository description.
