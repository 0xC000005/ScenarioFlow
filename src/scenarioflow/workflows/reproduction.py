"""Single native entry point for planning, resuming, and aggregating reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .multiseed import (
    CANDIDATE_IDS,
    ROLLOUT_SEEDS,
    TRAIN_SEEDS,
    ProtocolValidationError,
    aggregate_results,
    build_protocol,
)

RUN_MANIFEST_SCHEMA = "scenarioflow.multiseed-node.v1"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _validate_identity(
    *,
    candidate: str,
    train_seed: int | None,
    rollout_seed: int,
    result: Mapping[str, Any],
) -> None:
    if candidate not in CANDIDATE_IDS:
        raise ProtocolValidationError(f"unknown candidate {candidate!r}")
    if rollout_seed not in ROLLOUT_SEEDS:
        raise ProtocolValidationError(
            f"rollout seed must be one of {list(ROLLOUT_SEEDS)}"
        )
    expects_training = candidate not in {
        "random_walk",
        "bootstrap",
        "historical_sim",
        "pca_var",
        "garch_ccc",
        "filtered_hs",
    }
    if expects_training and train_seed not in TRAIN_SEEDS:
        raise ProtocolValidationError(
            f"training seed must be one of {list(TRAIN_SEEDS)}"
        )
    if not expects_training and train_seed is not None:
        raise ProtocolValidationError(f"{candidate} must not invent a training seed")
    if result.get("baseline_name") != candidate:
        raise ProtocolValidationError("result candidate identity mismatch")
    if result.get("mode") != "current_realvix_39_state_panel":
        raise ProtocolValidationError("result is not the published validation protocol")
    if result.get("n_windows") != 441 or result.get("n_samples") != 64:
        raise ProtocolValidationError(
            "result is not a complete 441-window, 64-scenario deck"
        )


def record_completed_node(
    *,
    output_dir: str | Path,
    candidate: str,
    train_seed: int | None,
    rollout_seed: int,
    result: Mapping[str, Any],
    garch_contract_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically seal a node; exact repeats resume and any drift is rejected."""
    _validate_identity(
        candidate=candidate,
        train_seed=train_seed,
        rollout_seed=rollout_seed,
        result=result,
    )
    if candidate == "garch_ccc" and (
        not isinstance(garch_contract_report, Mapping)
        or garch_contract_report.get("overall_pass") is not True
        or garch_contract_report.get("immutable_authority") is not True
    ):
        raise ProtocolValidationError(
            "garch_ccc requires the passing immutable published-row contract"
        )
    root = Path(output_dir).expanduser().resolve()
    result_text = (
        json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    result_sha256 = _digest_text(result_text)
    authority_hash = (
        None
        if garch_contract_report is None
        else _digest_text(_canonical_json(garch_contract_report))
    )
    manifest_core = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "candidate": candidate,
        "train_seed": train_seed,
        "rollout_seed": rollout_seed,
        "benchmark_split": "published_validation",
        "result_path": "results_current_panel.json",
        "result_sha256": result_sha256,
        "garch_contract_sha256": authority_hash,
    }
    result_path = root / "results_current_panel.json"
    manifest_path = root / "run_manifest.json"
    if result_path.exists() or manifest_path.exists():
        if not result_path.is_file() or not manifest_path.is_file():
            raise ProtocolValidationError("partial completed node cannot be resumed")
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                "completed node manifest is unreadable"
            ) from error
        existing_result_text = result_path.read_text(encoding="utf-8")
        if existing_manifest != manifest_core or existing_result_text != result_text:
            raise ProtocolValidationError(
                "refusing to replace a different completed node"
            )
        return {**manifest_core, "resumed": True}
    root.mkdir(parents=True, exist_ok=True)
    _write_atomic(result_path, result_text)
    _write_atomic(
        manifest_path, json.dumps(manifest_core, indent=2, sort_keys=True) + "\n"
    )
    return {**manifest_core, "resumed": False}


def collect_completed_runs(output_root: str | Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for manifest_path in sorted(Path(output_root).rglob("run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
            raise ProtocolValidationError(f"unexpected run manifest: {manifest_path}")
        result_path = manifest_path.parent / str(manifest.get("result_path"))
        text = result_path.read_text(encoding="utf-8")
        if _digest_text(text) != manifest.get("result_sha256"):
            raise ProtocolValidationError(
                f"completed result checksum mismatch: {result_path}"
            )
        runs.append(
            {
                "candidate": manifest["candidate"],
                "train_seed": manifest["train_seed"],
                "rollout_seed": manifest["rollout_seed"],
                "result": json.loads(text),
            }
        )
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    plan = subcommands.add_parser("plan", help="create or resume the frozen local DAG")
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--source-manifest", required=True)
    plan.add_argument("--artifact-manifest", required=True)
    node = subcommands.add_parser(
        "run-node", help="seal an explicitly produced package result"
    )
    node.add_argument("--candidate", required=True, choices=CANDIDATE_IDS)
    node.add_argument("--train-seed", type=int)
    node.add_argument("--rollout-seed", required=True, type=int)
    node.add_argument("--result-input", required=True)
    node.add_argument("--garch-contract-report")
    node.add_argument("--output-dir", required=True)
    aggregate = subcommands.add_parser(
        "aggregate", help="aggregate a complete resumed DAG"
    )
    aggregate.add_argument("--output-root", required=True)
    aggregate.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        payload = build_protocol(
            output_root=args.output_root,
            source_manifest=args.source_manifest,
            artifact_manifest=args.artifact_manifest,
        )
        print(
            json.dumps(
                {
                    "contract_hash": payload["contract_hash"],
                    "node_count": len(payload["nodes"]),
                }
            )
        )
        return
    if args.command == "run-node":
        result = json.loads(Path(args.result_input).read_text(encoding="utf-8"))
        authority = (
            None
            if args.garch_contract_report is None
            else json.loads(
                Path(args.garch_contract_report).read_text(encoding="utf-8")
            )
        )
        manifest = record_completed_node(
            output_dir=args.output_dir,
            candidate=args.candidate,
            train_seed=args.train_seed,
            rollout_seed=args.rollout_seed,
            result=result,
            garch_contract_report=authority,
        )
        print(json.dumps(manifest, sort_keys=True))
        return
    summary = aggregate_results(collect_completed_runs(args.output_root))
    output = Path(args.output)
    _write_atomic(output, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
