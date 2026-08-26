"""Aggregate the completed 3x5 final benchmark grid with fail-closed inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import CANDIDATE_IDS, ROLLOUT_SEEDS, TRAIN_SEEDS
from .reporting import (
    METRIC_PATHS,
    TABLE_THRESHOLD_RULES,
    metric_values,
    threshold_status,
)


class AggregationContractError(ValueError):
    """Raised before output when the completed grid violates its contract."""


def moving_block_bootstrap_mean(
    values: np.ndarray,
    *,
    block_length: int = 30,
    n_bootstrap: int = 10_000,
    seed: int = 20260805,
) -> dict[str, float]:
    """Chronology-aware uncertainty for one paired per-window series."""
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size < 1 or block_length < 1:
        raise ValueError("values and block_length must be positive")
    generator = np.random.default_rng(seed)
    starts = np.arange(max(series.size - block_length + 1, 1))
    draws = np.empty(n_bootstrap)
    blocks = int(np.ceil(series.size / block_length))
    for draw in range(n_bootstrap):
        sampled = []
        for start in generator.choice(starts, size=blocks, replace=True):
            block = np.take(
                series,
                np.arange(int(start), int(start) + block_length) % series.size,
            )
            sampled.append(block)
        draws[draw] = np.concatenate(sampled)[: series.size].mean()
    return {
        "mean": float(series.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def _read_result(node: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        Path(path)
        for path in node.get("resume", {}).get("completion_artifacts", [])
        if Path(path).name == "results_current_panel.json"
    ]
    if len(candidates) != 1 or not candidates[0].is_file():
        raise AggregationContractError(
            f"node {node.get('id')} has no completed result JSON"
        )
    document = json.loads(candidates[0].read_text(encoding="utf-8"))
    if "baselines" in document:
        document = document["baselines"][str(node["candidate"])]
    if document.get("mode") != "current_realvix_39_state_panel":
        raise AggregationContractError(f"panel mode mismatch for {node.get('id')}")
    return document


def aggregate_study(
    *, protocol_manifest: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    document = json.loads(Path(protocol_manifest).read_text(encoding="utf-8"))
    protocol = document.get("protocol", {})
    if tuple(protocol.get("train_seeds", ())) != TRAIN_SEEDS:
        raise AggregationContractError("training seed schedule drifted")
    if tuple(protocol.get("rollout_seeds", ())) != ROLLOUT_SEEDS:
        raise AggregationContractError("rollout seed schedule drifted")
    if tuple(protocol.get("candidate_ids", ())) != CANDIDATE_IDS:
        raise AggregationContractError("candidate schedule drifted")
    nodes = [
        node
        for node in document.get("commands", [])
        if node.get("action") == "evaluate"
    ]
    if len(nodes) != 120:
        raise AggregationContractError("final protocol must contain 120 evaluations")
    rows = []
    for node in sorted(nodes, key=lambda value: str(value["id"])):
        metrics = metric_values(_read_result(node))
        thresholds = {
            name: threshold_status(metrics[name], rule)
            for name, rule in TABLE_THRESHOLD_RULES.items()
        }
        rows.append(
            {
                "node_id": node["id"],
                "candidate": node["candidate"],
                "training_seed": node.get("training_seed"),
                "rollout_seed": node["rollout_seed"],
                "metrics": metrics,
                "thresholds": thresholds,
                "pass_all": all(thresholds.values()),
            }
        )
    summaries = {}
    for candidate in CANDIDATE_IDS:
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        summaries[candidate] = {
            "n_runs": len(candidate_rows),
            "pass_all_rate": float(
                np.mean([row["pass_all"] for row in candidate_rows])
            ),
            "metrics": {
                name: {
                    "mean": float(
                        np.mean([row["metrics"][name] for row in candidate_rows])
                    ),
                    "std": (
                        float(
                            np.std(
                                [row["metrics"][name] for row in candidate_rows], ddof=1
                            )
                        )
                        if len(candidate_rows) > 1
                        else 0.0
                    ),
                }
                for name, _path in METRIC_PATHS
            },
        }
    summary = {
        "schema_version": "scenarioflow.multiseed-aggregate.v1",
        "protocol_dag_sha256": protocol.get("executable_dag_sha256"),
        "seed_schedule": {
            "training_seeds": list(TRAIN_SEEDS),
            "rollout_seeds": list(ROLLOUT_SEEDS),
        },
        "per_runs": rows,
        "candidate_summaries": summaries,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "per-run-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "node_id",
            "candidate",
            "training_seed",
            "rollout_seed",
            *[name for name, _path in METRIC_PATHS],
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "node_id": row["node_id"],
                    "candidate": row["candidate"],
                    "training_seed": row["training_seed"],
                    "rollout_seed": row["rollout_seed"],
                    **row["metrics"],
                }
            )
    markdown = [
        "# Final multi-seed benchmark",
        "",
        "| Candidate | Runs | Pass-all rate | CRPS mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATE_IDS:
        item = summaries[candidate]
        markdown.append(
            f"| {candidate} | {item['n_runs']} | {item['pass_all_rate']:.1%} | {item['metrics']['CRPS']['mean']:.6g} |"
        )
    (output / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    aggregate_study(
        protocol_manifest=args.protocol_manifest, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
