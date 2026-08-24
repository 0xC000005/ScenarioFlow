"""Export frozen ICAIF and FMTS benchmark tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from ..evaluation.current_panel import (
    CURRENT_PANEL_RESULT_KEYS,
    METRIC_PATHS,
    MODE,
    TABLE_THRESHOLD_RULES,
    threshold_status,
    validate_result_schema,
)
from ..workflows.multiseed import CANDIDATE_IDS
from .authority import AuthorityError, verify_release_authorities
from .multiseed import METRIC_SPECS, MultiseedExportError, render_multiseed_table

BENCHMARK_SCHEMA = "scenarioflow.paper-benchmark.v1"

SCENARIOFLOW_CURRENT_ID = (
    "release.results.frozen.benchmark.current-panel.scenarioflow.results.json"
)
SCENARIOFLOW_GRAPH_CURRENT_ID = (
    "release.results.frozen.benchmark.current-panel.scenarioflow-graph.results.json"
)
SCENARIOFLOW_REPLAY_ID = (
    "release.results.frozen.benchmark.checkpoint-replay.scenarioflow.results.json"
)
BASELINE_SUMMARY_ID = (
    "release.results.frozen.benchmark.checkpoint-replay.baseline-summary.json"
)
G2_ID = "release.results.frozen.benchmark.current-panel.g2-deepvar-history-scale.results.json"
GAUSSIAN_SUMMARY_ID = (
    "release.results.frozen.benchmark.checkpoint-replay.gaussian-ccc.summary.json"
)
GAUSSIAN_MANIFEST_ID = (
    "release.results.frozen.benchmark.checkpoint-replay.gaussian-ccc.manifest.json"
)
GAUSSIAN_RESULT_ID = (
    "release.results.frozen.benchmark.checkpoint-replay.gaussian-ccc.results.json"
)
GAUSSIAN_REPORT_ID = "release.results.frozen.benchmark.checkpoint-replay.gaussian-ccc.contract-report.json"
FOUNDATION_BENCHMARK_ID = "release.results.frozen.foundation.benchmark-manifest.json"
FOUNDATION_MANIFEST_ID = (
    "release.results.frozen.foundation.canonical-eval.manifest.json"
)
FOUNDATION_SUMMARY_ID = "release.results.frozen.foundation.canonical-eval.summary.json"
MOIRAI_ID = "release.results.frozen.foundation.canonical-eval.moirai.results.json"
TOTO_ID = "release.results.frozen.foundation.canonical-eval.toto.results.json"
MULTISEED_ID = "release.results.frozen.multiseed.summary.json"

AUTHORITY_IDENTITIES: dict[str, tuple[int, str]] = {
    SCENARIOFLOW_CURRENT_ID: (
        94_580,
        "83b06b8cc263583b2b6b978195d63c375d3203fa82ecab3432522a3a13915fb9",
    ),
    SCENARIOFLOW_GRAPH_CURRENT_ID: (
        94_436,
        "61d9db2d669529e797416c11c7849c3bb7fe67f782686457c83552ffa7ea6b63",
    ),
    SCENARIOFLOW_REPLAY_ID: (
        94_580,
        "d743f4461b98373213aa0856c717ba5d517b587722b06761c713b1b230ad7a35",
    ),
    BASELINE_SUMMARY_ID: (
        935_832,
        "c0a1eac0d65de3201108f4ff4429ee47d541db40abc5da959317531175705304",
    ),
    G2_ID: (
        91_637,
        "532f9ce92f57a734692bf78e09c5b30a87d23ceda349a1a1150acdbfe066b0e5",
    ),
    GAUSSIAN_SUMMARY_ID: (
        104_375,
        "d1431f0dc94974301f0bd8642666ba70c8d093bb06cad8a43311f8900b33a771",
    ),
    GAUSSIAN_MANIFEST_ID: (
        13_850,
        "ea16b3a2f59fc066ab6c394190ac15fd70a66e47c2aa5d70b07111700646882d",
    ),
    GAUSSIAN_RESULT_ID: (
        91_037,
        "4749ef068a1510595a68e2227e9fe1be8cabc6927fcb8ad39b4b0b1271369f5a",
    ),
    GAUSSIAN_REPORT_ID: (
        39_950,
        "70c9ac714a4f81b182746810791658bea35d14c4d28bfde762e0377eb5f3190a",
    ),
    FOUNDATION_BENCHMARK_ID: (
        64_865,
        "8f1e4bd59f0ab38a36fffdafbbba59fa27442aba48946469350a11c9b9a91f19",
    ),
    FOUNDATION_MANIFEST_ID: (
        13_850,
        "ea16b3a2f59fc066ab6c394190ac15fd70a66e47c2aa5d70b07111700646882d",
    ),
    FOUNDATION_SUMMARY_ID: (
        210_156,
        "8a200ecf735dfd42ae11f5cc325b84b27880f690307547ae8443ea7ed94a36a6",
    ),
    MOIRAI_ID: (
        92_318,
        "3d43bbac6b6682647fd60e5d947e2b5c7f97fd56d22700f349af311177552f35",
    ),
    TOTO_ID: (
        91_205,
        "7cb04e6694766f5b344ce0d2e6b4b260eddcef47524abd1a019866f6bca9ec7f",
    ),
    MULTISEED_ID: (
        692_525,
        "4ebb4d26666cf3d05f8f85c03f678584c23be799479e94f328078fd6783eb3fb",
    ),
}

GOLDEN_IDENTITIES = {
    "fmts-benchmark-full.tex": (
        5_638,
        "89a2614a6c292dd472ebf373cf751bc96d0a28484cc8aefdc7dc0270ea39d5f7",
    ),
    "fmts-benchmark-short.tex": (
        4_379,
        "ba7d806d76d74c67bb2ed7104d541f1b36f831ccdda2c0b0ab75b0ca2d7ff2ed",
    ),
    "icaif-benchmark.tex": (
        4_212,
        "915f6a3cafd464b9d0e911fdb77a5530044dc09a9f29f44b485ebdddafb54943",
    ),
}

ICAIF_CANDIDATES = tuple(
    candidate for candidate in CANDIDATE_IDS if candidate != "deepvar_sni"
)
_FMTS_CORE: list[str] = []
for _candidate in CANDIDATE_IDS:
    if _candidate in {"scenarioflow_graph", "deepvar_sni"}:
        continue
    _FMTS_CORE.append(_candidate)
    if _candidate == "deepvar":
        _FMTS_CORE.append("deepvar_sni")
FMTS_CANDIDATES = tuple([*_FMTS_CORE, "moirai_1_1_finetuned", "toto_1_0_finetuned"])
CSV_CANDIDATES = (*FMTS_CANDIDATES, "scenarioflow_graph")

DISPLAY_NAMES = {
    "scenarioflow": "ScenarioFlow",
    "scenarioflow_graph": "ScenarioFlow + graph",
    "random_walk": "Random walk",
    "bootstrap": "Bootstrap",
    "historical_sim": "Historical simulation",
    "pca_var": "PCA-VAR",
    "garch_ccc": "GARCH-CCC",
    "filtered_hs": "Filtered historical simulation",
    "deepvar": "DeepVAR",
    "deepvar_sni": "DeepVAR + history scale",
    "timegrad": "TimeGrad",
    "path_diffusion": "CSDI",
    "moirai_1_1_finetuned": "Moirai",
    "toto_1_0_finetuned": "Toto",
}

FMTS_ECON_PATH = (
    "current_panel_iv",
    "iv_ewma_economic_link",
    "slope_ratio",
)

PANEL_A_METRICS = tuple(name for name, _path in METRIC_PATHS[:8])
PANEL_B_METRICS = tuple(name for name, _path in METRIC_PATHS[8:])
SHORT_METRICS = (
    "CRPS",
    "Energy Score",
    "Corr score",
    "Factor KS",
    "Tail q99",
    "Cov90",
    "CalErr",
    "Level KS",
    "Width rho",
    "MR ratio",
    "Econ ratio",
    "Jump KS",
    "Kurtosis",
)

FULL_LABELS = {
    "scenarioflow": r"\scenarioflow{}",
    "scenarioflow_graph": r"\scenarioflow{} + graph",
    "random_walk": "Random walk",
    "bootstrap": "Bootstrap",
    "historical_sim": "Historical sim.",
    "pca_var": "PCA-VAR",
    "garch_ccc": "GARCH-CCC",
    "filtered_hs": "Filtered hist. sim.",
    "deepvar": "DeepVAR",
    "deepvar_sni": "DeepVAR + history scale",
    "timegrad": "TimeGrad",
    "path_diffusion": "CSDI",
    "moirai_1_1_finetuned": "Moirai",
    "toto_1_0_finetuned": r"Toto$^\dagger$",
}
SHORT_LABELS = {
    **FULL_LABELS,
    "pca_var": "PCA--VAR",
    "garch_ccc": "GARCH--CCC",
}

# The accepted short-paper layout wraps each row at fixed semantic boundaries.
# Keeping this separate from value formatting makes line wrapping deterministic.
SHORT_ROW_GROUPS = {
    "scenarioflow": (4, 5, 5),
    "random_walk": (4, 4, 4, 2),
    "bootstrap": (4, 4, 6),
    "historical_sim": (4, 5, 5),
    "pca_var": (4, 6, 4),
    "garch_ccc": (4, 5, 3, 2),
    "filtered_hs": (4, 6, 4),
    "deepvar": (4, 6, 4),
    "deepvar_sni": (4, 6, 4),
    "timegrad": (4, 6, 4),
    "path_diffusion": (4, 3, 4, 3),
    "moirai_1_1_finetuned": (4, 5, 3, 2),
    "toto_1_0_finetuned": (1, 2, 2, 4, 5),
}


class BenchmarkExportError(ValueError):
    """Raised when table authorities or requested outputs are invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected: tuple[int, str], *, label: str) -> Path:
    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise BenchmarkExportError(f"{label} authority is not a regular file")
        if source.stat().st_size != expected[0] or _sha256(source) != expected[1]:
            raise BenchmarkExportError(f"{label} authority identity changed")
    except OSError as error:
        raise BenchmarkExportError(f"{label} authority is unreadable") from error
    return source


def authority_paths_from_manifest(
    manifest: Path, *, root: Path | None = None
) -> dict[str, Path]:
    """Verify the release manifest and return paths keyed by authority ID."""
    manifest_path = Path(manifest)
    release_root = (
        Path(root)
        if root is not None
        else manifest_path.expanduser().resolve().parent.parent
    )
    try:
        authorities = verify_release_authorities(release_root, manifest_path)
    except AuthorityError as error:
        raise BenchmarkExportError(str(error)) from error
    return {
        item.authority_id: release_root / item.released_path for item in authorities
    }


def _verified_authorities(authorities: Mapping[str, Path]) -> dict[str, Path]:
    verified: dict[str, Path] = {}
    for authority_id, identity in AUTHORITY_IDENTITIES.items():
        try:
            path = authorities[authority_id]
        except KeyError as error:
            raise BenchmarkExportError(
                f"missing benchmark authority: {authority_id}"
            ) from error
        verified[authority_id] = _verify_file(Path(path), identity, label=authority_id)
    return verified


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkExportError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: {
                "NaN": float("nan"),
                "Infinity": float("inf"),
                "-Infinity": float("-inf"),
            }[value],
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkExportError(f"invalid JSON authority: {path.name}") from error
    if not isinstance(document, dict):
        raise BenchmarkExportError(f"JSON authority is not an object: {path.name}")
    return document


def _path_value(result: Mapping[str, Any], path: Sequence[str]) -> float:
    current: object = result
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise BenchmarkExportError("missing metric path: " + ".".join(path))
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise BenchmarkExportError("metric is not numeric: " + ".".join(path))
    value = float(current)
    if not math.isfinite(value):
        raise BenchmarkExportError("metric is not finite: " + ".".join(path))
    return value


def _validate_toto(result: Mapping[str, Any]) -> None:
    missing = [key for key in CURRENT_PANEL_RESULT_KEYS if key not in result]
    if missing:
        raise BenchmarkExportError(f"Toto result is missing keys: {missing}")
    if (
        result.get("baseline_name") != "toto_1_0_finetuned"
        or result.get("mode") != MODE
        or result.get("n_windows") != 441
        or result.get("n_samples") != 64
        or not isinstance(result.get("state_specs"), list)
        or len(result["state_specs"]) != 39
    ):
        raise BenchmarkExportError("Toto result contract changed")
    if result.get("joint_panel_quality", {}).get("finite_rate") == 1.0:
        raise BenchmarkExportError("Toto is no longer the frozen invalid-deck row")
    nonfinite_paths = (
        ("panel_b_crps", "overall"),
        ("panel_b_crps", "factor_crps"),
        ("panel_b_energy_score", "overall"),
        ("panel_b_variogram_score", "variogram_score"),
    )
    for path in nonfinite_paths:
        current: object = result
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                raise BenchmarkExportError("Toto invalidity evidence changed")
            current = current[key]
        if not isinstance(current, (int, float)) or math.isfinite(float(current)):
            raise BenchmarkExportError("Toto invalidity evidence changed")
    _path_value(result, ("panel_b_crps", "iv_crps"))
    for _metric, path in METRIC_PATHS[8:]:
        _path_value(result, path)


def _metrics(
    result: Mapping[str, Any], *, toto: bool = False
) -> dict[str, float | None]:
    if toto:
        _validate_toto(result)
        available = {"IV CRPS", *(name for name, _path in METRIC_PATHS[8:])}
        return {
            name: _path_value(result, path) if name in available else None
            for name, path in METRIC_PATHS
        }
    try:
        validate_result_schema(result)
    except ValueError as error:
        raise BenchmarkExportError("current-panel result contract changed") from error
    return {name: _path_value(result, path) for name, path in METRIC_PATHS}


def _statuses(metrics: Mapping[str, float | None]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for metric in TABLE_THRESHOLD_RULES:
        value = metrics.get(metric)
        statuses[metric] = (
            "invalid" if value is None else str(threshold_status(metric, value))
        )
    return statuses


def _row(
    candidate_id: str,
    result: Mapping[str, Any],
    *,
    authority_ids: Mapping[str, str],
    toto: bool = False,
) -> dict[str, Any]:
    metrics = _metrics(result, toto=toto)
    fmts_metrics = dict(metrics)
    fmts_metrics["Econ ratio"] = _path_value(result, FMTS_ECON_PATH)
    thresholds = _statuses(metrics)
    fmts_thresholds = _statuses(fmts_metrics)
    return {
        "candidate_id": candidate_id,
        "display_name": DISPLAY_NAMES[candidate_id],
        "authority_ids": dict(authority_ids),
        "complete_deck_valid": not toto,
        "metrics": metrics,
        "fmts_metrics": fmts_metrics,
        "thresholds": thresholds,
        "fmts_thresholds": fmts_thresholds,
        "threshold_failure_count": sum(
            status != "pass" for status in thresholds.values()
        ),
        "fmts_threshold_failure_count": sum(
            status != "pass" for status in fmts_thresholds.values()
        ),
    }


def _build_document(paths: Mapping[str, Path]) -> dict[str, Any]:
    baseline_summary = _load_json(paths[BASELINE_SUMMARY_ID])
    if baseline_summary.get("mode") != MODE or not isinstance(
        baseline_summary.get("baselines"), dict
    ):
        raise BenchmarkExportError("baseline summary contract changed")
    baseline_documents = baseline_summary["baselines"]

    scenarioflow_replay = _load_json(paths[SCENARIOFLOW_REPLAY_ID])
    scenarioflow_current = _load_json(paths[SCENARIOFLOW_CURRENT_ID])
    graph_current = _load_json(paths[SCENARIOFLOW_GRAPH_CURRENT_ID])
    if {
        name: _path_value(scenarioflow_replay, path) for name, path in METRIC_PATHS
    } != {name: _path_value(scenarioflow_current, path) for name, path in METRIC_PATHS}:
        raise BenchmarkExportError("ScenarioFlow venue authorities disagree")

    gaussian_summary = _load_json(paths[GAUSSIAN_SUMMARY_ID])
    gaussian_manifest = _load_json(paths[GAUSSIAN_MANIFEST_ID])
    gaussian_result = _load_json(paths[GAUSSIAN_RESULT_ID])
    gaussian_report = _load_json(paths[GAUSSIAN_REPORT_ID])
    if (
        gaussian_summary.get("baselines", {}).get("garch_ccc") != gaussian_result
        or gaussian_manifest.get("mode") != MODE
        or gaussian_report.get("candidate") != "garch_ccc"
        or gaussian_report.get("overall_pass") is not True
    ):
        raise BenchmarkExportError("dedicated Gaussian-CCC contract changed")

    foundation_benchmark = _load_json(paths[FOUNDATION_BENCHMARK_ID])
    foundation_manifest = _load_json(paths[FOUNDATION_MANIFEST_ID])
    foundation_summary = _load_json(paths[FOUNDATION_SUMMARY_ID])
    moirai = _load_json(paths[MOIRAI_ID])
    toto = _load_json(paths[TOTO_ID])
    foundation_rows = foundation_summary.get("baselines", {})
    if (
        foundation_benchmark.get("schema_version")
        != "foundation-current-panel-benchmark-v1"
        or foundation_manifest.get("mode") != MODE
        or foundation_summary.get("mode") != MODE
        or set(foundation_rows) != {"moirai_1_1_finetuned", "toto_1_0_finetuned"}
        or foundation_rows["moirai_1_1_finetuned"].get("baseline_name")
        != moirai.get("baseline_name")
        or foundation_rows["toto_1_0_finetuned"].get("baseline_name")
        != toto.get("baseline_name")
    ):
        raise BenchmarkExportError("foundation canonical-evaluation contract changed")

    row_inputs: dict[str, tuple[Mapping[str, Any], dict[str, str], bool]] = {
        "scenarioflow": (
            scenarioflow_replay,
            {"fmts": SCENARIOFLOW_REPLAY_ID, "icaif": SCENARIOFLOW_CURRENT_ID},
            False,
        ),
        "scenarioflow_graph": (
            graph_current,
            {"icaif": SCENARIOFLOW_GRAPH_CURRENT_ID},
            False,
        ),
        "deepvar_sni": (
            _load_json(paths[G2_ID]),
            {"fmts": G2_ID},
            False,
        ),
        "garch_ccc": (
            gaussian_result,
            {"fmts": GAUSSIAN_RESULT_ID, "icaif": GAUSSIAN_RESULT_ID},
            False,
        ),
        "moirai_1_1_finetuned": (moirai, {"fmts": MOIRAI_ID}, False),
        "toto_1_0_finetuned": (toto, {"fmts": TOTO_ID}, True),
    }
    for candidate in (
        "random_walk",
        "bootstrap",
        "historical_sim",
        "pca_var",
        "filtered_hs",
        "deepvar",
        "timegrad",
        "path_diffusion",
    ):
        try:
            result = baseline_documents[candidate]
        except KeyError as error:
            raise BenchmarkExportError(
                f"baseline summary is missing {candidate}"
            ) from error
        row_inputs[candidate] = (
            result,
            {"fmts": BASELINE_SUMMARY_ID, "icaif": BASELINE_SUMMARY_ID},
            False,
        )

    rows = [
        _row(
            candidate,
            row_inputs[candidate][0],
            authority_ids=row_inputs[candidate][1],
            toto=row_inputs[candidate][2],
        )
        for candidate in CSV_CANDIDATES
    ]
    metric_lineage = {name: list(path) for name, path in METRIC_PATHS}
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "candidate_order": {
            "fmts": list(FMTS_CANDIDATES),
            "icaif": list(ICAIF_CANDIDATES),
        },
        "metric_order": [name for name, _path in METRIC_PATHS],
        "metric_lineage": {
            "task3_canonical": metric_lineage,
            "venues": {
                "fmts": {"Econ ratio": list(FMTS_ECON_PATH)},
                "icaif": {"Econ ratio": list(dict(METRIC_PATHS)["Econ ratio"])},
                "multiseed": {"Econ ratio": list(dict(METRIC_PATHS)["Econ ratio"])},
            },
        },
        "threshold_rules": TABLE_THRESHOLD_RULES,
        "venue_metric_overrides": {"fmts": {"Econ ratio": list(FMTS_ECON_PATH)}},
        "gaussian_contract_authorities": [
            GAUSSIAN_SUMMARY_ID,
            GAUSSIAN_MANIFEST_ID,
            GAUSSIAN_RESULT_ID,
            GAUSSIAN_REPORT_ID,
        ],
        "rows": rows,
    }


def _assert_output_is_safe(output_dir: Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    golden = (repository / "paper_assets/tables/golden").resolve()
    frozen = (repository / "results/frozen").resolve()
    if output == golden or golden in output.parents:
        raise BenchmarkExportError("refusing to overwrite protected golden tables")
    if output == frozen or frozen in output.parents:
        raise BenchmarkExportError("refusing to overwrite frozen evidence")
    return output


def _golden_payload(name: str) -> bytes:
    repository = Path(__file__).resolve().parents[3]
    path = repository / "paper_assets/tables/golden" / name
    _verify_file(path, GOLDEN_IDENTITIES[name], label=f"{name} golden")
    return path.read_bytes()


def _path_entry_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _cleanup_paths(paths: Sequence[Path]) -> list[OSError]:
    errors: list[OSError] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            errors.append(error)
    return errors


def _preflight_output_paths(destinations: Sequence[Path]) -> None:
    if not destinations:
        raise BenchmarkExportError("benchmark output set is empty")
    output = destinations[0].parent
    try:
        output.mkdir(parents=True, exist_ok=True)
        if not output.is_dir():
            raise NotADirectoryError
        for destination in destinations:
            if destination.parent != output:
                raise BenchmarkExportError("benchmark outputs must share one directory")
            if destination.is_dir():
                raise IsADirectoryError
            if (
                _path_entry_exists(destination)
                and not destination.is_file()
                and not destination.is_symlink()
            ):
                raise OSError
    except OSError as error:
        raise BenchmarkExportError("benchmark output path is unusable") from error


def _stage_output(destination: Path, payload: bytes) -> Path:
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.stage.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            staged = Path(handle.name)
            handle.write(payload)
        return staged
    except OSError:
        if staged is not None:
            _cleanup_paths((staged,))
        raise


def _reserve_backup(destination: Path) -> Path:
    backup: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.restore.",
            suffix=".backup",
            delete=False,
        ) as handle:
            backup = Path(handle.name)
        return backup
    except OSError:
        if backup is not None:
            _cleanup_paths((backup,))
        raise


def _write_output_batch(payloads: Mapping[Path, bytes]) -> None:
    destinations = tuple(payloads)
    _preflight_output_paths(destinations)
    staged: dict[Path, Path] = {}
    try:
        for destination, payload in payloads.items():
            staged[destination] = _stage_output(destination, payload)
    except OSError as error:
        _cleanup_paths(tuple(staged.values()))
        raise BenchmarkExportError("benchmark output path is unusable") from error

    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for destination in destinations:
            if not _path_entry_exists(destination):
                continue
            backup = _reserve_backup(destination)
            try:
                destination.replace(backup)
            except OSError:
                _cleanup_paths((backup,))
                raise
            backups[destination] = backup
        for destination in destinations:
            staged[destination].replace(destination)
            committed.append(destination)
    except OSError as error:
        rollback_errors = _cleanup_paths(tuple(reversed(committed)))
        failed_restores: dict[Path, Path] = {}
        for destination in reversed(destinations):
            backup = backups.get(destination)
            if backup is None:
                continue
            try:
                backup.replace(destination)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
                failed_restores[destination] = backup
        rollback_errors.extend(_cleanup_paths(tuple(staged.values())))
        preserved = set(failed_restores.values())
        rollback_errors.extend(
            _cleanup_paths(
                tuple(backup for backup in backups.values() if backup not in preserved)
            )
        )
        if failed_restores:
            recovery_paths = "; ".join(
                sorted(str(path) for path in failed_restores.values())
            )
            message = (
                "benchmark output commit failed and rollback was incomplete; "
                f"original output recovery backup(s) preserved at: {recovery_paths}"
            )
        elif rollback_errors:
            message = "benchmark output commit failed and rollback was incomplete"
        else:
            message = "benchmark output commit failed; previous outputs restored"
        raise BenchmarkExportError(message) from error

    # Every destination is committed at this point. Cleanup is best effort so a
    # stale backup cannot relabel a successful export as a failed transaction.
    _cleanup_paths(tuple(backups.values()))
    _cleanup_paths(tuple(staged.values()))


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(document: Mapping[str, Any]) -> bytes:
    metric_names = [name for name, _metric_path in METRIC_PATHS]
    fields = [
        "candidate_id",
        "display_name",
        "complete_deck_valid",
        "threshold_failure_count",
        "fmts_threshold_failure_count",
        "authority_ids",
        *metric_names,
        "FMTS Econ ratio",
    ]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in document["rows"]:
        writer.writerow(
            {
                "candidate_id": row["candidate_id"],
                "display_name": row["display_name"],
                "complete_deck_valid": str(row["complete_deck_valid"]).lower(),
                "threshold_failure_count": row["threshold_failure_count"],
                "fmts_threshold_failure_count": row["fmts_threshold_failure_count"],
                "authority_ids": json.dumps(
                    row["authority_ids"], sort_keys=True, separators=(",", ":")
                ),
                **{
                    metric: (
                        ""
                        if row["metrics"][metric] is None
                        else repr(row["metrics"][metric])
                    )
                    for metric in metric_names
                },
                "FMTS Econ ratio": repr(row["fmts_metrics"]["Econ ratio"]),
            }
        )
    return handle.getvalue().encode("utf-8")


def _validated_render_rows(
    document: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if document.get("schema_version") != BENCHMARK_SCHEMA:
        raise BenchmarkExportError("benchmark document schema changed")
    if document.get("candidate_order") != {
        "fmts": list(FMTS_CANDIDATES),
        "icaif": list(ICAIF_CANDIDATES),
    }:
        raise BenchmarkExportError("benchmark candidate order changed")
    if document.get("metric_order") != [name for name, _path in METRIC_PATHS]:
        raise BenchmarkExportError("benchmark metric order changed")
    source_rows = document.get("rows")
    if not isinstance(source_rows, list):
        raise BenchmarkExportError("benchmark rows are not a list")

    rows: dict[str, Mapping[str, Any]] = {}
    expected_metrics = {name for name, _path in METRIC_PATHS}
    expected_thresholds = set(TABLE_THRESHOLD_RULES)
    for raw_row in source_rows:
        if not isinstance(raw_row, Mapping):
            raise BenchmarkExportError("benchmark row is not an object")
        candidate = raw_row.get("candidate_id")
        if candidate not in CSV_CANDIDATES or candidate in rows:
            raise BenchmarkExportError("benchmark row identity changed")
        if not isinstance(raw_row.get("complete_deck_valid"), bool):
            raise BenchmarkExportError(f"{candidate}: invalid complete-deck flag")
        for metric_key, status_key, count_key in (
            ("metrics", "thresholds", "threshold_failure_count"),
            ("fmts_metrics", "fmts_thresholds", "fmts_threshold_failure_count"),
        ):
            metrics = raw_row.get(metric_key)
            statuses = raw_row.get(status_key)
            if not isinstance(metrics, Mapping) or set(metrics) != expected_metrics:
                raise BenchmarkExportError(f"{candidate}: metric set changed")
            if (
                not isinstance(statuses, Mapping)
                or set(statuses) != expected_thresholds
            ):
                raise BenchmarkExportError(f"{candidate}: threshold set changed")
            for metric, value in metrics.items():
                if value is None:
                    if raw_row["complete_deck_valid"]:
                        raise BenchmarkExportError(
                            f"{candidate}: valid deck has an undefined {metric}"
                        )
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise BenchmarkExportError(
                        f"{candidate}: {metric} is not a finite number"
                    )
            expected_statuses = _statuses(metrics)
            if dict(statuses) != expected_statuses:
                raise BenchmarkExportError(f"{candidate}: threshold status changed")
            expected_count = sum(
                status != "pass" for status in expected_statuses.values()
            )
            if raw_row.get(count_key) != expected_count:
                raise BenchmarkExportError(f"{candidate}: failure count changed")
        rows[str(candidate)] = raw_row

    if set(rows) != set(CSV_CANDIDATES):
        raise BenchmarkExportError("benchmark row set changed")
    return rows


def _best_candidates(
    rows: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[str],
    metric: str,
    *,
    venue: str,
) -> set[str]:
    metric_key = "fmts_metrics" if venue == "fmts" else "metrics"
    values = {
        candidate: float(rows[candidate][metric_key][metric])
        for candidate in candidates
        if rows[candidate]["complete_deck_valid"]
        and rows[candidate][metric_key][metric] is not None
    }
    if not values:
        return set()
    _heading, _digits, direction, target = METRIC_SPECS[metric]
    if direction == "min":
        distances = {
            candidate: value - min(values.values())
            for candidate, value in values.items()
        }
    elif direction == "max":
        distances = {
            candidate: max(values.values()) - value
            for candidate, value in values.items()
        }
    else:
        distances = {
            candidate: abs(value - float(target)) for candidate, value in values.items()
        }
    best_distance = min(distances.values())
    return {
        candidate
        for candidate, distance in distances.items()
        if math.isclose(distance, best_distance, rel_tol=1e-12, abs_tol=1e-12)
    }


def _format_metric_cell(
    row: Mapping[str, Any],
    metric: str,
    *,
    venue: str,
    best: bool,
    short: bool = False,
) -> str:
    metric_key = "fmts_metrics" if venue == "fmts" else "metrics"
    status_key = "fmts_thresholds" if venue == "fmts" else "thresholds"
    value = row[metric_key][metric]
    if value is None:
        undefined = r"\texttt{undef.}" if short else "undef."
        return rf"\textcolor{{red}}{{{undefined}}}"
    _heading, digits, _direction, _target = METRIC_SPECS[metric]
    text = f"{float(value):.{digits}f}"
    if best:
        text = rf"\textbf{{{text}}}"
    if row[status_key].get(metric) == "fail":
        text = rf"\textcolor{{red}}{{{text}}}"
    return text


def _render_full_rows(
    rows: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[str],
    metrics: Sequence[str],
    *,
    venue: str,
) -> str:
    best = {
        metric: _best_candidates(rows, candidates, metric, venue=venue)
        for metric in metrics
    }
    lines = []
    for candidate in candidates:
        cells = [
            _format_metric_cell(
                rows[candidate],
                metric,
                venue=venue,
                best=candidate in best[metric],
            )
            for metric in metrics
        ]
        lines.append(f"{FULL_LABELS[candidate]} & {' & '.join(cells)} " + r"\\")
    return "\n".join(lines)


def _render_short_rows(rows: Mapping[str, Mapping[str, Any]]) -> str:
    best = {
        metric: _best_candidates(rows, FMTS_CANDIDATES, metric, venue="fmts")
        for metric in SHORT_METRICS
    }
    rendered_rows: list[str] = []
    for candidate in FMTS_CANDIDATES:
        row = rows[candidate]
        cells = [
            _format_metric_cell(
                row,
                metric,
                venue="fmts",
                best=candidate in best[metric],
                short=True,
            )
            for metric in SHORT_METRICS
        ]
        failures = int(row["fmts_threshold_failure_count"])
        cells.append(
            rf"\PassCount{{{failures}}}"
            if failures == 0
            else rf"\FailCount{{{failures}}}"
        )
        groups = SHORT_ROW_GROUPS[candidate]
        if sum(groups) != len(cells):
            raise BenchmarkExportError(f"{candidate}: short-row layout changed")
        offset = 0
        physical_lines: list[str] = []
        for group_index, group_size in enumerate(groups):
            group = cells[offset : offset + group_size]
            prefix = f"{SHORT_LABELS[candidate]} & " if group_index == 0 else "& "
            suffix = " " + r"\\" if group_index == len(groups) - 1 else ""
            physical_lines.append(prefix + " & ".join(group) + suffix)
            offset += group_size
        rendered_rows.append("\n".join(physical_lines))
    return "\n".join(rendered_rows)


def _replace_row_blocks(template: str, bodies: Sequence[str]) -> str:
    marker = "\n".join((r"\midrule", r"\scenarioflow{}"))
    if template.count(marker) != len(bodies):
        raise BenchmarkExportError("golden table row markers changed")
    pieces: list[str] = []
    cursor = 0
    for body in bodies:
        marker_start = template.index(marker, cursor)
        row_start = marker_start + len(r"\midrule" + "\n")
        row_end = template.index(r"\bottomrule", row_start)
        pieces.append(template[cursor:row_start])
        pieces.append(body + "\n")
        cursor = row_end
    pieces.append(template[cursor:])
    return "".join(pieces)


def render_benchmark_fragments(document: Mapping[str, Any]) -> dict[str, str]:
    """Render numerical rows into the three verified paper table layouts."""
    rows = _validated_render_rows(document)
    full_template = _golden_payload("fmts-benchmark-full.tex").decode("utf-8")
    short_template = _golden_payload("fmts-benchmark-short.tex").decode("utf-8")
    icaif_template = _golden_payload("icaif-benchmark.tex").decode("utf-8")
    return {
        "fmts-benchmark-full.tex": _replace_row_blocks(
            full_template,
            (
                _render_full_rows(
                    rows,
                    FMTS_CANDIDATES,
                    PANEL_A_METRICS,
                    venue="fmts",
                ),
                _render_full_rows(
                    rows,
                    FMTS_CANDIDATES,
                    PANEL_B_METRICS,
                    venue="fmts",
                ),
            ),
        ),
        "fmts-benchmark-short.tex": _replace_row_blocks(
            short_template, (_render_short_rows(rows),)
        ),
        "icaif-benchmark.tex": _replace_row_blocks(
            icaif_template,
            (
                _render_full_rows(
                    rows,
                    ICAIF_CANDIDATES,
                    PANEL_A_METRICS,
                    venue="icaif",
                ),
                _render_full_rows(
                    rows,
                    ICAIF_CANDIDATES,
                    PANEL_B_METRICS,
                    venue="icaif",
                ),
            ),
        ),
    }


def _benchmark_export_payloads(
    authorities: Mapping[str, Path], output_dir: Path
) -> tuple[dict[str, Path], dict[Path, bytes]]:
    """Validate authorities and prepare the complete benchmark output set."""
    try:
        output = _assert_output_is_safe(output_dir)
    except (OSError, RuntimeError) as error:
        raise BenchmarkExportError("benchmark output path is unusable") from error
    verified = _verified_authorities(authorities)
    document = _build_document(verified)
    paths = {
        "benchmark_json": output / "benchmark.json",
        "benchmark_csv": output / "benchmark.csv",
        "fmts_full_tex": output / "fmts-benchmark-full.tex",
        "fmts_short_tex": output / "fmts-benchmark-short.tex",
        "icaif_tex": output / "icaif-benchmark.tex",
    }
    fragments = render_benchmark_fragments(document)
    payloads = {
        paths["benchmark_json"]: _json_bytes(document),
        paths["benchmark_csv"]: _csv_bytes(document),
    }
    for output_key, golden_name in (
        ("fmts_full_tex", "fmts-benchmark-full.tex"),
        ("fmts_short_tex", "fmts-benchmark-short.tex"),
        ("icaif_tex", "icaif-benchmark.tex"),
    ):
        payload = fragments[golden_name].encode("utf-8")
        if payload != _golden_payload(golden_name):
            raise BenchmarkExportError(
                f"rendered {golden_name} drifted from its verified golden"
            )
        payloads[paths[output_key]] = payload
    return paths, payloads


def export_benchmark_tables(
    authorities: Mapping[str, Path], output_dir: Path
) -> dict[str, Path]:
    """Validate frozen result authorities and export reader-facing table views."""
    paths, payloads = _benchmark_export_payloads(authorities, output_dir)
    _write_output_batch(payloads)
    return paths


def _parser(*, prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "python -m scenarioflow.paper.tables",
) -> int:
    parser = _parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        root = args.authority.expanduser().resolve().parent.parent
        authorities = authority_paths_from_manifest(args.authority, root=root)
        paths, payloads = _benchmark_export_payloads(authorities, args.output)
        multiseed_path = paths["benchmark_json"].parent / "fmts-multiseed.tex"
        paths["fmts_multiseed_tex"] = multiseed_path
        payloads[multiseed_path] = render_multiseed_table(
            authorities[MULTISEED_ID]
        ).encode("utf-8")
        _write_output_batch(payloads)
    except (BenchmarkExportError, MultiseedExportError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {name: str(path) for name, path in sorted(paths.items())},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
