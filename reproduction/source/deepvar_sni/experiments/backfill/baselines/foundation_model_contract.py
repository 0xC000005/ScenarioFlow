"""Sealed native-sample artifacts for external foundation-model baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np


SCHEMA_VERSION = "foundation-current-panel-v1"
EXPECTED_VARIABLES = 39
EXPECTED_HISTORY = 30
EXPECTED_HORIZON = 30
EXPECTED_EVALUATION_WINDOWS = 441
EXPECTED_SCENARIOS = 64
EXPECTED_STATE_SPECIFICATION_HASH = (
    "65540401c58143dc51a2c907916c9b4903b484c3c3ad0ba9ce1c062b96fc6076"
)

_RUN_KINDS = frozenset({"smoke", "canonical", "seed_sensitivity", "precision_control"})
_KNOWN_MODEL_NAMES = frozenset(
    {
        "moirai_1_1_zero_shot",
        "moirai_1_1_finetuned",
        "toto_1_0_zero_shot",
        "toto_1_0_finetuned",
    }
)
_SEALED_FILENAMES = frozenset({"histories.npy", "samples.npy", "manifest.json"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_kind",
        "model_name",
        "model_revision",
        "source_revision",
        "checkpoint_sha256",
        "model_config_sha256",
        "panel_manifest_sha256",
        "license_id",
        "train_seed",
        "rollout_seed",
        "state_specification_hash",
        "history_shape",
        "sample_shape",
        "history_sha256",
        "sample_sha256",
        "run_metadata",
    }
)
_HEX_DIGEST_LENGTH = 64


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _HEX_DIGEST_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_known_model_name(value: object) -> bool:
    return isinstance(value, str) and value in _KNOWN_MODEL_NAMES


def _validate_array_inputs(
    history_increment: np.ndarray,
    sample_increment: np.ndarray,
    run_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(history_increment, np.ndarray) or not isinstance(
        sample_increment, np.ndarray
    ):
        raise ValueError("history_increment and sample_increment must be numpy arrays")
    if history_increment.dtype != np.float32 or sample_increment.dtype != np.float32:
        raise ValueError("history_increment and sample_increment must be float32")
    if history_increment.ndim != 3 or history_increment.shape[1:] != (
        EXPECTED_HISTORY,
        EXPECTED_VARIABLES,
    ):
        raise ValueError("history_increment shape must be (windows, 30, 39)")
    if sample_increment.ndim != 4 or sample_increment.shape[2:] != (
        EXPECTED_HORIZON,
        EXPECTED_VARIABLES,
    ):
        raise ValueError("sample_increment shape must be (windows, scenarios, 30, 39)")
    if history_increment.shape[0] == 0 or sample_increment.shape[1] == 0:
        raise ValueError("history_increment and sample_increment must not be empty")
    if history_increment.shape[0] != sample_increment.shape[0]:
        raise ValueError("history_increment and sample_increment window counts differ")
    if (
        not np.isfinite(history_increment).all()
        or not np.isfinite(sample_increment).all()
    ):
        raise ValueError("history_increment and sample_increment must be finite")
    if run_kind != "smoke" and (
        history_increment.shape[0] != EXPECTED_EVALUATION_WINDOWS
        or sample_increment.shape[1] != EXPECTED_SCENARIOS
    ):
        raise ValueError(
            "non-smoke artifacts must use canonical window and scenario counts"
        )

    histories = np.ascontiguousarray(history_increment)
    samples = np.ascontiguousarray(sample_increment)
    history_hashes = [_array_sha256(row) for row in histories]
    if len(set(history_hashes)) != len(history_hashes):
        raise ValueError("duplicate history hashes are not allowed")
    return histories, samples


def _artifact_manifest(
    *,
    run_kind: str,
    model_name: str,
    model_revision: str,
    source_revision: str,
    checkpoint_sha256: str,
    model_config_sha256: str,
    panel_manifest_sha256: str,
    license_id: str,
    train_seed: int,
    rollout_seed: int,
    state_specification_hash: str,
    histories: np.ndarray,
    samples: np.ndarray,
    run_metadata: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_kind": run_kind,
        "model_name": model_name,
        "model_revision": model_revision,
        "source_revision": source_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "model_config_sha256": model_config_sha256,
        "panel_manifest_sha256": panel_manifest_sha256,
        "license_id": license_id,
        "train_seed": train_seed,
        "rollout_seed": rollout_seed,
        "state_specification_hash": state_specification_hash,
        "history_shape": list(histories.shape),
        "sample_shape": list(samples.shape),
        "history_sha256": _array_sha256(histories),
        "sample_sha256": _array_sha256(samples),
        "run_metadata": run_metadata or {},
    }


def _read_existing_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "existing output directory has no readable manifest"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError("existing output directory manifest is not an object")
    return value


def _existing_artifact_matches(
    path: Path, expected_manifest: dict[str, object]
) -> bool:
    try:
        if {entry.name for entry in path.iterdir()} != _SEALED_FILENAMES:
            return False
        history_shape = expected_manifest["history_shape"]
        sample_shape = expected_manifest["sample_shape"]
        if not isinstance(history_shape, list) or not isinstance(sample_shape, list):
            return False
        histories, samples, manifest = load_foundation_artifact(
            path,
            expected_model_name=expected_manifest["model_name"],
            expected_model_revision=expected_manifest["model_revision"],
            expected_source_revision=expected_manifest["source_revision"],
            expected_train_seed=expected_manifest["train_seed"],
            expected_rollout_seed=expected_manifest["rollout_seed"],
            expected_checkpoint_sha256=expected_manifest["checkpoint_sha256"],
            expected_model_config_sha256=expected_manifest["model_config_sha256"],
            expected_panel_manifest_sha256=expected_manifest["panel_manifest_sha256"],
            expected_state_specification_hash=expected_manifest[
                "state_specification_hash"
            ],
            expected_window_count=history_shape[0],
            expected_scenario_count=sample_shape[1],
        )
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return (
        manifest == expected_manifest
        and _array_sha256(histories) == expected_manifest["history_sha256"]
        and _array_sha256(samples) == expected_manifest["sample_sha256"]
    )


def write_foundation_artifact(
    output_dir: str | Path,
    *,
    run_kind: Literal["smoke", "canonical", "seed_sensitivity", "precision_control"],
    model_name: str,
    model_revision: str,
    source_revision: str,
    checkpoint_sha256: str,
    model_config_sha256: str,
    panel_manifest_sha256: str,
    license_id: str,
    train_seed: int,
    rollout_seed: int,
    state_specification_hash: str,
    history_increment: np.ndarray,
    sample_increment: np.ndarray,
    run_metadata: dict[str, object] | None = None,
) -> Path:
    """Write float32 histories/samples plus a hash-bound JSON manifest."""
    if run_kind not in _RUN_KINDS:
        raise ValueError("run_kind is not supported")
    if not _is_known_model_name(model_name):
        raise ValueError("model_name is not a known Moirai or Toto model")
    if not all(
        isinstance(value, str) and value
        for value in (model_revision, source_revision, license_id)
    ):
        raise ValueError(
            "model revision, source revision, and license_id must be non-empty"
        )
    for field, value in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("model_config_sha256", model_config_sha256),
        ("panel_manifest_sha256", panel_manifest_sha256),
        ("state_specification_hash", state_specification_hash),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    if state_specification_hash != EXPECTED_STATE_SPECIFICATION_HASH:
        raise ValueError(
            "state_specification_hash does not match the frozen state specification"
        )
    if (
        not isinstance(train_seed, int)
        or isinstance(train_seed, bool)
        or not isinstance(rollout_seed, int)
        or isinstance(rollout_seed, bool)
    ):
        raise ValueError("train_seed and rollout_seed must be integers")
    if run_metadata is not None and not isinstance(run_metadata, dict):
        raise ValueError("run_metadata must be a dictionary")

    histories, samples = _validate_array_inputs(
        history_increment, sample_increment, run_kind
    )
    manifest = _artifact_manifest(
        run_kind=run_kind,
        model_name=model_name,
        model_revision=model_revision,
        source_revision=source_revision,
        checkpoint_sha256=checkpoint_sha256,
        model_config_sha256=model_config_sha256,
        panel_manifest_sha256=panel_manifest_sha256,
        license_id=license_id,
        train_seed=train_seed,
        rollout_seed=rollout_seed,
        state_specification_hash=state_specification_hash,
        histories=histories,
        samples=samples,
        run_metadata=run_metadata,
    )
    try:
        manifest_text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError("run_metadata must be JSON serializable") from error

    artifact_dir = Path(output_dir)
    if artifact_dir.exists():
        if artifact_dir.is_dir() and _existing_artifact_matches(artifact_dir, manifest):
            return artifact_dir
        if not artifact_dir.is_dir() or any(artifact_dir.iterdir()):
            raise RuntimeError(
                "existing output directory does not match requested artifact"
            )
    else:
        artifact_dir.mkdir(parents=True)
    np.save(artifact_dir / "histories.npy", histories)
    np.save(artifact_dir / "samples.npy", samples)
    (artifact_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    return artifact_dir


def _require_manifest_value(manifest: dict[str, object], field: str) -> object:
    if field not in manifest:
        raise RuntimeError(f"manifest missing {field}")
    return manifest[field]


def _require_equal(manifest: dict[str, object], field: str, expected: object) -> None:
    if _require_manifest_value(manifest, field) != expected:
        raise RuntimeError(f"{field} does not match expected artifact")


def _validate_loaded_arrays(
    histories: np.ndarray,
    samples: np.ndarray,
    manifest: dict[str, object],
    expected_window_count: int,
    expected_scenario_count: int,
) -> None:
    if histories.dtype != np.float32 or samples.dtype != np.float32:
        raise RuntimeError("stored histories and samples must be float32")
    history_shape = (expected_window_count, EXPECTED_HISTORY, EXPECTED_VARIABLES)
    sample_shape = (
        expected_window_count,
        expected_scenario_count,
        EXPECTED_HORIZON,
        EXPECTED_VARIABLES,
    )
    if histories.shape != history_shape:
        raise RuntimeError("history_shape does not match expected artifact")
    if samples.shape != sample_shape:
        raise RuntimeError("sample_shape does not match expected artifact")
    if _require_manifest_value(manifest, "history_shape") != list(history_shape):
        raise RuntimeError("history_shape does not match stored histories")
    if _require_manifest_value(manifest, "sample_shape") != list(sample_shape):
        raise RuntimeError("sample_shape does not match stored samples")
    if not np.isfinite(histories).all() or not np.isfinite(samples).all():
        raise RuntimeError("stored histories and samples must be finite")
    if _require_manifest_value(manifest, "history_sha256") != _array_sha256(histories):
        raise RuntimeError("history_sha256 does not match stored histories")
    if _require_manifest_value(manifest, "sample_sha256") != _array_sha256(samples):
        raise RuntimeError("sample_sha256 does not match stored samples")
    hashes = [_array_sha256(row) for row in histories]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("duplicate history hashes are not allowed")


def load_foundation_artifact(
    artifact_dir: str | Path,
    *,
    expected_model_name: str,
    expected_model_revision: str,
    expected_source_revision: str,
    expected_train_seed: int,
    expected_rollout_seed: int,
    expected_checkpoint_sha256: str,
    expected_model_config_sha256: str,
    expected_panel_manifest_sha256: str,
    expected_state_specification_hash: str = EXPECTED_STATE_SPECIFICATION_HASH,
    expected_window_count: int = EXPECTED_EVALUATION_WINDOWS,
    expected_scenario_count: int = EXPECTED_SCENARIOS,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Validate schema, shapes, hashes, finiteness, identity, and seed before loading."""
    if expected_window_count <= 0 or expected_scenario_count <= 0:
        raise RuntimeError("expected window and scenario counts must be positive")
    path = Path(artifact_dir)
    manifest = _read_existing_manifest(path / "manifest.json")
    if set(manifest) != _MANIFEST_FIELDS:
        raise RuntimeError("manifest fields do not match the sealed schema")
    _require_equal(manifest, "schema_version", SCHEMA_VERSION)
    run_kind = _require_manifest_value(manifest, "run_kind")
    if run_kind not in _RUN_KINDS:
        raise RuntimeError("run_kind is not supported")
    if run_kind != "smoke" and (
        expected_window_count != EXPECTED_EVALUATION_WINDOWS
        or expected_scenario_count != EXPECTED_SCENARIOS
    ):
        raise RuntimeError("reduced counts are permitted only for smoke artifacts")
    model_name = _require_manifest_value(manifest, "model_name")
    if not _is_known_model_name(model_name):
        raise RuntimeError("model_name is not a known Moirai or Toto model")
    _require_equal(manifest, "model_name", expected_model_name)
    _require_equal(manifest, "model_revision", expected_model_revision)
    _require_equal(manifest, "source_revision", expected_source_revision)
    _require_equal(manifest, "train_seed", expected_train_seed)
    _require_equal(manifest, "rollout_seed", expected_rollout_seed)
    _require_equal(manifest, "checkpoint_sha256", expected_checkpoint_sha256)
    _require_equal(manifest, "model_config_sha256", expected_model_config_sha256)
    _require_equal(manifest, "panel_manifest_sha256", expected_panel_manifest_sha256)
    if expected_state_specification_hash != EXPECTED_STATE_SPECIFICATION_HASH:
        raise RuntimeError(
            "expected state_specification_hash does not match the frozen state specification"
        )
    _require_equal(
        manifest, "state_specification_hash", EXPECTED_STATE_SPECIFICATION_HASH
    )
    for field in (
        "checkpoint_sha256",
        "model_config_sha256",
        "panel_manifest_sha256",
        "state_specification_hash",
    ):
        if not _is_sha256(_require_manifest_value(manifest, field)):
            raise RuntimeError(f"{field} is not a SHA-256 digest")
    if not isinstance(
        _require_manifest_value(manifest, "train_seed"), int
    ) or isinstance(manifest["train_seed"], bool):
        raise RuntimeError("train_seed must be an integer")
    if not isinstance(
        _require_manifest_value(manifest, "rollout_seed"), int
    ) or isinstance(manifest["rollout_seed"], bool):
        raise RuntimeError("rollout_seed must be an integer")
    license_id = _require_manifest_value(manifest, "license_id")
    if not isinstance(license_id, str) or not license_id:
        raise RuntimeError("license_id must be a non-empty string")
    if not isinstance(_require_manifest_value(manifest, "run_metadata"), dict):
        raise RuntimeError("run_metadata must be an object")
    try:
        histories = np.load(path / "histories.npy", allow_pickle=False)
        samples = np.load(path / "samples.npy", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError("unable to load sealed histories or samples") from error
    _validate_loaded_arrays(
        histories,
        samples,
        manifest,
        expected_window_count,
        expected_scenario_count,
    )
    return histories, samples, manifest


class PrecomputedJointScenarioBaseline:
    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        expected_model_name: str,
        expected_model_revision: str,
        expected_source_revision: str,
        expected_train_seed: int,
        expected_rollout_seed: int,
        expected_checkpoint_sha256: str,
        expected_model_config_sha256: str,
        expected_panel_manifest_sha256: str,
        expected_window_count: int = EXPECTED_EVALUATION_WINDOWS,
        expected_scenario_count: int = EXPECTED_SCENARIOS,
    ) -> None:
        histories, samples, manifest = load_foundation_artifact(
            artifact_dir,
            expected_model_name=expected_model_name,
            expected_model_revision=expected_model_revision,
            expected_source_revision=expected_source_revision,
            expected_train_seed=expected_train_seed,
            expected_rollout_seed=expected_rollout_seed,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_model_config_sha256=expected_model_config_sha256,
            expected_panel_manifest_sha256=expected_panel_manifest_sha256,
            expected_window_count=expected_window_count,
            expected_scenario_count=expected_scenario_count,
        )
        self._samples = samples
        self._manifest = manifest
        self._row_by_history_hash = {
            hashlib.sha256(row.tobytes()).hexdigest(): i
            for i, row in enumerate(histories)
        }

    def sample_joint(
        self,
        history_changes: np.ndarray,
        *,
        n_samples: int,
        **_: object,
    ) -> np.ndarray:
        if not isinstance(history_changes, np.ndarray):
            raise RuntimeError("history_changes must be a numpy float32 array")
        if history_changes.dtype != np.float32:
            raise RuntimeError("history_changes must be float32")
        if history_changes.ndim != 3 or history_changes.shape[1:] != (
            EXPECTED_HISTORY,
            EXPECTED_VARIABLES,
        ):
            raise RuntimeError("history_changes shape must be (batch, 30, 39)")
        if history_changes.shape[0] == 0:
            raise RuntimeError("history_changes batch must not be empty")
        if not np.isfinite(history_changes).all():
            raise RuntimeError("history_changes must be finite")
        history = np.ascontiguousarray(history_changes)
        if n_samples != self._samples.shape[1]:
            raise RuntimeError("requested sample count does not match sealed artifact")
        rows = []
        for item in history:
            digest = hashlib.sha256(item.tobytes()).hexdigest()
            if digest not in self._row_by_history_hash:
                raise RuntimeError("incoming history is absent from sealed artifact")
            rows.append(self._row_by_history_hash[digest])
        return self._samples[np.asarray(rows, dtype=np.int64)].copy()
