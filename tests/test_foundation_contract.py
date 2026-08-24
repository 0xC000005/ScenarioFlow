from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scenarioflow.baselines.foundation import (
    EXPECTED_STATE_SPECIFICATION_HASH,
    MOIRAI_REVISION,
    TOTO_REVISION,
    FoundationArtifactAuthority,
    PrecomputedJointScenarioBaseline,
    load_foundation_artifact,
    write_foundation_artifact,
)
from scenarioflow.evaluation.current_panel import create_baseline
from scenarioflow.workflows.foundation import foundation_subprocess_command


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _sealed_fixture(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, FoundationArtifactAuthority]:
    histories = np.arange(2 * 30 * 39, dtype=np.float32).reshape(2, 30, 39)
    samples = np.arange(2 * 3 * 30 * 39, dtype=np.float32).reshape(2, 3, 30, 39)
    write_foundation_artifact(
        path,
        run_kind="smoke",
        model_name="moirai_1_1_finetuned",
        model_revision=MOIRAI_REVISION,
        source_revision=_digest("source"),
        checkpoint_sha256=_digest("checkpoint"),
        model_config_sha256=_digest("config"),
        panel_manifest_sha256=_digest("panel"),
        license_id="Apache-2.0",
        train_seed=1001,
        rollout_seed=2001,
        state_specification_hash=EXPECTED_STATE_SPECIFICATION_HASH,
        history_increment=histories,
        sample_increment=samples,
    )
    authority = FoundationArtifactAuthority(
        manifest_sha256=hashlib.sha256(
            (path / "manifest.json").read_bytes()
        ).hexdigest(),
        model_name="moirai_1_1_finetuned",
        model_revision=MOIRAI_REVISION,
        source_revision=_digest("source"),
        train_seed=1001,
        rollout_seed=2001,
        checkpoint_sha256=_digest("checkpoint"),
        model_config_sha256=_digest("config"),
        panel_manifest_sha256=_digest("panel"),
        expected_window_count=2,
        expected_scenario_count=3,
    )
    return histories, samples, authority


def test_sealed_foundation_deck_hashes_shapes_identity_and_seed(tmp_path: Path) -> None:
    artifact = tmp_path / "deck"
    histories, samples, authority = _sealed_fixture(artifact)
    loaded_history, loaded_samples, manifest = load_foundation_artifact(
        artifact,
        expected_manifest_sha256=authority.manifest_sha256,
        expected_model_name="moirai_1_1_finetuned",
        expected_model_revision=MOIRAI_REVISION,
        expected_source_revision=_digest("source"),
        expected_train_seed=1001,
        expected_rollout_seed=2001,
        expected_checkpoint_sha256=_digest("checkpoint"),
        expected_model_config_sha256=_digest("config"),
        expected_panel_manifest_sha256=_digest("panel"),
        expected_window_count=2,
        expected_scenario_count=3,
    )
    np.testing.assert_array_equal(loaded_history, histories)
    np.testing.assert_array_equal(loaded_samples, samples)
    assert manifest["model_revision"] == MOIRAI_REVISION

    baseline = PrecomputedJointScenarioBaseline.from_authority(artifact, authority)
    np.testing.assert_array_equal(
        baseline.sample_joint(histories[[1]], n_samples=3), samples[[1]]
    )


def test_sealed_foundation_deck_rejects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "deck"
    _, _, authority = _sealed_fixture(artifact)
    samples = np.load(artifact / "samples.npy", allow_pickle=False)
    samples[0, 0, 0, 0] += 1
    np.save(artifact / "samples.npy", samples)
    with pytest.raises(RuntimeError, match="sample_sha256"):
        PrecomputedJointScenarioBaseline.from_authority(artifact, authority)


def test_foundation_manifest_cannot_self_authorize_and_factory_requires_authority(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "deck"
    histories, _, authority = _sealed_fixture(artifact)
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["license_id"] = "changed"
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="external authority"):
        PrecomputedJointScenarioBaseline.from_authority(artifact, authority)
    with pytest.raises(ValueError, match="external foundation authority"):
        create_baseline(
            "moirai_1_1_finetuned",
            histories[:, -1],
            foundation_artifact=artifact,
        )


def test_pinned_revisions_and_missing_isolated_environment_are_actionable(
    tmp_path: Path,
) -> None:
    assert MOIRAI_REVISION == "cfd46d4510ed8896f263116f32928eede05b0a75"
    assert TOTO_REVISION == "46bd92adeeef6b3c2afb21008659f607fa203e38"
    with pytest.raises(FileNotFoundError, match="uv sync"):
        foundation_subprocess_command(
            model="moirai",
            environment_dir=tmp_path / "missing",
            backend_module="provider.runner",
            backend_args=("--model-root", str(tmp_path / "models")),
        )
