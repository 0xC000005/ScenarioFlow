"""Frozen numerical contract for the legacy ScenarioFlow reproduction routes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scenarioflow.reproduction.finetune_rollouts import stage2_optimizer_update_hash
from scenarioflow.reproduction.inference import (
    generate_deck,
    load_checkpoint_strict,
    velocity_output,
)
from scenarioflow.reproduction.losses import (
    channel_energy_loss,
    horizon_path_weights,
    joint_energy_loss,
)
from scenarioflow.reproduction.metrics import evaluate_deck
from scenarioflow.reproduction.panel import (
    prepare_panel_blocks,
    select_normalized_innovation_scope,
)
from scenarioflow.reproduction.train_next_step import (
    stage1_loss,
    stage1_optimizer_update_hash,
)
from scenarioflow.state import encode_state, reconstruct_state_from_increments

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests/fixtures/reproduction"
JSON_FIXTURE = FIXTURE_DIR / "golden-v1.json"
NPZ_FIXTURE = FIXTURE_DIR / "golden-v1.npz"
ARTIFACT_MANIFEST = ROOT / "artifacts/manifest-v1.tsv"

REQUIRED_ARRAY_SHAPES = {
    "raw_history": (2, 30, 39),
    "raw_future": (2, 30, 39),
    "state_encoding": (2, 60, 39),
    "history_normalized": (2, 30, 39),
    "future_normalized": (2, 30, 39),
    "normalization_center": (2, 39),
    "normalization_scale": (2, 39),
    "reconstructed_future": (2, 30, 39),
    "stage1_loss": (1,),
    "joint_energy_loss": (3,),
    "channel_energy_loss": (3,),
    "core_velocity": (2, 39),
    "graph_velocity": (2, 39),
    "rollout": (2, 2, 30, 39),
    "parameter_update_hashes": (2, 32),
    "metrics": (8,),
    "screen_decisions": (4,),
}
CANONICAL_NPZ_SHA256 = (
    "8c826d8826e158f29643a3aad8e1586fc6e2e409b622cdf5f107515d77d5628f"
)
CANONICAL_ARRAY_SHA256 = {
    "channel_energy_loss": "858e2e420805f40cee98b9cad182d97e63b4a473e405e58c334cc42297c6ff73",
    "core_velocity": "8bd596e3f931bdbc637df9e7ba8a60f6a02db2011bfa208baf17a5887fb565a2",
    "future_normalized": "89b577bc550add36bc3821ed32df56de808b4a0c13b87776c8867bb1f46f0a4f",
    "graph_velocity": "2e094658a8e8fae32f4c61bdd141f9dfe0a34907c4ada6742641afe55288bc58",
    "history_normalized": "f3eeb3328dc095cdae9d8c6cbd32ba8cf4bc47108436dcb033bd9bb242730eec",
    "joint_energy_loss": "ce194770f1390d034b50d15e4411f510a53f66fd156c51214428920d8a61e256",
    "metrics": "df2785eb8149730c43b12baabee6ce92759bda91241cd0270c49c00eb4bfe271",
    "normalization_center": "6287e17fe5efcb191aec8a2578e685d0a1040a0ab53bd1f02c7f49cc6c8fced7",
    "normalization_scale": "48812857b105f7b69e707310e5f323eb148b3dd881e1870a691d73ea5066c3ff",
    "parameter_update_hashes": "9df12ec968c8f429760d5c6218c832051dfe4fccfc125f10aca6228f8c02c108",
    "raw_future": "ed74af340948979c0d65a83f02fb76933ecbf18312b83f641aec7b2e3ccf0eb0",
    "raw_history": "b407293b8d3c5105da4387aaadb941b4d8910cb38f16c6b4c317a9fbe0cb85d6",
    "reconstructed_future": "7136176752b31db3741510a84a4efb6b71fbe49b0e17072c4cd9ea547e879de3",
    "rollout": "e37cd660f586bc49922c1696b02e8358fa8399f69deb14539804e61691f46c3a",
    "screen_decisions": "67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450",
    "stage1_loss": "c031d406e9145d4454124d5e8e48fc5256f9102d391744c09c17a995c0bfd1bb",
    "state_encoding": "25f0544ddf4382f2712e55ceccaf64f322eda5efcb6ad622a2e13074efca9465",
}
BYTE_ARRAY_DTYPES = {
    "parameter_update_hashes": np.dtype(np.uint8),
    "screen_decisions": np.dtype(np.int8),
}
SELECTED_PARAMETERS = (
    "feature_proj.weight",
    "memory.layers.0.self_attn.in_proj_weight",
    "velocity.out.3.weight",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_golden() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load the immutable JSON metadata and compact numerical fixture arrays."""
    assert JSON_FIXTURE.is_file(), f"missing golden metadata: {JSON_FIXTURE}"
    assert NPZ_FIXTURE.is_file(), f"missing golden arrays: {NPZ_FIXTURE}"
    metadata: dict[str, object] = json.loads(JSON_FIXTURE.read_text(encoding="utf-8"))
    with np.load(NPZ_FIXTURE, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return metadata, arrays


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _install_replacement_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    array_name: str,
    replacement: np.ndarray,
) -> None:
    metadata, arrays = load_golden()
    arrays[array_name] = replacement
    npz_path = tmp_path / "golden-v1.npz"
    np.savez(npz_path, **arrays)
    metadata["npz_sha256"] = _sha256(npz_path)
    metadata["arrays"] = {
        name: {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        }
        for name, values in arrays.items()
    }
    json_path = tmp_path / "golden-v1.json"
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setitem(globals(), "JSON_FIXTURE", json_path)
    monkeypatch.setitem(globals(), "NPZ_FIXTURE", npz_path)


def test_contract_rejects_self_consistent_replacement_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed values cannot become canonical by editing both fixture files."""
    _metadata, arrays = load_golden()
    replacement = arrays["stage1_loss"].copy()
    replacement[0] = np.nextafter(replacement[0], np.float32(np.inf))
    _install_replacement_fixture(
        tmp_path,
        monkeypatch,
        array_name="stage1_loss",
        replacement=replacement,
    )

    with pytest.raises(AssertionError, match="canonical array hash"):
        test_frozen_numerical_parity_contract_is_complete_and_self_authenticating()


def test_contract_rejects_float64_numerical_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protocol float32 means fixture metadata cannot bless float64 arrays."""
    _metadata, arrays = load_golden()
    _install_replacement_fixture(
        tmp_path,
        monkeypatch,
        array_name="stage1_loss",
        replacement=arrays["stage1_loss"].astype(np.float64),
    )

    with pytest.raises(AssertionError, match="float32"):
        test_frozen_numerical_parity_contract_is_complete_and_self_authenticating()


def test_contract_rejects_reencoded_npz_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editable metadata cannot bless a different container with identical arrays."""
    metadata, arrays = load_golden()
    npz_path = tmp_path / "golden-v1.npz"
    np.savez_compressed(npz_path, **arrays)
    metadata["npz_sha256"] = _sha256(npz_path)
    json_path = tmp_path / "golden-v1.json"
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setitem(globals(), "JSON_FIXTURE", json_path)
    monkeypatch.setitem(globals(), "NPZ_FIXTURE", npz_path)

    with pytest.raises(AssertionError, match="canonical NPZ hash"):
        test_frozen_numerical_parity_contract_is_complete_and_self_authenticating()


def test_frozen_numerical_parity_contract_is_complete_and_self_authenticating() -> None:
    """Dropping or changing any parity surface must fail before legacy deletion."""
    metadata, arrays = load_golden()

    assert metadata["schema_version"] == "scenarioflow.reproduction.golden.v1"
    assert metadata["protocol"] == {
        "device": "cpu",
        "dtype": "float32",
        "training_seed": 1001,
        "sampling_seed": 2001,
        "validation_window_indices": [4010, 4011],
        "validation_windows": 2,
        "generated_paths": 2,
        "forecast_steps": 30,
    }
    assert metadata["npz_sha256"] == _sha256(NPZ_FIXTURE)

    assert set(arrays) == set(REQUIRED_ARRAY_SHAPES)
    for name, shape in REQUIRED_ARRAY_SHAPES.items():
        assert arrays[name].shape == shape, name
        assert np.isfinite(arrays[name]).all(), name
        expected_dtype = BYTE_ARRAY_DTYPES.get(name, np.dtype(np.float32))
        assert (
            arrays[name].dtype == expected_dtype
        ), f"{name} must stay {expected_dtype.name}"
        actual_hash = hashlib.sha256(arrays[name].tobytes(order="C")).hexdigest()
        assert (
            actual_hash == CANONICAL_ARRAY_SHA256[name]
        ), f"{name} canonical array hash changed"
    assert _sha256(NPZ_FIXTURE) == CANONICAL_NPZ_SHA256, "canonical NPZ hash changed"
    assert np.all(arrays["normalization_scale"] > 0.0)
    np.testing.assert_allclose(
        arrays["reconstructed_future"], arrays["raw_future"], rtol=0.0, atol=2e-5
    )

    array_metadata = metadata["arrays"]
    assert isinstance(array_metadata, dict)
    assert set(array_metadata) == set(arrays)
    for name, values in arrays.items():
        record = array_metadata[name]
        assert isinstance(record, dict)
        assert record == {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        }

    manifest = {row["artifact_id"]: row for row in _manifest_rows(ARTIFACT_MANIFEST)}
    checkpoints = metadata["checkpoints"]
    assert isinstance(checkpoints, dict)
    assert set(checkpoints) == {
        "core.stage1",
        "core.stage2",
        "core.adaptive-graph",
    }
    for artifact_id, record in checkpoints.items():
        assert isinstance(record, dict)
        authority = manifest[artifact_id]
        assert record["relative_path"] == authority["relative_path"]
        assert record["sha256"] == authority["sha256"]

    sources = metadata["sources"]
    assert isinstance(sources, list) and sources
    source_identities = set()
    for source in sources:
        assert isinstance(source, dict)
        assert set(source) == {"snapshot", "source_revision", "path", "sha256"}
        assert all(source[key] for key in ("snapshot", "source_revision", "path"))
        assert len(source["sha256"]) == 64
        assert source["sha256"] == source["sha256"].lower()
        int(source["sha256"], 16)
        source_identities.add((source["snapshot"], source["path"]))
    assert len(source_identities) == len(sources)


@pytest.fixture(scope="module")
def extracted_core_inputs():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _metadata, golden = load_golden()
    _columns, panel_metadata, train_block, val_block = prepare_panel_blocks(
        data_dir=ROOT / "data"
    )
    assert val_block.indices[:2].tolist() == [4010, 4011]
    selected = select_normalized_innovation_scope(
        val_block,
        "joint38",
        25,
        scale_half_life=None,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
    )
    arrays = tuple(
        value[:2] if isinstance(value, np.ndarray) else value for value in selected
    )
    return golden, panel_metadata, train_block, arrays


def test_extracted_panel_and_state_paths_match_every_golden_array(
    extracted_core_inputs,
) -> None:
    golden, panel_metadata, _train_block, selected = extracted_core_inputs
    (
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        _drift,
        history_raw,
        specs,
    ) = selected
    future_raw = golden["raw_future"]
    state_encoding = encode_state(
        np.concatenate([history_raw, future_raw], axis=1), specs
    ).astype(np.float32)
    reconstructed = reconstruct_state_from_increments(
        history_raw[:, -1],
        future_norm * scale[:, None] + center[:, None],
        specs,
    ).astype(np.float32)

    assert panel_metadata == load_golden()[0]["panel_metadata"]
    for actual, name in (
        (history_raw, "raw_history"),
        (history_level, "state_encoding"),
        (history_norm, "history_normalized"),
        (future_norm, "future_normalized"),
        (center, "normalization_center"),
        (scale, "normalization_scale"),
        (reconstructed, "reconstructed_future"),
    ):
        expected = golden[name]
        if name == "state_encoding":
            expected = expected[:, :30]
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(future_level, golden["state_encoding"][:, 30:])
    np.testing.assert_array_equal(state_encoding, golden["state_encoding"])


def test_extracted_training_losses_and_updates_match_golden(
    extracted_core_inputs,
) -> None:
    golden, _panel_metadata, _train_block, selected = extracted_core_inputs
    tensors = tuple(torch.from_numpy(value) for value in selected[:7])
    stage1_checkpoint = ROOT / "artifacts/cache/models/core/stage1/best_model.pt"
    stage2_checkpoint = ROOT / "artifacts/cache/models/core/stage2/best_model.pt"

    np.testing.assert_array_equal(
        stage1_loss(stage1_checkpoint, tensors, seed=1001), golden["stage1_loss"]
    )
    updates = np.stack(
        [
            stage1_optimizer_update_hash(
                stage1_checkpoint,
                tensors,
                selected_parameters=SELECTED_PARAMETERS,
                seed=1001,
                learning_rate=7e-4,
                weight_decay=1e-5,
                clip_grad_norm=1.0,
            ),
            stage2_optimizer_update_hash(
                stage2_checkpoint,
                tensors,
                selected_parameters=SELECTED_PARAMETERS,
                seed=1001,
            ),
        ]
    )
    np.testing.assert_array_equal(updates, golden["parameter_update_hashes"])


def test_extracted_checkpoints_velocity_rollout_losses_and_metrics_match_golden(
    extracted_core_inputs,
) -> None:
    golden, _panel_metadata, train_block, selected = extracted_core_inputs
    (
        history_level,
        history_norm,
        _future_level,
        future_norm,
        center,
        scale,
        drift,
        history_raw,
        specs,
    ) = selected
    tensors = tuple(torch.from_numpy(value) for value in selected[:7])
    stage2_checkpoint = ROOT / "artifacts/cache/models/core/stage2/best_model.pt"
    graph_checkpoint = ROOT / "artifacts/cache/models/core/adaptive_graph/best_model.pt"

    core_model, _ = load_checkpoint_strict(stage2_checkpoint, torch.device("cpu"))
    graph_model, _ = load_checkpoint_strict(graph_checkpoint, torch.device("cpu"))
    np.testing.assert_array_equal(
        velocity_output(
            core_model,
            tensors[0],
            tensors[1],
            tensors[4],
            tensors[5],
            tensors[6],
            seed=1001,
        ),
        golden["core_velocity"],
    )
    np.testing.assert_array_equal(
        velocity_output(
            graph_model,
            tensors[0],
            tensors[1],
            tensors[4],
            tensors[5],
            tensors[6],
            seed=1001,
        ),
        golden["graph_velocity"],
    )
    rollout = generate_deck(
        core_model,
        history_level,
        history_norm,
        center,
        scale,
        drift,
        history_raw,
        specs,
        samples=2,
        n_steps=30,
        batch_size=2,
        chunk_size=2,
        temperature=1.0,
        seed=2001,
        device=torch.device("cpu"),
    )
    assert rollout.shape == (2, 2, 30, 39)
    np.testing.assert_array_equal(rollout, golden["rollout"])

    torch.manual_seed(2001)
    with torch.no_grad():
        sampled_increment = core_model.sample_batched(
            tensors[0],
            tensors[1],
            tensors[4],
            tensors[5],
            drift_feature=tensors[6],
            n_samples=2,
            n_steps=30,
            chunk_size=2,
            temperature=1.0,
        ).numpy()
    sampled_normalized = (sampled_increment - center[:, None, None]) / scale[
        :, None, None
    ]
    weights = horizon_path_weights(
        30, end_weight=1.2, device=torch.device("cpu"), dtype=torch.float32
    )
    joint = joint_energy_loss(
        torch.from_numpy(sampled_normalized.astype(np.float32)),
        tensors[3],
        eps=1e-6,
        horizon_weights=weights,
    )
    channel = channel_energy_loss(
        torch.from_numpy(sampled_normalized.astype(np.float32)),
        tensors[3],
        eps=1e-6,
        horizon_weights=weights,
    )
    np.testing.assert_allclose(
        np.asarray([value.item() for value in joint], dtype=np.float32),
        golden["joint_energy_loss"],
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray([value.item() for value in channel], dtype=np.float32),
        golden["channel_energy_loss"],
        rtol=0.0,
        atol=1e-7,
    )
    metrics, screens = evaluate_deck(
        raw_history=history_raw,
        raw_future=golden["raw_future"],
        sample_raw=rollout,
        train_block=train_block,
        factor_names=[spec.name for spec in specs[25:]],
        iv_count=25,
    )
    np.testing.assert_allclose(metrics, golden["metrics"], rtol=0.0, atol=2e-8)
    np.testing.assert_array_equal(screens, golden["screen_decisions"])
