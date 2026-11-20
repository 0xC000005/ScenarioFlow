"""Shared test configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_CHECKPOINTS = (
    ROOT / "artifacts/cache/models/core/stage1/best_model.pt",
    ROOT / "artifacts/cache/models/core/stage2/best_model.pt",
    ROOT / "artifacts/cache/models/core/adaptive_graph/best_model.pt",
)
NEEDS_CORE_CHECKPOINTS = {
    "test_generate_deck_batch_size_one_matches_frozen_window_batching",
    "test_inference_cli_batch_size_one_uses_frozen_batch_assignment",
    "test_finetune_command_applies_nondefault_settings_before_recording",
    "test_extracted_training_losses_and_updates_match_golden",
    "test_extracted_checkpoints_velocity_rollout_losses_and_metrics_match_golden",
    "test_setup_check_validates_packaged_data_and_manifest",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip checkpoint-replay tests until the released weights are fetched."""
    if all(path.is_file() for path in CORE_CHECKPOINTS):
        return
    skip = pytest.mark.skip(
        reason="run ./scripts/setup.sh to fetch the released core checkpoints"
    )
    for item in items:
        if item.originalname in NEEDS_CORE_CHECKPOINTS:
            item.add_marker(skip)
