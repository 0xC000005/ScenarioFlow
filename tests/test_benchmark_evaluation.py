from types import SimpleNamespace

import numpy as np

from scenarioflow.benchmark.evaluation import evaluate_samples
from scenarioflow.state import UnifiedVariableSpec


def test_evaluation_can_write_protocol_artifacts_at_output_root(tmp_path) -> None:
    rng = np.random.default_rng(17)
    specs = [
        UnifiedVariableSpec(
            name=f"variable_{index}",
            source_column=f"variable_{index}",
            source_index=index,
            transform="level",
        )
        for index in range(39)
    ]
    history = rng.normal(size=(3, 30, 39)).astype(np.float32)
    future = rng.normal(size=(3, 30, 39)).astype(np.float32)
    samples = rng.normal(size=(3, 4, 30, 39)).astype(np.float32)
    block = SimpleNamespace(
        history_state=history,
        future_state=future,
        indices=np.arange(3),
        specs=specs,
    )

    evaluate_samples(
        candidate="scenarioflow",
        train_block=block,
        validation_block=block,
        sample_raw=samples,
        output_dir=tmp_path,
        artifact_dir=tmp_path,
        save_window_scores=True,
    )

    assert (tmp_path / "results_current_panel.json").is_file()
    assert (tmp_path / "window_scores.npz").is_file()
    assert not (tmp_path / "scenarioflow").exists()
