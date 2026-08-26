"""Contracts for the archived original experiment routes."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = ROOT / "scripts/run_reproduction.py"
    spec = importlib.util.spec_from_file_location("scenarioflow_release_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_dag_uses_the_original_training_and_evaluation_files() -> None:
    payload = _runner().materialize_payload()
    nodes = {node["id"]: node for node in payload["commands"]}
    assert len(nodes) == 141
    assert nodes["train:scenarioflow:stage1:1001"]["command"][1] == (
        "experiments/backfill/block_ar/"
        "train_662a_state_aware_normalized_innovation_flow.py"
    )
    assert nodes["train:scenarioflow:stage2:1001"]["command"][1] == (
        "experiments/backfill/block_ar/"
        "train_666a_normalized_innovation_rollout_energy_finetune.py"
    )
    assert nodes["eval:scenarioflow:train_1001:rollout_2001"]["command"][1] == (
        "experiments/backfill/baselines/evaluate_sni_current_panel_metrics.py"
    )
    for node in nodes.values():
        working_directory = Path(node["working_directory"])
        assert (working_directory / node["command"][1]).is_file()
        assert "src/scenarioflow" not in " ".join(node["command"])


def test_release_dag_keeps_the_paper_contract_and_seed_grid() -> None:
    payload = _runner().materialize_payload()
    protocol = payload["protocol"]
    assert protocol["train_seeds"] == [1001, 1002, 1003]
    assert protocol["rollout_seeds"] == [2001, 2002, 2003, 2004, 2005]
    assert protocol["contract"] == {
        "mode": "current_realvix_39_state_panel",
        "n_market_variables": 39,
        "n_evaluation_windows": 441,
        "n_scenarios_per_window": 64,
        "history_length": 30,
        "future_length": 30,
        "threshold_contract_hash": (
            "68b5ab9ca4497ff5ec07b43a184c5ab1470d3d8aba5dd21ccb428b00ff6afc09"
        ),
        "state_specification_hash": (
            "65540401c58143dc51a2c907916c9b4903b484c3c3ad0ba9ce1c062b96fc6076"
        ),
    }


def test_friendly_wrappers_only_select_archived_nodes() -> None:
    wrappers = {
        "train_next_step.sh": "train:scenarioflow:stage1:${SEED}",
        "finetune_rollouts.sh": "train:scenarioflow:stage2:${SEED}",
        "evaluate_benchmark.sh": "--node-id \"$1\"",
        "reproduce.sh": "--resume-all",
    }
    for filename, expected in wrappers.items():
        contents = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert "run_reproduction.py" in contents
        assert expected in contents


def test_scenarioflow_inference_wrapper_uses_the_released_model() -> None:
    script = ROOT / "scripts/run_inference.sh"
    contents = script.read_text(encoding="utf-8")

    assert "artifacts/cache/models/core/stage2/best_model.pt" in contents
    assert "evaluate_sni_current_panel_metrics.py" in contents
    assert "OPENAI_API_KEY" not in contents
    assert "codex" not in contents.lower()


def test_setup_uses_the_v1_github_release_by_default() -> None:
    contents = (ROOT / "scripts/setup.sh").read_text(encoding="utf-8")

    assert (
        'DEFAULT_BASE_URL="https://github.com/0xC000005/ScenarioFlow/'
        'releases/download/v1.0.0"'
    ) in contents
    assert 'BASE_URL="${SCENARIOFLOW_ARTIFACT_BASE_URL:-$DEFAULT_BASE_URL}"' in contents


def test_setup_installs_the_locked_base_project_and_fetches_core(
    tmp_path: Path,
) -> None:
    """The default install must stay minimal while installing the released model."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "uv-calls.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$UV_CALLS_FILE\"\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    artifact_root = tmp_path / "artifacts"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "UV_CALLS_FILE": str(calls),
    }

    result = subprocess.run(
        [
            str(ROOT / "scripts" / "setup.sh"),
            "--artifact-root",
            str(artifact_root),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "sync --locked",
        (
            "run scenarioflow fetch --profile core --base-url "
            "https://github.com/0xC000005/ScenarioFlow/releases/download/v1.0.0 "
            f"--artifact-root {artifact_root}"
        ),
        f"run scenarioflow setup --check --artifact-root {artifact_root}",
    ]


def test_setup_repairs_data_links_flattened_by_source_zip(tmp_path: Path) -> None:
    """Browser-downloaded source ZIPs must retain working reproduction routes."""
    release = tmp_path / "release"
    scripts = release / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/setup.sh", scripts / "setup.sh")
    (release / "data").mkdir()

    for snapshot in ("scenarioflow", "baselines", "garch_ccc", "deepvar_sni"):
        source = release / "reproduction/source" / snapshot
        source.mkdir(parents=True)
        (source / "data").write_text("../../../data", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        [str(scripts / "setup.sh")],
        cwd=release,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for snapshot in ("scenarioflow", "baselines", "garch_ccc", "deepvar_sni"):
        data = release / "reproduction/source" / snapshot / "data"
        assert data.is_symlink()
        assert os.readlink(data) == "../../../data"


def test_installation_check_probes_data_through_preserved_source_links() -> None:
    contents = (ROOT / "scripts/check_installation.sh").read_text(encoding="utf-8")

    assert "vol_surface_with_ret.npz" in contents
    assert "preserved data routes: ok" in contents


def test_setup_explains_manual_release_fallback_when_remote_fetch_fails(
    tmp_path: Path,
) -> None:
    """A private or unavailable release must leave the user a usable next step."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == run\\ scenarioflow\\ fetch* ]]; then\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        [str(ROOT / "scripts" / "setup.sh")],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Automatic artifact download failed." in result.stderr
    assert (
        "https://github.com/0xC000005/ScenarioFlow/releases/tag/v1.0.0"
        in result.stderr
    )
    assert "scenarioflow-core-v1.0.0.zip" in result.stderr
    assert (
        "./scripts/setup.sh --profile core "
        "--bundle-dir /path/to/downloaded-release-assets"
        in result.stderr
    )
