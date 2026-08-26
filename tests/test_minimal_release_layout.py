"""Release-layout contract for the extracted numerical core."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from scenarioflow.reproduction import finetune_rollouts, inference, train_next_step
from scenarioflow.benchmark.guard import tracked_numbered_experiments

ROOT = Path(__file__).resolve().parents[1]


def _shipped_python_files() -> list[Path]:
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if tracked.returncode == 0:
        return [
            ROOT / Path(filename)
            for filename in tracked.stdout.decode().split("\0")
            if filename.endswith(".py")
        ]

    ignored = {".git", ".pytest_cache", ".venv", "__pycache__", "build"}
    return [
        path
        for path in ROOT.rglob("*.py")
        if not ignored.intersection(path.relative_to(ROOT).parts)
    ]


def test_release_contains_only_the_clean_numerical_core() -> None:
    """Shipping an archived tree or numbered experiment entrypoint is a release bug."""
    shipped_files = _shipped_python_files()
    assert not [
        path
        for path in shipped_files
        if path.relative_to(ROOT).parts[:2] == ("reproduction", "source")
    ]

    assert tracked_numbered_experiments(ROOT) == []


def test_public_cli_parsers_supply_overridable_core_defaults() -> None:
    """The clean CLIs own defaults so public wrappers pass only user intent."""
    stage1 = train_next_step.build_parser()
    assert stage1.parse_args([]).output_dir == ROOT / "build/training/stage1"
    assert stage1.parse_args(["--output-dir", "custom/stage1"]).output_dir == Path(
        "custom/stage1"
    )

    stage2 = finetune_rollouts.build_parser()
    assert stage2.parse_args([]).checkpoint == (
        ROOT / "artifacts/cache/models/core/stage1/best_model.pt"
    )
    assert stage2.parse_args([]).output_dir == ROOT / "build/training/stage2"
    assert stage2.parse_args(
        ["--checkpoint", "custom/stage1.pt", "--output_dir", "custom/stage2"]
    ).output_dir == Path("custom/stage2")

    numerical_inference = inference.build_parser()
    assert numerical_inference.parse_args([]).checkpoint == (
        ROOT / "artifacts/cache/models/core/stage2/best_model.pt"
    )
    assert numerical_inference.parse_args(
        ["--output-dir", "custom/inference"]
    ).output_dir == Path("custom/inference")


@pytest.mark.parametrize(
    ("module", "downstream_name"),
    (
        (train_next_step, "prepare_panel_blocks"),
        (finetune_rollouts, "load_model"),
    ),
)
def test_public_trainers_reject_unknown_options_before_execution(
    monkeypatch: pytest.MonkeyPatch, module, downstream_name: str
) -> None:
    """A misspelled public option must be reported instead of being ignored."""

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("unknown options must stop before training setup")

    monkeypatch.setattr(module, downstream_name, must_not_run)
    with pytest.raises(SystemExit) as error:
        module.main(["--not-a-real-option"])
    assert error.value.code == 2


def test_public_wrappers_forward_custom_inference_without_default_checkpoint(
    tmp_path: Path,
) -> None:
    """A supplied checkpoint reaches inference even when the released default is absent."""
    release = tmp_path / "release"
    scripts = release / "scripts"
    scripts.mkdir(parents=True)
    for name in ("train_next_step.sh", "finetune_rollouts.sh", "run_inference.sh"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)

    python = release / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    calls = tmp_path / "calls.txt"
    python.write_text(
        "#!/usr/bin/env bash\n" 'printf \'%s\\n\' "$*" >> "$SCENARIOFLOW_TEST_CALLS"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    environment = {"SCENARIOFLOW_TEST_CALLS": str(calls)}
    for name, override in (
        ("train_next_step.sh", "--output-dir custom/stage1"),
        ("finetune_rollouts.sh", "--output-dir custom/stage2"),
        (
            "run_inference.sh",
            "--checkpoint custom/stage2.pt --output-dir custom/inference",
        ),
    ):
        result = subprocess.run(
            [str(scripts / name), *override.split()],
            cwd=release,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    assert calls.read_text(encoding="utf-8").splitlines() == [
        "-m scenarioflow.reproduction.train_next_step --output-dir custom/stage1",
        "-m scenarioflow.reproduction.finetune_rollouts --output-dir custom/stage2",
        "-m scenarioflow.reproduction.inference "
        "--checkpoint custom/stage2.pt --output-dir custom/inference",
    ]
