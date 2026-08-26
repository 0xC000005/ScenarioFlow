"""Release guards for the compact benchmark output and source layout."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Iterable


class UnsafeOutputRoot(ValueError):
    """Raised when a benchmark root overlaps released evidence or source data."""


def assert_isolated_output_root(
    root: str | Path, protected: Iterable[str | Path], output: str | Path
) -> Path:
    repository = Path(root).resolve()
    candidate = Path(output).expanduser().resolve()
    if repository != candidate and repository not in candidate.parents:
        raise UnsafeOutputRoot("benchmark output must remain below the repository root")
    for supplied in protected:
        protected_path = (repository / supplied).resolve()
        if (
            candidate == protected_path
            or candidate in protected_path.parents
            or protected_path in candidate.parents
        ):
            raise UnsafeOutputRoot(
                f"benchmark output {candidate} overlaps protected {protected_path}"
            )
    return candidate


def tracked_numbered_experiments(root: str | Path) -> list[Path]:
    """Return tracked numbered experiment scripts, which must remain absent."""
    repository = Path(root).resolve()
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository,
        capture_output=True,
        check=True,
    )
    paths = [Path(value) for value in completed.stdout.decode().split("\0") if value]
    prefixes = ("analyze_", "audit_", "evaluate_", "plot_", "test_", "train_")
    return [
        path
        for path in paths
        if path.suffix == ".py"
        and "tests/fixtures" not in path.as_posix()
        and any(
            path.stem.startswith(prefix)
            and path.stem.removeprefix(prefix)[:1].isdigit()
            for prefix in prefixes
        )
    ]
