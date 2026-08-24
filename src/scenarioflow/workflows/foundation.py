"""Explicit subprocess boundary for optional isolated foundation environments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..baselines.foundation import MOIRAI_REVISION, TOTO_REVISION


def foundation_subprocess_command(
    *,
    model: str,
    environment_dir: str | Path,
    backend_module: str,
    backend_args: Sequence[str],
) -> list[str]:
    """Return an isolated ``uv run`` command without importing provider frameworks."""
    if model not in {"moirai", "toto"}:
        raise ValueError("model must be 'moirai' or 'toto'")
    environment = Path(environment_dir)
    pyproject = environment / "pyproject.toml"
    lockfile = environment / "uv.lock"
    if not pyproject.is_file() or not lockfile.is_file():
        raise FileNotFoundError(
            f"missing isolated {model} environment at {environment}; run "
            f"`uv sync --project {environment} --frozen` first"
        )
    if not backend_module or backend_module.startswith("."):
        raise ValueError("backend_module must be an explicit importable module")
    revision = MOIRAI_REVISION if model == "moirai" else TOTO_REVISION
    return [
        "uv",
        "run",
        "--project",
        str(environment),
        "--frozen",
        "python",
        "-m",
        backend_module,
        "--upstream-revision",
        revision,
        *map(str, backend_args),
    ]
