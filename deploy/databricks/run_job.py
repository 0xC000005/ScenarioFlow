#!/usr/bin/env python3
"""Run the existing ScenarioFlow numerical scripts on Databricks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence


def _replace_empty_directory_with_link(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        if not link.is_dir() or any(link.iterdir()):
            raise RuntimeError(f"refusing to replace non-empty runtime path: {link}")
        link.rmdir()
    link.symlink_to(target, target_is_directory=True)


def _commands(root: Path, operation: str, seed: int) -> list[list[str]]:
    scripts = root / "scripts"
    if operation == "train":
        return [
            [str(scripts / "train_next_step.sh"), "--seed", str(seed)],
            [str(scripts / "finetune_rollouts.sh"), "--seed", str(seed)],
        ]
    if operation == "inference":
        return [[str(scripts / "run_inference.sh"), "--seed", str(seed)]]
    return [[str(scripts / "reproduce.sh")]]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation", choices=("train", "inference", "reproduce"), required=True
    )
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    default_root = Path(__file__).resolve().parents[2]
    root = Path(os.environ.get("SCENARIOFLOW_ROOT", default_root)).resolve()
    commands = _commands(root, args.operation, args.seed)
    summary = {
        "operation": args.operation,
        "commands": commands,
        "artifact_root": str(args.artifact_root),
        "output_root": str(args.output_root),
    }
    if args.dry_run:
        print(json.dumps(summary))
        return 0

    _replace_empty_directory_with_link(
        root / "artifacts/cache", args.artifact_root.resolve()
    )
    _replace_empty_directory_with_link(root / "build", args.output_root.resolve())
    for command in commands:
        subprocess.run(command, cwd=root, check=True, env=os.environ.copy())
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
