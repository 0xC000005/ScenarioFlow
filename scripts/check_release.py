#!/usr/bin/env python3
"""Check a candidate ScenarioFlow release for secrets, host paths, and large files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable


MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
SKIP_SUFFIXES = {".npz", ".parquet", ".pyc", ".pyo"}
SKIP_DIRECTORY_NAMES = {".git", ".superpowers", ".venv", "__pycache__"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
HOST_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_.-])/(?:home|Users|private|tmp)/[^\s\"']+")
WINDOWS_HOST_PATH_PATTERN = re.compile(r"\b[A-Za-z]:\\Users\\[^\s\"']+")


def _release_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if not path.is_file() or any(part in SKIP_DIRECTORY_NAMES for part in relative_parts):
            continue
        yield path


def _text_files(root: Path) -> Iterable[Path]:
    for path in _release_files(root):
        if path.suffix in SKIP_SUFFIXES:
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        yield path


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _secret_issues(root: Path) -> list[str]:
    issues: list[str] = []
    for path in _text_files(root):
        contents = path.read_text(encoding="utf-8")
        if any(pattern.search(contents) for pattern in SECRET_PATTERNS):
            issues.append(f"credential-shaped value: {_relative(root, path)}")
    return issues


def _host_path_issues(root: Path) -> list[str]:
    issues: list[str] = []
    for path in _text_files(root):
        contents = path.read_text(encoding="utf-8")
        if HOST_PATH_PATTERN.search(contents) or WINDOWS_HOST_PATH_PATTERN.search(contents):
            issues.append(f"absolute host path: {_relative(root, path)}")
    return issues


def _large_file_issues(root: Path) -> list[str]:
    return [
        f"file exceeds GitHub 100 MiB limit: {_relative(root, path)}"
        for path in _release_files(root)
        if path.stat().st_size > MAX_GITHUB_FILE_BYTES
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--all", action="store_true", help="run every public-release check")
    parser.add_argument("--secrets", action="store_true", help="scan text files for credential-shaped values")
    parser.add_argument("--absolute-paths", action="store_true", help="scan text files for host paths")
    parser.add_argument("--size", action="store_true", help="reject files larger than GitHub's 100 MiB limit")
    args = parser.parse_args()
    root = args.root.resolve()
    checks = args.all or args.secrets or args.absolute_paths or args.size
    if not checks:
        parser.error("select --all or at least one check")
    if not root.is_dir():
        parser.error(f"release root does not exist: {root}")

    issues: list[str] = []
    if args.all or args.secrets:
        issues.extend(_secret_issues(root))
    if args.all or args.absolute_paths:
        issues.extend(_host_path_issues(root))
    if args.all or args.size:
        issues.extend(_large_file_issues(root))
    if issues:
        print("release checks failed")
        print("\n".join(issues))
        return 1
    print("release checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
