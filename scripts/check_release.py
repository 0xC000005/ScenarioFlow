#!/usr/bin/env python3
"""Check a candidate ScenarioFlow release for secrets, host paths, and large files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Iterable, Iterator

MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
SKIP_SUFFIXES = {".npz", ".parquet", ".pyc", ".pyo"}
SKIP_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".superpowers",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
SKIP_RELATIVE_PREFIXES = {("artifacts", "cache")}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
GENERIC_UNIX_ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w./-])/(?!/|\s)")
TEXT_UNIX_HOST_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:home|Users|private|tmp|root|var|mnt|workspace|opt|srv)/"
    r"[^\s\"']+"
)
DOCKERFILE_HOST_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:home|Users|private)/[^\s\"']+"
)
WINDOWS_DRIVE_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
WINDOWS_UNC_PATH_PATTERN = re.compile(
    r"(?<!\\)\\\\[A-Za-z0-9._-]+[\\/][A-Za-z0-9._-]{2,}"
)
LOCAL_FILE_URI_PATTERN = re.compile(r"(?i)(?<![A-Za-z0-9+.-])file:[\\/]+")
TRIPLE_SLASH_URI_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]*:/{3,}"
)


def _release_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        ignored_prefix = any(
            len(relative_parts) > len(prefix)
            and relative_parts[: len(prefix)] == prefix
            for prefix in SKIP_RELATIVE_PREFIXES
        )
        if (
            not path.is_file()
            or ignored_prefix
            or any(part in SKIP_DIRECTORY_NAMES for part in relative_parts)
        ):
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


def _json_strings(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)
    elif isinstance(value, str):
        yield value


def _portable_authority_paths(root: Path) -> set[str]:
    manifest = root / "results" / "FROZEN_AUTHORITY.tsv"
    if not manifest.is_file():
        return set()
    try:
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            return {
                row["released_path"]
                for row in rows
                if row.get("transformation") == "portable-json-v1"
                and row.get("released_path")
            }
    except (csv.Error, KeyError, UnicodeDecodeError):
        return set()


def _contains_host_path(path: Path, contents: str, *, strict_json: bool) -> bool:
    if path.name == "Dockerfile" or path.name.startswith("Dockerfile."):
        return bool(
            DOCKERFILE_HOST_PATH_PATTERN.search(contents)
            or WINDOWS_DRIVE_PATH_PATTERN.search(contents)
            or WINDOWS_UNC_PATH_PATTERN.search(contents)
            or LOCAL_FILE_URI_PATTERN.search(contents)
        )
    if path.suffix == ".json":
        try:
            document = json.loads(contents)
        except json.JSONDecodeError:
            pass
        else:
            patterns = (
                (
                    GENERIC_UNIX_ABSOLUTE_PATH_PATTERN,
                    WINDOWS_DRIVE_PATH_PATTERN,
                    WINDOWS_UNC_PATH_PATTERN,
                    LOCAL_FILE_URI_PATTERN,
                    TRIPLE_SLASH_URI_PATTERN,
                )
                if strict_json
                else (
                    TEXT_UNIX_HOST_PATH_PATTERN,
                    WINDOWS_DRIVE_PATH_PATTERN,
                    WINDOWS_UNC_PATH_PATTERN,
                    LOCAL_FILE_URI_PATTERN,
                )
            )
            return any(
                any(pattern.search(value) for pattern in patterns)
                for value in _json_strings(document)
            )
    return bool(
        TEXT_UNIX_HOST_PATH_PATTERN.search(contents)
        or WINDOWS_DRIVE_PATH_PATTERN.search(contents)
        or WINDOWS_UNC_PATH_PATTERN.search(contents)
        or LOCAL_FILE_URI_PATTERN.search(contents)
    )


def _secret_issues(root: Path) -> list[str]:
    issues: list[str] = []
    for path in _text_files(root):
        contents = path.read_text(encoding="utf-8")
        if any(pattern.search(contents) for pattern in SECRET_PATTERNS):
            issues.append(f"credential-shaped value: {_relative(root, path)}")
    return issues


def _host_path_issues(root: Path) -> list[str]:
    issues: list[str] = []
    portable_authorities = _portable_authority_paths(root)
    for path in _text_files(root):
        contents = path.read_text(encoding="utf-8")
        if _contains_host_path(
            path,
            contents,
            strict_json=_relative(root, path) in portable_authorities,
        ):
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
    parser.add_argument(
        "--all", action="store_true", help="run every public-release check"
    )
    parser.add_argument(
        "--secrets",
        action="store_true",
        help="scan text files for credential-shaped values",
    )
    parser.add_argument(
        "--absolute-paths", action="store_true", help="scan text files for host paths"
    )
    parser.add_argument(
        "--size",
        action="store_true",
        help="reject files larger than GitHub's 100 MiB limit",
    )
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
