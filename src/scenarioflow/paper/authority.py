"""Load and verify the immutable evidence shipped with ScenarioFlow."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Literal


SCHEMA_VERSION = "scenarioflow.frozen-authority.v1"
MANIFEST_COLUMNS = (
    "schema_version",
    "authority_id",
    "category",
    "released_path",
    "released_bytes",
    "released_sha256",
    "frozen_source_path",
    "frozen_bytes",
    "frozen_sha256",
    "transformation",
)
PORTABLE_JSON_SOURCES = frozenset(
    {
        "results/fmts_multiseed_20260805/aggregate/summary.json",
        "results/fmts_multiseed_20260805/command_dag.json",
        "results/fmts_multiseed_20260805/protocol_manifest.json",
        "results/fmts_reproduction_20260805/reproduction_report.json",
        "results/foundation_baselines_current_panel_20260810/benchmark_manifest.json",
        "results/fmts_reproduction_20260805/reports/layer_a_garch_ccc/garch_ccc_contract_report.json",
    }
)
MANAGED_RELEASE_ROOTS = (
    PurePosixPath("results/frozen"),
    PurePosixPath("metadata/models"),
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_AUTHORITY_ID = re.compile(r"[a-z0-9][a-z0-9.-]*")
_UNIX_HOST_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:home|Users|private|tmp)/[^\s\"']+"
)
_WINDOWS_HOST_PATH = re.compile(r"\b[A-Za-z]:\\Users\\[^\s\"']+")
_PRIVATE_TRANSCRIPT = re.compile(
    r"/(?:home/[^/]+|Users/[^/]+)/\.claude/projects/" r"[A-Za-z0-9._/-]+\.jsonl"
)
_TEMPORARY_WORKTREE = re.compile(
    r"/(?:t)mp/(?P<identifier>fmts-window-scores-[A-Za-z0-9._-]+)"
)

_PATH_ROOT_REWRITES = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (
            r"/(?:home/[^/]+|Users/[^/]+)/Documents/vol-surface-vae-pub"
            r"(?:(?P<separator>/)|(?=$|[\s\"'`,;:)\]}}]))",
            "frozen-source://",
        ),
        (
            r"/(?:home/[^/]+|Users/[^/]+)/Documents/"
            r"vol-surface-vae-pub-paper-backups"
            r"(?:(?P<separator>/)|(?=$|[\s\"'`,;:)\]}}]))",
            "paper-authority://",
        ),
        (
            r"/(?:home/[^/]+|Users/[^/]+)/\.cache/vol-surface-vae-pub"
            r"(?:(?P<separator>/)|(?=$|[\s\"'`,;:)\]}}]))",
            "artifact-cache://",
        ),
    )
)

Transformation = Literal["identity", "portable-json-v1"]


class AuthorityError(RuntimeError):
    """Raised when released evidence does not match its frozen authority."""


@dataclass(frozen=True)
class FrozenAuthority:
    """One immutable mapping from frozen provenance to a public file."""

    authority_id: str
    category: str
    released_path: Path
    released_bytes: int
    released_sha256: str
    frozen_source_path: str
    frozen_bytes: int
    frozen_sha256: str
    transformation: Transformation


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AuthorityError(f"{label} must be a normalized relative path")
    return path


def _parse_positive_int(value: str, *, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise AuthorityError(f"{label} must be a positive integer") from error
    if parsed <= 0 or str(parsed) != value:
        raise AuthorityError(f"{label} must be a positive integer")
    return parsed


def _parse_sha256(value: str, *, label: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise AuthorityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def load_authorities(path: Path) -> tuple[FrozenAuthority, ...]:
    """Load and validate a frozen-authority TSV manifest."""
    manifest = Path(path)
    if manifest.is_symlink() or not manifest.is_file():
        raise AuthorityError("frozen-authority manifest must be a regular file")
    try:
        with manifest.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
                raise AuthorityError("frozen-authority manifest columns are invalid")
            raw_rows = list(reader)
    except UnicodeDecodeError as error:
        raise AuthorityError("frozen-authority manifest must be UTF-8") from error
    if not raw_rows:
        raise AuthorityError("frozen-authority manifest must not be empty")

    authorities: list[FrozenAuthority] = []
    authority_ids: set[str] = set()
    released_paths: set[str] = set()
    frozen_paths: set[str] = set()
    for line_number, row in enumerate(raw_rows, start=2):
        if row["schema_version"] != SCHEMA_VERSION:
            raise AuthorityError(f"line {line_number}: unsupported schema version")
        authority_id = row["authority_id"]
        if _AUTHORITY_ID.fullmatch(authority_id) is None:
            raise AuthorityError(f"line {line_number}: invalid authority ID")
        if authority_id in authority_ids:
            raise AuthorityError(f"line {line_number}: duplicate authority ID")
        authority_ids.add(authority_id)

        released_path = _safe_relative_path(
            row["released_path"], label=f"line {line_number} released path"
        )
        released_string = released_path.as_posix()
        if released_string in released_paths:
            raise AuthorityError(f"line {line_number}: duplicate released path")
        released_paths.add(released_string)
        if not any(
            released_path == root or released_path.is_relative_to(root)
            for root in MANAGED_RELEASE_ROOTS
        ):
            raise AuthorityError(
                f"line {line_number}: released path is outside managed roots"
            )

        frozen_path = _safe_relative_path(
            row["frozen_source_path"],
            label=f"line {line_number} frozen source path",
        ).as_posix()
        if frozen_path in frozen_paths:
            raise AuthorityError(f"line {line_number}: duplicate frozen source path")
        frozen_paths.add(frozen_path)

        category = row["category"]
        if not category or category.strip() != category:
            raise AuthorityError(f"line {line_number}: invalid category")
        transformation = row["transformation"]
        if transformation not in {"identity", "portable-json-v1"}:
            raise AuthorityError(f"line {line_number}: invalid transformation")
        if (
            transformation == "portable-json-v1"
            and frozen_path not in PORTABLE_JSON_SOURCES
        ):
            raise AuthorityError(
                f"line {line_number}: portable projection was not approved"
            )

        released_bytes = _parse_positive_int(
            row["released_bytes"], label=f"line {line_number} released bytes"
        )
        frozen_bytes = _parse_positive_int(
            row["frozen_bytes"], label=f"line {line_number} frozen bytes"
        )
        released_sha256 = _parse_sha256(
            row["released_sha256"], label=f"line {line_number} released SHA-256"
        )
        frozen_sha256 = _parse_sha256(
            row["frozen_sha256"], label=f"line {line_number} frozen SHA-256"
        )
        if transformation == "identity" and (
            released_bytes != frozen_bytes or released_sha256 != frozen_sha256
        ):
            raise AuthorityError(
                f"line {line_number}: identity release must match frozen identity"
            )

        authorities.append(
            FrozenAuthority(
                authority_id=authority_id,
                category=category,
                released_path=Path(released_string),
                released_bytes=released_bytes,
                released_sha256=released_sha256,
                frozen_source_path=frozen_path,
                frozen_bytes=frozen_bytes,
                frozen_sha256=frozen_sha256,
                transformation=transformation,  # type: ignore[arg-type]
            )
        )
    return tuple(authorities)


def _project_string(value: str) -> str:
    projected = value
    for pattern, replacement in _PATH_ROOT_REWRITES:
        projected = pattern.sub(lambda _: replacement, projected)

    projected = _TEMPORARY_WORKTREE.sub(
        lambda match: f"legacy-worktree://{match.group('identifier')}", projected
    )

    def replace_private(match: re.Match[str]) -> str:
        digest = hashlib.sha256(match.group(0).encode()).hexdigest()
        return f"private-provenance://{digest}"

    projected = _PRIVATE_TRANSCRIPT.sub(replace_private, projected)
    if _UNIX_HOST_PATH.search(projected) or _WINDOWS_HOST_PATH.search(projected):
        raise AuthorityError("unrecognized absolute host path in portable JSON")
    return projected


def _project_value(value: object) -> object:
    if isinstance(value, dict):
        projected: dict[str, object] = {}
        for key, item in value.items():
            projected_key = _project_string(key)
            if projected_key in projected:
                raise AuthorityError(
                    "portable-json-v1 path rewrite caused a key collision"
                )
            projected[projected_key] = _project_value(item)
        return projected
    if isinstance(value, list):
        return [_project_value(item) for item in value]
    if isinstance(value, str):
        return _project_string(value)
    return value


def project_portable_json_v1(payload: bytes) -> bytes:
    """Rewrite approved host paths while preserving the JSON data model."""
    try:
        source = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorityError("portable-json-v1 input must be UTF-8 JSON") from error
    projected = _project_value(source)
    try:
        serialized = json.dumps(
            projected,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except ValueError as error:
        raise AuthorityError(
            "portable-json-v1 input contains non-finite data"
        ) from error
    return f"{serialized}\n".encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lexical_root(root: Path) -> Path:
    supplied = root.expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    lexical = Path(os.path.abspath(supplied))
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise AuthorityError("release root or an ancestor is a symlink")
    if not lexical.is_dir():
        raise AuthorityError("release root must be a directory")
    return lexical


def _safe_released_file(root: Path, relative: Path) -> Path:
    target = root.joinpath(*relative.parts)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise AuthorityError(
                f"released authority contains a symlink: {relative.as_posix()}"
            )
    return target


def _managed_files(root: Path) -> set[str]:
    files: set[str] = set()
    for managed in MANAGED_RELEASE_ROOTS:
        managed_path = root.joinpath(*managed.parts)
        if not managed_path.exists():
            continue
        if managed_path.is_symlink():
            raise AuthorityError(f"managed release root is a symlink: {managed}")
        for path in managed_path.rglob("*"):
            if path.is_symlink():
                raise AuthorityError(
                    f"managed release entry is a symlink: {path.relative_to(root)}"
                )
            if path.is_file():
                files.add(path.relative_to(root).as_posix())
    return files


def verify_release_authorities(root: Path, manifest: Path) -> list[FrozenAuthority]:
    """Verify every managed released file and reject missing or extra files."""
    release_root = _lexical_root(Path(root))
    authorities = load_authorities(Path(manifest))
    expected_paths = {item.released_path.as_posix() for item in authorities}
    actual_paths = _managed_files(release_root)
    missing = sorted(expected_paths - actual_paths)
    if missing:
        raise AuthorityError(f"missing released authority: {missing[0]}")
    extra = sorted(actual_paths - expected_paths)
    if extra:
        raise AuthorityError(f"unexpected managed release file: {extra[0]}")

    for item in authorities:
        released = _safe_released_file(release_root, item.released_path)
        if (
            not released.is_file()
            or released.stat().st_size != item.released_bytes
            or _sha256_file(released) != item.released_sha256
        ):
            raise AuthorityError(
                f"released authority size or SHA-256 mismatch: {item.authority_id}"
            )
        if item.transformation == "portable-json-v1":
            try:
                document = json.loads(released.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AuthorityError(
                    f"portable authority is not valid JSON: {item.authority_id}"
                ) from error
            for value in _walk_strings(document):
                if _UNIX_HOST_PATH.search(value) or _WINDOWS_HOST_PATH.search(value):
                    raise AuthorityError(
                        f"portable authority contains a host path: {item.authority_id}"
                    )
    return list(authorities)


def _walk_strings(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value
