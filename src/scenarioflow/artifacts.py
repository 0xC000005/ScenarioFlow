"""Strict checksum-pinned resolution of ScenarioFlow release artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Collection, Iterable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen


SCHEMA_VERSION = "scenarioflow.artifacts.v1"
MANIFEST_COLUMNS = (
    "schema_version",
    "artifact_id",
    "profile",
    "category",
    "relative_path",
    "bytes",
    "sha256",
    "frozen_source_path",
    "url_path",
)
PROFILES = frozenset(
    {"core", "baselines", "multiseed", "foundation", "nl-runtime", "evidence", "all"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_ID = re.compile(r"[a-z0-9][a-z0-9.-]*")
ArtifactState = Literal["missing", "partial", "present", "corrupt", "unavailable"]


class ArtifactError(RuntimeError):
    """Raised when an artifact authority or payload cannot be trusted."""


@dataclass(frozen=True)
class ArtifactRecord:
    """One immutable row from the external release manifest."""

    artifact_id: str
    profiles: frozenset[str]
    category: str
    relative_path: str
    expected_bytes: int
    sha256: str
    frozen_source_path: str
    url_path: str


@dataclass(frozen=True)
class ArtifactStatus:
    """Safe integrity status without host paths or payload contents."""

    state: ArtifactState
    artifact_id: str
    expected_bytes: int
    sha256: str
    relative_path: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "artifact_id": self.artifact_id,
            "bytes": self.expected_bytes,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "state": self.state,
        }


def _safe_posix_path(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or str(path) != value
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ArtifactError(f"{label} must be a normalized relative path")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, record: ArtifactRecord) -> bool:
    return (
        not path.is_symlink()
        and path.is_file()
        and path.stat().st_size == record.expected_bytes
        and _sha256_file(path) == record.sha256
    )


def _root_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve(strict=False)


def _safe_target(root: Path, relative_path: str) -> Path:
    target = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ArtifactError("artifact destination escapes the artifact root") from error
    current = root
    if current.is_symlink():
        raise ArtifactError("artifact root must not be a symlink")
    for component in PurePosixPath(relative_path).parts:
        current = current / component
        if current.is_symlink():
            raise ArtifactError(
                f"artifact destination contains a symlink: {relative_path}"
            )
    return target


def _safe_bundle_source(bundle_root: Path, url_path: str) -> Path:
    source = _safe_target(bundle_root, url_path)
    if source.is_symlink():
        raise ArtifactError("local bundle member must not be a symlink")
    return source


class ArtifactResolver:
    """Resolve or fetch files authenticated solely by a committed TSV manifest."""

    def __init__(self, records: Iterable[ArtifactRecord]) -> None:
        self.records = tuple(records)
        self._by_id = {record.artifact_id: record for record in self.records}
        self.profiles = frozenset(
            profile for record in self.records for profile in record.profiles
        )

    @classmethod
    def from_tsv(cls, manifest_path: str | Path) -> "ArtifactResolver":
        path = Path(manifest_path)
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
                    raise ArtifactError("artifact manifest columns do not match v1")
                rows = list(reader)
        except ArtifactError:
            raise
        except (OSError, csv.Error) as error:
            raise ArtifactError("artifact manifest is unreadable") from error
        if not rows:
            raise ArtifactError("artifact manifest has no rows")

        records: list[ArtifactRecord] = []
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for index, row in enumerate(rows, start=2):
            if row["schema_version"] != SCHEMA_VERSION:
                raise ArtifactError(f"row {index} has an unsupported schema version")
            artifact_id = row["artifact_id"]
            if not _ARTIFACT_ID.fullmatch(artifact_id):
                raise ArtifactError(f"row {index} has an invalid artifact ID")
            if artifact_id in seen_ids:
                raise ArtifactError(f"duplicate artifact ID: {artifact_id}")
            seen_ids.add(artifact_id)

            relative = str(
                _safe_posix_path(row["relative_path"], label="relative path")
            )
            if relative in seen_paths:
                raise ArtifactError(f"duplicate artifact path: {relative}")
            seen_paths.add(relative)
            frozen_source = str(
                _safe_posix_path(row["frozen_source_path"], label="frozen source path")
            )
            url_path = str(_safe_posix_path(row["url_path"], label="URL path"))
            if url_path != quote(relative, safe="/._-"):
                raise ArtifactError(
                    f"row {index} URL path does not encode its relative path"
                )

            category = row["category"].strip()
            if not category or category == "data" or relative.startswith("data/"):
                raise ArtifactError(
                    "data artifacts must not enter the external manifest"
                )
            digest = row["sha256"]
            if _SHA256.fullmatch(digest) is None:
                raise ArtifactError(
                    f"row {index} SHA-256 must be 64 lowercase hex characters"
                )
            try:
                expected_bytes = int(row["bytes"])
            except ValueError as error:
                raise ArtifactError(
                    f"row {index} byte size is not an integer"
                ) from error
            if expected_bytes <= 0:
                raise ArtifactError(f"row {index} byte size must be positive")

            profiles = frozenset(item.strip() for item in row["profile"].split(","))
            if not profiles or "" in profiles or not profiles.issubset(PROFILES):
                raise ArtifactError(f"row {index} has an unknown artifact profile")
            if "all" not in profiles:
                raise ArtifactError(f"row {index} must belong to the all profile")
            records.append(
                ArtifactRecord(
                    artifact_id=artifact_id,
                    profiles=profiles,
                    category=category,
                    relative_path=relative,
                    expected_bytes=expected_bytes,
                    sha256=digest,
                    frozen_source_path=frozen_source,
                    url_path=url_path,
                )
            )
        return cls(records)

    def record(self, artifact_id: str) -> ArtifactRecord:
        try:
            return self._by_id[artifact_id]
        except KeyError as error:
            raise ArtifactError(f"unknown artifact ID: {artifact_id}") from error

    def for_profile(self, profile: str) -> tuple[ArtifactRecord, ...]:
        if profile not in self.profiles:
            raise ArtifactError(f"unknown artifact profile: {profile}")
        return tuple(record for record in self.records if profile in record.profiles)

    def select(
        self,
        *,
        profile: str | None = None,
        artifact_ids: Collection[str] = (),
    ) -> tuple[ArtifactRecord, ...]:
        if profile is not None and artifact_ids:
            raise ArtifactError("choose either an artifact profile or artifact IDs")
        if artifact_ids:
            selected: list[ArtifactRecord] = []
            seen: set[str] = set()
            for artifact_id in artifact_ids:
                if artifact_id not in seen:
                    selected.append(self.record(artifact_id))
                    seen.add(artifact_id)
            return tuple(selected)
        return self.for_profile(profile or "core")

    def status(self, artifact_id: str, artifact_root: str | Path) -> ArtifactStatus:
        record = self.record(artifact_id)
        root = _root_path(artifact_root)
        try:
            final = _safe_target(root, record.relative_path)
            part = final.with_name(final.name + ".part")
            if final.exists() or final.is_symlink():
                state: ArtifactState = (
                    "present" if _verified(final, record) else "corrupt"
                )
            elif part.exists() or part.is_symlink():
                state = (
                    "partial"
                    if part.is_file() and not part.is_symlink()
                    else "unavailable"
                )
            else:
                state = "missing"
        except (ArtifactError, OSError):
            state = "unavailable"
        return ArtifactStatus(
            state=state,
            artifact_id=record.artifact_id,
            expected_bytes=record.expected_bytes,
            sha256=record.sha256,
            relative_path=record.relative_path,
        )

    def verify(
        self, artifact_ids: Collection[str], artifact_root: str | Path
    ) -> list[ArtifactStatus]:
        return [self.status(artifact_id, artifact_root) for artifact_id in artifact_ids]

    def resolve(self, artifact_id: str, artifact_root: str | Path) -> Path:
        record = self.record(artifact_id)
        root = _root_path(artifact_root)
        target = _safe_target(root, record.relative_path)
        status = self.status(artifact_id, root)
        if status.state != "present":
            raise ArtifactError(
                f"artifact {artifact_id} is {status.state}: {record.relative_path}"
            )
        return target

    def fetch(
        self,
        artifact_ids: Collection[str],
        artifact_root: str | Path,
        *,
        base_url: str | None = None,
        bundle_dir: str | Path | None = None,
        resume: bool = True,
        repair: bool = False,
        offline: bool = False,
    ) -> list[Path]:
        if base_url is not None and bundle_dir is not None:
            raise ArtifactError("--base-url and --bundle-dir are mutually exclusive")
        if offline and (base_url is not None or bundle_dir is not None):
            raise ArtifactError("offline mode does not accept a download source")
        selected = [self.record(artifact_id) for artifact_id in artifact_ids]
        if not selected:
            raise ArtifactError("no artifacts were selected")
        root = _root_path(artifact_root)
        pending: list[ArtifactRecord] = []
        for record in selected:
            status = self.status(record.artifact_id, root)
            if status.state == "present":
                continue
            if status.state == "unavailable":
                # Re-run the guard to return the specific safe-path error.
                _safe_target(root, record.relative_path)
                raise ArtifactError(
                    f"artifact destination is unavailable: {record.relative_path}"
                )
            if status.state == "corrupt" and not repair:
                raise ArtifactError(
                    f"artifact {record.artifact_id} is corrupt; rerun with --repair"
                )
            pending.append(record)

        if not pending:
            return [self.resolve(record.artifact_id, root) for record in selected]
        if offline:
            names = ", ".join(record.artifact_id for record in pending)
            raise ArtifactError(f"offline mode cannot fetch missing artifacts: {names}")
        if base_url is None and bundle_dir is None:
            base_url = os.environ.get("SCENARIOFLOW_ARTIFACT_BASE_URL") or None
        if base_url is None and bundle_dir is None:
            raise ArtifactError(
                "no artifact source configured; provide --base-url or --bundle-dir"
            )

        bundle_root = None if bundle_dir is None else _root_path(bundle_dir)
        for record in pending:
            target = _safe_target(root, record.relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            _safe_target(root, record.relative_path)
            if bundle_root is not None:
                self._fetch_from_bundle(record, target, bundle_root)
            else:
                assert base_url is not None
                self._fetch_remote(record, target, base_url, resume=resume)
        # Preserve caller selection order even when some files were already present.
        return [self.resolve(record.artifact_id, root) for record in selected]

    @staticmethod
    def _failed_part(part: Path) -> Path:
        failed = part.with_name(part.name + ".failed")
        if failed.exists() or failed.is_symlink():
            if failed.is_symlink() or not failed.is_file():
                raise ArtifactError("failed-part destination is unsafe")
            failed.unlink()
        os.replace(part, failed)
        return failed

    @staticmethod
    def _source_failure(record: ArtifactRecord, reason: str) -> ArtifactError:
        return ArtifactError(
            f"{reason} for {record.artifact_id}: {record.url_path} "
            f"(expected sha256 {record.sha256})"
        )

    def _install_part(self, record: ArtifactRecord, target: Path, part: Path) -> None:
        if not _verified(part, record):
            self._failed_part(part)
            raise self._source_failure(
                record, "artifact failed size/checksum verification"
            )
        if target.is_symlink():
            raise ArtifactError(
                f"artifact destination contains a symlink: {record.relative_path}"
            )
        os.replace(part, target)

    def _fetch_from_bundle(
        self, record: ArtifactRecord, target: Path, bundle_root: Path
    ) -> None:
        source = _safe_bundle_source(bundle_root, record.url_path)
        if not source.is_file():
            raise self._source_failure(record, "local bundle member is missing")
        if not _verified(source, record):
            raise self._source_failure(record, "local bundle member failed checksum")
        part = target.with_name(target.name + ".part")
        if part.exists() or part.is_symlink():
            if part.is_symlink() or not part.is_file():
                raise ArtifactError("artifact partial destination is unsafe")
            part.unlink()
        with source.open("rb") as read_handle, part.open("xb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle, length=1024 * 1024)
            write_handle.flush()
            os.fsync(write_handle.fileno())
        self._install_part(record, target, part)

    @staticmethod
    def _open_remote(url: str, *, start: int | None):
        headers = {"Accept-Encoding": "identity"}
        if start is not None:
            headers["Range"] = f"bytes={start}-"
        request = Request(url, headers=headers)
        try:
            return urlopen(request, timeout=60)
        except HTTPError as error:
            if start is not None and error.code == 416:
                error.close()
                return None
            raise ArtifactError("remote artifact request failed") from error
        except (URLError, OSError) as error:
            raise ArtifactError("remote artifact request failed") from error

    def _fetch_remote(
        self,
        record: ArtifactRecord,
        target: Path,
        base_url: str,
        *,
        resume: bool,
    ) -> None:
        part = target.with_name(target.name + ".part")
        if part.is_symlink() or (part.exists() and not part.is_file()):
            raise ArtifactError("artifact partial destination is unsafe")
        if not resume and part.exists():
            part.unlink()
        start = part.stat().st_size if part.exists() else 0
        if start >= record.expected_bytes:
            part.unlink()
            start = 0
        url = urljoin(base_url.rstrip("/") + "/", record.url_path)
        try:
            response = self._open_remote(url, start=start if start else None)
        except ArtifactError as error:
            raise self._source_failure(
                record, "remote artifact is unavailable"
            ) from error
        if response is None:
            part.unlink(missing_ok=True)
            start = 0
            try:
                response = self._open_remote(url, start=None)
            except ArtifactError as error:
                raise self._source_failure(
                    record, "remote artifact is unavailable"
                ) from error
            if response is None:  # pragma: no cover - no Range header was sent
                raise ArtifactError("remote artifact request failed")
        status = int(getattr(response, "status", response.getcode()))
        append = False
        if start:
            if status == 206:
                expected_range = (
                    f"bytes {start}-{record.expected_bytes - 1}/{record.expected_bytes}"
                )
                append = response.headers.get("Content-Range") == expected_range
                if not append:
                    response.close()
                    part.unlink(missing_ok=True)
                    try:
                        response = self._open_remote(url, start=None)
                    except ArtifactError as error:
                        raise self._source_failure(
                            record, "remote artifact is unavailable"
                        ) from error
                    status = int(getattr(response, "status", response.getcode()))
            elif status == 200:
                append = False
            else:
                response.close()
                raise ArtifactError("remote server returned an invalid resume response")
        if not start and status != 200:
            response.close()
            raise ArtifactError("remote server did not return a complete artifact")
        if start and not append and status != 200:
            response.close()
            raise ArtifactError("remote server did not restart the artifact")
        encoding = (response.headers.get("Content-Encoding") or "identity").lower()
        if encoding != "identity":
            response.close()
            raise ArtifactError("remote artifact must use identity encoding")

        mode = "ab" if append else "wb"
        try:
            with response, part.open(mode) as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                    if handle.tell() > record.expected_bytes:
                        raise ArtifactError(
                            "remote artifact exceeded its declared size"
                        )
                handle.flush()
                os.fsync(handle.fileno())
        except ArtifactError:
            if part.exists():
                self._failed_part(part)
            raise
        except (OSError, HTTPError, URLError) as error:
            # Keep an interrupted .part so a later call can resume it.
            raise ArtifactError("remote artifact download was interrupted") from error
        if part.stat().st_size < record.expected_bytes:
            raise self._source_failure(
                record, "remote artifact download was interrupted"
            )
        self._install_part(record, target, part)


def default_manifest_path() -> Path:
    """Locate the tracked manifest for an editable/native installation."""
    configured = os.environ.get("SCENARIOFLOW_ARTIFACT_MANIFEST")
    if configured:
        return Path(configured).expanduser()
    repository_manifest = (
        Path(__file__).resolve().parents[2] / "artifacts" / "manifest-v1.tsv"
    )
    if repository_manifest.is_file():
        return repository_manifest
    current_manifest = Path.cwd() / "artifacts" / "manifest-v1.tsv"
    return current_manifest
