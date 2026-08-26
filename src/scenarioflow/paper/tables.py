"""Identity verification for preserved paper tables and safe output writes."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Mapping

PRESERVED_TABLE_CONTRACTS = {
    "fmts.benchmark-full": {
        "released_path": "paper_assets/tables/golden/fmts-benchmark-full.tex",
        "bytes": 5_436,
        "sha256": "521f596246bbd74b9ed2b2735977d6ba606b4bc91f36cdfba8b2d06baf9492fb",
        "source_authority": (
            "selected-source.tar.zst:paper/camera_ready/neurips_workshop/"
            "main_v4_fmts_table1.tex"
        ),
    },
    "fmts.benchmark-short": {
        "released_path": "paper_assets/tables/golden/fmts-benchmark-short.tex",
        "bytes": 4_231,
        "sha256": "b505c03adbc507c01b8d0fffb1444ab4a0b85a06b32a73998d0a2dbae27c5e0d",
        "source_authority": (
            "selected-source.tar.zst:paper/camera_ready/neurips_workshop/"
            "main_v4_fmts_table1_short.tex"
        ),
    },
    "icaif.benchmark": {
        "released_path": "paper_assets/tables/golden/icaif-benchmark.tex",
        "bytes": 4_212,
        "sha256": "915f6a3cafd464b9d0e911fdb77a5530044dc09a9f29f44b485ebdddafb54943",
        "source_authority": "paper-authority:main_v2_reframed.tex#tab:baseline-comparison",
    },
}


class BenchmarkExportError(ValueError):
    """Raised when a paper output or preserved benchmark view is unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lexical_path(path: Path) -> Path:
    supplied = path.expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    lexical = Path(os.path.abspath(supplied))
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise BenchmarkExportError("paper output path or an ancestor is a symlink")
    return lexical


def _write_output_batch(payloads: Mapping[Path, bytes]) -> None:
    """Atomically replace regular caller-selected output files."""
    staged: dict[Path, Path] = {}
    try:
        for supplied, payload in payloads.items():
            destination = _lexical_path(Path(supplied))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not destination.is_file():
                raise BenchmarkExportError("paper output path is unusable")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload)
                staged[destination] = Path(handle.name)
        for destination, temporary in staged.items():
            temporary.replace(destination)
    except (OSError, BenchmarkExportError) as error:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        if isinstance(error, BenchmarkExportError):
            raise
        raise BenchmarkExportError("paper output path is unusable") from error


def verify_preserved_table_views(
    repository_root: Path, asset_manifest: Path
) -> dict[str, Path]:
    """Verify final benchmark views without claiming to regenerate their rows."""
    try:
        with Path(asset_manifest).open(encoding="utf-8", newline="") as handle:
            rows = {
                row["asset_id"]: row for row in csv.DictReader(handle, delimiter="\t")
            }
    except (OSError, csv.Error, KeyError) as error:
        raise BenchmarkExportError("paper asset manifest is unreadable") from error

    verified: dict[str, Path] = {}
    root = Path(repository_root)
    for asset_id, expected in PRESERVED_TABLE_CONTRACTS.items():
        row = rows.get(asset_id)
        if (
            row is None
            or row.get("released_path") != expected["released_path"]
            or row.get("bytes") != str(expected["bytes"])
            or row.get("sha256") != expected["sha256"]
            or row.get("mode") != "preserved-only"
            or row.get("generator") != ""
            or row.get("source_authority") != expected["source_authority"]
        ):
            raise BenchmarkExportError(
                f"preserved benchmark contract changed: {asset_id}"
            )
        path = root / str(expected["released_path"])
        try:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != expected["bytes"]
                or _sha256(path) != expected["sha256"]
            ):
                raise BenchmarkExportError(
                    f"preserved benchmark identity changed: {asset_id}"
                )
        except OSError as error:
            raise BenchmarkExportError(
                f"preserved benchmark is unreadable: {asset_id}"
            ) from error
        verified[asset_id] = path
    return verified
