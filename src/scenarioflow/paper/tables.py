"""Identity verification for preserved paper tables and safe output writes."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from ..artifacts import ArtifactResolver
from ..benchmark.reporting import metric_values
from .authority import AuthorityError, FrozenAuthority, load_authorities

SCENARIOFLOW_AUTHORITY_ID = (
    "release.results.frozen.benchmark.current-panel.scenarioflow.results.json"
)
SCENARIOFLOW_GRAPH_AUTHORITY_ID = (
    "release.results.frozen.benchmark.current-panel.scenarioflow-graph.results.json"
)

PRESERVED_TABLE_CONTRACTS = {
    "fmts.benchmark-full": {
        "released_path": "paper_assets/tables/golden/fmts-benchmark-full.tex",
        "bytes": 5_436,
        "sha256": "521f596246bbd74b9ed2b2735977d6ba606b4bc91f36cdfba8b2d06baf9492fb",
        "source_authority": (
            "selected-source.tar.zst:paper/camera_ready/neurips_workshop/"
            "main_v4_fmts_table1.tex"
        ),
        "row_authority_ids": (SCENARIOFLOW_AUTHORITY_ID,),
    },
    "fmts.benchmark-short": {
        "released_path": "paper_assets/tables/golden/fmts-benchmark-short.tex",
        "bytes": 4_231,
        "sha256": "b505c03adbc507c01b8d0fffb1444ab4a0b85a06b32a73998d0a2dbae27c5e0d",
        "source_authority": (
            "selected-source.tar.zst:paper/camera_ready/neurips_workshop/"
            "main_v4_fmts_table1_short.tex"
        ),
        "row_authority_ids": (SCENARIOFLOW_AUTHORITY_ID,),
    },
    "icaif.benchmark": {
        "released_path": "paper_assets/tables/golden/icaif-benchmark.tex",
        "bytes": 4_212,
        "sha256": "915f6a3cafd464b9d0e911fdb77a5530044dc09a9f29f44b485ebdddafb54943",
        "source_authority": "paper-authority:main_v2_reframed.tex#tab:baseline-comparison",
        "row_authority_ids": (
            SCENARIOFLOW_AUTHORITY_ID,
            SCENARIOFLOW_GRAPH_AUTHORITY_ID,
        ),
    },
}

DOLLAR_LIQUIDITY_ASSET_IDS = (
    "fmts.dollar-liquidity-full",
    "fmts.dollar-liquidity-short",
    "icaif.dollar-liquidity",
)
DOLLAR_LIQUIDITY_ARTIFACT_IDS = (
    "evidence.nl.dollar-liquidity-cohesive",
    "evidence.nl.dollar-liquidity-start-only",
)
DOLLAR_LIQUIDITY_REPORT_AUTHORITY_IDS = (
    "release.results.frozen.dollar-liquidity.narrative-report.json",
    "release.results.frozen.dollar-liquidity.start-only-report.json",
    "release.results.frozen.dollar-liquidity.fixed-start-comparison.json",
)

_PANEL_A_METRICS = (
    "CRPS",
    "IV CRPS",
    "Factor CRPS",
    "Energy Score",
    "Variogram",
    "Corr score",
    "Factor KS",
    "Tail q99",
)
_PANEL_B_METRICS = (
    "Cov90",
    "CalErr",
    "Level KS",
    "Median dev",
    "Width rho",
    "MR ratio",
    "Econ ratio",
    "Jump KS",
    "Kurtosis",
    "ACF corr",
)
_SHORT_METRICS = (
    "CRPS",
    "Energy Score",
    "Corr score",
    "Factor KS",
    "Tail q99",
    "Cov90",
    "CalErr",
    "Level KS",
    "Width rho",
    "MR ratio",
    "Econ ratio",
    "Jump KS",
    "Kurtosis",
)
_DECIMALS = {
    "CRPS": 4,
    "IV CRPS": 4,
    "Factor CRPS": 4,
    "Energy Score": 3,
    "Variogram": 4,
    "Corr score": 3,
    "Factor KS": 3,
    "Tail q99": 3,
    **{name: 3 for name in _PANEL_B_METRICS},
}
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\.\d+|\d+)")


class BenchmarkExportError(ValueError):
    """Raised when a paper output or preserved benchmark view is unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_rows(path: Path) -> dict[str, dict[str, str]]:
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            return {
                row["asset_id"]: row for row in csv.DictReader(handle, delimiter="\t")
            }
    except (OSError, csv.Error, KeyError) as error:
        raise BenchmarkExportError("paper asset manifest is unreadable") from error


def _authority_map(root: Path) -> dict[str, FrozenAuthority]:
    try:
        authorities = load_authorities(root / "results" / "FROZEN_AUTHORITY.tsv")
    except AuthorityError as error:
        raise BenchmarkExportError(str(error)) from error
    return {authority.authority_id: authority for authority in authorities}


def _verified_authority_path(
    root: Path, authorities: Mapping[str, FrozenAuthority], authority_id: str
) -> Path:
    authority = authorities.get(authority_id)
    if authority is None:
        raise BenchmarkExportError(f"missing frozen authority: {authority_id}")
    path = root / authority.released_path
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != authority.released_bytes
            or _sha256(path) != authority.released_sha256
        ):
            raise BenchmarkExportError(f"frozen authority changed: {authority_id}")
    except OSError as error:
        raise BenchmarkExportError(
            f"frozen authority is unreadable: {authority_id}"
        ) from error
    return path


def _source_tokens(source: str) -> list[str]:
    return source.split(";") if source else []


def _benchmark_source_contract(
    row: Mapping[str, str], expected: Mapping[str, Any]
) -> dict[str, str]:
    tokens = _source_tokens(row.get("source_authority", ""))
    declared = f"declared-provenance:{expected['source_authority']}"
    if not tokens or tokens[0] != declared:
        raise BenchmarkExportError("preserved benchmark declared provenance changed")
    row_authorities: dict[str, str] = {}
    for token in tokens[1:]:
        if not token.startswith("frozen-row:") or "@" not in token:
            raise BenchmarkExportError("preserved benchmark row authority changed")
        authority_id, digest = token.removeprefix("frozen-row:").rsplit("@", 1)
        row_authorities[authority_id] = digest
    if set(row_authorities) != set(expected["row_authority_ids"]):
        raise BenchmarkExportError("preserved benchmark row authority changed")
    return row_authorities


def _logical_model_rows(text: str, label: str) -> list[str]:
    rows: list[str] = []
    current: list[str] | None = None
    prefix = f"{label} &"
    for line in text.splitlines():
        stripped = line.strip()
        if current is None and stripped.startswith(prefix):
            current = [stripped]
        elif current is not None:
            current.append(stripped)
        if current is not None and stripped.endswith(r"\\"):
            rows.append(" ".join(current))
            current = None
    return rows


def _formatted_metrics(
    document: Mapping[str, Any], names: tuple[str, ...], *, fmts: bool
) -> list[str]:
    try:
        values = metric_values(document)
        if fmts:
            values["Econ ratio"] = float(
                document["current_panel_iv"]["iv_ewma_economic_link"]["slope_ratio"]
            )
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkExportError(
            "preserved benchmark row authority is invalid"
        ) from error
    return [f"{values[name]:.{_DECIMALS[name]}f}" for name in names]


def _verify_model_rows(
    asset_id: str,
    table: str,
    documents: Mapping[str, Mapping[str, Any]],
) -> None:
    checks: list[tuple[str, str, tuple[str, ...], bool]]
    if asset_id == "fmts.benchmark-full":
        checks = [
            (r"\scenarioflow{}", SCENARIOFLOW_AUTHORITY_ID, _PANEL_A_METRICS, False),
            (r"\scenarioflow{}", SCENARIOFLOW_AUTHORITY_ID, _PANEL_B_METRICS, True),
        ]
    elif asset_id == "fmts.benchmark-short":
        checks = [(r"\scenarioflow{}", SCENARIOFLOW_AUTHORITY_ID, _SHORT_METRICS, True)]
    else:
        checks = [
            (r"\scenarioflow{}", SCENARIOFLOW_AUTHORITY_ID, _PANEL_A_METRICS, False),
            (r"\scenarioflow{}", SCENARIOFLOW_AUTHORITY_ID, _PANEL_B_METRICS, False),
            (
                r"\scenarioflow{} + graph",
                SCENARIOFLOW_GRAPH_AUTHORITY_ID,
                _PANEL_A_METRICS,
                False,
            ),
            (
                r"\scenarioflow{} + graph",
                SCENARIOFLOW_GRAPH_AUTHORITY_ID,
                _PANEL_B_METRICS,
                False,
            ),
        ]
    offsets: dict[str, int] = {}
    for label, authority_id, metrics, fmts in checks:
        rows = _logical_model_rows(table, label)
        offset = offsets.get(label, 0)
        if offset >= len(rows):
            raise BenchmarkExportError(
                f"preserved benchmark row contract changed: {asset_id} {label}"
            )
        actual = _NUMBER.findall(rows[offset].split("&", 1)[1])
        expected = _formatted_metrics(documents[authority_id], metrics, fmts=fmts)
        if actual[: len(expected)] != expected:
            raise BenchmarkExportError(
                f"preserved benchmark row contract changed: {asset_id} {label}"
            )
        offsets[label] = offset + 1


def verify_dollar_liquidity_authorities(
    repository_root: Path,
    asset_manifest: Path,
    artifact_resolver: ArtifactResolver,
) -> None:
    """Verify every array/report authority declared by the dollar table views."""
    root = Path(repository_root)
    rows = _asset_rows(asset_manifest)
    authorities = _authority_map(root)
    artifact_tokens = {
        f"artifact:{artifact_id}" for artifact_id in DOLLAR_LIQUIDITY_ARTIFACT_IDS
    }
    for artifact_id in DOLLAR_LIQUIDITY_ARTIFACT_IDS:
        artifact_resolver.record(artifact_id)
    frozen_tokens = set()
    for authority_id in DOLLAR_LIQUIDITY_REPORT_AUTHORITY_IDS:
        _verified_authority_path(root, authorities, authority_id)
        frozen_tokens.add(
            f"frozen:{authority_id}@{authorities[authority_id].released_sha256}"
        )
    expected_tokens = artifact_tokens | frozen_tokens
    for asset_id in DOLLAR_LIQUIDITY_ASSET_IDS:
        row = rows.get(asset_id)
        actual_tokens = (
            set(_source_tokens(row.get("source_authority", ""))) if row else set()
        )
        if actual_tokens != expected_tokens:
            raise BenchmarkExportError(
                f"dollar-liquidity authority contract changed: {asset_id}"
            )


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
    """Verify identities, declared provenance, and surviving result rows."""
    rows = _asset_rows(asset_manifest)
    verified: dict[str, Path] = {}
    root = Path(repository_root)
    authorities = _authority_map(root)
    for asset_id, expected in PRESERVED_TABLE_CONTRACTS.items():
        row = rows.get(asset_id)
        if (
            row is None
            or row.get("released_path") != expected["released_path"]
            or row.get("mode") != "preserved-only"
            or row.get("generator") != ""
        ):
            raise BenchmarkExportError(
                f"preserved benchmark contract changed: {asset_id}"
            )
        row_authorities = _benchmark_source_contract(row, expected)
        documents: dict[str, Mapping[str, Any]] = {}
        for authority_id, declared_digest in row_authorities.items():
            authority = authorities.get(authority_id)
            if authority is None or authority.released_sha256 != declared_digest:
                raise BenchmarkExportError("preserved benchmark row authority changed")
            authority_path = _verified_authority_path(root, authorities, authority_id)
            try:
                document = json.loads(authority_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BenchmarkExportError(
                    "preserved benchmark row authority is invalid"
                ) from error
            if not isinstance(document, dict):
                raise BenchmarkExportError(
                    "preserved benchmark row authority is invalid"
                )
            documents[authority_id] = document
        path = root / str(expected["released_path"])
        try:
            table = path.read_text(encoding="utf-8")
            _verify_model_rows(asset_id, table, documents)
            if (
                path.is_symlink()
                or not path.is_file()
                or row.get("bytes") != str(expected["bytes"])
                or row.get("sha256") != expected["sha256"]
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


def materialize_preserved_table_views(
    repository_root: Path, asset_manifest: Path, output_dir: Path
) -> dict[str, Path]:
    """Place the verified camera-ready benchmark views in a caller-owned directory."""
    verified = verify_preserved_table_views(repository_root, asset_manifest)
    destination = Path(output_dir)
    outputs = {
        asset_id: destination / source.name for asset_id, source in verified.items()
    }
    _write_output_batch(
        {
            outputs[asset_id]: source.read_bytes()
            for asset_id, source in verified.items()
        }
    )
    return outputs
