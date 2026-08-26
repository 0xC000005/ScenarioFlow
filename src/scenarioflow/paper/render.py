"""Render numerical paper assets from checksum-pinned released evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..artifacts import ArtifactError, ArtifactResolver
from .dollar_liquidity import (
    DollarLiquidityExportError,
    export_dollar_liquidity_tables,
)
from .multiseed import MultiseedExportError, export_multiseed_table
from .plots import (
    PaperPlotError,
    plot_iv_cross_cell_sensitivity,
    plot_iv_marginal_daily_changes,
    plot_joint_factor_fans,
)
from .tables import (
    BenchmarkExportError,
    _write_output_batch,
    verify_preserved_table_views,
)

PaperMode = Literal["all", "tables", "plots"]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
IV_ARRAYS_ID = "paper.icaif.iv-plot-arrays"
JOINT_ARRAYS_ID = "paper.icaif.joint-factor-fan-arrays"
NARRATIVE_ARRAYS_ID = "evidence.nl.dollar-liquidity-cohesive"
START_ONLY_ARRAYS_ID = "evidence.nl.dollar-liquidity-start-only"

EXACT_OUTPUT_IDENTITIES = {
    "joint-factor-fans-selection.json": (
        1_536,
        "c30a1b42580d6e6686924ae3ef33e4dbe5436f95af11d6c89e331057fedb79f1",
    ),
    "fmts-multiseed.tex": (
        7_398,
        "a04e9cdf9fc8c0cb36e1606e479d894b7255c91311ae678e257e690546878192",
    ),
    "fmts-dollar-liquidity-full.tex": (
        2_503,
        "6c92a8310dc69502bfc132c49c84dbf2bbf8b28fdc230bd43df058423b1696bb",
    ),
    "fmts-dollar-liquidity-short.tex": (
        2_360,
        "db6e6e6aff6ba2f37da608a050cea94449bbb17b391ee38498311a42b9cfa0dc",
    ),
    "icaif-dollar-liquidity.tex": (
        2_422,
        "a49f1a6a9c2e0363d0ef86e30198ed909166dfd8fbf0a80cde9b8c22941bc767",
    ),
}

FIGURE_GOLDENS = {
    "iv-marginal-daily-changes.png": "iv-marginal-daily-changes.png",
    "iv-cross-cell-sensitivity.png": "iv-cross-cell-sensitivity.png",
    "joint-factor-fans.png": "joint-factor-fans.png",
}
FIGURE_DIMENSION_TOLERANCE = 1
FIGURE_THUMBNAIL_SIZE = (128, 128)
FIGURE_NORMALIZED_MAE_LIMIT = 0.045


class PaperRenderError(ValueError):
    """Raised when released evidence cannot produce the declared paper output."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_root(path: Path) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    output = Path(os.path.abspath(supplied))
    current = Path(output.anchor)
    for component in output.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise PaperRenderError("paper output path or an ancestor is a symlink")
    for protected in (
        REPOSITORY_ROOT / "paper_assets",
        REPOSITORY_ROOT / "results" / "frozen",
    ):
        protected = Path(os.path.abspath(protected))
        if output == protected or protected in output.parents:
            raise PaperRenderError("refusing to overwrite frozen paper evidence")
    return output


def _load_npz(path: Path, *, schema: str, keys: set[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != keys:
                raise PaperRenderError(f"released array schema changed: {path.name}")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, PaperRenderError):
            raise
        raise PaperRenderError(
            f"released arrays are unreadable: {path.name}"
        ) from error
    if (
        arrays["schema_version"].shape != ()
        or str(arrays["schema_version"].item()) != schema
    ):
        raise PaperRenderError(f"released array schema changed: {path.name}")
    return arrays


def _verify_figure_content(name: str, rendered: Path, golden: Path) -> None:
    """Check stable visual content without pinning PNG encoder metadata."""
    try:
        from PIL import Image

        with Image.open(rendered) as rendered_image, Image.open(golden) as golden_image:
            rendered_size = rendered_image.size
            golden_size = golden_image.size
            if any(
                abs(current - expected) > FIGURE_DIMENSION_TOLERANCE
                for current, expected in zip(rendered_size, golden_size, strict=True)
            ):
                raise PaperRenderError(f"rendered figure dimensions changed: {name}")
            resampling = Image.Resampling.LANCZOS
            current_pixels = np.asarray(
                rendered_image.convert("RGB").resize(FIGURE_THUMBNAIL_SIZE, resampling),
                dtype=np.float64,
            )
            golden_pixels = np.asarray(
                golden_image.convert("RGB").resize(FIGURE_THUMBNAIL_SIZE, resampling),
                dtype=np.float64,
            )
    except (ImportError, OSError, ValueError) as error:
        if isinstance(error, PaperRenderError):
            raise
        raise PaperRenderError(f"rendered figure is unreadable: {name}") from error
    normalized_mae = float(np.mean(np.abs(current_pixels - golden_pixels)) / 255.0)
    if normalized_mae > FIGURE_NORMALIZED_MAE_LIMIT:
        raise PaperRenderError(
            f"rendered figure content changed: {name} "
            f"(normalized MAE {normalized_mae:.4f})"
        )


def _verify_outputs(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        golden_name = FIGURE_GOLDENS.get(path.name)
        if golden_name is not None:
            _verify_figure_content(
                name,
                path,
                REPOSITORY_ROOT
                / "paper_assets"
                / "figures"
                / "generated"
                / golden_name,
            )
            continue
        expected = EXACT_OUTPUT_IDENTITIES.get(path.name)
        if expected is None:
            continue
        try:
            if path.stat().st_size != expected[0] or _sha256(path) != expected[1]:
                raise PaperRenderError(f"rendered output identity changed: {name}")
        except OSError as error:
            raise PaperRenderError(f"rendered output is unreadable: {name}") from error


def _render_plots(
    resolver: ArtifactResolver, artifact_root: Path, output: Path
) -> dict[str, Path]:
    iv_path = resolver.resolve(IV_ARRAYS_ID, artifact_root)
    joint_path = resolver.resolve(JOINT_ARRAYS_ID, artifact_root)
    iv = _load_npz(
        iv_path,
        schema="scenarioflow.paper.iv-plot-arrays.v1",
        keys={
            "schema_version",
            "history",
            "future",
            "samples",
            "seed",
            "checkpoint_sha256",
            "device",
        },
    )
    joint = _load_npz(
        joint_path,
        schema="scenarioflow.paper.joint-factor-fan-arrays.v1",
        keys={
            "schema_version",
            "history",
            "future",
            "samples",
            "spec_names",
            "end_dates",
            "selection_json",
            "seed",
            "checkpoint_sha256",
            "device",
        },
    )

    destination = output / "plots"
    paths = {
        "icaif.iv-marginal": destination / "iv-marginal-daily-changes.png",
        "icaif.iv-cross-cell": destination / "iv-cross-cell-sensitivity.png",
        "icaif.joint-factor-fans": destination / "joint-factor-fans.png",
        "icaif.joint-factor-fans-selection": (
            destination / "joint-factor-fans-selection.json"
        ),
    }
    plot_iv_marginal_daily_changes(
        iv["history"], iv["future"], iv["samples"], paths["icaif.iv-marginal"]
    )
    plot_iv_cross_cell_sensitivity(
        iv["history"], iv["future"], iv["samples"], paths["icaif.iv-cross-cell"]
    )
    plot_joint_factor_fans(
        joint["history"],
        joint["future"],
        joint["samples"],
        [str(value) for value in joint["spec_names"].tolist()],
        [str(value) for value in joint["end_dates"].tolist()],
        paths["icaif.joint-factor-fans"],
    )
    selection = joint["selection_json"]
    if selection.shape != ():
        raise PaperRenderError("joint factor-fan selection evidence changed")
    selection_payload = str(selection.item()).encode("utf-8")
    _write_output_batch({paths["icaif.joint-factor-fans-selection"]: selection_payload})
    _verify_outputs(paths)
    return paths


def _render_tables(
    resolver: ArtifactResolver, artifact_root: Path, output: Path
) -> dict[str, Path]:
    narrative_arrays = resolver.resolve(NARRATIVE_ARRAYS_ID, artifact_root)
    start_only_arrays = resolver.resolve(START_ONLY_ARRAYS_ID, artifact_root)
    destination = output / "tables"
    multiseed = export_multiseed_table(
        REPOSITORY_ROOT / "results" / "frozen" / "multiseed" / "summary.json",
        destination / "fmts-multiseed.tex",
    )
    dollar = export_dollar_liquidity_tables(
        REPOSITORY_ROOT
        / "results"
        / "frozen"
        / "dollar-liquidity"
        / "narrative-report.json",
        narrative_arrays,
        REPOSITORY_ROOT
        / "results"
        / "frozen"
        / "dollar-liquidity"
        / "start-only-report.json",
        start_only_arrays,
        REPOSITORY_ROOT
        / "results"
        / "frozen"
        / "dollar-liquidity"
        / "fixed-start-comparison.json",
        destination,
    )
    paths = {
        "fmts.multiseed": multiseed,
        "fmts.dollar-liquidity-full": dollar["fmts_full_tex"],
        "fmts.dollar-liquidity-short": dollar["fmts_short_tex"],
        "icaif.dollar-liquidity": dollar["icaif_tex"],
        "evidence.dollar-liquidity-json": dollar["json"],
        "evidence.dollar-liquidity-csv": dollar["csv"],
    }
    _verify_outputs(paths)
    return paths


def render_paper_assets(
    *,
    mode: PaperMode,
    artifact_manifest: Path,
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Render supported numerical outputs and verify preserved benchmark views."""
    if mode not in {"all", "tables", "plots"}:
        raise PaperRenderError(f"unknown paper rendering mode: {mode}")
    output = _output_root(output_dir)
    resolver = ArtifactResolver.from_tsv(artifact_manifest)
    required = []
    if mode in {"all", "plots"}:
        required.extend((IV_ARRAYS_ID, JOINT_ARRAYS_ID))
    if mode in {"all", "tables"}:
        required.extend((NARRATIVE_ARRAYS_ID, START_ONLY_ARRAYS_ID))
    resolved = [
        resolver.resolve(artifact_id, artifact_root) for artifact_id in required
    ]
    if len(resolved) != len(required):  # pragma: no cover - resolver is fail-closed
        raise PaperRenderError("required released paper evidence is unavailable")

    preserved = verify_preserved_table_views(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "paper_assets" / "ASSETS.tsv"
    )
    generated: dict[str, Path] = {}
    try:
        if mode in {"all", "tables"}:
            generated.update(_render_tables(resolver, artifact_root, output))
        if mode in {"all", "plots"}:
            generated.update(_render_plots(resolver, artifact_root, output))
    except (
        ArtifactError,
        BenchmarkExportError,
        DollarLiquidityExportError,
        MultiseedExportError,
        PaperPlotError,
    ) as error:
        raise PaperRenderError(str(error)) from error
    return {
        "mode": mode,
        "output_dir": str(output),
        "generated": {key: str(path) for key, path in sorted(generated.items())},
        "preserved_only": {key: str(path) for key, path in sorted(preserved.items())},
    }


def render_report_json(report: dict[str, Any]) -> str:
    """Serialize one stable CLI response."""
    return json.dumps(report, sort_keys=True)
