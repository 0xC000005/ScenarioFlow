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
    verify_dollar_liquidity_authorities,
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

FIGURE_PANEL_GRIDS = {
    "iv-marginal-daily-changes.png": (3, 3),
    "iv-cross-cell-sensitivity.png": (2, 4),
    "joint-factor-fans.png": (2, 3),
}
FIGURE_DIMENSION_TOLERANCE = 1
FIGURE_TILE_SIZE = (192, 192)
FIGURE_TILE_MAE_LIMIT = 0.12
FIGURE_TILE_CORRELATION_MIN = 0.42
FIGURE_EDGE_DENSITY_MIN = 0.75
FIGURE_EDGE_DENSITY_MAX = 1.55
FIGURE_EDGE_THRESHOLD = 0.20


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
    """Check every panel's pixels and edges without pinning PNG metadata."""
    try:
        from PIL import Image
        from scipy import ndimage

        with Image.open(rendered) as rendered_image, Image.open(golden) as golden_image:
            rendered_size = rendered_image.size
            golden_size = golden_image.size
            if any(
                abs(current - expected) > FIGURE_DIMENSION_TOLERANCE
                for current, expected in zip(rendered_size, golden_size, strict=True)
            ):
                raise PaperRenderError(f"rendered figure dimensions changed: {name}")
            resampling = Image.Resampling.LANCZOS
            golden_rgb = golden_image.convert("RGB")
            rendered_rgb = rendered_image.convert("RGB").resize(
                golden_rgb.size, resampling
            )
            grid = FIGURE_PANEL_GRIDS.get(golden.name)
            if grid is None:
                raise PaperRenderError(f"rendered figure has no panel grid: {name}")
            rows, columns = grid
            width, height = golden_rgb.size
            x_edges = [index * width // columns for index in range(columns + 1)]
            y_edges = [index * height // rows for index in range(rows + 1)]
            for row in range(rows):
                for column in range(columns):
                    box = (
                        x_edges[column],
                        y_edges[row],
                        x_edges[column + 1],
                        y_edges[row + 1],
                    )
                    current_tile = rendered_rgb.crop(box)
                    golden_tile = golden_rgb.crop(box)
                    current_native = np.asarray(current_tile, dtype=np.int16)
                    golden_native = np.asarray(golden_tile, dtype=np.int16)
                    normalized_mae = float(
                        np.mean(np.abs(current_native - golden_native)) / 255.0
                    )
                    if normalized_mae > FIGURE_TILE_MAE_LIMIT:
                        raise PaperRenderError(
                            f"rendered figure tile agreement changed: {name} "
                            f"tile {row},{column} (MAE {normalized_mae:.4f})"
                        )

                    current_luminance = (
                        np.asarray(
                            current_tile.convert("L").resize(
                                FIGURE_TILE_SIZE, resampling
                            ),
                            dtype=np.float64,
                        )
                        / 255.0
                    )
                    golden_luminance = (
                        np.asarray(
                            golden_tile.convert("L").resize(
                                FIGURE_TILE_SIZE, resampling
                            ),
                            dtype=np.float64,
                        )
                        / 255.0
                    )
                    current_blurred = ndimage.gaussian_filter(current_luminance, 1.0)
                    golden_blurred = ndimage.gaussian_filter(golden_luminance, 1.0)
                    current_edges = np.hypot(
                        ndimage.sobel(current_blurred, axis=0),
                        ndimage.sobel(current_blurred, axis=1),
                    )
                    golden_edges = np.hypot(
                        ndimage.sobel(golden_blurred, axis=0),
                        ndimage.sobel(golden_blurred, axis=1),
                    )
                    current_density = float(
                        np.mean(current_edges > FIGURE_EDGE_THRESHOLD)
                    )
                    golden_density = float(
                        np.mean(golden_edges > FIGURE_EDGE_THRESHOLD)
                    )
                    edge_ratio = current_density / max(golden_density, 1e-12)
                    if not (
                        FIGURE_EDGE_DENSITY_MIN <= edge_ratio <= FIGURE_EDGE_DENSITY_MAX
                    ):
                        raise PaperRenderError(
                            f"rendered figure edge agreement changed: {name} "
                            f"tile {row},{column} (density ratio {edge_ratio:.3f})"
                        )

                    current_centered = current_blurred - current_blurred.mean()
                    golden_centered = golden_blurred - golden_blurred.mean()
                    denominator = float(
                        np.sqrt(
                            np.sum(current_centered * current_centered)
                            * np.sum(golden_centered * golden_centered)
                        )
                    )
                    correlation = (
                        float(np.sum(current_centered * golden_centered) / denominator)
                        if denominator > 1e-12
                        else 0.0
                    )
                    if correlation < FIGURE_TILE_CORRELATION_MIN:
                        raise PaperRenderError(
                            f"rendered figure tile agreement changed: {name} "
                            f"tile {row},{column} (correlation {correlation:.3f})"
                        )
    except (ImportError, OSError, ValueError) as error:
        if isinstance(error, PaperRenderError):
            raise
        raise PaperRenderError(f"rendered figure is unreadable: {name}") from error


def _verify_outputs(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        if path.name in FIGURE_PANEL_GRIDS:
            _verify_figure_content(
                name,
                path,
                REPOSITORY_ROOT / "paper_assets" / "figures" / "generated" / path.name,
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

    generated: dict[str, Path] = {}
    try:
        if mode in {"all", "tables"}:
            verify_dollar_liquidity_authorities(
                REPOSITORY_ROOT,
                REPOSITORY_ROOT / "paper_assets" / "ASSETS.tsv",
                resolver,
            )
        preserved = verify_preserved_table_views(
            REPOSITORY_ROOT, REPOSITORY_ROOT / "paper_assets" / "ASSETS.tsv"
        )
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
        "preserved_only": {
            key: {
                "path": str(path),
                "verification": (
                    "exact identity; declared paper-source provenance; "
                    "frozen benchmark row checked"
                ),
            }
            for key, path in sorted(preserved.items())
        },
    }


def render_report_json(report: dict[str, Any]) -> str:
    """Serialize one stable CLI response."""
    return json.dumps(report, sort_keys=True)
