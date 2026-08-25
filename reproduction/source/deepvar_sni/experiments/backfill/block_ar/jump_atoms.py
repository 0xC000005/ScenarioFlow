from __future__ import annotations

import math
from typing import Any

import torch

from experiments.backfill.block_ar.basis_geometry import BasisGeometry


def representative_spatial_indices(grid_h: int, grid_w: int) -> dict[str, int]:
    n_cells = grid_h * grid_w
    low = 0
    mid = min(n_cells - 1, grid_w + 1 if grid_h > 1 and grid_w > 1 else max(1, n_cells // 2))
    high = n_cells - 1
    return {"low": low, "mid": mid, "high": high}


def build_block_band_jump_atoms(
    geometry: BasisGeometry,
    n_blocks: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if geometry.n_frames % n_blocks != 0:
        raise ValueError(f"n_frames={geometry.n_frames} must be divisible by n_blocks={n_blocks}")
    block_len = geometry.n_frames // n_blocks
    spatial_indices = representative_spatial_indices(geometry.grid_h, geometry.grid_w)
    atoms = []
    metadata = []

    for block in range(n_blocks):
        time_mask = torch.zeros(geometry.n_frames, dtype=geometry.basis_time.dtype)
        time_mask[block * block_len : (block + 1) * block_len] = 1.0 / math.sqrt(block_len)
        for band in ("low", "mid", "high"):
            spatial_pattern = geometry.basis_space[spatial_indices[band]]
            atom_time_cell = time_mask[:, None] * spatial_pattern[None, :]
            atom_basis = geometry.to_basis(atom_time_cell.unsqueeze(0)).reshape(-1)
            atom_basis = atom_basis / atom_basis.norm().clamp_min(1e-8)
            atoms.append(atom_basis)
            metadata.append(
                {
                    "block": int(block),
                    "band": band,
                    "spatial_basis_index": int(spatial_indices[band]),
                }
            )

    return torch.stack(atoms, dim=0), metadata
