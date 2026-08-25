from __future__ import annotations

import math

import torch
import torch.nn as nn


def orthonormal_dct_matrix(n: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Orthonormal DCT-II matrix with rows as basis vectors."""
    k = torch.arange(n, device=device, dtype=dtype or torch.float32).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype or torch.float32).unsqueeze(0)
    mat = torch.cos(math.pi / n * (i + 0.5) * k)
    mat[0] = mat[0] / math.sqrt(n)
    if n > 1:
        mat[1:] = mat[1:] * math.sqrt(2.0 / n)
    return mat


def spatial_dct2_basis(h: int, w: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (basis, frequency_score) for a 2D DCT basis on an h x w grid."""
    bh = orthonormal_dct_matrix(h, device=device, dtype=dtype)
    bw = orthonormal_dct_matrix(w, device=device, dtype=dtype)
    basis = torch.kron(bh, bw)  # (h*w, h*w)

    scores = []
    max_score = max((h - 1) + (w - 1), 1)
    for r in range(h):
        for c in range(w):
            scores.append((r + c) / max_score)
    freq_score = torch.tensor(scores, device=device, dtype=dtype or torch.float32)
    return basis, freq_score


class BasisGeometry(nn.Module):
    """Orthonormal spatial-temporal basis with simple low/mid/high frequency bands."""

    def __init__(
        self,
        n_frames: int,
        grid_h: int = 5,
        grid_w: int = 5,
        low_scale_clip: float = 1.2,
        mid_scale_clip: float = 0.7,
        high_scale_clip: float = 0.35,
    ):
        super().__init__()
        self.n_frames = int(n_frames)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.n_cells = self.grid_h * self.grid_w
        self.dim = self.n_frames * self.n_cells

        b_time = orthonormal_dct_matrix(self.n_frames)
        b_space, spatial_score = spatial_dct2_basis(self.grid_h, self.grid_w)
        temporal_score = torch.arange(self.n_frames, dtype=torch.float32) / max(self.n_frames - 1, 1)

        joint_score = 0.5 * temporal_score.unsqueeze(1) + 0.5 * spatial_score.unsqueeze(0)
        band_id = torch.zeros_like(joint_score, dtype=torch.long)
        band_id[joint_score > 1.0 / 3.0] = 1
        band_id[joint_score > 2.0 / 3.0] = 2

        scale_clip_grid = torch.full((self.n_frames, self.n_cells), low_scale_clip, dtype=torch.float32)
        scale_clip_grid[band_id == 1] = mid_scale_clip
        scale_clip_grid[band_id == 2] = high_scale_clip

        self.register_buffer("basis_time", b_time)
        self.register_buffer("basis_time_t", b_time.transpose(0, 1))
        self.register_buffer("basis_space", b_space)
        self.register_buffer("basis_space_t", b_space.transpose(0, 1))
        self.register_buffer("joint_score", joint_score)
        self.register_buffer("band_id", band_id)
        self.register_buffer("scale_clip_grid", scale_clip_grid)

    def to_basis(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., T, C) -> coeffs (..., T, C)."""
        orig_shape = x.shape
        x = x.reshape(-1, self.n_frames, self.n_cells)
        tmp = torch.einsum("tu,buc->btc", self.basis_time, x)
        coeff = torch.einsum("btc,cv->btv", tmp, self.basis_space_t)
        return coeff.reshape(orig_shape)

    def from_basis(self, coeff: torch.Tensor) -> torch.Tensor:
        """coeffs: (..., T, C) -> x (..., T, C)."""
        orig_shape = coeff.shape
        coeff = coeff.reshape(-1, self.n_frames, self.n_cells)
        tmp = torch.einsum("tu,buc->btc", self.basis_time_t, coeff)
        x = torch.einsum("btc,cv->btv", tmp, self.basis_space)
        return x.reshape(orig_shape)

    def clip_vector(self) -> torch.Tensor:
        return self.scale_clip_grid.reshape(-1)

    def high_band_mask(self) -> torch.Tensor:
        return (self.band_id == 2).float().reshape(-1)

    def mid_band_mask(self) -> torch.Tensor:
        return (self.band_id == 1).float().reshape(-1)

    def low_band_mask(self) -> torch.Tensor:
        return (self.band_id == 0).float().reshape(-1)
