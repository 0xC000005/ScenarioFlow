"""Per-cell quantile mapping of daily IV changes.

Transforms generated daily changes to match GT daily change distributions
per cell, fitted from training data. Applied post-generation at inference time.
"""

import numpy as np


class QuantileMapper:
    """Per-cell quantile mapping of daily IV changes.

    Loads a fitted quantile map (.npz) with gen_quantiles and gt_quantiles,
    both shape (25, N_quantiles). Applies monotone per-cell transformation
    to daily changes, then reconstructs IV levels.
    """

    def __init__(self, quantile_map_path: str, alpha: float = 1.0):
        """Load fitted quantile map.

        Args:
            quantile_map_path: path to .npz file
            alpha: blending factor [0, 1]. 1.0 = full mapping, 0.0 = identity.
                   Intermediate values blend: target = gen + alpha*(gt - gen)
        """
        data = np.load(quantile_map_path)
        self.gen_quantiles = data["gen_quantiles"]  # (25, N_q)
        gt_raw = data["gt_quantiles"]               # (25, N_q)
        # Blend target quantiles: gen + alpha*(gt - gen)
        self.gt_quantiles = self.gen_quantiles + alpha * (gt_raw - self.gen_quantiles)
        self.alpha = alpha
        assert self.gen_quantiles.shape == self.gt_quantiles.shape
        self.n_cells = self.gen_quantiles.shape[0]
        # Shape-only mode: quantiles are in standardized space
        self.shape_only = bool(data.get("shape_only", False))
        if self.shape_only:
            self.gen_mean = data["gen_mean"]  # (25,)
            self.gen_std = data["gen_std"]    # (25,)

    @staticmethod
    def _interp_with_extrapolation(x, xp, fp):
        """np.interp with linear extrapolation beyond fitted range."""
        result = np.interp(x, xp, fp)
        # Extrapolate below
        below = x < xp[0]
        if np.any(below):
            slope_lo = (fp[1] - fp[0]) / (xp[1] - xp[0]) if xp[1] != xp[0] else 0.0
            result[below] = fp[0] + slope_lo * (x[below] - xp[0])
        # Extrapolate above
        above = x > xp[-1]
        if np.any(above):
            slope_hi = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) if xp[-1] != xp[-2] else 0.0
            result[above] = fp[-1] + slope_hi * (x[above] - xp[-1])
        return result

    def map_changes(self, changes: np.ndarray) -> np.ndarray:
        """Map daily changes per cell.

        Args:
            changes: (..., 5, 5) daily changes in IV space
        Returns:
            mapped: same shape, with per-cell quantile mapping applied
        """
        orig_shape = changes.shape
        spatial = changes.reshape(-1, 5, 5)  # (M, 5, 5)
        mapped = np.empty_like(spatial)
        for r in range(5):
            for c in range(5):
                idx = r * 5 + c
                vals = spatial[:, r, c]
                if self.shape_only:
                    # Standardize → map in z-space → destandardize
                    z = (vals - self.gen_mean[idx]) / max(self.gen_std[idx], 1e-10)
                    z_mapped = self._interp_with_extrapolation(
                        z, self.gen_quantiles[idx], self.gt_quantiles[idx]
                    )
                    mapped[:, r, c] = z_mapped * self.gen_std[idx] + self.gen_mean[idx]
                else:
                    mapped[:, r, c] = self._interp_with_extrapolation(
                        vals, self.gen_quantiles[idx], self.gt_quantiles[idx]
                    )
        return mapped.reshape(orig_shape)

    @staticmethod
    def _reflect_clip(x, lo, hi):
        """Reflect values into [lo, hi] instead of hard clipping."""
        width = hi - lo
        shifted = x - lo
        shifted = shifted % (2 * width)
        return np.where(shifted > width, 2 * width - shifted, shifted) + lo

    def apply(self, samples: np.ndarray, history: np.ndarray,
              reflect: bool = False) -> np.ndarray:
        """Apply quantile mapping to sample trajectories.

        Args:
            samples: (N, S, T, 5, 5) generated IV surfaces in [0, 1]
            history: (N, H, 5, 5) history IV surfaces in [0, 1]
            reflect: if True, use reflecting boundaries instead of hard clip
        Returns:
            mapped_samples: (N, S, T, 5, 5) with mapped daily changes
        """
        N, S, T, H, W = samples.shape
        anchor = history[:, -1:]  # (N, 1, 5, 5)
        anchor_exp = np.broadcast_to(
            anchor[:, np.newaxis, :, :, :], (N, S, 1, H, W)
        ).copy()
        full_traj = np.concatenate([anchor_exp, samples], axis=2)  # (N, S, T+1, 5, 5)
        changes = np.diff(full_traj, axis=2)  # (N, S, T, 5, 5)
        mapped_changes = self.map_changes(changes)
        mapped_samples = anchor_exp + np.cumsum(mapped_changes, axis=2)
        if reflect:
            return self._reflect_clip(mapped_samples, 0.0, 1.0)
        return np.clip(mapped_samples, 0.0, 1.0)
