"""Classical baselines for IV surface scenario generation.

All baselines implement the same interface as SinglePassBlockAR:
    model.sample(history, n_samples=50) -> (B, n_samples, T_fut, 5, 5) in [0, 1]

Joint 38-d interface (for multi-factor benchmark):
    model.sample_joint(history_changes, n_samples) -> (B, n_samples, T_fut, D)
    Returns daily changes (not levels). Caller handles IV reconstruction.

Baselines:
    1. RandomWalk — persist last surface + scaled Gaussian noise
    2. HistoricalSimulation — sample matching windows from training data
    3. UnconditionalBootstrap — resample daily changes (preserving cross-cell), cumsum
    4. PCAVARBaseline — PCA on surfaces, VAR(1) on factors, bootstrap residuals
    5. GARCHCCCBaseline — EWMA variance per cell + constant conditional correlation
    6. DCCGARCHBaseline — GARCH per cell + Dynamic Conditional Correlation
    7. FilteredHistoricalSimulation — EWMA variance + bootstrap standardized residuals
"""

import numpy as np
import torch
from scipy import optimize, stats as sp_stats


def denormalize_iv(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]"""
    return (x + 1.0) / 2.0


class BaselineModel:
    """Mixin providing eval()/train() no-ops for compatibility with test suites."""

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def sample_batched(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


# =============================================================================
# 1. Random Walk
# =============================================================================

class RandomWalkBaseline(BaselineModel):
    """Persist last observed surface + Gaussian noise scaled by historical vol.

    The simplest possible baseline. Each ensemble member evolves the last
    observed surface as a random walk with iid Gaussian increments whose
    scale matches the empirical daily change std per cell.
    """

    def __init__(self, surfaces: np.ndarray, future_len: int = 30):
        self.future_len = future_len
        daily_changes = np.diff(surfaces, axis=0)  # (N-1, 5, 5)
        self.cell_std = daily_changes.std(axis=0)  # (5, 5)

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history)  # (B, T, 5, 5)
        B = history_denorm.shape[0]
        last_surface = history_denorm[:, -1]  # (B, 5, 5)
        device = history.device

        cell_std = torch.tensor(
            self.cell_std, dtype=history.dtype, device=device
        )  # (5, 5)

        noise = torch.randn(
            B, n_samples, self.future_len, 5, 5,
            dtype=history.dtype, device=device,
        )
        increments = noise * cell_std[None, None, None, :, :]
        cum_increments = increments.cumsum(dim=2)

        samples = last_surface[:, None, None, :, :] + cum_increments
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 2. Historical Simulation (conditional on mean IV + vol-of-vol)
# =============================================================================

class HistoricalSimulation(BaselineModel):
    """Sample actual historical 30-day windows as scenarios.

    For each test history, matches training windows by (mean IV, vol-of-vol,
    term structure slope) and samples subsequent futures weighted by similarity.
    This is the standard Filtered Historical Simulation approach used in
    risk management (without the GARCH filter).
    """

    def __init__(
        self,
        surfaces: np.ndarray,
        history_len: int = 30,
        future_len: int = 30,
    ):
        self.history_len = history_len
        self.future_len = future_len

        n_windows = len(surfaces) - history_len - future_len + 1
        self.hist_windows = np.zeros(
            (n_windows, history_len, 5, 5), dtype=np.float32
        )
        self.fut_windows = np.zeros(
            (n_windows, future_len, 5, 5), dtype=np.float32
        )

        for i in range(n_windows):
            self.hist_windows[i] = surfaces[i : i + history_len]
            self.fut_windows[i] = surfaces[
                i + history_len : i + history_len + future_len
            ]

        # Multi-feature matching: (mean_iv, vol_of_vol, slope)
        # mean IV of last 5 days
        mean_iv = self.hist_windows[:, -5:].mean(axis=(1, 2, 3))  # (n,)
        # vol-of-vol: std of daily mean-IV changes
        mean_iv_ts = self.hist_windows.mean(axis=(2, 3))  # (n, H)
        daily_chg = np.diff(mean_iv_ts, axis=1)  # (n, H-1)
        vov = daily_chg.std(axis=1)  # (n,)
        # term structure slope: long tenor - short tenor (last day)
        last_day = self.hist_windows[:, -1]  # (n, 5, 5)
        slope = last_day[:, -1, 2] - last_day[:, 0, 2]  # (n,)  long-short ATM

        self.features = np.column_stack([mean_iv, vov, slope])  # (n, 3)
        # Normalize features for distance computation
        self.feat_mean = self.features.mean(axis=0)
        self.feat_std = self.features.std(axis=0) + 1e-8
        self.features_norm = (self.features - self.feat_mean) / self.feat_std

    def _compute_features(self, history_np: np.ndarray) -> np.ndarray:
        """Compute matching features from a single history window. (5,5) x T"""
        mean_iv = history_np[-5:].mean()
        mean_iv_ts = history_np.mean(axis=(1, 2))  # (T,)
        daily_chg = np.diff(mean_iv_ts)
        vov = daily_chg.std() if len(daily_chg) > 1 else 0.0
        slope = history_np[-1, -1, 2] - history_np[-1, 0, 2]
        feat = np.array([mean_iv, vov, slope])
        return (feat - self.feat_mean) / self.feat_std

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history).cpu().numpy()
        B = history_denorm.shape[0]

        all_samples = np.zeros(
            (B, n_samples, self.future_len, 5, 5), dtype=np.float32
        )

        for b in range(B):
            query = self._compute_features(history_denorm[b])
            distances = np.linalg.norm(
                self.features_norm - query[None, :], axis=1
            )
            # Softmax weighting (temperature = 1.0)
            weights = np.exp(-distances)
            weights /= weights.sum()

            indices = np.random.choice(
                len(self.fut_windows), size=n_samples, replace=True, p=weights
            )
            all_samples[b] = self.fut_windows[indices]

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 3. Unconditional Bootstrap (preserves cross-cell correlation per day)
# =============================================================================

class UnconditionalBootstrap(BaselineModel):
    """Resample daily IV changes from training data, cumsum to build paths.

    Each draw is a full (5,5) daily change, preserving cross-cell correlation
    within a day. Draws are iid across time (breaks temporal structure).
    This measures the value of conditioning — gap vs our model = conditionality.
    """

    def __init__(self, surfaces: np.ndarray, future_len: int = 30):
        self.future_len = future_len
        self.daily_changes = np.diff(surfaces, axis=0).astype(np.float32)

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history)
        B = history_denorm.shape[0]
        last_surface = history_denorm[:, -1].cpu().numpy()  # (B, 5, 5)

        n_changes = len(self.daily_changes)
        # Vectorized: draw all indices at once
        idx = np.random.randint(
            0, n_changes, size=(B, n_samples, self.future_len)
        )
        # Gather changes: (B, n_samples, T_fut, 5, 5)
        changes = self.daily_changes[idx]
        # Cumsum from last surface
        cum_changes = np.cumsum(changes, axis=2)
        all_samples = last_surface[:, None, None, :, :] + cum_changes

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 4. PCA + VAR(1) with bootstrapped residuals
# =============================================================================

class PCAVARBaseline(BaselineModel):
    """PCA on surfaces + VAR(1) on principal components + simulate forward.

    Standard quantitative finance approach:
    1. Flatten 5x5 surfaces to 25-dim vectors
    2. Fit PCA to reduce to n_components factors
    3. Fit VAR(1) on factor time series: z_t = A @ z_{t-1} + intercept + eps
    4. At inference: project history to PCA space, simulate VAR, reconstruct.
       Bootstrap actual residuals for ensemble diversity (preserves fat tails
       in factor space).
    """

    def __init__(
        self,
        surfaces: np.ndarray,
        future_len: int = 30,
        n_components: int = 5,
    ):
        self.future_len = future_len
        self.n_components = n_components

        flat = surfaces.reshape(-1, 25).astype(np.float64)
        self.mean = flat.mean(axis=0)
        centered = flat - self.mean

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.components = Vt[:n_components]  # (K, 25)
        self.explained_var = (S[:n_components] ** 2) / (len(flat) - 1)

        factors = centered @ self.components.T  # (N, K)

        # Fit VAR(1): z_t = A @ z_{t-1} + intercept + eps
        Z_prev = factors[:-1]
        Z_next = factors[1:]
        Z_aug = np.column_stack([Z_prev, np.ones(len(Z_prev))])
        coeffs, _, _, _ = np.linalg.lstsq(Z_aug, Z_next, rcond=None)
        self.A = coeffs[:n_components].T  # (K, K)
        self.intercept = coeffs[n_components]  # (K,)

        Z_pred = Z_prev @ self.A.T + self.intercept
        self.residuals = (Z_next - Z_pred).astype(np.float32)

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history).cpu().numpy()
        B = history_denorm.shape[:1][0]

        all_samples = np.zeros(
            (B, n_samples, self.future_len, 5, 5), dtype=np.float32
        )
        n_resid = len(self.residuals)

        for b in range(B):
            last_flat = history_denorm[b, -1].reshape(1, 25)
            z_init = ((last_flat - self.mean) @ self.components.T)[0]

            for s in range(n_samples):
                z = z_init.copy()
                path = np.zeros((self.future_len, 25), dtype=np.float64)

                for t in range(self.future_len):
                    resid_idx = np.random.randint(0, n_resid)
                    z = self.A @ z + self.intercept + self.residuals[resid_idx]
                    path[t] = z @ self.components + self.mean

                all_samples[b, s] = path.reshape(self.future_len, 5, 5)

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 5. GARCH(1,1) + CCC (Constant Conditional Correlation)
# =============================================================================

class GARCHCCCBaseline(BaselineModel):
    """GARCH(1,1) per cell + Constant Conditional Correlation + multivariate Student-t.

    Industry standard for multivariate volatility modeling (Bollerslev 1990).
    Uses the `arch` package for proper GARCH(1,1) MLE fitting per cell:
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

    Cross-cell correlation from standardized residuals (CCC).
    Proper multivariate Student-t via scale mixture: z = L @ g / sqrt(w/df),
    where g ~ N(0,I), w ~ chi2(df). This preserves both the correlation
    structure AND the per-marginal Student-t tails (Demarta & McNeil 2005).
    """

    def __init__(
        self,
        surfaces: np.ndarray,
        future_len: int = 30,
    ):
        from arch import arch_model

        self.future_len = future_len

        daily_changes = np.diff(surfaces, axis=0)  # (N-1, 5, 5)
        N = daily_changes.shape[0]
        self.cell_mean = daily_changes.mean(axis=0)  # (5, 5)

        # Fit GARCH(1,1) with Student-t per cell
        self.omega = np.zeros((5, 5))
        self.alpha = np.zeros((5, 5))
        self.beta = np.zeros((5, 5))
        self.df = np.zeros((5, 5))
        cond_vol_last = np.zeros((5, 5))  # for diagnostics

        std_residuals_all = np.zeros((N, 5, 5))

        print(f"    Fitting 25 GARCH(1,1) models...", end="", flush=True)
        for i in range(5):
            for j in range(5):
                series = (daily_changes[:, i, j] - self.cell_mean[i, j]) * 100
                am = arch_model(
                    series, vol="GARCH", p=1, q=1,
                    dist="StudentsT", mean="Zero", rescale=False,
                )
                res = am.fit(disp="off", show_warning=False)

                self.omega[i, j] = res.params["omega"] / 1e4  # undo *100 scaling
                self.alpha[i, j] = res.params["alpha[1]"]
                self.beta[i, j] = res.params["beta[1]"]
                self.df[i, j] = max(res.params["nu"], 2.1)  # df > 2 for finite var

                # Store conditional volatility and standardized residuals
                cond_vol = res.conditional_volatility / 100  # undo scaling
                cond_vol_last[i, j] = cond_vol[-1]
                resid = daily_changes[:, i, j] - self.cell_mean[i, j]
                std_residuals_all[:, i, j] = resid / np.maximum(cond_vol, 1e-10)

        print(" done")

        # CCC: correlation of standardized residuals
        std_flat = std_residuals_all.reshape(N, 25)
        self.std_residuals_train = std_flat.astype(np.float32)
        self.corr_matrix = np.corrcoef(std_flat.T)
        # Ensure PSD
        eigvals, eigvecs = np.linalg.eigh(self.corr_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        self.corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.cholesky_L = np.linalg.cholesky(self.corr_matrix)

        # Pooled df for correlation-preserving simulation
        self.pooled_df = float(np.median(self.df))

        # Store unconditional variance for initialization
        self.uncond_var = daily_changes.var(axis=0)

    def _garch_var_from_history(self, history_denorm: np.ndarray) -> np.ndarray:
        """Roll GARCH(1,1) forward through history to get conditional variance."""
        changes = np.diff(history_denorm, axis=0)
        residuals = changes - self.cell_mean

        # Initialize with unconditional variance
        var = self.uncond_var.copy()
        for t in range(len(residuals)):
            var = self.omega + self.alpha * residuals[t] ** 2 + self.beta * var
            var = np.maximum(var, 1e-12)
        return var

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history).cpu().numpy()
        B = history_denorm.shape[0]

        all_samples = np.zeros(
            (B, n_samples, self.future_len, 5, 5), dtype=np.float32
        )

        for b in range(B):
            last_surface = history_denorm[b, -1]
            cond_var = self._garch_var_from_history(history_denorm[b])

            for s in range(n_samples):
                surface = last_surface.copy()
                var = cond_var.copy()

                for t in range(self.future_len):
                    # Proper multivariate Student-t via scale mixture
                    # (Demarta & McNeil 2005): X = L @ Z / sqrt(W/df)
                    # where Z ~ N(0,I), W ~ chi2(df)
                    # This gives X ~ multivariate-t with correlation R and df degrees
                    g = np.random.standard_normal(25)
                    w = np.random.chisquare(self.pooled_df)
                    z_corr = self.cholesky_L @ g / np.sqrt(w / self.pooled_df)
                    z_grid = z_corr.reshape(5, 5)

                    std = np.sqrt(var)
                    change = self.cell_mean + std * z_grid
                    surface = surface + change

                    # GARCH variance update
                    resid = change - self.cell_mean
                    var = self.omega + self.alpha * resid ** 2 + self.beta * var
                    var = np.maximum(var, 1e-12)

                    all_samples[b, s, t] = surface

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 6. GARCH(1,1) + Dynamic Conditional Correlation (DCC)
# =============================================================================

class DCCGARCHBaseline(GARCHCCCBaseline):
    """GARCH(1,1) per cell + Dynamic Conditional Correlation (DCC(1,1)).

    This is the natural dynamic-correlation extension of GARCHCCCBaseline:
        H_t = D_t R_t D_t
        Q_t = (1-a-b) * Qbar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
        R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)

    We keep the repo's existing per-cell Student-t GARCH fit and estimate the
    DCC parameters (a, b) on standardized residuals using a Gaussian
    quasi-likelihood over a subsample of the training history.
    """

    def __init__(
        self,
        surfaces: np.ndarray,
        future_len: int = 30,
        fit_sample_size: int = 1500,
    ):
        super().__init__(surfaces, future_len=future_len)

        z = self.std_residuals_train.astype(np.float64)
        z = z[np.all(np.isfinite(z), axis=1)]
        z = z - z.mean(axis=0, keepdims=True)
        z = z / np.maximum(z.std(axis=0, keepdims=True), 1e-6)

        if fit_sample_size and len(z) > fit_sample_size:
            idx = np.linspace(0, len(z) - 1, fit_sample_size).round().astype(int)
            z_fit = z[idx]
        else:
            z_fit = z

        self.q_bar = self._project_psd(np.corrcoef(z_fit.T))
        self.dcc_a, self.dcc_b = self._fit_dcc_params(z_fit, self.q_bar)

        print(
            f"    DCC params: a={self.dcc_a:.4f}, b={self.dcc_b:.4f}, "
            f"a+b={self.dcc_a + self.dcc_b:.4f}"
        )

    @staticmethod
    def _project_psd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        matrix = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, eps)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    @classmethod
    def _corr_from_q(cls, q: np.ndarray) -> np.ndarray:
        q = cls._project_psd(q)
        diag = np.sqrt(np.maximum(np.diag(q), 1e-10))
        corr = q / np.outer(diag, diag)
        corr = 0.5 * (corr + corr.T)
        corr = cls._project_psd(corr)
        diag = np.sqrt(np.maximum(np.diag(corr), 1e-10))
        corr = corr / np.outer(diag, diag)
        np.fill_diagonal(corr, 1.0)
        return corr

    @classmethod
    def _dcc_negloglik(
        cls,
        params: tuple[float, float] | np.ndarray,
        z: np.ndarray,
        q_bar: np.ndarray,
    ) -> float:
        a, b = float(params[0]), float(params[1])
        if a < 0.0 or b < 0.0 or a + b >= 0.995:
            return 1e12

        q_t = q_bar.copy()
        total = 0.0
        ident = np.eye(q_bar.shape[0], dtype=np.float64)

        for z_t in z:
            r_t = cls._corr_from_q(q_t)
            try:
                chol = np.linalg.cholesky(r_t + 1e-10 * ident)
            except np.linalg.LinAlgError:
                return 1e12

            logdet = 2.0 * np.log(np.diag(chol)).sum()
            solved = np.linalg.solve(chol, z_t)
            quad = float(solved @ solved)
            total += logdet + quad

            q_t = (1.0 - a - b) * q_bar + a * np.outer(z_t, z_t) + b * q_t
            q_t = 0.5 * (q_t + q_t.T)

        return 0.5 * total / max(len(z), 1)

    @classmethod
    def _fit_dcc_params(
        cls,
        z: np.ndarray,
        q_bar: np.ndarray,
    ) -> tuple[float, float]:
        grid = [
            (a, b)
            for a in (0.01, 0.02, 0.03, 0.05, 0.08, 0.10)
            for b in (0.85, 0.90, 0.94, 0.97)
            if a + b < 0.995
        ]
        best = min(grid, key=lambda ab: cls._dcc_negloglik(ab, z, q_bar))

        result = optimize.minimize(
            lambda x: cls._dcc_negloglik(x, z, q_bar),
            x0=np.array(best, dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(1e-4, 0.25), (1e-4, 0.995)],
            options={"maxiter": 40},
        )

        if result.success:
            a_opt, b_opt = map(float, result.x)
            if a_opt >= 0.0 and b_opt >= 0.0 and a_opt + b_opt < 0.995:
                return a_opt, b_opt

        return float(best[0]), float(best[1])

    def _dcc_state_from_history(
        self, history_denorm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        changes = np.diff(history_denorm, axis=0)
        residuals = changes - self.cell_mean

        var = self.uncond_var.copy()
        q_t = self.q_bar.copy()

        for resid in residuals:
            std = np.sqrt(np.maximum(var, 1e-12))
            z_t = np.clip((resid / std).reshape(25), -10.0, 10.0)
            q_t = (
                (1.0 - self.dcc_a - self.dcc_b) * self.q_bar
                + self.dcc_a * np.outer(z_t, z_t)
                + self.dcc_b * q_t
            )
            q_t = 0.5 * (q_t + q_t.T)

            var = self.omega + self.alpha * resid ** 2 + self.beta * var
            var = np.maximum(var, 1e-12)

        return var, q_t

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history).cpu().numpy()
        bsz = history_denorm.shape[0]

        all_samples = np.zeros(
            (bsz, n_samples, self.future_len, 5, 5), dtype=np.float32
        )
        ident = np.eye(25, dtype=np.float64)

        for b in range(bsz):
            last_surface = history_denorm[b, -1]
            cond_var, q_hist = self._dcc_state_from_history(history_denorm[b])

            for s in range(n_samples):
                surface = last_surface.copy()
                var = cond_var.copy()
                q_t = q_hist.copy()

                for t in range(self.future_len):
                    r_t = self._corr_from_q(q_t)
                    chol = np.linalg.cholesky(r_t + 1e-10 * ident)

                    g = np.random.standard_normal(25)
                    w = np.random.chisquare(self.pooled_df)
                    z_corr = chol @ g / np.sqrt(w / self.pooled_df)
                    z_grid = z_corr.reshape(5, 5)

                    std = np.sqrt(var)
                    change = self.cell_mean + std * z_grid
                    surface = surface + change

                    resid = change - self.cell_mean
                    z_t = np.clip((resid / np.maximum(std, 1e-12)).reshape(25), -10.0, 10.0)
                    q_t = (
                        (1.0 - self.dcc_a - self.dcc_b) * self.q_bar
                        + self.dcc_a * np.outer(z_t, z_t)
                        + self.dcc_b * q_t
                    )
                    q_t = 0.5 * (q_t + q_t.T)

                    var = self.omega + self.alpha * resid ** 2 + self.beta * var
                    var = np.maximum(var, 1e-12)

                    all_samples[b, s, t] = surface

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 7. Filtered Historical Simulation (EWMA + bootstrap actual residuals)
# =============================================================================

class FilteredHistoricalSimulation(BaselineModel):
    """GARCH(1,1)-Filtered Historical Simulation (Barone-Adesi et al. 1999).

    Reference-faithful implementation:
    1. Fit GARCH(1,1) per cell on training data → standardized residuals
    2. At inference: roll GARCH through history to initialize conditional variance
    3. Draw standardized residuals from the training-set pool (with replacement)
    4. Re-scale by current GARCH conditional std → preserves fat tails AND vol clustering

    Cross-cell correlation is preserved automatically because each draw is a
    full 25-dim standardized residual vector from the same training day.
    """

    def __init__(
        self,
        surfaces: np.ndarray,
        future_len: int = 30,
    ):
        from arch import arch_model

        self.future_len = future_len

        daily_changes = np.diff(surfaces, axis=0)  # (N-1, 5, 5)
        N = daily_changes.shape[0]

        self.cell_mean = daily_changes.mean(axis=0)
        residuals = daily_changes - self.cell_mean

        # Fit GARCH(1,1) per cell for variance filtering
        self.omega = np.zeros((5, 5))
        self.alpha = np.zeros((5, 5))
        self.beta = np.zeros((5, 5))

        var_series = np.zeros((N, 5, 5), dtype=np.float64)

        print(f"    Fitting 25 GARCH(1,1) models for FHS...", end="", flush=True)
        for i in range(5):
            for j in range(5):
                series = (daily_changes[:, i, j] - self.cell_mean[i, j]) * 100
                am = arch_model(
                    series, vol="GARCH", p=1, q=1,
                    dist="Normal", mean="Zero", rescale=False,
                )
                res = am.fit(disp="off", show_warning=False)

                self.omega[i, j] = res.params["omega"] / 1e4
                self.alpha[i, j] = res.params["alpha[1]"]
                self.beta[i, j] = res.params["beta[1]"]

                cond_vol = res.conditional_volatility / 100
                var_series[:, i, j] = np.maximum(cond_vol ** 2, 1e-12)

        print(" done")

        # Store standardized residuals for bootstrapping
        std_series = np.sqrt(var_series)
        self.std_residuals = (residuals / std_series).astype(np.float32)  # (N, 5, 5)

        # Store unconditional variance for initialization
        self.uncond_var = daily_changes.var(axis=0)

    def _garch_var_from_history(self, history_denorm: np.ndarray) -> np.ndarray:
        """Roll GARCH(1,1) forward through history to get conditional variance."""
        changes = np.diff(history_denorm, axis=0)
        residuals = changes - self.cell_mean

        var = self.uncond_var.copy()
        for t in range(len(residuals)):
            var = self.omega + self.alpha * residuals[t] ** 2 + self.beta * var
            var = np.maximum(var, 1e-12)
        return var

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history).cpu().numpy()
        B = history_denorm.shape[0]

        n_resid = len(self.std_residuals)
        all_samples = np.zeros(
            (B, n_samples, self.future_len, 5, 5), dtype=np.float32
        )

        for b in range(B):
            last_surface = history_denorm[b, -1]
            cond_var = self._garch_var_from_history(history_denorm[b])

            for s in range(n_samples):
                surface = last_surface.copy()
                var = cond_var.copy()

                for t in range(self.future_len):
                    # Bootstrap a standardized residual (full 5x5, preserves correlation)
                    idx = np.random.randint(0, n_resid)
                    z = self.std_residuals[idx]  # (5, 5)

                    # Re-scale by current GARCH conditional std
                    std = np.sqrt(var)
                    change = self.cell_mean + std * z
                    surface = surface + change

                    # GARCH variance update
                    resid = change - self.cell_mean
                    var = self.omega + self.alpha * resid ** 2 + self.beta * var
                    var = np.maximum(var, 1e-12)

                    all_samples[b, s, t] = surface

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)
