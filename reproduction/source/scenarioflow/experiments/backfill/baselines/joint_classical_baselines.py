"""Joint D-dim classical baselines for multi-factor benchmark.

All classes operate on D-dimensional daily changes (D=38 for IV+factors).
Constructor takes daily_changes: (N, D).
sample_joint(history_changes, n_samples) -> (B, n_samples, T_fut, D) changes.
Caller is responsible for IV reconstruction (cumsum + anchor).

Baselines:
    1. JointRandomWalk — joint Gaussian with full DxD covariance
    2. JointBootstrap — resample whole D-dim daily change vectors
    3. JointHistoricalSim — nonparametric conditional resampler with factor matching
    4. JointPCAVAR — PCA on D-dim changes, VAR(1) on components
    5. JointGARCHCCC — GARCH(1,1) per dim + CCC + multivariate Student-t
    6. JointFilteredHS — GARCH(1,1) per dim + bootstrap standardized residuals
"""

import numpy as np
from .classical_baselines import BaselineModel


class JointRandomWalk(BaselineModel):
    """Joint Gaussian random walk on D-dim daily changes.

    Uses full DxD covariance estimated from training changes (not per-dim iid).
    This preserves cross-dimension dependence in the noise.
    """

    def __init__(self, daily_changes: np.ndarray, future_len: int = 30):
        self.future_len = future_len
        self.D = daily_changes.shape[1]
        self.cov = np.cov(daily_changes.T)  # (D, D)
        self.chol = np.linalg.cholesky(
            self.cov + 1e-10 * np.eye(self.D)
        )  # (D, D)

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 50, **kwargs
    ) -> np.ndarray:
        """
        Args:
            history_changes: (B, T_hist, D) — ignored (random walk is unconditional)
            n_samples: number of ensemble members
        Returns:
            changes: (B, n_samples, T_fut, D) daily changes
        """
        B = history_changes.shape[0]
        z = np.random.standard_normal(
            (B, n_samples, self.future_len, self.D)
        )
        changes = z @ self.chol.T
        return changes.astype(np.float32)


class JointBootstrap(BaselineModel):
    """Resample whole D-dim daily change vectors from training data.

    Preserves all cross-dimension correlation within each day.
    Draws are iid across time (breaks temporal dependence).
    """

    def __init__(self, daily_changes: np.ndarray, future_len: int = 30):
        self.future_len = future_len
        self.daily_changes = daily_changes.astype(np.float32)  # (N, D)
        self.D = daily_changes.shape[1]

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 50, **kwargs
    ) -> np.ndarray:
        B = history_changes.shape[0]
        N = len(self.daily_changes)
        idx = np.random.randint(0, N, size=(B, n_samples, self.future_len))
        return self.daily_changes[idx]  # (B, n_samples, T_fut, D)


class JointHistoricalSim(BaselineModel):
    """Nonparametric conditional resampler on D-dim daily changes.

    Matches test windows to training windows using a small past-state feature
    vector (IV summary + factor regime indicators). Returns full D-dim futures.
    """

    def __init__(
        self,
        daily_changes: np.ndarray,
        surfaces: np.ndarray,
        factor_levels: np.ndarray,
        history_len: int = 30,
        future_len: int = 30,
    ):
        self.history_len = history_len
        self.future_len = future_len
        self.D = daily_changes.shape[1]

        n_windows = len(daily_changes) - history_len - future_len + 1
        self.hist_changes = np.zeros(
            (n_windows, history_len, self.D), dtype=np.float32
        )
        self.fut_changes = np.zeros(
            (n_windows, future_len, self.D), dtype=np.float32
        )

        for i in range(n_windows):
            self.hist_changes[i] = daily_changes[i : i + history_len]
            self.fut_changes[i] = daily_changes[
                i + history_len : i + history_len + future_len
            ]

        # 6 matching features:
        # changes[i] = surfaces[i+1] - surfaces[i], so history changes [i, i+H)
        # correspond to surfaces [i+1, i+H+1). Forecast origin = surfaces[i + H].
        features_list = []
        for i in range(n_windows):
            surf_window = surfaces[i + 1 : i + history_len + 1]  # (H, 5, 5)
            flev_window = factor_levels[i + 1 : i + history_len + 1]  # (H, 13)

            # IV features
            mean_iv = surf_window[-5:].mean()
            mean_iv_ts = surf_window.mean(axis=(1, 2))
            vov = np.diff(mean_iv_ts).std() if len(mean_iv_ts) > 1 else 0.0
            slope = surf_window[-1, -1, 2] - surf_window[-1, 0, 2]

            # Factor features
            # SPX logret is dim 25 in daily_changes (first factor return column)
            spx_rets = self.hist_changes[i, -5:, 25]
            mean_abs_spx = np.abs(spx_rets).mean()
            # level_columns: spx,usdcad,usdjpy,dxy,copper,wheat,crude_oil,
            #   us2y,us10y,aaa_oas,bbb_oas,nikkei,gold
            yc_slope = flev_window[-1, 8] - flev_window[-1, 7]  # us10y - us2y
            credit_width = flev_window[-1, 10] - flev_window[-1, 9]  # bbb - aaa

            features_list.append([
                mean_iv, vov, slope, mean_abs_spx, yc_slope, credit_width
            ])

        self.features = np.array(features_list, dtype=np.float64)
        self.feat_mean = self.features.mean(axis=0)
        self.feat_std = self.features.std(axis=0) + 1e-8
        self.features_norm = (self.features - self.feat_mean) / self.feat_std

    def _compute_features(
        self, history_changes, surfaces, factor_levels
    ):
        mean_iv = surfaces[-5:].mean()
        mean_iv_ts = surfaces.mean(axis=(1, 2))
        vov = np.diff(mean_iv_ts).std() if len(mean_iv_ts) > 1 else 0.0
        slope = surfaces[-1, -1, 2] - surfaces[-1, 0, 2]

        spx_rets = history_changes[-5:, 25]
        mean_abs_spx = np.abs(spx_rets).mean()
        yc_slope = factor_levels[-1, 8] - factor_levels[-1, 7]
        credit_width = factor_levels[-1, 10] - factor_levels[-1, 9]

        feat = np.array([mean_iv, vov, slope, mean_abs_spx, yc_slope, credit_width])
        return (feat - self.feat_mean) / self.feat_std

    def sample_joint(
        self,
        history_changes: np.ndarray,
        n_samples: int = 50,
        surfaces: np.ndarray = None,
        factor_levels: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Args:
            history_changes: (B, T_hist, D) daily changes
            surfaces: (B, T_hist, 5, 5) surface levels for feature computation
            factor_levels: (B, T_hist, 13) factor levels for regime features
        Returns:
            changes: (B, n_samples, T_fut, D) daily changes
        """
        B = history_changes.shape[0]
        all_samples = np.zeros(
            (B, n_samples, self.future_len, self.D), dtype=np.float32
        )

        for b in range(B):
            query = self._compute_features(
                history_changes[b], surfaces[b], factor_levels[b]
            )
            distances = np.linalg.norm(
                self.features_norm - query[None, :], axis=1
            )
            weights = np.exp(-distances)
            weights /= weights.sum()

            indices = np.random.choice(
                len(self.fut_changes), size=n_samples, replace=True, p=weights
            )
            all_samples[b] = self.fut_changes[indices]

        return all_samples


class JointPCAVAR(BaselineModel):
    """Joint PCA-VAR on D-dim standardized daily changes.

    Factors are endogenous (not exogenous). PCA reduces dimensionality,
    VAR(1) models temporal dynamics in factor space.
    """

    def __init__(
        self,
        daily_changes: np.ndarray,
        future_len: int = 30,
        n_components: int = 10,
        explained_var_cap: float = 0.95,
    ):
        self.future_len = future_len
        self.D = daily_changes.shape[1]

        self.mean = daily_changes.mean(axis=0)
        self.std = daily_changes.std(axis=0) + 1e-8
        standardized = (daily_changes - self.mean) / self.std

        U, S, Vt = np.linalg.svd(standardized, full_matrices=False)
        explained = (S ** 2) / (S ** 2).sum()
        cumulative = np.cumsum(explained)
        n_by_cap = int(np.searchsorted(cumulative, explained_var_cap) + 1)
        self.n_components = min(n_components, n_by_cap, len(S))
        self.components = Vt[: self.n_components]  # (K, D)
        print(f"    PCA-VAR: {self.n_components} components, "
              f"{cumulative[self.n_components - 1]:.1%} variance explained")

        factors = standardized @ self.components.T  # (N, K)

        Z_prev = factors[:-1]
        Z_next = factors[1:]
        Z_aug = np.column_stack([Z_prev, np.ones(len(Z_prev))])
        coeffs, _, _, _ = np.linalg.lstsq(Z_aug, Z_next, rcond=None)
        self.A = coeffs[: self.n_components].T
        self.intercept = coeffs[self.n_components]
        self.residuals = (Z_next - (Z_prev @ self.A.T + self.intercept)).astype(
            np.float32
        )

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 50, **kwargs
    ) -> np.ndarray:
        B = history_changes.shape[0]
        all_samples = np.zeros(
            (B, n_samples, self.future_len, self.D), dtype=np.float32
        )
        n_resid = len(self.residuals)

        for b in range(B):
            last_std = (history_changes[b, -1] - self.mean) / self.std
            z_init = last_std @ self.components.T

            for s in range(n_samples):
                z = z_init.copy()
                for t in range(self.future_len):
                    resid_idx = np.random.randint(0, n_resid)
                    z = self.A @ z + self.intercept + self.residuals[resid_idx]
                    reconstructed_std = z @ self.components
                    all_samples[b, s, t] = reconstructed_std * self.std + self.mean

        return all_samples


def _fit_garch_per_dim(residuals, D, uncond_var, student_t=True):
    """Fit GARCH(1,1) per dimension with EWMA fallback.

    Returns omega, alpha, beta, df (if student_t), std_residuals, n_failed.
    """
    from arch import arch_model

    N = residuals.shape[0]
    omega = np.zeros(D)
    alpha = np.zeros(D)
    beta = np.zeros(D)
    df = np.zeros(D) if student_t else None
    std_residuals = np.zeros((N, D), dtype=np.float64)
    n_failed = 0

    dist = "StudentsT" if student_t else "Normal"

    for d in range(D):
        series = residuals[:, d] * 100
        try:
            am = arch_model(
                series, vol="GARCH", p=1, q=1,
                dist=dist, mean="Zero", rescale=False,
            )
            res = am.fit(disp="off", show_warning=False)
            omega[d] = res.params["omega"] / 1e4
            alpha[d] = res.params["alpha[1]"]
            beta[d] = res.params["beta[1]"]
            if student_t:
                df[d] = max(res.params["nu"], 2.1)
            cond_vol = res.conditional_volatility / 100
            std_residuals[:, d] = residuals[:, d] / np.maximum(cond_vol, 1e-10)
        except Exception:
            n_failed += 1
            lam = 0.94
            omega[d] = uncond_var[d] * (1 - lam)
            alpha[d] = 1 - lam
            beta[d] = lam
            if student_t:
                df[d] = 5.0
            var_t = uncond_var[d]
            for t in range(N):
                std_residuals[t, d] = residuals[t, d] / max(np.sqrt(var_t), 1e-10)
                var_t = lam * var_t + (1 - lam) * residuals[t, d] ** 2

    return omega, alpha, beta, df, std_residuals, n_failed


class JointGARCHCCC(BaselineModel):
    """GARCH(1,1) per dimension + CCC on D-dim daily changes.

    Uses multivariate Student-t via scale mixture (Demarta & McNeil 2005).
    Failed GARCH fits fall back to EWMA(lambda=0.94).
    """

    def __init__(self, daily_changes: np.ndarray, future_len: int = 30):
        self.future_len = future_len
        self.D = daily_changes.shape[1]

        self.dim_mean = daily_changes.mean(axis=0)
        residuals = daily_changes - self.dim_mean
        self.uncond_var = daily_changes.var(axis=0)

        print(f"    Fitting {self.D} GARCH(1,1)-t models...", end="", flush=True)
        (self.omega, self.alpha, self.beta, self.df,
         std_residuals, n_failed) = _fit_garch_per_dim(
            residuals, self.D, self.uncond_var, student_t=True
        )
        if n_failed:
            print(f" ({n_failed} EWMA fallbacks)", end="")
        print(" done")

        # CCC correlation matrix with PSD projection
        self.corr_matrix = np.corrcoef(std_residuals.T)
        eigvals, eigvecs = np.linalg.eigh(self.corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        self.corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.cholesky_L = np.linalg.cholesky(self.corr_matrix)
        self.pooled_df = float(np.median(self.df))

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 50, **kwargs
    ) -> np.ndarray:
        B = history_changes.shape[0]
        all_samples = np.zeros(
            (B, n_samples, self.future_len, self.D), dtype=np.float32
        )

        for b in range(B):
            resid_hist = history_changes[b] - self.dim_mean
            var = self.uncond_var.copy()
            for t in range(len(resid_hist)):
                var = self.omega + self.alpha * resid_hist[t] ** 2 + self.beta * var
                var = np.maximum(var, 1e-12)

            for s in range(n_samples):
                var_t = var.copy()
                for t in range(self.future_len):
                    g = np.random.standard_normal(self.D)
                    w = np.random.chisquare(self.pooled_df)
                    z_corr = self.cholesky_L @ g / np.sqrt(w / self.pooled_df)
                    std_t = np.sqrt(var_t)
                    change = self.dim_mean + std_t * z_corr
                    all_samples[b, s, t] = change
                    resid = change - self.dim_mean
                    var_t = self.omega + self.alpha * resid ** 2 + self.beta * var_t
                    var_t = np.maximum(var_t, 1e-12)

        return all_samples


class JointFilteredHS(BaselineModel):
    """GARCH(1,1)-Filtered Historical Simulation on D-dim daily changes.

    Failed GARCH fits fall back to EWMA(lambda=0.94).
    """

    def __init__(self, daily_changes: np.ndarray, future_len: int = 30):
        self.future_len = future_len
        self.D = daily_changes.shape[1]
        N = daily_changes.shape[0]

        self.dim_mean = daily_changes.mean(axis=0)
        residuals = daily_changes - self.dim_mean
        self.uncond_var = daily_changes.var(axis=0)

        print(f"    Fitting {self.D} GARCH(1,1) models for FHS...", end="", flush=True)
        (self.omega, self.alpha, self.beta, _,
         std_residuals, n_failed) = _fit_garch_per_dim(
            residuals, self.D, self.uncond_var, student_t=False
        )
        if n_failed:
            print(f" ({n_failed} EWMA fallbacks)", end="")
        print(" done")

        self.std_residuals = std_residuals.astype(np.float32)  # (N, D)

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 50, **kwargs
    ) -> np.ndarray:
        B = history_changes.shape[0]
        N_resid = len(self.std_residuals)
        all_samples = np.zeros(
            (B, n_samples, self.future_len, self.D), dtype=np.float32
        )

        for b in range(B):
            resid_hist = history_changes[b] - self.dim_mean
            var = self.uncond_var.copy()
            for t in range(len(resid_hist)):
                var = self.omega + self.alpha * resid_hist[t] ** 2 + self.beta * var
                var = np.maximum(var, 1e-12)

            for s in range(n_samples):
                var_t = var.copy()
                for t in range(self.future_len):
                    idx = np.random.randint(0, N_resid)
                    z = self.std_residuals[idx]  # (D,) preserves cross-dim correlation
                    std_t = np.sqrt(var_t)
                    change = self.dim_mean + std_t * z
                    all_samples[b, s, t] = change
                    resid = change - self.dim_mean
                    var_t = self.omega + self.alpha * resid ** 2 + self.beta * var_t
                    var_t = np.maximum(var_t, 1e-12)

        return all_samples
