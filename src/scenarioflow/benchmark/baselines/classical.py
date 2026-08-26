"""Final joint 39-state classical benchmark implementations.

The GARCH-CCC sampler is the pinned Gaussian-innovation revision
``da91172``.  It deliberately does not restore the earlier Student-t scale
mixture that was unstable on the mixed-transform panel.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _rng(kwargs: dict[str, Any]) -> np.random.Generator | None:
    value = kwargs.get("rng")
    if value is not None and not isinstance(value, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    return value


class JointRandomWalk:
    """Unconditional Gaussian increment walk with full-panel covariance."""

    def __init__(self, daily_changes: np.ndarray, future_len: int = 30) -> None:
        values = np.asarray(daily_changes, dtype=np.float64)
        self.future_len = int(future_len)
        self.dimension = int(values.shape[1])
        covariance = np.cov(values.T)
        self.cholesky = np.linalg.cholesky(covariance + 1e-10 * np.eye(self.dimension))

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 64, **kwargs: Any
    ) -> np.ndarray:
        generator = _rng(kwargs)
        shape = (
            int(history_changes.shape[0]),
            int(n_samples),
            self.future_len,
            self.dimension,
        )
        noise = (
            generator.standard_normal(shape)
            if generator is not None
            else np.random.standard_normal(shape)
        )
        return (noise @ self.cholesky.T).astype(np.float32)


class JointBootstrap:
    """IID resampling of whole daily panel vectors."""

    def __init__(self, daily_changes: np.ndarray, future_len: int = 30) -> None:
        self.future_len = int(future_len)
        self.daily_changes = np.asarray(daily_changes, dtype=np.float32)
        self.dimension = int(self.daily_changes.shape[1])

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 64, **kwargs: Any
    ) -> np.ndarray:
        generator = _rng(kwargs)
        shape = (
            int(history_changes.shape[0]),
            int(n_samples),
            self.future_len,
        )
        indices = (
            generator.integers(0, len(self.daily_changes), size=shape)
            if generator is not None
            else np.random.randint(0, len(self.daily_changes), size=shape)
        )
        return self.daily_changes[indices]


class WindowHistoricalSimulation:
    """Soft-nearest-neighbour resampling of complete historical futures."""

    def __init__(
        self,
        history_increment: np.ndarray,
        future_increment: np.ndarray,
        history_state: np.ndarray,
        *,
        iv_count: int = 25,
    ) -> None:
        self.history_increment = np.asarray(history_increment, dtype=np.float32)
        self.future_increment = np.asarray(future_increment, dtype=np.float32)
        self.history_state = np.asarray(history_state, dtype=np.float32)
        self.iv_count = int(iv_count)
        self.future_len = int(self.future_increment.shape[1])
        self.dimension = int(self.future_increment.shape[2])
        self.features = np.asarray(
            [
                self._features(self.history_increment[index], self.history_state[index])
                for index in range(self.history_increment.shape[0])
            ],
            dtype=np.float64,
        )
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0) + 1e-8
        self.normalized_features = (self.features - self.mean) / self.std

    def _features(
        self, history_increment: np.ndarray, history_state: np.ndarray
    ) -> np.ndarray:
        iv = history_state[:, : self.iv_count].reshape(history_state.shape[0], 5, 5)
        mean_iv = float(iv[-5:].mean())
        mean_iv_time = iv.mean(axis=(1, 2))
        vol_of_vol = (
            float(np.diff(mean_iv_time).std()) if mean_iv_time.shape[0] > 1 else 0.0
        )
        slope = float(iv[-1, -1, 2] - iv[-1, 0, 2])
        spx_activity = float(np.abs(history_increment[-5:, self.iv_count]).mean())
        factor = history_state[:, self.iv_count :]
        yield_slope = float(factor[-1, 8] - factor[-1, 7])
        credit_width = float(factor[-1, 10] - factor[-1, 9])
        return np.asarray(
            [mean_iv, vol_of_vol, slope, spx_activity, yield_slope, credit_width]
        )

    def sample_joint(
        self,
        history_changes: np.ndarray,
        n_samples: int = 64,
        *,
        history_state: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        generator = _rng(kwargs)
        output = np.empty(
            (
                history_changes.shape[0],
                int(n_samples),
                self.future_len,
                self.dimension,
            ),
            dtype=np.float32,
        )
        for batch in range(history_changes.shape[0]):
            query = (
                self._features(history_changes[batch], history_state[batch]) - self.mean
            ) / self.std
            logits = -np.linalg.norm(self.normalized_features - query[None, :], axis=1)
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= weights.sum()
            indices = (
                generator.choice(
                    len(self.future_increment),
                    size=int(n_samples),
                    replace=True,
                    p=weights,
                )
                if generator is not None
                else np.random.choice(
                    len(self.future_increment),
                    size=int(n_samples),
                    replace=True,
                    p=weights,
                )
            )
            output[batch] = self.future_increment[indices]
        return output


class JointPCAVAR:
    """PCA reduction followed by a VAR(1) and residual bootstrap."""

    def __init__(
        self,
        daily_changes: np.ndarray,
        future_len: int = 30,
        n_components: int = 10,
        explained_var_cap: float = 0.95,
    ) -> None:
        values = np.asarray(daily_changes, dtype=np.float64)
        self.future_len = int(future_len)
        self.dimension = int(values.shape[1])
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0) + 1e-8
        standardized = (values - self.mean) / self.std
        _u, singular, vectors = np.linalg.svd(standardized, full_matrices=False)
        cumulative = np.cumsum(singular**2 / np.sum(singular**2))
        by_cap = int(np.searchsorted(cumulative, float(explained_var_cap)) + 1)
        self.n_components = min(int(n_components), by_cap, len(singular))
        self.components = vectors[: self.n_components]
        factors = standardized @ self.components.T
        augmented = np.column_stack([factors[:-1], np.ones(len(factors) - 1)])
        coefficients, *_ = np.linalg.lstsq(augmented, factors[1:], rcond=None)
        self.transition = coefficients[: self.n_components].T
        self.intercept = coefficients[self.n_components]
        self.residuals = (
            factors[1:] - factors[:-1] @ self.transition.T - self.intercept
        ).astype(np.float32)

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 64, **kwargs: Any
    ) -> np.ndarray:
        generator = _rng(kwargs)
        output = np.empty(
            (
                history_changes.shape[0],
                int(n_samples),
                self.future_len,
                self.dimension,
            ),
            dtype=np.float32,
        )
        for batch in range(history_changes.shape[0]):
            initial = (
                (history_changes[batch, -1] - self.mean) / self.std
            ) @ self.components.T
            for sample in range(int(n_samples)):
                state = initial.copy()
                for step in range(self.future_len):
                    index = (
                        int(generator.integers(0, len(self.residuals)))
                        if generator is not None
                        else int(np.random.randint(0, len(self.residuals)))
                    )
                    state = (
                        self.transition @ state + self.intercept + self.residuals[index]
                    )
                    output[batch, sample, step] = (
                        state @ self.components * self.std + self.mean
                    )
        return output


def _fit_garch_per_dimension(
    residuals: np.ndarray, unconditional_variance: np.ndarray, *, student_t: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from arch import arch_model
    except ImportError as error:
        raise ImportError(
            "GARCH baselines require `uv sync --extra baseline-classical`"
        ) from error
    dimension = residuals.shape[1]
    omega = np.empty(dimension)
    alpha = np.empty(dimension)
    beta = np.empty(dimension)
    standardized = np.empty_like(residuals, dtype=np.float64)
    distribution = "StudentsT" if student_t else "Normal"
    for channel in range(dimension):
        series = residuals[:, channel] * 100.0
        try:
            fit = arch_model(
                series,
                vol="GARCH",
                p=1,
                q=1,
                dist=distribution,
                mean="Zero",
                rescale=False,
            ).fit(disp="off", show_warning=False)
            omega[channel] = fit.params["omega"] / 1e4
            alpha[channel] = fit.params["alpha[1]"]
            beta[channel] = fit.params["beta[1]"]
            volatility = fit.conditional_volatility / 100.0
            standardized[:, channel] = residuals[:, channel] / np.maximum(
                volatility, 1e-10
            )
        except Exception:
            decay = 0.94
            omega[channel] = unconditional_variance[channel] * (1.0 - decay)
            alpha[channel] = 1.0 - decay
            beta[channel] = decay
            variance = unconditional_variance[channel]
            for time in range(residuals.shape[0]):
                standardized[time, channel] = residuals[time, channel] / max(
                    np.sqrt(variance), 1e-10
                )
                variance = (
                    decay * variance + (1.0 - decay) * residuals[time, channel] ** 2
                )
    return omega, alpha, beta, standardized


class _GARCHBase:
    def __init__(self, daily_changes: np.ndarray, future_len: int) -> None:
        values = np.asarray(daily_changes, dtype=np.float64)
        self.future_len = int(future_len)
        self.dimension = int(values.shape[1])
        self.mean = values.mean(axis=0)
        self.unconditional_variance = values.var(axis=0)
        self.residuals = values - self.mean

    def _initial_variance(self, history: np.ndarray) -> np.ndarray:
        variance = self.unconditional_variance.copy()
        for residual in history - self.mean:
            variance = self.omega + self.alpha * residual**2 + self.beta * variance
            variance = np.maximum(variance, 1e-12)
        return variance


class JointGaussianGARCHCCC(_GARCHBase):
    """Per-channel GARCH(1,1) with pinned Gaussian CCC innovations."""

    def __init__(self, daily_changes: np.ndarray, future_len: int = 30) -> None:
        super().__init__(daily_changes, future_len)
        self.omega, self.alpha, self.beta, standardized = _fit_garch_per_dimension(
            self.residuals, self.unconditional_variance, student_t=True
        )
        correlation = np.corrcoef(standardized.T)
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        correlation = (
            eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-6)) @ eigenvectors.T
        )
        self.cholesky = np.linalg.cholesky(correlation)

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 64, **kwargs: Any
    ) -> np.ndarray:
        generator = _rng(kwargs)
        output = np.empty(
            (history_changes.shape[0], int(n_samples), self.future_len, self.dimension),
            dtype=np.float32,
        )
        for batch, history in enumerate(history_changes):
            initial = self._initial_variance(history)
            for sample in range(int(n_samples)):
                variance = initial.copy()
                for step in range(self.future_len):
                    gaussian = (
                        generator.standard_normal(self.dimension)
                        if generator is not None
                        else np.random.standard_normal(self.dimension)
                    )
                    change = self.mean + np.sqrt(variance) * (self.cholesky @ gaussian)
                    output[batch, sample, step] = change
                    residual = change - self.mean
                    variance = (
                        self.omega + self.alpha * residual**2 + self.beta * variance
                    )
                    variance = np.maximum(variance, 1e-12)
        return output


class JointFilteredHistoricalSimulation(_GARCHBase):
    """Per-channel GARCH filtering with joint residual-vector resampling."""

    def __init__(self, daily_changes: np.ndarray, future_len: int = 30) -> None:
        super().__init__(daily_changes, future_len)
        self.omega, self.alpha, self.beta, standardized = _fit_garch_per_dimension(
            self.residuals, self.unconditional_variance, student_t=False
        )
        self.standardized_residuals = standardized.astype(np.float32)

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 64, **kwargs: Any
    ) -> np.ndarray:
        generator = _rng(kwargs)
        output = np.empty(
            (history_changes.shape[0], int(n_samples), self.future_len, self.dimension),
            dtype=np.float32,
        )
        for batch, history in enumerate(history_changes):
            initial = self._initial_variance(history)
            for sample in range(int(n_samples)):
                variance = initial.copy()
                for step in range(self.future_len):
                    index = (
                        int(generator.integers(0, len(self.standardized_residuals)))
                        if generator is not None
                        else int(np.random.randint(0, len(self.standardized_residuals)))
                    )
                    change = (
                        self.mean
                        + np.sqrt(variance) * self.standardized_residuals[index]
                    )
                    output[batch, sample, step] = change
                    residual = change - self.mean
                    variance = (
                        self.omega + self.alpha * residual**2 + self.beta * variance
                    )
                    variance = np.maximum(variance, 1e-12)
        return output


# Reader-facing aliases match the paper candidate IDs.
JointGARCHCCC = JointGaussianGARCHCCC
JointFilteredHS = JointFilteredHistoricalSimulation
