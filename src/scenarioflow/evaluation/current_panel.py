"""Published current-panel validation contract and common metric battery."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats as sp_stats

from ..baselines.classical import (
    JointBootstrap,
    JointFilteredHS,
    JointGARCHDCC,
    JointPCAVAR,
    JointRandomWalk,
)
from ..baselines.deep import load_current_panel_deep_baseline
from ..baselines.foundation import PrecomputedJointScenarioBaseline
from ..baselines.g2 import DeepVARSNIInterfaceBaseline
from ..data import load_aligned_iv_factor_panel, official_train_val_indices
from ..state import build_increment_coordinate_block, reconstruct_state_from_increments

RESULT_SCHEMA_VERSION = "scenarioflow.current-panel-result.v1"
MODE = "current_realvix_39_state_panel"


@dataclass(frozen=True)
class ValidationProtocol:
    mode: str = MODE
    state_count: int = 39
    iv_count: int = 25
    history_len: int = 30
    future_len: int = 30
    test_start: int = 4511
    val_size: int = 441
    n_samples: int = 64
    evaluation_seed: int = 772


PUBLISHED_VALIDATION_PROTOCOL = ValidationProtocol()

CLASSICAL_BASELINES = (
    "random_walk",
    "bootstrap",
    "historical_sim",
    "pca_var",
    "garch_ccc",
    "garch_dcc",
    "filtered_hs",
)
DEEP_BASELINES = ("deepvar", "timegrad", "path_diffusion", "cond_wgan")
FOUNDATION_BASELINES = ("moirai_1_1_finetuned", "toto_1_0_finetuned")
REFERENCE_BASELINES = ("sni_joint",)
BASELINES = (
    CLASSICAL_BASELINES + DEEP_BASELINES + FOUNDATION_BASELINES + REFERENCE_BASELINES
)

CURRENT_PANEL_RESULT_KEYS = (
    "schema_version",
    "baseline_name",
    "mode",
    "generation_time_s",
    "n_windows",
    "n_samples",
    "state_specs",
    "current_panel_iv",
    "current_panel_iv_metrics",
    "joint_panel_quality",
    "panel_b_crps",
    "panel_b_energy_score",
    "panel_b_variogram_score",
    "panel_b_correlation",
    "panel_b_factor_ks",
)

DISPLAYED_IV_METRIC_KEYS = (
    "cov90",
    "calibration_error",
    "level_ks_median",
    "median_abs_dev",
    "width_rho",
    "mr_ratio",
    "economic_ratio",
    "jump_ks",
    "kurtosis_ratio",
    "acf_corr",
)

TABLE_THRESHOLD_RULES: dict[str, dict[str, float | str]] = {
    "Cov90": {"kind": "band", "lo": 0.75, "hi": 0.98},
    "CalErr": {"kind": "upper", "hi": 0.20},
    "Level KS": {"kind": "upper", "hi": 0.20},
    "Width rho": {"kind": "lower", "lo": 0.20},
    "MR ratio": {"kind": "band", "lo": 0.25, "hi": 1.75},
    "Econ ratio": {"kind": "band", "lo": 0.75, "hi": 1.30},
    "Jump KS": {"kind": "upper", "hi": 0.50},
    "Kurtosis": {"kind": "band", "lo": 0.50, "hi": 1.50},
    "Corr score": {"kind": "lower", "lo": 0.30},
    "Factor KS": {"kind": "upper", "hi": 0.15},
    "Tail q99": {"kind": "band", "lo": 0.50, "hi": 1.50},
}

METRIC_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CRPS", ("panel_b_crps", "overall")),
    ("IV CRPS", ("panel_b_crps", "iv_crps")),
    ("Factor CRPS", ("panel_b_crps", "factor_crps")),
    ("Energy Score", ("panel_b_energy_score", "overall")),
    ("Variogram", ("panel_b_variogram_score", "variogram_score")),
    ("Corr score", ("panel_b_correlation", "corr_score")),
    ("Factor KS", ("panel_b_factor_ks", "mean_statistic")),
    ("Tail q99", ("joint_panel_quality", "factor_tail_q99_ratio_median")),
    ("Cov90", ("current_panel_iv_metrics", "cov90")),
    ("CalErr", ("current_panel_iv_metrics", "calibration_error")),
    ("Level KS", ("current_panel_iv_metrics", "level_ks_median")),
    ("Median dev", ("current_panel_iv_metrics", "median_abs_dev")),
    ("Width rho", ("current_panel_iv_metrics", "width_rho")),
    ("MR ratio", ("current_panel_iv_metrics", "mr_ratio")),
    ("Econ ratio", ("current_panel_iv_metrics", "economic_ratio")),
    ("Jump KS", ("current_panel_iv_metrics", "jump_ks")),
    ("Kurtosis", ("current_panel_iv_metrics", "kurtosis_ratio")),
    ("ACF corr", ("current_panel_iv_metrics", "acf_corr")),
)


class ResultContractError(ValueError):
    """Raised when an evaluator result drifts from the published schema."""


class PublishedRowOnlyError(ValueError):
    """Raised when current code is asked to impersonate the frozen Gaussian row."""


class WindowHistoricalSim:
    """State-matched resampler over whole training future-increment windows."""

    def __init__(
        self,
        history_increment: np.ndarray,
        future_increment: np.ndarray,
        history_state: np.ndarray,
        *,
        iv_count: int,
        seed: int | None = None,
    ) -> None:
        self.history_increment = np.asarray(history_increment, dtype=np.float32)
        self.future_increment = np.asarray(future_increment, dtype=np.float32)
        self.history_state = np.asarray(history_state, dtype=np.float32)
        self.iv_count = int(iv_count)
        self.future_len = int(self.future_increment.shape[1])
        self.dimension = int(self.future_increment.shape[-1])
        self.rng = np.random.default_rng(seed)
        self.features = np.asarray(
            [
                self._features(self.history_increment[index], self.history_state[index])
                for index in range(self.history_increment.shape[0])
            ],
            dtype=np.float64,
        )
        self.feature_mean = self.features.mean(axis=0)
        self.feature_std = self.features.std(axis=0) + 1e-8
        self.normalized_features = (
            self.features - self.feature_mean
        ) / self.feature_std

    def _features(
        self, history_increment: np.ndarray, history_state: np.ndarray
    ) -> np.ndarray:
        iv = history_state[:, : self.iv_count].reshape(history_state.shape[0], 5, 5)
        mean_iv = float(iv[-5:].mean())
        mean_iv_series = iv.mean(axis=(1, 2))
        volatility_of_volatility = (
            float(np.diff(mean_iv_series).std()) if mean_iv_series.shape[0] > 1 else 0.0
        )
        slope = float(iv[-1, -1, 2] - iv[-1, 0, 2])
        spx_activity = float(np.abs(history_increment[-5:, self.iv_count]).mean())
        factors = history_state[:, self.iv_count :]
        yield_slope = (
            float(factors[-1, 8] - factors[-1, 7]) if factors.shape[-1] > 8 else 0.0
        )
        credit_width = (
            float(factors[-1, 10] - factors[-1, 9]) if factors.shape[-1] > 10 else 0.0
        )
        return np.asarray(
            [
                mean_iv,
                volatility_of_volatility,
                slope,
                spx_activity,
                yield_slope,
                credit_width,
            ],
            dtype=np.float64,
        )

    def sample_joint(
        self,
        history_changes: np.ndarray,
        *,
        n_samples: int,
        history_state: np.ndarray,
        **_: Any,
    ) -> np.ndarray:
        changes = np.asarray(history_changes, dtype=np.float32)
        states = np.asarray(history_state, dtype=np.float32)
        output = np.zeros(
            (changes.shape[0], int(n_samples), self.future_len, self.dimension),
            dtype=np.float32,
        )
        for batch_index in range(changes.shape[0]):
            query = (
                self._features(changes[batch_index], states[batch_index])
                - self.feature_mean
            ) / self.feature_std
            distance = np.linalg.norm(self.normalized_features - query[None, :], axis=1)
            logits = -distance
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= weights.sum()
            selected = self.rng.choice(
                self.future_increment.shape[0],
                size=int(n_samples),
                replace=True,
                p=weights,
            )
            output[batch_index] = self.future_increment[selected]
        return output


def _path_value(payload: Mapping[str, Any], path: Sequence[str]) -> object:
    current: object = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ResultContractError("missing metric path: " + ".".join(path))
        current = current[key]
    return current


def validate_result_schema(
    result: Mapping[str, Any], *, canonical_counts: bool = True
) -> None:
    missing = [key for key in CURRENT_PANEL_RESULT_KEYS if key not in result]
    if missing:
        raise ResultContractError(f"result missing keys: {missing}")
    if result["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResultContractError("unexpected result schema_version")
    if result["mode"] != MODE:
        raise ResultContractError("unexpected evaluator mode")
    if not isinstance(result["state_specs"], list) or len(result["state_specs"]) != 39:
        raise ResultContractError("current-panel result must contain 39 state specs")
    if canonical_counts and (result["n_windows"] != 441 or result["n_samples"] != 64):
        raise ResultContractError("published validation result must use 441x64 deck")
    iv_metrics = result["current_panel_iv_metrics"]
    if not isinstance(iv_metrics, Mapping):
        raise ResultContractError("current_panel_iv_metrics must be an object")
    missing_iv = [key for key in DISPLAYED_IV_METRIC_KEYS if key not in iv_metrics]
    if missing_iv:
        raise ResultContractError(f"result missing displayed IV metrics: {missing_iv}")
    for _name, path in METRIC_PATHS:
        value = _path_value(result, path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ResultContractError("metric is not numeric: " + ".".join(path))
        if not math.isfinite(float(value)):
            raise ResultContractError("metric is not finite: " + ".".join(path))


def threshold_status(metric: str, value: float) -> str | None:
    rule = TABLE_THRESHOLD_RULES.get(metric)
    if rule is None:
        return None
    kind = rule["kind"]
    if kind == "upper":
        failed = value > float(rule["hi"])
    elif kind == "lower":
        failed = value < float(rule["lo"])
    elif kind == "band":
        failed = value < float(rule["lo"]) or value > float(rule["hi"])
    else:
        raise ValueError(f"unsupported threshold kind {kind!r}")
    return "fail" if failed else "pass"


def build_run_manifest(
    *,
    columns: list[str],
    metadata: Mapping[str, Any],
    state_specs: list[Mapping[str, Any]],
    n_train_windows: int,
    n_val_windows: int,
    n_samples: int,
    seed: int,
    history_alignment_error: float,
    future_alignment_error: float,
) -> dict[str, Any]:
    return {
        "mode": MODE,
        "benchmark_split": "published_validation",
        "columns": list(columns),
        "metadata": dict(metadata),
        "state_specs": list(state_specs),
        "n_train_windows": int(n_train_windows),
        "n_val_windows": int(n_val_windows),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "iv_alignment": {
            "history_max_abs_error": float(history_alignment_error),
            "future_max_abs_error": float(future_alignment_error),
        },
    }


def compute_crps(
    samples_delta: np.ndarray, gt_delta: np.ndarray, *, iv_count: int
) -> dict[str, Any]:
    horizons = (1, 7, 14, 30)
    n, k, t_len, d = samples_delta.shape
    per_h: dict[str, list[float]] = {}
    for horizon in horizons:
        if horizon > t_len:
            continue
        t = horizon - 1
        samples = samples_delta[:, :, t, :]
        truth = gt_delta[:, t, :]
        mae = np.abs(samples - truth[:, None, :]).mean(axis=1)
        spread = np.zeros((n, d), dtype=np.float64)
        for index in range(k):
            spread += np.abs(samples[:, index : index + 1, :] - samples).sum(axis=1)
        spread /= float(k * k)
        per_h[str(horizon)] = (mae - 0.5 * spread).mean(axis=0).tolist()
    per_dim = np.asarray(list(per_h.values()), dtype=np.float64).mean(axis=0)
    return {
        "per_horizon": per_h,
        "per_dimension": per_dim.tolist(),
        "overall": float(per_dim.mean()),
        "iv_crps": float(per_dim[:iv_count].mean()),
        "factor_crps": float(per_dim[iv_count:].mean()),
    }


def compute_energy_score(
    samples_delta: np.ndarray, gt_delta: np.ndarray, train_delta: np.ndarray
) -> dict[str, Any]:
    mean = train_delta.mean(axis=0)
    std = train_delta.std(axis=0) + 1e-8
    samples_z = (samples_delta - mean[None, None, None, :]) / std[None, None, None, :]
    truth_z = (gt_delta - mean[None, None, :]) / std[None, None, :]
    n, k, t_len, _ = samples_delta.shape
    per_h: dict[str, float] = {}
    for horizon in (1, 7, 14, 30):
        if horizon > t_len:
            continue
        t = horizon - 1
        samples = samples_z[:, :, t, :]
        truth = truth_z[:, t, :]
        term1 = np.linalg.norm(samples - truth[:, None, :], axis=2).mean(axis=1)
        spread = np.zeros(n, dtype=np.float64)
        for index in range(k):
            spread += np.linalg.norm(
                samples[:, index : index + 1, :] - samples, axis=2
            ).sum(axis=1)
        spread /= float(k * k)
        per_h[str(horizon)] = float((term1 - 0.5 * spread).mean())
    return {"per_horizon": per_h, "overall": float(np.mean(list(per_h.values())))}


def compute_variogram_score(
    samples_delta: np.ndarray,
    gt_delta: np.ndarray,
    train_delta: np.ndarray,
    *,
    p: float = 0.5,
) -> dict[str, Any]:
    mean = train_delta.mean(axis=0)
    std = train_delta.std(axis=0) + 1e-8
    t = min(samples_delta.shape[2], 30) - 1
    samples = (samples_delta[:, :, t, :] - mean[None, None, :]) / std[None, None, :]
    truth = (gt_delta[:, t, :] - mean[None, :]) / std[None, :]
    sample_vario = np.mean(
        np.abs(samples[:, :, :, None] - samples[:, :, None, :]) ** p, axis=1
    )
    truth_vario = np.abs(truth[:, :, None] - truth[:, None, :]) ** p
    upper = np.triu(np.ones((samples.shape[-1], samples.shape[-1]), dtype=bool), k=1)
    return {
        "variogram_score": float(((sample_vario - truth_vario) ** 2)[:, upper].mean()),
        "p": float(p),
        "horizon": int(t + 1),
    }


def compute_correlation_score(
    samples_delta: np.ndarray, gt_delta: np.ndarray, *, iv_count: int
) -> dict[str, Any]:
    truth = gt_delta[..., iv_count:].reshape(-1, gt_delta.shape[-1] - iv_count)
    generated = samples_delta[..., iv_count:].reshape(-1, gt_delta.shape[-1] - iv_count)
    truth_corr = np.nan_to_num(np.corrcoef(truth.T), nan=0.0)
    generated_corr = np.nan_to_num(np.corrcoef(generated.T), nan=0.0)
    frobenius = float(np.linalg.norm(generated_corr - truth_corr, ord="fro"))
    ratio = frobenius / max(float(np.linalg.norm(truth_corr, ord="fro")), 1e-12)
    return {
        "frobenius_error": frobenius,
        "frobenius_ratio": ratio,
        "corr_score": float(1.0 - ratio),
    }


def compute_factor_ks(
    samples_delta: np.ndarray,
    gt_delta: np.ndarray,
    factor_names: list[str],
    *,
    iv_count: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    statistics: list[float] = []
    for index, name in enumerate(factor_names):
        generated = samples_delta[..., iv_count + index].reshape(-1)
        truth = gt_delta[..., iv_count + index].reshape(-1)
        statistic, p_value = sp_stats.ks_2samp(generated, truth)
        statistics.append(float(statistic))
        rows[name] = {"statistic": float(statistic), "p_value": float(p_value)}
    return {
        "per_factor": rows,
        "mean_statistic": float(np.mean(statistics)) if statistics else None,
        "median_statistic": float(np.median(statistics)) if statistics else None,
        "worst_statistic": float(np.max(statistics)) if statistics else None,
    }


def prepare_validation_blocks(
    *,
    data_dir: str | Path,
    protocol: ValidationProtocol = PUBLISHED_VALIDATION_PROTOCOL,
) -> tuple[np.ndarray, list[str], Any, Any]:
    """Build the exact pre-test train and published 441-window validation blocks."""
    panel, columns, _dates = load_aligned_iv_factor_panel(data_dir=data_dir)
    train_indices, val_indices = official_train_val_indices(
        test_start=protocol.test_start,
        val_size=protocol.val_size,
        history_len=protocol.history_len,
        future_len=protocol.future_len,
    )
    train = build_increment_coordinate_block(
        panel,
        columns,
        train_indices,
        history_len=protocol.history_len,
        future_len=protocol.future_len,
        iv_count=protocol.iv_count,
        iv_transform="bounded_logit",
    )
    validation = build_increment_coordinate_block(
        panel,
        columns,
        val_indices,
        history_len=protocol.history_len,
        future_len=protocol.future_len,
        iv_count=protocol.iv_count,
        iv_transform="bounded_logit",
    )
    if len(validation.specs) != protocol.state_count:
        raise RuntimeError(f"expected {protocol.state_count} states")
    return panel, columns, train, validation


def create_baseline(
    name: str,
    train_daily_increment: np.ndarray,
    *,
    future_len: int = 30,
    seed: int = 772,
    checkpoint: str | Path | None = None,
    foundation_artifact: str | Path | None = None,
    train_block: Any | None = None,
    device: str = "cpu",
) -> Any:
    if name == "random_walk":
        return JointRandomWalk(train_daily_increment, future_len=future_len, seed=seed)
    if name == "bootstrap":
        return JointBootstrap(train_daily_increment, future_len=future_len, seed=seed)
    if name == "historical_sim":
        if train_block is None:
            raise ValueError("historical_sim requires the prepared training block")
        return WindowHistoricalSim(
            train_block.history_increment,
            train_block.future_increment,
            train_block.history_state,
            iv_count=25,
            seed=seed,
        )
    if name == "pca_var":
        return JointPCAVAR(train_daily_increment, future_len=future_len, seed=seed)
    if name == "garch_ccc":
        raise PublishedRowOnlyError(
            "garch_ccc is the frozen Gaussian published row; validate its pinned "
            "artifacts with scenarioflow.evaluation.garch_ccc_contract"
        )
    if name == "garch_dcc":
        return JointGARCHDCC(train_daily_increment, future_len=future_len, seed=seed)
    if name == "filtered_hs":
        return JointFilteredHS(train_daily_increment, future_len=future_len, seed=seed)
    if name in DEEP_BASELINES:
        if checkpoint is None:
            raise FileNotFoundError(f"{name} requires an explicit checkpoint")
        return load_current_panel_deep_baseline(checkpoint, device=device, seed=seed)
    if name == "deepvar_sni":
        if checkpoint is None:
            raise FileNotFoundError("deepvar_sni requires an explicit checkpoint")
        return DeepVARSNIInterfaceBaseline.load(checkpoint, device=device, seed=seed)
    if name in FOUNDATION_BASELINES:
        if foundation_artifact is None:
            raise FileNotFoundError(f"{name} requires an explicit sealed artifact")
        return PrecomputedJointScenarioBaseline.from_manifest(foundation_artifact)
    raise ValueError(f"unknown or externally supplied baseline {name!r}")


def _serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_result(output_dir: str | Path, result: Mapping[str, Any]) -> Path:
    validate_result_schema(result)
    destination = Path(output_dir) / str(result["baseline_name"])
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "results_current_panel.json"
    path.write_text(
        json.dumps(_serializable(result), indent=2) + "\n", encoding="utf-8"
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe-contract", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.describe_contract:
        raise SystemExit(
            "use --describe-contract; execution is orchestrated by workflows.reproduction"
        )
    print(
        json.dumps(
            {
                "protocol": asdict(PUBLISHED_VALIDATION_PROTOCOL),
                "candidates": list(BASELINES),
                "metric_paths": [(name, list(path)) for name, path in METRIC_PATHS],
                "thresholds": TABLE_THRESHOLD_RULES,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
