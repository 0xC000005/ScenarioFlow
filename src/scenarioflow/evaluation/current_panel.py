"""Published current-panel validation contract and common metric battery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy import stats as sp_stats

from ..baselines.classical import (
    JointBootstrap,
    JointFilteredHS,
    JointGARCHDCC,
    JointPCAVAR,
    JointRandomWalk,
)
from ..baselines.deep import load_current_panel_deep_baseline
from ..baselines.foundation import (
    FoundationArtifactAuthority,
    PrecomputedJointScenarioBaseline,
)
from ..baselines.g2 import DeepVARSNIInterfaceBaseline
from ..data import load_aligned_iv_factor_panel, official_train_val_indices
from ..state import (
    build_increment_coordinate_block,
    normalize_increment_windows,
    reconstruct_state_from_increments,
    state_panel_from_specs,
)
from ..model import load_model as load_scenarioflow_model
from .iv_suite import run_full_iv_suite

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


class ScenarioFlowCheckpointBaseline:
    """Adapter from a package checkpoint to the shared increment sampler."""

    def __init__(self, checkpoint: str | Path, *, device: str, seed: int) -> None:
        self.device = torch.device(device)
        self.model, _ = load_scenarioflow_model(checkpoint, self.device)
        self.model.eval()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        self._rng_state = generator.get_state()

    def sample_joint(
        self,
        history_changes: np.ndarray,
        *,
        n_samples: int,
        history_state: np.ndarray,
        **_: Any,
    ) -> np.ndarray:
        history = np.asarray(history_changes, dtype=np.float32)
        normalized, _future, center, scale = normalize_increment_windows(
            history,
            history,
            half_life=None,
            scale_floor=1e-4,
            center_mode="zero",
        )
        outputs = []
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self._rng_state, self.device)
            else:
                torch.set_rng_state(self._rng_state)
            for window in range(history.shape[0]):
                draws = []
                for _sample in range(n_samples):
                    draw = self.model.sample_batched(
                        torch.from_numpy(
                            np.asarray(
                                history_state[window : window + 1], dtype=np.float32
                            )
                        ).to(self.device),
                        torch.from_numpy(normalized[window : window + 1]).to(
                            self.device
                        ),
                        torch.from_numpy(center[window : window + 1]).to(self.device),
                        torch.from_numpy(scale[window : window + 1]).to(self.device),
                        n_samples=1,
                        n_steps=self.model.cfg.future_len,
                        chunk_size=1,
                    )
                    draws.append(draw.detach().cpu().numpy()[0, 0])
                outputs.append(draws)
            self._rng_state = (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else torch.get_rng_state()
            )
        return np.asarray(outputs, dtype=np.float32)


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
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
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
    if provenance is not None:
        manifest["provenance"] = dict(provenance)
    return manifest


def compute_block_alignment(panel: np.ndarray, block: Any) -> tuple[float, float]:
    """Recompute raw state windows from the source panel and report max errors."""
    state_panel = state_panel_from_specs(np.asarray(panel), block.specs)
    history_len = int(block.history_state.shape[1])
    future_len = int(block.future_state.shape[1])
    expected_history = np.stack(
        [state_panel[start : start + history_len] for start in block.indices]
    )
    expected_future = np.stack(
        [
            state_panel[start + history_len : start + history_len + future_len]
            for start in block.indices
        ]
    )
    return (
        float(np.max(np.abs(expected_history - block.history_state))),
        float(np.max(np.abs(expected_future - block.future_state))),
    )


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
    foundation_authority: FoundationArtifactAuthority | None = None,
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
    if name in {"scenarioflow", "scenarioflow_graph"}:
        if checkpoint is None:
            raise FileNotFoundError(f"{name} requires an explicit checkpoint")
        return ScenarioFlowCheckpointBaseline(checkpoint, device=device, seed=seed)
    if name in FOUNDATION_BASELINES:
        if foundation_artifact is None:
            raise FileNotFoundError(f"{name} requires an explicit sealed artifact")
        if foundation_authority is None:
            raise ValueError(f"{name} requires external foundation authority")
        return PrecomputedJointScenarioBaseline.from_authority(
            foundation_artifact, foundation_authority
        )
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
    validate_result_schema(
        result,
        canonical_counts=(result["n_windows"] == 441 and result["n_samples"] == 64),
    )
    destination = Path(output_dir) / str(result["baseline_name"])
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "results_current_panel.json"
    path.write_text(
        json.dumps(_serializable(result), indent=2) + "\n", encoding="utf-8"
    )
    return path


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if left.size < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    left_rank = np.argsort(np.argsort(left)).astype(np.float64)
    right_rank = np.argsort(np.argsort(right)).astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 3 or np.std(left[keep]) < 1e-12 or np.std(right[keep]) < 1e-12:
        return 0.0
    value = float(np.corrcoef(left[keep], right[keep])[0, 1])
    return value if np.isfinite(value) else 0.0


def _finite_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 1.0 if abs(numerator) < 1e-12 else float(numerator / 1e-12)
    return float(numerator / denominator)


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    right_values = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    left_values = left_values[np.isfinite(left_values)]
    right_values = right_values[np.isfinite(right_values)]
    if left_values.size == 0 or right_values.size == 0:
        return float("nan")
    values = np.concatenate([left_values, right_values])
    left_cdf = np.searchsorted(left_values, values, side="right") / left_values.size
    right_cdf = np.searchsorted(right_values, values, side="right") / right_values.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def evaluate_iv_suite(
    conditional_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
    *,
    shuffled_samples: np.ndarray,
    conditionality_batch_size: int = 32,
    returns: np.ndarray,
    window_indices: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Compute the complete published IV diagnostic payload on raw IV levels."""
    return run_full_iv_suite(
        conditional_samples,
        ground_truth,
        history,
        shuffled_samples=shuffled_samples,
        conditionality_batch_size=conditionality_batch_size,
        returns=returns,
        window_indices=window_indices,
    )


def summarize_joint_quality(
    raw_history: np.ndarray,
    raw_future: np.ndarray,
    sample_raw: np.ndarray,
    factor_names: Sequence[str],
    *,
    iv_count: int,
) -> dict[str, Any]:
    anchor = raw_history[:, -1]
    truth_delta = np.diff(np.concatenate([anchor[:, None], raw_future], axis=1), axis=1)
    sample_anchor = np.repeat(anchor[:, None, None], sample_raw.shape[1], axis=1)
    sample_delta = np.diff(np.concatenate([sample_anchor, sample_raw], axis=2), axis=2)
    factor_slice = slice(iv_count, raw_future.shape[-1])
    gt_factor_delta = truth_delta[..., factor_slice]
    gen_factor_delta = sample_delta[..., factor_slice]
    gt_factor_level = raw_future[..., factor_slice]
    gen_factor_level = sample_raw[..., factor_slice]
    factor_ks, tail_ratios, per_factor = [], [], []
    for position, name in enumerate(factor_names):
        observed = gt_factor_delta[..., position]
        generated = gen_factor_delta[..., position]
        ks_value = _ks_statistic(observed, generated)
        q99_ratio = float(np.quantile(np.abs(generated), 0.99)) / max(
            float(np.quantile(np.abs(observed), 0.99)), 1e-12
        )
        factor_ks.append(ks_value)
        tail_ratios.append(q99_ratio)
        per_factor.append(
            {
                "name": str(name),
                "gt_min": float(np.nanmin(gt_factor_level[..., position])),
                "gt_max": float(np.nanmax(gt_factor_level[..., position])),
                "gen_min": float(np.nanmin(gen_factor_level[..., position])),
                "gen_max": float(np.nanmax(gen_factor_level[..., position])),
                "ks_delta": ks_value,
                "q99_abs_delta_ratio": q99_ratio,
            }
        )

    def safe_corrcoef(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        keep = values.std(axis=0) > 1e-12
        output = np.eye(values.shape[1], dtype=np.float64)
        if keep.sum() >= 2:
            output[np.ix_(keep, keep)] = np.nan_to_num(
                np.corrcoef(values[:, keep], rowvar=False), nan=0.0
            )
        return output

    def corr_similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
        upper = np.triu_indices(left.shape[0], 1)
        left_values, right_values = left[upper], right[upper]
        return {
            "upper_corr": _safe_corr(left_values, right_values),
            "mae": float(np.mean(np.abs(left_values - right_values))),
            "frobenius": float(
                np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12)
            ),
            "gt_mean_abs": float(np.mean(np.abs(left_values))),
            "gen_mean_abs": float(np.mean(np.abs(right_values))),
        }

    gt_all = truth_delta.reshape(-1, truth_delta.shape[-1])
    gen_all = sample_delta.reshape(-1, sample_delta.shape[-1])
    gt_corr, gen_corr = safe_corrcoef(gt_all), safe_corrcoef(gen_all)
    factor_corr = corr_similarity(
        gt_corr[iv_count:, iv_count:], gen_corr[iv_count:, iv_count:]
    )
    gt_cross, gen_cross = gt_corr[:iv_count, iv_count:], gen_corr[:iv_count, iv_count:]
    cross = {
        "matrix_corr": _safe_corr(gt_cross, gen_cross),
        "mae": float(np.mean(np.abs(gt_cross - gen_cross))),
        "gt_mean_abs": float(np.mean(np.abs(gt_cross))),
        "gen_mean_abs": float(np.mean(np.abs(gen_cross))),
    }
    median = np.median(sample_raw, axis=1)
    rolled = np.roll(median, 1, axis=0)
    conditional_mae = float(np.mean(np.abs(median - raw_future)))
    rolled_mae = float(np.mean(np.abs(rolled - raw_future)))
    history_activity = np.mean(np.diff(raw_history, axis=1) ** 2, axis=(1, 2))
    future_activity = np.mean(truth_delta**2, axis=(1, 2))
    deck_width = (
        np.quantile(sample_raw, 0.95, axis=1) - np.quantile(sample_raw, 0.05, axis=1)
    ).mean(axis=(1, 2))
    return {
        "finite_rate": float(np.isfinite(sample_raw).mean()),
        "n_windows": int(raw_future.shape[0]),
        "n_samples": int(sample_raw.shape[1]),
        "future_len": int(raw_future.shape[1]),
        "n_factors": int(len(factor_names)),
        "factor_delta_ks_mean": float(np.mean(factor_ks)),
        "factor_delta_ks_pass_020": int(np.sum(np.asarray(factor_ks) < 0.20)),
        "factor_tail_q99_ratio_median": float(np.median(tail_ratios)),
        "factor_tail_q99_pass_05_20": int(
            np.sum((np.asarray(tail_ratios) >= 0.5) & (np.asarray(tail_ratios) <= 2.0))
        ),
        "factor_factor_corr": factor_corr,
        "iv_factor_corr": cross,
        "conditional_panel": {
            "conditional_median_mae": conditional_mae,
            "rolled_median_mae": rolled_mae,
            "median_mae_reduction_vs_rolled_pct": float(
                (rolled_mae - conditional_mae) / max(rolled_mae, 1e-12) * 100.0
            ),
            "history_activity_width_spearman": _safe_spearman(
                history_activity, deck_width
            ),
            "future_activity_width_spearman": _safe_spearman(
                future_activity, deck_width
            ),
        },
        "per_factor": per_factor,
    }


def evaluate_baseline(
    *,
    name: str,
    model: Any,
    train_block: Any,
    validation_block: Any,
    n_samples: int,
    batch_size: int,
    returns: np.ndarray,
    iv_count: int = 25,
    shuffle_seed: int = 772,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Sample, reconstruct, and score one prepared current-panel baseline."""

    def sample_order(order: np.ndarray) -> np.ndarray:
        generated: list[np.ndarray] = []
        for start in range(0, order.size, batch_size):
            positions = order[start : start + batch_size]
            kwargs = (
                {"history_state": validation_block.history_state[positions]}
                if name in {"historical_sim", "scenarioflow", "scenarioflow_graph"}
                else {}
            )
            generated.append(
                np.asarray(
                    model.sample_joint(
                        validation_block.history_increment[positions],
                        n_samples=n_samples,
                        **kwargs,
                    ),
                    dtype=np.float32,
                )
            )
        increments = np.concatenate(generated, axis=0)
        return reconstruct_state_from_increments(
            validation_block.history_state[order, -1],
            increments,
            validation_block.specs,
        )

    started = time.perf_counter()
    matched_order = np.arange(validation_block.history_increment.shape[0])
    sample_raw = sample_order(matched_order)
    shuffle_generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
    shuffled_chunks: list[np.ndarray] = []
    for start in range(0, min(matched_order.size, batch_size * 5), batch_size):
        base = matched_order[start : start + batch_size]
        local = torch.randperm(base.size, generator=shuffle_generator).numpy()
        if base.size > 1 and np.array_equal(local, np.arange(base.size)):
            local = np.roll(local, 1)
        shuffled_chunks.append(sample_order(base[local]))
    shuffled_raw = np.concatenate(shuffled_chunks, axis=0)
    raw_history = validation_block.history_state.astype(np.float32)
    raw_future = validation_block.future_state.astype(np.float32)
    anchor = raw_history[:, -1]
    gt_delta = np.diff(np.concatenate([anchor[:, None], raw_future], axis=1), axis=1)
    sample_anchor = np.repeat(anchor[:, None, None], n_samples, axis=1)
    sample_delta = np.diff(np.concatenate([sample_anchor, sample_raw], axis=2), axis=2)
    train_anchor = train_block.history_state[:, -1]
    train_delta = np.diff(
        np.concatenate([train_anchor[:, None], train_block.future_state], axis=1),
        axis=1,
    ).reshape(-1, len(train_block.specs))
    iv_samples = sample_raw[..., :iv_count]
    iv_truth = raw_future[..., :iv_count]
    iv_history = raw_history[..., :iv_count]
    iv_suite, iv_metrics = evaluate_iv_suite(
        iv_samples,
        iv_truth,
        iv_history,
        shuffled_samples=shuffled_raw[..., :iv_count],
        conditionality_batch_size=batch_size,
        returns=returns,
        window_indices=validation_block.indices,
    )
    factor_names = [spec.name for spec in validation_block.specs[iv_count:]]
    result = {
        "baseline_name": name,
        "mode": MODE,
        "generation_time_s": float(time.perf_counter() - started),
        "n_windows": int(raw_future.shape[0]),
        "n_samples": int(n_samples),
        "state_specs": [asdict(spec) for spec in validation_block.specs],
        "current_panel_iv": iv_suite,
        "current_panel_iv_metrics": iv_metrics,
        "joint_panel_quality": summarize_joint_quality(
            raw_history,
            raw_future,
            sample_raw,
            factor_names,
            iv_count=iv_count,
        ),
        "panel_b_crps": compute_crps(sample_delta, gt_delta, iv_count=iv_count),
        "panel_b_energy_score": compute_energy_score(
            sample_delta, gt_delta, train_delta
        ),
        "panel_b_variogram_score": compute_variogram_score(
            sample_delta, gt_delta, train_delta
        ),
        "panel_b_correlation": compute_correlation_score(
            sample_delta, gt_delta, iv_count=iv_count
        ),
        "panel_b_factor_ks": compute_factor_ks(
            sample_delta, gt_delta, factor_names, iv_count=iv_count
        ),
    }
    return result, {
        "samples_delta": sample_delta,
        "ground_truth_delta": gt_delta,
        "train_delta": train_delta,
        "conditional_iv_samples": iv_samples.reshape(
            raw_future.shape[0], n_samples, raw_future.shape[1], 5, 5
        ),
        "ground_truth_iv": iv_truth.reshape(
            raw_future.shape[0], raw_future.shape[1], 5, 5
        ),
        "history_iv": iv_history.reshape(
            raw_history.shape[0], raw_history.shape[1], 5, 5
        ),
        "raw_history": raw_history,
        "sample_raw": sample_raw,
        "window_indices": validation_block.indices,
        "variable_names": [spec.name for spec in validation_block.specs],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe-contract", action="store_true")
    parser.add_argument(
        "--baseline",
        choices=BASELINES + ("deepvar_sni", "scenarioflow", "scenarioflow_graph"),
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--foundation-artifact")
    parser.add_argument("--foundation-authority")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=441)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=772)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-window-scores", action="store_true")
    parser.add_argument("--protocol")
    parser.add_argument("--node-id")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.describe_contract:
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
        return
    if args.baseline is None or args.data_dir is None or args.output_dir is None:
        raise SystemExit("--baseline, --data-dir, and --output-dir are required")
    if args.n_samples <= 0 or args.batch_size <= 0:
        raise SystemExit("--n-samples and --batch-size must be positive")
    if not 1 <= args.max_windows <= PUBLISHED_VALIDATION_PROTOCOL.val_size:
        raise SystemExit("--max-windows must be between 1 and 441")

    panel, columns, train, validation = prepare_validation_blocks(
        data_dir=args.data_dir
    )
    if args.max_windows < validation.indices.size:
        count = args.max_windows
        validation = replace(
            validation,
            history_increment=validation.history_increment[:count],
            future_increment=validation.future_increment[:count],
            history_state=validation.history_state[:count],
            future_state=validation.future_state[:count],
            indices=validation.indices[:count],
        )
    authority = None
    if args.foundation_authority is not None:
        document = json.loads(
            Path(args.foundation_authority).read_text(encoding="utf-8")
        )
        authority = FoundationArtifactAuthority(**document)
    train_daily = train.history_increment[:, -1, :]
    model = create_baseline(
        args.baseline,
        train_daily,
        future_len=PUBLISHED_VALIDATION_PROTOCOL.future_len,
        seed=args.seed,
        checkpoint=args.checkpoint,
        foundation_artifact=args.foundation_artifact,
        foundation_authority=authority,
        train_block=train,
        device=args.device,
    )
    result, artifact_inputs = evaluate_baseline(
        name=args.baseline,
        model=model,
        train_block=train,
        validation_block=validation,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        returns=np.load(
            Path(args.data_dir) / "vol_surface_with_ret.npz", allow_pickle=False
        )["ret"],
        shuffle_seed=args.seed,
    )
    result_path = write_result(args.output_dir, result)
    metadata = {
        "train_indices_start": int(train.indices[0]),
        "train_indices_end": int(train.indices[-1]),
        "val_indices_start": int(validation.indices[0]),
        "val_indices_end": int(validation.indices[-1]),
        "iv_transform": "bounded_logit",
        "positive_level_policy": "reference_based",
    }
    history_error, future_error = compute_block_alignment(panel, validation)
    if history_error > 1e-6 or future_error > 1e-6:
        raise RuntimeError(
            f"IV alignment failure: history={history_error}, future={future_error}"
        )
    data_files = (
        "vol_surface_with_ret.npz",
        "spx_vol_surface_history_full_data_fixed.parquet",
        "multi_factor_levels.parquet",
        "multi_factor_returns.parquet",
    )
    provenance = {
        "data_artifacts": {
            name: hashlib.sha256((Path(args.data_dir) / name).read_bytes()).hexdigest()
            for name in data_files
        },
        "checkpoint": (
            None
            if args.checkpoint is None
            else {
                "path": str(Path(args.checkpoint).resolve()),
                "sha256": hashlib.sha256(
                    Path(args.checkpoint).read_bytes()
                ).hexdigest(),
            }
        ),
        "evaluator": {
            "baseline": args.baseline,
            "batch_size": args.batch_size,
            "max_windows": args.max_windows,
            "n_samples": args.n_samples,
            "seed": args.seed,
        },
    }
    manifest = build_run_manifest(
        columns=columns,
        metadata=metadata,
        state_specs=[asdict(spec) for spec in validation.specs],
        n_train_windows=len(train.indices),
        n_val_windows=len(validation.indices),
        n_samples=args.n_samples,
        seed=args.seed,
        history_alignment_error=history_error,
        future_alignment_error=future_error,
        provenance=provenance,
    )
    if len(validation.indices) != 441 or args.n_samples != 64:
        manifest["benchmark_split"] = "published_validation_prefix_smoke"
    candidate_dir = result_path.parent
    (candidate_dir / "manifest.json").write_text(
        json.dumps(_serializable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.save_window_scores:
        from .window_scores import maybe_write_window_score_artifact

        maybe_write_window_score_artifact(
            enabled=True,
            output_path=candidate_dir / "window_scores.npz",
            existing_result=result,
            **artifact_inputs,
        )
    if (args.protocol is None) != (args.node_id is None):
        raise SystemExit("--protocol and --node-id must be supplied together")
    if args.protocol is not None:
        from ..workflows.reproduction import record_completed_node

        protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
        record_completed_node(
            output_dir=args.output_dir,
            protocol=protocol,
            node_id=args.node_id,
            result=result,
            evaluation_manifest=manifest,
        )
    print(result_path)


if __name__ == "__main__":
    main()
