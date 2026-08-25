from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    iv_to_unconstrained,
    unconstrained_to_iv,
)


def compute_vol_of_vol(history: np.ndarray) -> np.ndarray:
    mean_iv = history.mean(axis=(-1, -2))
    daily_changes = np.diff(mean_iv, axis=1)
    return daily_changes.std(axis=1)


def regime_bucket_from_vov(vov: np.ndarray, edges: np.ndarray) -> np.ndarray:
    low, high = float(edges[0]), float(edges[1])
    buckets = np.full(vov.shape, 1, dtype=np.int64)
    buckets[vov <= low] = 0
    buckets[vov >= high] = 2
    return buckets


def repeat_forward_outputs(outputs: tuple[torch.Tensor, ...], n_repeat: int) -> tuple[torch.Tensor, ...]:
    batch = outputs[0].shape[0]
    repeated: list[torch.Tensor] = []
    for tensor in outputs:
        expanded = tensor.unsqueeze(1).expand(batch, n_repeat, *tensor.shape[1:])
        repeated.append(expanded.reshape(batch * n_repeat, *tensor.shape[1:]))
    return tuple(repeated)


def encode_teacher_basis(
    model,
    target_u: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    (
        mu,
        time_factor,
        time_diag,
        cell_factor,
        cell_diag,
        scale,
        _flow_context,
        base_local_delta,
        block_logits,
    ) = outputs
    basis = model.teacher_basis_flat_from_outputs(
        target_u=target_u,
        mu=mu,
        time_factor=time_factor,
        time_diag=time_diag,
        cell_factor=cell_factor,
        cell_diag=cell_diag,
        scale=scale,
        base_local_delta=base_local_delta,
        block_logits=block_logits,
    )
    return basis.view(target_u.shape[0], model.decoder.n_frames, model.decoder.n_cells)


def decode_teacher_basis(
    model,
    basis: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    (
        mu,
        time_factor,
        time_diag,
        cell_factor,
        cell_diag,
        scale,
        _flow_context,
        base_local_delta,
        block_logits,
    ) = outputs
    batch, n_frames, n_cells = basis.shape
    cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
    chol_t = torch.linalg.cholesky(cov_t)
    chol_c = torch.linalg.cholesky(cov_c)
    base_white = model.path_geometry.from_basis(basis)
    white_blocks = base_white.view(batch, model.decoder.n_blocks, model.decoder.block_len, n_cells)
    map_assign = block_logits.argmax(dim=-1)
    map_factors, _map_logdet, _map_offdiag_rms = model.decoder.build_template_factors(map_assign)
    lhs = white_blocks.permute(0, 1, 3, 2).reshape(
        batch * model.decoder.n_blocks,
        n_cells,
        model.decoder.block_len,
    )
    factor_flat = map_factors.reshape(
        batch * model.decoder.n_blocks,
        n_cells,
        n_cells,
    )
    routed_white = torch.matmul(factor_flat, lhs)
    routed_white = routed_white.reshape(
        batch,
        model.decoder.n_blocks,
        n_cells,
        model.decoder.block_len,
    ).permute(0, 1, 3, 2)
    routed_white = routed_white.reshape(batch, n_frames, n_cells)
    temp = torch.einsum("bij,bjk->bik", chol_t, routed_white)
    noise = torch.einsum("btj,bcj->btc", temp, chol_c)
    shared_local_delta = model.decoder.build_shared_local_delta(base_local_delta)
    local_scale = torch.exp(0.5 * shared_local_delta)
    return mu + noise * local_scale


@dataclass
class TailCalFitResult:
    gen_quantiles: np.ndarray
    gt_quantiles: np.ndarray
    global_gen_quantiles: np.ndarray
    global_gt_quantiles: np.ndarray
    quantile_levels: np.ndarray
    vov_edges: np.ndarray
    block_bounds: np.ndarray
    future_len: int
    support_lo: float
    support_hi: float
    support_eps: float
    log_shift_clip: float
    fit_summary: dict[str, Any]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            gen_quantiles=self.gen_quantiles,
            gt_quantiles=self.gt_quantiles,
            global_gen_quantiles=self.global_gen_quantiles,
            global_gt_quantiles=self.global_gt_quantiles,
            quantile_levels=self.quantile_levels,
            vov_edges=self.vov_edges,
            block_bounds=self.block_bounds,
            future_len=np.array(self.future_len, dtype=np.int64),
            support_lo=np.array(self.support_lo, dtype=np.float32),
            support_hi=np.array(self.support_hi, dtype=np.float32),
            support_eps=np.array(self.support_eps, dtype=np.float32),
            log_shift_clip=np.array(self.log_shift_clip, dtype=np.float32),
            fit_summary_json=np.array(json.dumps(self.fit_summary), dtype=object),
        )


class TeacherBasisBlockTailCalibrator:
    def __init__(self, tailcal_path: str | Path, alpha: float = 1.0):
        data = np.load(tailcal_path, allow_pickle=True)
        self.gen_quantiles = data["gen_quantiles"]
        gt_raw = data["gt_quantiles"]
        self.gt_quantiles = self.gen_quantiles + alpha * (gt_raw - self.gen_quantiles)
        self.global_gen_quantiles = data["global_gen_quantiles"]
        global_gt_raw = data["global_gt_quantiles"]
        self.global_gt_quantiles = self.global_gen_quantiles + alpha * (global_gt_raw - self.global_gen_quantiles)
        self.quantile_levels = data["quantile_levels"]
        self.vov_edges = data["vov_edges"]
        self.block_bounds = data["block_bounds"]
        self.future_len = int(data["future_len"])
        self.support_lo = float(data["support_lo"])
        self.support_hi = float(data["support_hi"])
        self.support_eps = float(data["support_eps"])
        self.log_shift_clip = float(data["log_shift_clip"])
        self.alpha = alpha

    @staticmethod
    def _interp_with_extrapolation(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        result = np.interp(x, xp, fp)
        below = x < xp[0]
        if np.any(below):
            slope_lo = (fp[1] - fp[0]) / (xp[1] - xp[0]) if xp[1] != xp[0] else 0.0
            result[below] = fp[0] + slope_lo * (x[below] - xp[0])
        above = x > xp[-1]
        if np.any(above):
            slope_hi = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) if xp[-1] != xp[-2] else 0.0
            result[above] = fp[-1] + slope_hi * (x[above] - xp[-1])
        return result

    def _map_log_radius(self, log_r: np.ndarray, bucket: int, block_idx: int) -> np.ndarray:
        xp = self.gen_quantiles[bucket, block_idx]
        fp = self.gt_quantiles[bucket, block_idx]
        if not np.all(np.isfinite(xp)) or np.ptp(xp) < 1e-6:
            xp = self.global_gen_quantiles[block_idx]
            fp = self.global_gt_quantiles[block_idx]
        mapped = self._interp_with_extrapolation(log_r, xp, fp)
        shift = np.clip(mapped - log_r, -self.log_shift_clip, self.log_shift_clip)
        return log_r + shift

    def apply(
        self,
        samples: np.ndarray,
        history: np.ndarray,
        model,
        device: str,
        batch_size: int = 8,
    ) -> np.ndarray:
        if samples.shape[2] != self.future_len:
            raise ValueError(f"Tail calibrator expects future_len={self.future_len}, got {samples.shape[2]}")
        model.eval()
        vov = compute_vol_of_vol(history)
        regime_bucket = regime_bucket_from_vov(vov, self.vov_edges)
        calibrated_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, samples.shape[0], batch_size):
                end = min(start + batch_size, samples.shape[0])
                hist_t = torch.tensor(history[start:end], dtype=torch.float32, device=device)
                samp_t = torch.tensor(samples[start:end], dtype=torch.float32, device=device)
                batch = end - start
                n_samples = samp_t.shape[1]
                outputs = model.forward_from_history(hist_t)
                outputs_rep = repeat_forward_outputs(outputs, n_samples)
                future_u = iv_to_unconstrained(
                    samp_t.reshape(batch * n_samples, self.future_len, -1),
                    lo=self.support_lo,
                    hi=self.support_hi,
                    eps=self.support_eps,
                )
                basis = encode_teacher_basis(model, future_u, outputs_rep)
                basis_np = basis.detach().cpu().numpy()
                basis_cal = basis_np.copy()
                bucket_rep = np.repeat(regime_bucket[start:end], n_samples)
                for block_idx in range(len(self.block_bounds) - 1):
                    lo = int(self.block_bounds[block_idx])
                    hi = int(self.block_bounds[block_idx + 1])
                    block = basis_np[:, lo:hi, :]
                    flat = block.reshape(block.shape[0], -1)
                    r = np.linalg.norm(flat, axis=1)
                    log_r = np.log(np.clip(r, 1e-8, None))
                    for bucket in range(3):
                        mask = bucket_rep == bucket
                        if not np.any(mask):
                            continue
                        mapped_log_r = self._map_log_radius(log_r[mask], bucket, block_idx)
                        scale = np.exp(mapped_log_r - log_r[mask]).astype(np.float32)
                        basis_cal[mask, lo:hi, :] *= scale[:, None, None]
                basis_cal_t = torch.tensor(basis_cal, dtype=basis.dtype, device=device)
                calibrated_u = decode_teacher_basis(model, basis_cal_t, outputs_rep)
                calibrated_iv = unconstrained_to_iv(
                    calibrated_u,
                    lo=self.support_lo,
                    hi=self.support_hi,
                )
                calibrated_batches.append(
                    calibrated_iv.view(batch, n_samples, self.future_len, 5, 5).cpu().numpy()
                )
        return np.concatenate(calibrated_batches, axis=0)


def fit_tailcal_map(
    gen_log_r: np.ndarray,
    gt_log_r: np.ndarray,
    regime_bucket: np.ndarray,
    block_idx: np.ndarray,
    quantile_levels: np.ndarray,
    vov_edges: np.ndarray,
    block_bounds: np.ndarray,
    future_len: int,
    support_lo: float,
    support_hi: float,
    support_eps: float,
    log_shift_clip: float,
) -> TailCalFitResult:
    n_buckets = 3
    n_blocks = len(block_bounds) - 1
    gen_quantiles = np.zeros((n_buckets, n_blocks, len(quantile_levels)), dtype=np.float32)
    gt_quantiles = np.zeros((n_buckets, n_blocks, len(quantile_levels)), dtype=np.float32)
    global_gen_quantiles = np.zeros((n_blocks, len(quantile_levels)), dtype=np.float32)
    global_gt_quantiles = np.zeros((n_blocks, len(quantile_levels)), dtype=np.float32)

    fit_summary: dict[str, Any] = {"counts": {}}
    for b in range(n_blocks):
        gmask = block_idx == b
        global_gen_quantiles[b] = np.quantile(gen_log_r[gmask], quantile_levels).astype(np.float32)
        global_gt_quantiles[b] = np.quantile(gt_log_r[gmask], quantile_levels).astype(np.float32)
        fit_summary["counts"][f"global_block_{b}"] = int(gmask.sum())
        for k in range(n_buckets):
            mask = (regime_bucket == k) & (block_idx == b)
            fit_summary["counts"][f"bucket_{k}_block_{b}"] = int(mask.sum())
            if mask.sum() < max(32, len(quantile_levels)):
                gen_quantiles[k, b] = global_gen_quantiles[b]
                gt_quantiles[k, b] = global_gt_quantiles[b]
            else:
                gen_quantiles[k, b] = np.quantile(gen_log_r[mask], quantile_levels).astype(np.float32)
                gt_quantiles[k, b] = np.quantile(gt_log_r[mask], quantile_levels).astype(np.float32)

    fit_summary["vov_edges"] = [float(vov_edges[0]), float(vov_edges[1])]
    fit_summary["block_bounds"] = [int(x) for x in block_bounds.tolist()]
    fit_summary["log_shift_clip"] = float(log_shift_clip)
    return TailCalFitResult(
        gen_quantiles=gen_quantiles,
        gt_quantiles=gt_quantiles,
        global_gen_quantiles=global_gen_quantiles,
        global_gt_quantiles=global_gt_quantiles,
        quantile_levels=quantile_levels.astype(np.float32),
        vov_edges=vov_edges.astype(np.float32),
        block_bounds=block_bounds.astype(np.int64),
        future_len=future_len,
        support_lo=support_lo,
        support_hi=support_hi,
        support_eps=support_eps,
        log_shift_clip=log_shift_clip,
        fit_summary=fit_summary,
    )
