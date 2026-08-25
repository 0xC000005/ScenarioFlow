#!/usr/bin/env python
"""
182a_v0: Pathwise residual-law model on top of the 179b backbone.

Keep:
  - support-aware transform
  - explicit mean-reverting mean path
  - structured covariance with exact block covariance-mixture semantics

Change:
  - replace residual patching with one conditional path transport over the
    full whitened residual path in geometry-aware basis space
  - train that transport with conditional flow matching from a structured
    smooth-plus-jump prior

v0 is intentionally staged:
  - backbone frozen
  - train only the path transport and prior parameters
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.basis_geometry import BasisGeometry
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    effective_rank,
    iv_to_unconstrained,
    make_serializable,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_179b_basis_centered_residual_transport import (
    BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel,
)
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import (
    evaluate_joint_subset,
)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half, 1)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class AdaLNPathBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.SiLU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.mod1 = nn.Linear(cond_dim, 2 * d_model)
        self.mod2 = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.mod1.weight)
        nn.init.zeros_(self.mod1.bias)
        nn.init.zeros_(self.mod2.weight)
        nn.init.zeros_(self.mod2.bias)

    def _adaln(self, x: torch.Tensor, mod: nn.Linear, cond: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        shift, scale = mod(cond).chunk(2, dim=-1)
        return norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self._adaln(x, self.mod1, cond, self.norm1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self._adaln(x, self.mod2, cond, self.norm2)
        x = x + self.ff(h)
        return x


class ConditionalPathFlowTransformer(nn.Module):
    def __init__(
        self,
        n_frames: int,
        n_cells: int,
        context_dim: int,
        d_model: int = 192,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_mult: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.n_cells = n_cells
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim + time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.input_proj = nn.Linear(n_cells, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_frames, d_model))
        self.blocks = nn.ModuleList(
            [AdaLNPathBlock(d_model=d_model, n_heads=n_heads, ff_mult=ff_mult, cond_dim=d_model) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_cells)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t_flat: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch = z_t_flat.shape[0]
        x = z_t_flat.view(batch, self.n_frames, self.n_cells)
        h = self.input_proj(x) + self.pos_embed
        cond = self.context_proj(torch.cat([context, self.time_embed(t)], dim=-1))
        for block in self.blocks:
            h = block(h, cond)
        out = self.out_proj(self.out_norm(h))
        return out.reshape(batch, self.n_frames * self.n_cells)


class StructuredBasisSmoothJumpPrior(nn.Module):
    def __init__(
        self,
        geometry: BasisGeometry,
        low_std: float = 1.00,
        mid_std: float = 0.75,
        high_std: float = 0.35,
        low_jump_prob: float = 0.005,
        mid_jump_prob: float = 0.025,
        high_jump_prob: float = 0.070,
        low_jump_scale: float = 0.10,
        mid_jump_scale: float = 0.30,
        high_jump_scale: float = 0.75,
    ):
        super().__init__()
        self.geometry = geometry
        self.register_buffer("band_id_flat", geometry.band_id.reshape(-1), persistent=False)
        self.smooth_log_std = nn.Parameter(
            torch.log(torch.tensor([low_std, mid_std, high_std], dtype=torch.float32))
        )
        self.jump_logit = nn.Parameter(
            torch.tensor(
                [
                    _logit(low_jump_prob),
                    _logit(mid_jump_prob),
                    _logit(high_jump_prob),
                ],
                dtype=torch.float32,
            )
        )
        self.jump_log_scale = nn.Parameter(
            torch.log(torch.tensor([low_jump_scale, mid_jump_scale, high_jump_scale], dtype=torch.float32))
        )

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        band_id = self.band_id_flat.to(device=device)
        smooth_std = torch.exp(self.smooth_log_std).to(device=device, dtype=dtype)[band_id]
        jump_prob = torch.sigmoid(self.jump_logit).to(device=device, dtype=dtype)[band_id]
        jump_scale = torch.exp(self.jump_log_scale).to(device=device, dtype=dtype)[band_id]

        smooth = torch.randn(batch_size, band_id.numel(), device=device, dtype=dtype) * smooth_std.unsqueeze(0)
        active = torch.bernoulli(jump_prob.expand(batch_size, -1)).to(dtype)
        laplace = torch.distributions.Laplace(
            torch.zeros((), device=device, dtype=dtype),
            torch.ones((), device=device, dtype=dtype),
        )
        jump = active * laplace.sample((batch_size, band_id.numel())) * jump_scale.unsqueeze(0)
        z0 = smooth + jump
        stats = {
            "prior_smooth_std_mean": smooth_std.mean(),
            "prior_jump_prob_mean": jump_prob.mean(),
            "prior_jump_scale_mean": jump_scale.mean(),
            "prior_active_rate": active.mean(),
            "prior_jump_abs_mean": jump.abs().mean(),
            "prior_std_mean": z0.std(dim=-1).mean(),
        }
        return z0, stats


class PathwiseResidualLawMeanRevertingCovarianceMixtureModel(
    BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel
):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        path_config: dict,
        prior_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-5,
        base_nu: float = 8.0,
        mix_chunk_size: int = 27,
    ):
        super().__init__(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            flow_config=flow_config,
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
            base_nu=base_nu,
            mix_chunk_size=mix_chunk_size,
        )
        self.path_config = dict(path_config)
        self.prior_config = dict(prior_config)
        self.n_ode_steps = int(path_config["n_ode_steps"])
        self.path_geometry = BasisGeometry(
            n_frames=decoder_config["n_frames"],
            grid_h=5,
            grid_w=5,
            low_scale_clip=flow_config.get("low_scale_clip", 1.2),
            mid_scale_clip=flow_config.get("mid_scale_clip", 0.7),
            high_scale_clip=flow_config.get("high_scale_clip", 0.35),
        )
        path_context_in = flow_config["context_dim"] + decoder_config["n_blocks"] * decoder_config["n_templates"] + 2
        self.path_context_adapter = nn.Sequential(
            nn.Linear(path_context_in, path_config["context_hidden_dim"]),
            nn.SiLU(),
            nn.Linear(path_config["context_hidden_dim"], path_config["context_dim"]),
        )
        self.path_transport = ConditionalPathFlowTransformer(
            n_frames=decoder_config["n_frames"],
            n_cells=decoder_config["n_cells"],
            context_dim=path_config["context_dim"],
            d_model=path_config["d_model"],
            n_heads=path_config["n_heads"],
            n_layers=path_config["n_layers"],
            ff_mult=path_config["ff_mult"],
            time_embed_dim=path_config["time_embed_dim"],
        )
        self.prior = StructuredBasisSmoothJumpPrior(self.path_geometry, **prior_config)

    def maybe_load_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        model_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key.startswith("path_context_adapter.") or key.startswith("path_transport.") or key.startswith("prior."):
                skipped.append(key)
                continue
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"  Warm start loaded from {ckpt_path}")
        print(
            f"  Warm start missing keys: {len(missing)} | unexpected keys: {len(unexpected)} | "
            f"shape-skipped: {len(skipped)}"
        )

    def build_path_context(
        self,
        flow_context: torch.Tensor,
        block_logits: torch.Tensor,
        base_local_delta: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        block_probs = F.softmax(block_logits, dim=-1).reshape(flow_context.shape[0], -1)
        local_rms = base_local_delta.pow(2).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        scale_feat = scale.unsqueeze(-1)
        raw = torch.cat([flow_context, block_probs, scale_feat, local_rms], dim=-1)
        return self.path_context_adapter(raw)

    def teacher_basis_flat_from_outputs(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        time_factor: torch.Tensor,
        time_diag: torch.Tensor,
        cell_factor: torch.Tensor,
        cell_diag: torch.Tensor,
        scale: torch.Tensor,
        base_local_delta: torch.Tensor,
        block_logits: torch.Tensor,
    ) -> torch.Tensor:
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        shared_local_delta = self.decoder.build_shared_local_delta(base_local_delta)
        shared_local_scale = torch.exp(0.5 * shared_local_delta)
        diff = (target_u - mu) / shared_local_scale
        white_t = torch.linalg.solve_triangular(chol_t, diff, upper=False)
        white_shared = torch.linalg.solve_triangular(chol_c, white_t.transpose(1, 2), upper=False).transpose(1, 2)
        white_blocks = white_shared.view(
            target_u.shape[0],
            self.decoder.n_blocks,
            self.decoder.block_len,
            self.decoder.n_cells,
        )
        map_assign = block_logits.argmax(dim=-1)
        map_factors, _map_logdet, _map_offdiag_rms = self.decoder.build_template_factors(map_assign)
        rhs_map = white_blocks.permute(0, 1, 3, 2).reshape(
            target_u.shape[0] * self.decoder.n_blocks,
            self.decoder.n_cells,
            self.decoder.block_len,
        )
        factor_map = map_factors.reshape(
            target_u.shape[0] * self.decoder.n_blocks,
            self.decoder.n_cells,
            self.decoder.n_cells,
        )
        base_blocks_map = torch.linalg.solve_triangular(factor_map, rhs_map, upper=False)
        base_blocks_map = base_blocks_map.reshape(
            target_u.shape[0],
            self.decoder.n_blocks,
            self.decoder.n_cells,
            self.decoder.block_len,
        ).permute(0, 1, 3, 2)
        base_white = base_blocks_map.reshape(target_u.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        return self.path_geometry.to_basis(base_white).reshape(target_u.shape[0], -1)

    def sample_basis_paths(self, path_context: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = path_context.shape[0]
        z, prior_stats = self.prior.sample(batch * n_samples, device=path_context.device, dtype=path_context.dtype)
        ctx = path_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
        dt = 1.0 / float(self.n_ode_steps)
        for step in range(self.n_ode_steps):
            t = torch.full(
                (batch * n_samples,),
                fill_value=step * dt,
                device=path_context.device,
                dtype=path_context.dtype,
            )
            v = self.path_transport(z, t, ctx)
            z = z + dt * v
        with torch.no_grad():
            stats = dict(prior_stats)
            stats["sample_basis_std_mean"] = z.std(dim=-1).mean()
        return z, stats

    @torch.no_grad()
    def sample_future_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            block_logits,
        ) = self.forward_from_history(history_01)
        batch, n_frames, n_cells = mu.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        path_context = self.build_path_context(flow_context, block_logits, base_local_delta, scale)
        basis_paths, _stats = self.sample_basis_paths(path_context, n_samples=n_samples)
        base_white_flat = self.path_geometry.from_basis(
            basis_paths.view(batch * n_samples, n_frames, n_cells)
        ).reshape(batch * n_samples, n_frames * n_cells)
        base_white = base_white_flat.view(batch * n_samples, self.decoder.n_blocks, self.decoder.block_len, n_cells)

        block_probs = F.softmax(block_logits, dim=-1)
        sampled_blocks = []
        for b in range(self.decoder.n_blocks):
            sampled = torch.multinomial(block_probs[:, b], num_samples=n_samples, replacement=True)
            sampled_blocks.append(sampled)
        sampled_assign = torch.stack(sampled_blocks, dim=-1)
        assign_flat = sampled_assign.reshape(batch * n_samples, self.decoder.n_blocks)
        sampled_factors, _logdet_cov, _offdiag_rms = self.decoder.build_template_factors(assign_flat)
        lhs = base_white.permute(0, 1, 3, 2).reshape(
            batch * n_samples * self.decoder.n_blocks,
            n_cells,
            self.decoder.block_len,
        )
        factor_flat = sampled_factors.reshape(
            batch * n_samples * self.decoder.n_blocks,
            n_cells,
            n_cells,
        )
        routed_white = torch.matmul(factor_flat, lhs)
        routed_white = routed_white.reshape(
            batch * n_samples,
            self.decoder.n_blocks,
            n_cells,
            self.decoder.block_len,
        ).permute(0, 1, 3, 2)
        routed_white = routed_white.reshape(batch, n_samples, n_frames, n_cells)

        temp = torch.einsum("bij,bsjk->bsik", chol_t, routed_white)
        noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
        shared_local_delta = self.decoder.build_shared_local_delta(base_local_delta)
        local_scale = torch.exp(0.5 * shared_local_delta).unsqueeze(1)
        return mu.unsqueeze(1) + noise * local_scale


def checkpoint_key(val_metrics: dict, joint_metrics: dict) -> tuple[float, float, float, float, float, float]:
    mr_ratio = float(joint_metrics.get("joint_sample_mr_ratio", float("nan")))
    jump_ks = float(joint_metrics.get("joint_pathwise_jump_ks", float("inf")))
    kurt = float(joint_metrics.get("joint_kurtosis_ratio", float("nan")))
    cov90 = float(joint_metrics.get("joint_cov90", float("nan")))
    tc = float(joint_metrics.get("joint_turb_calm_ratio", float("nan")))
    val_loss = float(val_metrics.get("val_flow_match_loss", float("inf")))
    mr_gap = 1e6 if not np.isfinite(mr_ratio) else abs(mr_ratio - 1.0)
    jump_gap = 1e6 if not np.isfinite(jump_ks) else jump_ks
    if not np.isfinite(kurt):
        kurt_gap = 1e6
    elif kurt < 0.5:
        kurt_gap = 0.5 - kurt
    elif kurt > 2.0:
        kurt_gap = kurt - 2.0
    else:
        kurt_gap = 0.0
    cov_gap = 1e6 if not np.isfinite(cov90) else abs(cov90 - 0.90)
    tc_gap = 1e6 if not np.isfinite(tc) else max(1.15 - tc, 0.0)
    return (mr_gap, jump_gap, kurt_gap, cov_gap, tc_gap, val_loss)


def flow_matching_loss(
    model: PathwiseResidualLawMeanRevertingCovarianceMixtureModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_u = iv_to_unconstrained(
        future_01,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    with torch.no_grad():
        (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            block_logits,
        ) = model.forward_from_history(history_01)
        target_basis = model.teacher_basis_flat_from_outputs(
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
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
    z0, prior_stats = model.prior.sample(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    t = torch.rand(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    z_t = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * target_basis
    target_v = target_basis - z0
    pred_v = model.path_transport(z_t, t, path_context)
    loss = F.mse_loss(pred_v, target_v)

    high_mask = model.path_geometry.high_band_mask().to(target_basis.device, dtype=target_basis.dtype)
    target_std = target_basis.std(dim=-1)
    pred_norm = pred_v.norm(dim=-1)
    target_jump_like = (target_basis.abs() > 2.5).float().mean(dim=-1)
    metrics = {
        "flow_match_loss": loss,
        "target_basis_std_mean": target_std.mean(),
        "prior_std_mean": prior_stats["prior_std_mean"],
        "prior_jump_prob_mean": prior_stats["prior_jump_prob_mean"],
        "prior_jump_scale_mean": prior_stats["prior_jump_scale_mean"],
        "prior_active_rate": prior_stats["prior_active_rate"],
        "pred_velocity_norm_mean": pred_norm.mean(),
        "target_jump_like_rate": target_jump_like.mean(),
        "target_high_band_abs_mean": ((target_basis.abs() * high_mask.unsqueeze(0)).sum(dim=-1) / high_mask.sum().clamp_min(1.0)).mean(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: PathwiseResidualLawMeanRevertingCovarianceMixtureModel,
    val_loader: DataLoader,
) -> dict[str, float]:
    model.eval()
    metric_names = [
        "flow_match_loss",
        "target_basis_std_mean",
        "prior_std_mean",
        "prior_jump_prob_mean",
        "prior_jump_scale_mean",
        "prior_active_rate",
        "pred_velocity_norm_mean",
        "target_jump_like_rate",
        "target_high_band_abs_mean",
    ]
    totals = {f"val_{k}": 0.0 for k in metric_names}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss, metrics = flow_matching_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    return {k: v / max(total_count, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="182a_v0: pathwise residual-law model")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_path", type=float, default=8e-4)
    parser.add_argument("--weight_decay_path", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--time_rank", type=int, default=6)
    parser.add_argument("--cell_rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--flow_context_dim", type=int, default=256)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_low_scale_clip", type=float, default=1.2)
    parser.add_argument("--flow_mid_scale_clip", type=float, default=0.7)
    parser.add_argument("--flow_high_scale_clip", type=float, default=0.35)
    parser.add_argument("--base_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-5)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--local_delta_clip", type=float, default=0.35)
    parser.add_argument("--n_blocks", type=int, default=5)
    parser.add_argument("--n_templates", type=int, default=3)
    parser.add_argument("--mix_chunk_size", type=int, default=27)
    parser.add_argument("--template_diag_clip", type=float, default=0.30)
    parser.add_argument("--template_offdiag_clip", type=float, default=0.18)
    parser.add_argument("--drift_strength_max", type=float, default=0.75)
    parser.add_argument("--equilibrium_offset_clip", type=float, default=0.20)
    parser.add_argument("--init_drift_strength", type=float, default=0.20)
    parser.add_argument("--path_context_dim", type=int, default=256)
    parser.add_argument("--path_context_hidden_dim", type=int, default=256)
    parser.add_argument("--path_d_model", type=int, default=192)
    parser.add_argument("--path_heads", type=int, default=4)
    parser.add_argument("--path_layers", type=int, default=4)
    parser.add_argument("--path_ff_mult", type=int, default=4)
    parser.add_argument("--path_time_embed_dim", type=int, default=64)
    parser.add_argument("--path_ode_steps", type=int, default=8)
    parser.add_argument("--prior_low_std", type=float, default=1.00)
    parser.add_argument("--prior_mid_std", type=float, default=0.75)
    parser.add_argument("--prior_high_std", type=float, default=0.35)
    parser.add_argument("--prior_low_jump_prob", type=float, default=0.005)
    parser.add_argument("--prior_mid_jump_prob", type=float, default=0.025)
    parser.add_argument("--prior_high_jump_prob", type=float, default=0.070)
    parser.add_argument("--prior_low_jump_scale", type=float, default=0.10)
    parser.add_argument("--prior_mid_jump_scale", type=float, default=0.30)
    parser.add_argument("--prior_high_jump_scale", type=float, default=0.75)
    parser.add_argument("--warm_start_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    hist_len = args.history_len
    future_len = args.future_len

    test_start = 4511
    max_train_idx = test_start - hist_len - future_len
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, hist_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, hist_len, future_len)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size,
        shuffle=False,
    )

    encoder_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=64,
        bottleneck_dim=128,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_frames=future_len,
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=128,
        time_rank=args.time_rank,
        cell_rank=args.cell_rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim,
        local_delta_clip=args.local_delta_clip,
        n_blocks=args.n_blocks,
        n_templates=args.n_templates,
        template_diag_clip=args.template_diag_clip,
        template_offdiag_clip=args.template_offdiag_clip,
        drift_strength_max=args.drift_strength_max,
        equilibrium_offset_clip=args.equilibrium_offset_clip,
        init_drift_strength=args.init_drift_strength,
    )
    flow_config = dict(
        dim=future_len * 25,
        context_dim=args.flow_context_dim,
        n_frames=future_len,
        grid_h=5,
        grid_w=5,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        low_scale_clip=args.flow_low_scale_clip,
        mid_scale_clip=args.flow_mid_scale_clip,
        high_scale_clip=args.flow_high_scale_clip,
    )
    path_config = dict(
        context_dim=args.path_context_dim,
        context_hidden_dim=args.path_context_hidden_dim,
        d_model=args.path_d_model,
        n_heads=args.path_heads,
        n_layers=args.path_layers,
        ff_mult=args.path_ff_mult,
        time_embed_dim=args.path_time_embed_dim,
        n_ode_steps=args.path_ode_steps,
    )
    prior_config = dict(
        low_std=args.prior_low_std,
        mid_std=args.prior_mid_std,
        high_std=args.prior_high_std,
        low_jump_prob=args.prior_low_jump_prob,
        mid_jump_prob=args.prior_mid_jump_prob,
        high_jump_prob=args.prior_high_jump_prob,
        low_jump_scale=args.prior_low_jump_scale,
        mid_jump_scale=args.prior_mid_jump_scale,
        high_jump_scale=args.prior_high_jump_scale,
    )

    model = PathwiseResidualLawMeanRevertingCovarianceMixtureModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        path_config=path_config,
        prior_config=prior_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
        mix_chunk_size=args.mix_chunk_size,
    ).to(device)

    if args.warm_start_path:
        model.maybe_load_warm_start(args.warm_start_path, device)

    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.flow.requires_grad_(False)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    n_path = sum(p.numel() for p in model.path_transport.parameters())
    n_ctx = sum(p.numel() for p in model.path_context_adapter.parameters())
    n_prior = sum(p.numel() for p in model.prior.parameters())
    print(f"\n{'=' * 72}")
    print("182a_v0: pathwise residual-law model")
    print(f"{'=' * 72}")
    print(f"  Frozen encoder params: {n_enc:,}")
    print(f"  Frozen decoder params: {n_dec:,}")
    print(f"  Frozen legacy residual flow params: {n_flow:,}")
    print(f"  Trainable path transport params: {n_path:,}")
    print(f"  Trainable context adapter params: {n_ctx:,}")
    print(f"  Trainable prior params: {n_prior:,}")
    print(f"  Total params: {n_enc + n_dec + n_flow + n_path + n_ctx + n_prior:,}")
    if args.warm_start_path:
        print(f"  Warm start: {args.warm_start_path}")

    optimizer = torch.optim.AdamW(
        list(model.path_transport.parameters())
        + list(model.path_context_adapter.parameters())
        + list(model.prior.parameters()),
        lr=args.lr_path,
        weight_decay=args.weight_decay_path,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metric_names = [
        "flow_match_loss",
        "target_basis_std_mean",
        "prior_std_mean",
        "prior_jump_prob_mean",
        "prior_jump_scale_mean",
        "prior_active_rate",
        "pred_velocity_norm_mean",
        "target_jump_like_rate",
        "target_high_band_abs_mean",
    ]

    best_key = None
    best_metrics = None
    history = []
    history_path = Path(args.output_dir) / "training_history.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        totals = {f"train_{k}": 0.0 for k in metric_names}
        nb = 0
        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = flow_matching_loss(model, history_01, future_01)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.path_transport.parameters())
                + list(model.path_context_adapter.parameters())
                + list(model.prior.parameters()),
                1.0,
            )
            optimizer.step()
            for key in totals:
                totals[key] += metrics[key.replace("train_", "")].item()
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in totals.items()}
        val_metrics = evaluate_teacher_forced(model, val_loader)
        joint_metrics = evaluate_joint_subset(
            model,
            val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )
        elapsed = time.time() - t0
        current_key = checkpoint_key(val_metrics, joint_metrics)
        is_best = best_key is None or current_key < best_key
        if is_best:
            best_key = current_key
            best_metrics = {**val_metrics, **joint_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_key": best_key,
                    "config": {
                        "type": "pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_182a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "base_nu": args.base_nu,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                        "mix_chunk_size": args.mix_chunk_size,
                        "frozen_backbone": True,
                    },
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        row = {"epoch": epoch, **train_metrics, **val_metrics, **joint_metrics}
        history.append(row)
        history_path.write_text(json.dumps(make_serializable(history), indent=2))
        print(
            f"Ep {epoch:3d}  train_fm={train_metrics['train_flow_match_loss']:.4f}  "
            f"val_fm={val_metrics['val_flow_match_loss']:.4f}  cov90={joint_metrics['joint_cov90']:.4f}  "
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  mr={joint_metrics['joint_sample_mr_ratio']:.3f}  "
            f"kurt={joint_metrics['joint_kurtosis_ratio']:.3f}  jumpKS={joint_metrics['joint_pathwise_jump_ks']:.3f}  "
            f"pJump={val_metrics['val_prior_jump_prob_mean']:.3f}  pAct={val_metrics['val_prior_active_rate']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "config": {
            "type": "pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_182a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "path": path_config,
            "prior": prior_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "base_nu": args.base_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "mix_chunk_size": args.mix_chunk_size,
            "frozen_backbone": True,
        },
        "best_key": best_key,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))

    if best_metrics is not None:
        print(
            "\nBest diagnostics: "
            f"joint_cov90={best_metrics['joint_cov90']:.4f}, "
            f"joint_turb_calm_ratio={best_metrics['joint_turb_calm_ratio']:.3f}, "
            f"joint_sample_mr_ratio={best_metrics['joint_sample_mr_ratio']:.3f}, "
            f"joint_kurtosis_ratio={best_metrics['joint_kurtosis_ratio']:.3f}, "
            f"joint_pathwise_jump_ks={best_metrics['joint_pathwise_jump_ks']:.3f}"
        )


if __name__ == "__main__":
    main()
