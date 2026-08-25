#!/usr/bin/env python
"""
191a: AR Student-t with exact source/sink local variance reallocation.

Minimal AR-family refinement of 169c:
  - keep multistep AR recursion
  - keep low-rank + diagonal Student-t density forecasting
  - add exact top-k recipient / donor variance reallocation per step
  - add mild rollout-aware training via scheduled self-conditioning

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_191a_ar_reallocation_student_t.py \
        --output_dir models/backfill/ar_reallocation_student_t_191a --device cuda
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    effective_rank,
    inverse_softplus,
    iv_to_unconstrained,
    make_serializable,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    SpatialShapeScaleStudentTDecoder,
    ShapeScaleStudentTARModel,
    build_multistep_windows,
    evaluate_rollout_subset,
)


def _scaled_logit(prob: float) -> float:
    prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    return math.log(prob / (1.0 - prob))


def topk_sparse_distribution(scores: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    if k >= scores.shape[-1]:
        return F.softmax(scores / temperature, dim=-1)
    top_vals, top_idx = torch.topk(scores, k=k, dim=-1)
    top_probs = F.softmax(top_vals / temperature, dim=-1)
    probs = torch.zeros_like(scores)
    probs.scatter_(dim=-1, index=top_idx, src=top_probs)
    return probs


def build_target_reallocation(
    residual_sq: torch.Tensor,
    base_var: torch.Tensor,
    k_plus: int,
    k_minus: int,
    target_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.log(residual_sq.clamp_min(1e-8)) - torch.log(base_var.clamp_min(1e-8))
    raw = raw - raw.mean(dim=-1, keepdim=True)
    raw = raw.clamp(min=-target_clip, max=target_clip)

    pos_mass = raw.clamp_min(0.0)
    neg_mass = (-raw).clamp_min(0.0)

    target_plus = topk_sparse_distribution(pos_mass, k=k_plus, temperature=1.0)
    target_minus = topk_sparse_distribution(neg_mass, k=k_minus, temperature=1.0)

    target_w = torch.exp(raw)
    target_w = target_w / target_w.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    return target_plus, target_minus, target_w, raw


def sparse_target_ce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mass = target.sum(dim=-1)
    ce = -(target * pred.clamp_min(1e-8).log()).sum(dim=-1)
    ce = torch.where(mass > 0, ce, torch.zeros_like(ce))
    return ce.mean()


class SpatialReallocationStudentTDecoder(SpatialShapeScaleStudentTDecoder):
    def __init__(
        self,
        *args,
        realloc_k_plus: int = 3,
        realloc_k_minus: int = 3,
        realloc_budget_max: float = 0.80,
        realloc_init_budget: float = 0.20,
        alloc_temperature: float = 0.65,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.realloc_k_plus = realloc_k_plus
        self.realloc_k_minus = realloc_k_minus
        self.realloc_budget_max = realloc_budget_max
        self.alloc_temperature = alloc_temperature

        d_model = self.input_proj.out_features
        self.pos_head = nn.Linear(d_model, 1)
        self.neg_head = nn.Linear(d_model, 1)
        self.budget_head = nn.Linear(d_model, 1)

        nn.init.zeros_(self.pos_head.weight)
        nn.init.zeros_(self.pos_head.bias)
        nn.init.zeros_(self.neg_head.weight)
        nn.init.zeros_(self.neg_head.bias)
        nn.init.zeros_(self.budget_head.weight)
        nn.init.constant_(self.budget_head.bias, _scaled_logit(realloc_init_budget / realloc_budget_max))

    def forward(  # type: ignore[override]
        self, cond: torch.Tensor, prev_u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(prev_u.unsqueeze(-1))
        h = h + self.cond_proj(cond).unsqueeze(1)
        h = h + self.spatial_pos

        for li, layer in enumerate(self.layers):
            ls_a = self.ls_params[2 * li]
            ls_f = self.ls_params[2 * li + 1]
            h_norm = layer["attn_norm"](h)
            attn_out, _ = layer["attn"](h_norm, h_norm, h_norm)
            h = h + ls_a * attn_out
            h = h + ls_f * layer["ff"](layer["ff_norm"](h))

        h = self.out_norm(h)
        pooled = h.mean(dim=1)

        mu = prev_u + self.mean_head(h).squeeze(-1)
        factor = self.factor_head(h)
        diag = F.softplus(self.diag_head(h).squeeze(-1)) + self.diag_floor
        scale = F.softplus(self.scale_head(pooled).squeeze(-1)) + self.scale_floor

        if self.fixed_nu is None:
            nu = F.softplus(self.nu_head(pooled).squeeze(-1)) + self.nu_floor
            nu = torch.clamp(nu, max=self.nu_max)
        else:
            nu = self.fixed_nu_value.expand(h.shape[0]).to(h.dtype)

        pos_scores = self.pos_head(h).squeeze(-1)
        neg_scores = self.neg_head(h).squeeze(-1)
        budget = self.realloc_budget_max * torch.sigmoid(self.budget_head(pooled).squeeze(-1))
        return mu, factor, diag, scale, nu, pos_scores, neg_scores, budget


class ReallocationStudentTARModel(ShapeScaleStudentTARModel):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        reallocation_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        super().__init__(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
        )
        self.decoder = SpatialReallocationStudentTDecoder(**decoder_config, **reallocation_config)
        self.reallocation_config = reallocation_config

    def cond_from_gru_outputs(self, gru_outputs: torch.Tensor) -> torch.Tensor:
        attn_logits = self.encoder.attn_proj(gru_outputs).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
        return self.encoder.bottleneck(pooled)

    def base_covariance(
        self,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)
        cov = factor_norm @ factor_norm.transpose(-1, -2)
        cov = cov + torch.diag_embed(diag_norm.pow(2) + self.cov_jitter)
        cov = cov * scale.pow(2).unsqueeze(-1).unsqueeze(-1)
        return cov

    def adjusted_components(
        self,
        factor: torch.Tensor,
        diag: torch.Tensor,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        budget: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        factor_norm, diag_norm, avg_var = self.normalized_components(factor, diag)
        pos_alloc = topk_sparse_distribution(
            pos_scores,
            k=self.decoder.realloc_k_plus,
            temperature=self.decoder.alloc_temperature,
        )
        neg_alloc = topk_sparse_distribution(
            neg_scores,
            k=self.decoder.realloc_k_minus,
            temperature=self.decoder.alloc_temperature,
        )
        weights = 1.0 + budget.unsqueeze(-1) * (pos_alloc - neg_alloc)
        weight_sqrt = weights.clamp_min(1e-6).sqrt()
        factor_adj = factor_norm * weight_sqrt.unsqueeze(-1)
        diag_adj = diag_norm * weight_sqrt
        return factor_adj, diag_adj, avg_var, pos_alloc, neg_alloc, weights

    def covariance_from_adjusted(
        self,
        factor_adj: torch.Tensor,
        diag_adj: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        cov = factor_adj @ factor_adj.transpose(-1, -2)
        cov = cov + torch.diag_embed(diag_adj.pow(2) + self.cov_jitter)
        cov = cov * scale.pow(2).unsqueeze(-1).unsqueeze(-1)
        return cov

    def student_t_nll_from_cov(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        cov: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        d = target_u.shape[-1]
        chol = torch.linalg.cholesky(cov)
        diff = (target_u - mu).unsqueeze(-1)
        solved = torch.cholesky_solve(diff, chol).squeeze(-1)
        mahal = (diff.squeeze(-1) * solved).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)

        nu = nu.clamp_min(self.decoder.nu_floor + 1e-6)
        pi = target_u.new_tensor(math.pi)
        log_norm = (
            torch.lgamma((nu + d) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * (d * torch.log(nu * pi) + logdet)
        )
        log_kernel = -0.5 * (nu + d) * torch.log1p(mahal / nu)
        return -(log_norm + log_kernel)

    def forward_from_history(  # type: ignore[override]
        self, history_01: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.encode(history_01)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        return self.decoder(cond, prev_u)

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:  # type: ignore[override]
        mu, factor, diag, scale, nu, pos_scores, neg_scores, budget = self.forward_from_history(history_01)
        batch_size, n_cells = mu.shape
        rank = factor.shape[-1]
        factor_adj, diag_adj, _avg_var, _plus, _minus, _weights = self.adjusted_components(
            factor, diag, pos_scores, neg_scores, budget
        )

        eps_lowrank = torch.randn(batch_size, n_samples, rank, device=mu.device, dtype=mu.dtype)
        eps_diag = torch.randn(batch_size, n_samples, n_cells, device=mu.device, dtype=mu.dtype)
        lowrank_noise = torch.einsum("bcr,bnr->bnc", factor_adj, eps_lowrank)
        diag_noise = diag_adj.unsqueeze(1) * eps_diag

        gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
        mix = gamma.sample((n_samples,)).transpose(0, 1).to(mu.dtype).clamp_min(1e-6)
        t_scale = torch.rsqrt(mix).unsqueeze(-1)

        total_noise = (lowrank_noise + diag_noise) * scale.unsqueeze(1).unsqueeze(-1)
        return mu.unsqueeze(1) + total_noise * t_scale


def reallocation_multistep_loss(
    model: ReallocationStudentTARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    objective_config: dict,
    self_feed_prob: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, hist_len = history_01.shape[:2]
    future_len = future_01.shape[1]
    n_cells = future_01.shape[-1]

    hist_norm = normalize_iv(history_01).reshape(batch_size, hist_len, n_cells)
    gru_outputs, gru_state = model.encoder.gru(hist_norm)
    prev_01 = history_01[:, -1].reshape(batch_size, n_cells)

    total_nll = 0.0
    total_plus = 0.0
    total_minus = 0.0
    total_weight = 0.0
    total_sign = 0.0
    total_mae = 0.0
    total_rank = 0.0
    total_scale = 0.0
    total_budget = 0.0
    total_top1_plus = 0.0
    total_top1_minus = 0.0
    total_overlap = 0.0

    pred_mean_path = []
    target_mean_path = []

    for step in range(future_len):
        cond = model.cond_from_gru_outputs(gru_outputs)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        mu, factor, diag, scale, nu, pos_scores, neg_scores, budget = model.decoder(cond, prev_u)
        factor_adj, diag_adj, _avg_var, pos_alloc, neg_alloc, weights = model.adjusted_components(
            factor, diag, pos_scores, neg_scores, budget
        )
        cov = model.covariance_from_adjusted(factor_adj, diag_adj, scale)
        base_cov = model.base_covariance(factor, diag, scale)
        base_var = torch.diagonal(base_cov, dim1=-2, dim2=-1)

        target_t = future_01[:, step, :]
        target_u = iv_to_unconstrained(
            target_t,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )

        nll_t = model.student_t_nll_from_cov(target_u, mu, cov, nu)
        mu_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)

        target_plus, target_minus, target_w, _raw = build_target_reallocation(
            residual_sq=(target_u - mu).pow(2),
            base_var=base_var,
            k_plus=model.decoder.realloc_k_plus,
            k_minus=model.decoder.realloc_k_minus,
            target_clip=objective_config["target_clip"],
        )

        plus_loss = sparse_target_ce(pos_alloc, target_plus)
        minus_loss = sparse_target_ce(neg_alloc, target_minus)
        weight_loss = F.smooth_l1_loss(
            weights.clamp_min(1e-6).log(),
            target_w.clamp_min(1e-6).log(),
        )
        sign_loss = F.relu(-(weights - 1.0) * (target_w - 1.0)).mean()

        total_nll = total_nll + nll_t.mean()
        total_plus = total_plus + plus_loss
        total_minus = total_minus + minus_loss
        total_weight = total_weight + weight_loss
        total_sign = total_sign + sign_loss
        total_mae = total_mae + (mu_iv - target_t).abs().mean()
        total_rank = total_rank + effective_rank(cov).mean()
        total_scale = total_scale + scale.mean()
        total_budget = total_budget + budget.mean()
        total_top1_plus = total_top1_plus + pos_alloc.max(dim=-1).values.mean()
        total_top1_minus = total_top1_minus + neg_alloc.max(dim=-1).values.mean()
        total_overlap = total_overlap + 0.5 * (
            torch.minimum(pos_alloc, target_plus).sum(dim=-1).mean()
            + torch.minimum(neg_alloc, target_minus).sum(dim=-1).mean()
        )

        pred_mean_path.append(mu_iv.mean(dim=-1))
        target_mean_path.append(target_t.mean(dim=-1))

        use_pred = (
            self_feed_prob > 0.0
            and step < future_len - 1
            and torch.rand((), device=history_01.device).item() < self_feed_prob
        )
        next_frame = mu_iv.detach() if use_pred else target_t
        next_norm = normalize_iv(next_frame).unsqueeze(1)
        next_out, gru_state = model.encoder.gru(next_norm, gru_state)
        gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
        prev_01 = next_frame

    pred_mean_path_t = torch.stack(pred_mean_path, dim=1)
    target_mean_path_t = torch.stack(target_mean_path, dim=1)
    path_mean_loss = F.smooth_l1_loss(pred_mean_path_t, target_mean_path_t)

    norm = 1.0 / future_len
    total_loss = (
        total_nll * norm
        + objective_config["plus_weight"] * total_plus * norm
        + objective_config["minus_weight"] * total_minus * norm
        + objective_config["weight_weight"] * total_weight * norm
        + objective_config["sign_weight"] * total_sign * norm
        + objective_config["path_mean_weight"] * path_mean_loss
    )

    metrics = {
        "total_loss": total_loss,
        "multistep_nll": total_nll * norm,
        "realloc_plus_loss": total_plus * norm,
        "realloc_minus_loss": total_minus * norm,
        "realloc_weight_loss": total_weight * norm,
        "realloc_sign_loss": total_sign * norm,
        "path_mean_loss": path_mean_loss,
        "multistep_mae": total_mae * norm,
        "pred_eff_rank": total_rank * norm,
        "scale_mean": total_scale * norm,
        "budget_mean": total_budget * norm,
        "alloc_top1_plus": total_top1_plus * norm,
        "alloc_top1_minus": total_top1_minus * norm,
        "alloc_overlap": total_overlap * norm,
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: ReallocationStudentTARModel,
    val_loader: DataLoader,
    objective_config: dict,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for history_01, future_01 in val_loader:
        loss, metrics = reallocation_multistep_loss(
            model,
            history_01,
            future_01,
            objective_config=objective_config,
            self_feed_prob=0.0,
        )
        batch_size = history_01.shape[0]
        total_count += batch_size
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v.item()) * batch_size
    if total_count == 0:
        return {f"val_{k}": float("nan") for k in [
            "total_loss", "multistep_nll", "realloc_plus_loss", "realloc_minus_loss",
            "realloc_weight_loss", "realloc_sign_loss", "path_mean_loss", "multistep_mae",
            "pred_eff_rank", "scale_mean", "budget_mean", "alloc_top1_plus",
            "alloc_top1_minus", "alloc_overlap",
        ]}
    return {f"val_{k}": v / total_count for k, v in totals.items()}


def maybe_load_warm_start(model: ReallocationStudentTARModel, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    current = model.state_dict()
    loadable = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    missing = sorted(set(current.keys()) - set(loadable.keys()))
    skipped = sorted(set(state.keys()) - set(loadable.keys()))
    model.load_state_dict(loadable, strict=False)
    print(f"Warm start loaded from {checkpoint_path}")
    print(f"  matched keys: {len(loadable)}")
    print(f"  missing new keys: {len(missing)}")
    print(f"  skipped old keys: {len(skipped)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="191a: AR Student-t with exact source/sink variance reallocation")
    parser.add_argument("--stage1_epochs", type=int, default=3)
    parser.add_argument("--stage2_epochs", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_stage1", type=float, default=2e-4)
    parser.add_argument("--lr_stage2", type=float, default=7e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.10)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--nu_floor", type=float, default=2.1)
    parser.add_argument("--nu_init", type=float, default=8.0)
    parser.add_argument("--nu_max", type=float, default=100.0)
    parser.add_argument("--fixed_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--realloc_k_plus", type=int, default=3)
    parser.add_argument("--realloc_k_minus", type=int, default=3)
    parser.add_argument("--realloc_budget_max", type=float, default=0.80)
    parser.add_argument("--realloc_init_budget", type=float, default=0.22)
    parser.add_argument("--alloc_temperature", type=float, default=0.65)
    parser.add_argument("--target_clip", type=float, default=1.0)
    parser.add_argument("--plus_weight", type=float, default=0.08)
    parser.add_argument("--minus_weight", type=float, default=0.08)
    parser.add_argument("--weight_weight", type=float, default=0.16)
    parser.add_argument("--sign_weight", type=float, default=0.08)
    parser.add_argument("--path_mean_weight", type=float, default=0.20)
    parser.add_argument("--self_feed_max", type=float, default=0.35)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--warm_start", type=str, default="models/backfill/student_t_169c/best_model.pt")
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
        train_indices = train_indices[:args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[:args.max_val_windows]

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
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=128,
        rank=args.rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        init_nu=args.nu_init,
        fixed_nu=args.fixed_nu,
    )
    realloc_config = dict(
        realloc_k_plus=args.realloc_k_plus,
        realloc_k_minus=args.realloc_k_minus,
        realloc_budget_max=args.realloc_budget_max,
        realloc_init_budget=args.realloc_init_budget,
        alloc_temperature=args.alloc_temperature,
    )
    objective_config = dict(
        plus_weight=args.plus_weight,
        minus_weight=args.minus_weight,
        weight_weight=args.weight_weight,
        sign_weight=args.sign_weight,
        path_mean_weight=args.path_mean_weight,
        target_clip=args.target_clip,
    )

    model = ReallocationStudentTARModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        reallocation_config=realloc_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    maybe_load_warm_start(model, args.warm_start)

    print("=" * 72)
    print("191a: AR Student-t with exact source/sink local variance reallocation")
    print("=" * 72)
    print(f"Train: {len(train_indices)} | Val: {len(val_indices)}")
    print(f"Stage1 epochs: {args.stage1_epochs} | Stage2 epochs: {args.stage2_epochs}")
    print(f"Exact top-k realloc: k+={args.realloc_k_plus}, k-={args.realloc_k_minus}")
    print(f"Budget max: {args.realloc_budget_max:.3f} | self-feed max: {args.self_feed_max:.2f}")

    realloc_params = (
        list(model.decoder.pos_head.parameters())
        + list(model.decoder.neg_head.parameters())
        + list(model.decoder.budget_head.parameters())
    )

    best_score = float("inf")
    best_metrics = None
    history = []

    total_epochs = args.stage1_epochs + args.stage2_epochs
    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        if epoch <= args.stage1_epochs:
            stage = 1
            model.encoder.requires_grad_(False)
            for p in model.decoder.parameters():
                p.requires_grad_(False)
            for p in realloc_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW(realloc_params, lr=args.lr_stage1, weight_decay=args.weight_decay)
            self_feed_prob = 0.0
        else:
            stage = 2
            model.requires_grad_(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=args.weight_decay)
            stage2_idx = epoch - args.stage1_epochs
            stage2_den = max(args.stage2_epochs - 1, 1)
            self_feed_prob = args.self_feed_max * (stage2_idx - 1) / stage2_den

        model.train()
        train_sums: dict[str, float] = {}
        nb = 0
        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = reallocation_multistep_loss(
                model,
                history_01,
                future_01,
                objective_config=objective_config,
                self_feed_prob=self_feed_prob,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k, v in metrics.items():
                train_sums[k] = train_sums.get(k, 0.0) + float(v.item())
            nb += 1

        train_metrics = {f"train_{k}": v / max(nb, 1) for k, v in train_sums.items()}
        val_metrics = evaluate_teacher_forced(model, val_loader, objective_config)
        rollout_metrics = evaluate_rollout_subset(
            model,
            val_loader,
            rollout_val_samples=args.rollout_val_samples,
            rollout_eval_limit=args.rollout_eval_limit,
        )
        elapsed = time.time() - t0

        score = (
            val_metrics["val_total_loss"]
            + 0.25 * abs(rollout_metrics["rollout_cov90"] - 0.90)
            + 0.15 * max(0.0, 1.15 - rollout_metrics["rollout_turb_calm_ratio"])
            + 0.10 * rollout_metrics["rollout_support_violation_rate"]
        )
        is_best = score < best_score
        if is_best:
            best_score = score
            best_metrics = {**val_metrics, **rollout_metrics, "score": score}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_total_loss": val_metrics["val_total_loss"],
                    "score": score,
                    "config": {
                        "type": "constrained_reallocation_multistep_student_t_191a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "reallocation": realloc_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                    },
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        row = {
            "epoch": epoch,
            "stage": stage,
            "self_feed_prob": self_feed_prob,
            **train_metrics,
            **val_metrics,
            **rollout_metrics,
            "selection_score": score,
        }
        history.append(row)
        print(
            f"Ep {epoch:3d}  stg={stage}  "
            f"train={train_metrics['train_total_loss']:.4f}  "
            f"val={val_metrics['val_total_loss']:.4f}  "
            f"roll_cov90={rollout_metrics['rollout_cov90']:.4f}  "
            f"roll_tc={rollout_metrics['rollout_turb_calm_ratio']:.3f}  "
            f"b={val_metrics['val_budget_mean']:.3f}  "
            f"pTop={val_metrics['val_alloc_top1_plus']:.3f}  "
            f"nTop={val_metrics['val_alloc_top1_minus']:.3f}  "
            f"ovlp={val_metrics['val_alloc_overlap']:.3f}  "
            f"sf={self_feed_prob:.2f}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "val_total_loss": history[-1]["val_total_loss"] if history else float("nan"),
        "score": history[-1]["selection_score"] if history else float("nan"),
        "config": {
            "type": "constrained_reallocation_multistep_student_t_191a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "reallocation": realloc_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
        },
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)

    if best_metrics is not None:
        print(
            f"Best score: {best_score:.4f} | "
            f"roll_cov90={best_metrics['rollout_cov90']:.4f} | "
            f"roll_tc={best_metrics['rollout_turb_calm_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
