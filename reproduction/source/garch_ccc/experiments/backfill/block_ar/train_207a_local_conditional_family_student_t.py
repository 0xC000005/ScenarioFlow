#!/usr/bin/env python
"""
207a: 201b AR training geometry with a calm base path plus local conditional family.

Keep:
  - Transformer history encoder
  - mean / covariance trunk
  - underfit-aware teacher-forced objective
  - self-fed rollout likelihood objective

Change only:
  - replace the single broad innovation law with
    base Student-t + local conditional scenario family
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    inverse_softplus,
    make_serializable,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    SpatialShapeScaleStudentTDecoder,
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import (
    TransformerRolloutTailStudentTARModel,
    evaluate_rollout_subset,
    maybe_load_decoder_warm_start,
)
from experiments.backfill.block_ar.train_201b_underfit_aware_selffed_rollout_student_t import (
    evaluate_selffed_rollout,
    evaluate_teacher_forced,
    selffed_rollout_likelihood_objective,
    teacher_forced_multistep_objective,
)


class LocalConditionalFamilyStudentTDecoder(SpatialShapeScaleStudentTDecoder):
    def __init__(
        self,
        *args,
        n_family: int = 3,
        family_scale_floor: float = 5e-4,
        init_gate_prob: float = 0.10,
        init_family_scale: float = 0.05,
        family_shape_scale: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_family = n_family
        self.family_scale_floor = family_scale_floor

        d_model = self.mean_head.in_features
        self.gate_logit_head = nn.Linear(d_model, 1)
        self.family_logit_head = nn.Linear(d_model, n_family)
        self.family_scale_head = nn.Linear(d_model, 1)
        self.family_shape_head = nn.Linear(d_model, n_family * self.n_cells)

        nn.init.zeros_(self.gate_logit_head.weight)
        nn.init.constant_(self.gate_logit_head.bias, float(np.log(init_gate_prob / (1.0 - init_gate_prob))))

        nn.init.zeros_(self.family_logit_head.weight)
        nn.init.zeros_(self.family_logit_head.bias)

        nn.init.zeros_(self.family_scale_head.weight)
        nn.init.constant_(
            self.family_scale_head.bias,
            inverse_softplus(max(init_family_scale - self.family_scale_floor, 1e-6)),
        )
        nn.init.normal_(self.family_shape_head.weight, mean=0.0, std=family_shape_scale)
        nn.init.zeros_(self.family_shape_head.bias)

    def normalize_family_shapes(self, family_shapes: torch.Tensor) -> torch.Tensor:
        family_shapes = torch.tanh(family_shapes)
        return family_shapes / family_shapes.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(
        self, cond: torch.Tensor, prev_u: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        gate_logit = self.gate_logit_head(pooled).squeeze(-1)
        family_logits = self.family_logit_head(pooled)
        family_scale = F.softplus(self.family_scale_head(pooled).squeeze(-1)) + self.family_scale_floor
        family_shapes = self.family_shape_head(pooled).view(pooled.shape[0], self.n_family, self.n_cells)
        family_shapes = self.normalize_family_shapes(family_shapes)

        if self.fixed_nu is None:
            nu = F.softplus(self.nu_head(pooled).squeeze(-1)) + self.nu_floor
            nu = torch.clamp(nu, max=self.nu_max)
        else:
            nu = self.fixed_nu_value.expand(h.shape[0]).to(h.dtype)
        return mu, factor, diag, scale, nu, gate_logit, family_logits, family_scale, family_shapes


class TransformerLocalConditionalFamilyARModel(TransformerRolloutTailStudentTARModel):
    def __init__(
        self,
        encoder_config: dict[str, Any],
        decoder_config: dict[str, Any],
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        family_keys = {
            "n_family",
            "family_scale_floor",
            "init_gate_prob",
            "init_family_scale",
            "family_shape_scale",
            "family_nu",
            "gate_temperature",
            "family_temperature",
            "gate_prob_eps",
        }
        base_decoder_config = {k: v for k, v in decoder_config.items() if k not in family_keys}
        super().__init__(
            encoder_config=encoder_config,
            decoder_config=base_decoder_config,
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
        )

        self.decoder = LocalConditionalFamilyStudentTDecoder(
            **base_decoder_config,
            n_family=decoder_config.get("n_family", 3),
            family_scale_floor=decoder_config.get("family_scale_floor", 5e-4),
            init_gate_prob=decoder_config.get("init_gate_prob", 0.10),
            init_family_scale=decoder_config.get("init_family_scale", 0.05),
            family_shape_scale=decoder_config.get("family_shape_scale", 0.05),
        )
        self.decoder_config = decoder_config
        self.family_nu = float(decoder_config.get("family_nu", 4.0))
        self.gate_temperature = float(decoder_config.get("gate_temperature", 0.5))
        self.family_temperature = float(decoder_config.get("family_temperature", 0.6))
        self.gate_prob_eps = float(decoder_config.get("gate_prob_eps", 1e-4))
        self._cached_family: dict[str, torch.Tensor] | None = None

    def forward_from_history(
        self, history_01: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        cond, attn = self.encode(history_01, return_attention=True)
        history_flat = history_01.reshape(history_01.shape[0], history_01.shape[1], -1)
        prev_01 = history_flat[:, -1]
        from experiments.backfill.block_ar.train_169a_transformed_student_t import iv_to_unconstrained

        prev_u = iv_to_unconstrained(
            prev_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        mu, factor, diag, scale, nu, gate_logit, family_logits, family_scale, family_shapes = self.decoder(cond, prev_u)
        self._cached_family = {
            "gate_logit": gate_logit,
            "family_logits": family_logits,
            "family_scale": family_scale,
            "family_shapes": family_shapes,
        }
        if return_attention:
            return mu, factor, diag, scale, nu, attn
        return mu, factor, diag, scale, nu

    def last_family_params(self) -> dict[str, torch.Tensor]:
        if self._cached_family is None:
            raise RuntimeError("Family params requested before forward_from_history")
        return self._cached_family

    def family_statistics_from_history(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        _ = self.forward_from_history(history_01)
        family = self.last_family_params()
        gate_prob = torch.sigmoid(family["gate_logit"]).clamp(self.gate_prob_eps, 1.0 - self.gate_prob_eps)
        family_probs = torch.softmax(family["family_logits"], dim=-1)
        family_entropy = -(family_probs * torch.log(family_probs.clamp_min(1e-8))).sum(dim=-1)
        top_cells = family["family_shapes"].abs().argmax(dim=-1)
        union_sizes = []
        pairwise_cos = []
        for b in range(top_cells.shape[0]):
            union_sizes.append(float(torch.unique(top_cells[b]).numel()))
            fam = family["family_shapes"][b]
            fam = fam / fam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            sims = fam @ fam.T
            mask = ~torch.eye(sims.shape[0], dtype=torch.bool, device=sims.device)
            pairwise_cos.append(float(sims[mask].mean()) if mask.any() else 1.0)
        return {
            "gate_prob_mean": gate_prob.mean(),
            "gate_prob_min": gate_prob.min(),
            "gate_prob_max": gate_prob.max(),
            "family_top1_mean": family_probs.max(dim=-1).values.mean(),
            "family_entropy_mean": family_entropy.mean(),
            "family_scale_mean": family["family_scale"].mean(),
            "family_union_size_mean": gate_prob.new_tensor(union_sizes).mean(),
            "family_pairwise_cos_mean": gate_prob.new_tensor(pairwise_cos).mean(),
        }

    def _student_t_log_prob(
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
        return log_norm + log_kernel

    def _student_t_log_prob_components(
        self,
        target_u: torch.Tensor,
        mu_components: torch.Tensor,
        cov: torch.Tensor,
        nu_components: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_components, d = mu_components.shape
        cov_components = cov[:, None, :, :].expand(batch_size, n_components, d, d).reshape(batch_size * n_components, d, d)
        chol = torch.linalg.cholesky(cov_components)
        diff = (target_u[:, None, :] - mu_components).reshape(batch_size * n_components, d, 1)
        solved = torch.cholesky_solve(diff, chol).reshape(batch_size, n_components, d)
        mahal = ((target_u[:, None, :] - mu_components) * solved).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1).reshape(batch_size, n_components)

        nu = nu_components.clamp_min(self.decoder.nu_floor + 1e-6)
        pi = target_u.new_tensor(math.pi)
        log_norm = (
            torch.lgamma((nu + d) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * (d * torch.log(nu * pi) + logdet)
        )
        log_kernel = -0.5 * (nu + d) * torch.log1p(mahal / nu)
        return log_norm + log_kernel

    def student_t_nll(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        family = self.last_family_params()
        cov = self.covariance(factor, diag, scale)
        base_logp = self._student_t_log_prob(target_u, mu, cov, nu)

        gate_prob = torch.sigmoid(family["gate_logit"]).clamp(self.gate_prob_eps, 1.0 - self.gate_prob_eps)
        family_probs = torch.softmax(family["family_logits"], dim=-1).clamp_min(1e-8)
        family_mu = mu[:, None, :] + family["family_scale"][:, None, None] * family["family_shapes"].to(dtype=mu.dtype)
        family_nu = target_u.new_full((target_u.shape[0], family_mu.shape[1]), self.family_nu)
        family_logp = self._student_t_log_prob_components(target_u, family_mu, cov, family_nu)

        base_logw = torch.log1p(-gate_prob).unsqueeze(-1)
        family_logw = torch.log(gate_prob).unsqueeze(-1) + torch.log(family_probs)
        total_logp = torch.logsumexp(
            torch.cat([base_logw + base_logp.unsqueeze(-1), family_logw + family_logp], dim=-1),
            dim=-1,
        )
        return -total_logp

    def _sample_base_u_from_params(
        self,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
        n_samples: int | None = None,
    ) -> torch.Tensor:
        batch_size, n_cells = mu.shape
        rank = factor.shape[-1]
        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)

        if n_samples is None:
            eps_lowrank = torch.randn(batch_size, rank, device=mu.device, dtype=mu.dtype)
            eps_diag = torch.randn(batch_size, n_cells, device=mu.device, dtype=mu.dtype)
            lowrank_noise = torch.einsum("bcr,br->bc", factor_norm, eps_lowrank)
            diag_noise = diag_norm * eps_diag
            gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
            mix = gamma.rsample().clamp_min(1e-6)
            t_scale = torch.rsqrt(mix).unsqueeze(-1)
            total_noise = (lowrank_noise + diag_noise) * scale.unsqueeze(-1)
            return mu + total_noise * t_scale

        eps_lowrank = torch.randn(batch_size, n_samples, rank, device=mu.device, dtype=mu.dtype)
        eps_diag = torch.randn(batch_size, n_samples, n_cells, device=mu.device, dtype=mu.dtype)
        lowrank_noise = torch.einsum("bcr,bnr->bnc", factor_norm, eps_lowrank)
        diag_noise = diag_norm.unsqueeze(1) * eps_diag
        gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
        mix = gamma.rsample((n_samples,)).transpose(0, 1).to(mu.dtype).clamp_min(1e-6)
        t_scale = torch.rsqrt(mix).unsqueeze(-1)
        total_noise = (lowrank_noise + diag_noise) * scale.unsqueeze(1).unsqueeze(-1)
        return mu.unsqueeze(1) + total_noise * t_scale

    def reparameterized_next_u(
        self,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        family = self.last_family_params()
        base_sample = self._sample_base_u_from_params(mu, factor, diag, scale, nu, n_samples=None)

        gate_prob = torch.sigmoid(family["gate_logit"]).clamp(self.gate_prob_eps, 1.0 - self.gate_prob_eps)
        gate = RelaxedBernoulli(
            temperature=gate_prob.new_tensor(self.gate_temperature),
            probs=gate_prob,
        ).rsample()
        family_weights = F.gumbel_softmax(family["family_logits"], tau=self.family_temperature, hard=False)
        family_shape = torch.einsum("bk,bkc->bc", family_weights, family["family_shapes"].to(dtype=mu.dtype))
        return base_sample + gate.unsqueeze(-1) * family["family_scale"].unsqueeze(-1) * family_shape

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mu, factor, diag, scale, nu = self.forward_from_history(history_01)
        family = self.last_family_params()
        base_samples = self._sample_base_u_from_params(mu, factor, diag, scale, nu, n_samples=n_samples)

        gate_prob = torch.sigmoid(family["gate_logit"]).clamp(self.gate_prob_eps, 1.0 - self.gate_prob_eps)
        gate = (torch.rand(mu.shape[0], n_samples, device=mu.device, dtype=mu.dtype) < gate_prob.unsqueeze(1)).to(mu.dtype)
        family_probs = torch.softmax(family["family_logits"], dim=-1)
        family_idx = torch.multinomial(family_probs, num_samples=n_samples, replacement=True)
        batch_index = torch.arange(mu.shape[0], device=mu.device)[:, None]
        chosen_shapes = family["family_shapes"][batch_index, family_idx]
        return base_samples + gate.unsqueeze(-1) * family["family_scale"].unsqueeze(1).unsqueeze(-1) * chosen_shapes


@torch.no_grad()
def evaluate_family_statistics(
    model: TransformerLocalConditionalFamilyARModel,
    val_loader: DataLoader,
) -> dict[str, float]:
    model.eval()
    totals = {
        "val_gate_prob_mean": 0.0,
        "val_gate_prob_min": 0.0,
        "val_gate_prob_max": 0.0,
        "val_family_top1_mean": 0.0,
        "val_family_entropy_mean": 0.0,
        "val_family_scale_mean": 0.0,
        "val_family_union_size_mean": 0.0,
        "val_family_pairwise_cos_mean": 0.0,
    }
    total_count = 0
    for history_01, _future_01 in val_loader:
        stats = model.family_statistics_from_history(history_01)
        bs = history_01.shape[0]
        totals["val_gate_prob_mean"] += stats["gate_prob_mean"].item() * bs
        totals["val_gate_prob_min"] += stats["gate_prob_min"].item() * bs
        totals["val_gate_prob_max"] += stats["gate_prob_max"].item() * bs
        totals["val_family_top1_mean"] += stats["family_top1_mean"].item() * bs
        totals["val_family_entropy_mean"] += stats["family_entropy_mean"].item() * bs
        totals["val_family_scale_mean"] += stats["family_scale_mean"].item() * bs
        totals["val_family_union_size_mean"] += stats["family_union_size_mean"].item() * bs
        totals["val_family_pairwise_cos_mean"] += stats["family_pairwise_cos_mean"].item() * bs
        total_count += bs
    return {k: v / max(total_count, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="207a local conditional family Student-t AR")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=3e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--enc_d_model", type=int, default=128)
    parser.add_argument("--enc_heads", type=int, default=4)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--enc_dropout", type=float, default=0.1)
    parser.add_argument("--dec_d_model", type=int, default=128)
    parser.add_argument("--dec_heads", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=4)
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
    parser.add_argument("--tail_weight", type=float, default=6.0)
    parser.add_argument("--tail_mae_weight", type=float, default=0.05)
    parser.add_argument("--underfit_z_gate", type=float, default=1.8)
    parser.add_argument("--overwidth_weight", type=float, default=0.02)
    parser.add_argument("--rollout_weight", type=float, default=0.25)
    parser.add_argument("--rollout_steps", type=int, default=5)
    parser.add_argument("--rollout_warmup_epochs", type=int, default=3)
    parser.add_argument("--rollout_ramp_epochs", type=int, default=3)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--decoder_warm_start", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument("--n_family", type=int, default=3)
    parser.add_argument("--family_scale_floor", type=float, default=5e-4)
    parser.add_argument("--init_gate_prob", type=float, default=0.10)
    parser.add_argument("--init_family_scale", type=float, default=0.05)
    parser.add_argument("--family_shape_scale", type=float, default=0.05)
    parser.add_argument("--family_nu", type=float, default=4.0)
    parser.add_argument("--gate_temperature", type=float, default=0.5)
    parser.add_argument("--family_temperature", type=float, default=0.6)
    parser.add_argument("--gate_prob_eps", type=float, default=1e-4)
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
    train_abs_delta = np.abs(np.diff(surfaces[:4511], axis=0).reshape(-1))
    q95_threshold = float(np.quantile(train_abs_delta, 0.95))
    q99_threshold = float(np.quantile(train_abs_delta, 0.99))

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

    encoder_config = dict(
        input_dim=25,
        d_model=args.enc_d_model,
        n_heads=args.enc_heads,
        n_layers=args.enc_layers,
        dropout=args.enc_dropout,
        bottleneck_dim=128,
        max_len=max(hist_len, 64),
    )
    decoder_config = dict(
        n_cells=25,
        d_model=args.dec_d_model,
        n_heads=args.dec_heads,
        n_layers=args.dec_layers,
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
        n_family=args.n_family,
        family_scale_floor=args.family_scale_floor,
        init_gate_prob=args.init_gate_prob,
        init_family_scale=args.init_family_scale,
        family_shape_scale=args.family_shape_scale,
        family_nu=args.family_nu,
        gate_temperature=args.gate_temperature,
        family_temperature=args.family_temperature,
        gate_prob_eps=args.gate_prob_eps,
    )
    objective_config = dict(
        q95_threshold=q95_threshold,
        q99_threshold=q99_threshold,
        tail_weight=args.tail_weight,
        tail_mae_weight=args.tail_mae_weight,
        underfit_z_gate=args.underfit_z_gate,
        overwidth_weight=args.overwidth_weight,
        rollout_steps=args.rollout_steps,
    )

    model = TransformerLocalConditionalFamilyARModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    maybe_load_decoder_warm_start(model, args.decoder_warm_start)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n{'=' * 76}")
    print("207a: Transformer AR Student-t with local conditional scenario family")
    print(f"{'=' * 76}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Total params:   {n_enc + n_dec:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Tail thresholds: q95={q95_threshold:.5f}, q99={q99_threshold:.5f}")
    print(f"  Underfit z-gate: {args.underfit_z_gate:.3f}")
    print(f"  Overwidth weight: {args.overwidth_weight}")
    print(f"  Self-fed rollout steps: {args.rollout_steps}, weight={args.rollout_weight}")
    print(f"  Local family size: {args.n_family} | init gate prob: {args.init_gate_prob:.3f} | init family scale: {args.init_family_scale:.4f}")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": args.lr_encoder,
                "weight_decay": args.weight_decay_encoder,
            },
            {
                "params": model.decoder.parameters(),
                "lr": args.lr_decoder,
                "weight_decay": args.weight_decay_decoder,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep = {
            "teacher_total_loss": 0.0,
            "multistep_nll": 0.0,
            "multistep_mae": 0.0,
            "overwidth_penalty": 0.0,
            "severity_mean": 0.0,
            "underfit_mean": 0.0,
            "step_weight_mean": 0.0,
            "scale_mean": 0.0,
            "shape_avg_var": 0.0,
            "nu_mean": 0.0,
            "attention_top1": 0.0,
            "rollout_total_loss": 0.0,
            "rollout_nll": 0.0,
            "rollout_mae": 0.0,
            "rollout_overwidth_penalty": 0.0,
            "rollout_severity_mean": 0.0,
            "rollout_underfit_mean": 0.0,
            "rollout_step_weight_mean": 0.0,
        }
        nb = 0

        if epoch <= args.rollout_warmup_epochs:
            rollout_scale = 0.0
        else:
            progress = (epoch - args.rollout_warmup_epochs) / max(args.rollout_ramp_epochs, 1)
            rollout_scale = float(min(max(progress, 0.0), 1.0))

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            teacher_loss, teacher_metrics = teacher_forced_multistep_objective(model, history_01, future_01, objective_config)
            loss = teacher_loss
            rollout_metrics = {
                "rollout_total_loss": torch.tensor(0.0, device=history_01.device),
                "rollout_nll": torch.tensor(0.0, device=history_01.device),
                "rollout_mae": torch.tensor(0.0, device=history_01.device),
                "rollout_overwidth_penalty": torch.tensor(0.0, device=history_01.device),
                "rollout_severity_mean": torch.tensor(0.0, device=history_01.device),
                "rollout_underfit_mean": torch.tensor(0.0, device=history_01.device),
                "rollout_step_weight_mean": torch.tensor(0.0, device=history_01.device),
            }
            if rollout_scale > 0.0 and args.rollout_weight > 0.0 and args.rollout_steps > 0:
                rollout_loss, rollout_metrics = selffed_rollout_likelihood_objective(model, history_01, future_01, objective_config)
                loss = loss + (args.rollout_weight * rollout_scale) * rollout_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in ep:
                if k in teacher_metrics:
                    ep[k] += teacher_metrics[k].item()
                elif k in rollout_metrics:
                    ep[k] += rollout_metrics[k].item()
            nb += 1

        scheduler.step()

        train_metrics = {f"train_{k}": v / max(nb, 1) for k, v in ep.items()}
        train_metrics["train_rollout_scale"] = rollout_scale

        val_teacher = evaluate_teacher_forced(model, val_loader, objective_config)
        val_rollout = evaluate_selffed_rollout(model, val_loader, objective_config)
        rollout_diag = evaluate_rollout_subset(
            model,
            val_loader,
            rollout_val_samples=args.rollout_val_samples,
            rollout_eval_limit=args.rollout_eval_limit,
        )
        event_diag = evaluate_family_statistics(model, val_loader)

        selection_score = val_teacher["val_teacher_total_loss"] + args.rollout_weight * val_rollout["val_rollout_total_loss"]

        elapsed = time.time() - t0
        is_best = selection_score < best_score
        if is_best:
            best_score = selection_score
            best_metrics = {
                **val_teacher,
                **val_rollout,
                **rollout_diag,
                **event_diag,
                "selection_score": selection_score,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_score": best_score,
                    "config": {
                        "type": "transformer_local_conditional_family_student_t_207a",
                        "encoder": encoder_config,
                        "decoder": decoder_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "cov_jitter": args.cov_jitter,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "objective": objective_config,
                    },
                    "metrics": best_metrics,
                },
                Path(args.output_dir) / "best_model.pt",
            )

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_teacher,
            **val_rollout,
            **rollout_diag,
            **event_diag,
            "selection_score": selection_score,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        with open(Path(args.output_dir) / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        print(
            f"Ep {epoch:>3d}  "
            f"train_tf={train_metrics['train_teacher_total_loss']:.4f}  "
            f"train_ro={train_metrics['train_rollout_total_loss']:.4f}  "
            f"val_tf={val_teacher['val_teacher_total_loss']:.4f}  "
            f"val_ro={val_rollout['val_rollout_total_loss']:.4f}  "
            f"roll_cov90={rollout_diag['rollout_cov90']:.4f}  "
            f"roll_mae={rollout_diag['rollout_mae']:.4f}  "
            f"roll_tc={rollout_diag['rollout_turb_calm_ratio']:.3f}  "
            f"rank={rollout_diag['rollout_rank_ratio_h30']:.2f}  "
            f"gate={event_diag['val_gate_prob_mean']:.3f}  "
            f"fam_top1={event_diag['val_family_top1_mean']:.3f}  "
            f"fam_scale={event_diag['val_family_scale_mean']:.4f}  "
            f"rs={rollout_scale:.2f}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "selection_score": best_score,
            "config": {
                "type": "transformer_local_conditional_family_student_t_207a",
                "encoder": encoder_config,
                "decoder": decoder_config,
                "support_lo": args.support_lo,
                "support_hi": args.support_hi,
                "support_eps": args.support_eps,
                "cov_jitter": args.cov_jitter,
                "history_len": hist_len,
                "future_len": future_len,
                "objective": objective_config,
            },
            "metrics": best_metrics,
        },
        Path(args.output_dir) / "final_model.pt",
    )


if __name__ == "__main__":
    main()
