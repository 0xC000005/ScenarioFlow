#!/usr/bin/env python
"""
172a: Conditional residual flow on top of the 170d structured joint base.

Keep 170d:
  - support-aware transformed-space density modeling
  - one-shot future block generation
  - proper joint likelihood
  - structured separable time/cell covariance

Change:
  - whiten residuals with the predicted 170d-style mean/covariance
  - model the whitened residual law with a conditional affine-coupling flow
  - use a heavy-tailed i.i.d. base distribution in flow space
"""

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

from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    effective_rank,
    inverse_softplus,
    iv_to_unconstrained,
    make_serializable,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_170b_whitened_flow import (
    ConditionalResidualFlow,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
    eff_rank_np,
)


class ResidualFlowStructuredJointDecoder(nn.Module):
    """Structured joint future decoder plus flow context head."""

    def __init__(
        self,
        n_frames: int = 30,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 128,
        time_rank: int = 6,
        cell_rank: int = 5,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        init_diag: float = 0.05,
        init_scale: float = 0.10,
        flow_context_dim: int = 256,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.n_cells = n_cells
        self.time_rank = time_rank
        self.cell_rank = cell_rank
        self.diag_floor = diag_floor
        self.scale_floor = scale_floor

        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, d_model) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_cells, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "temp_norm": nn.LayerNorm(d_model),
                        "temp_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                        "temp_ff_norm": nn.LayerNorm(d_model),
                        "temp_ff": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Linear(d_model * 4, d_model),
                        ),
                        "spat_norm": nn.LayerNorm(d_model),
                        "spat_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                        "spat_ff_norm": nn.LayerNorm(d_model),
                        "spat_ff": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Linear(d_model * 4, d_model),
                        ),
                    }
                )
            )

        self.out_norm = nn.LayerNorm(d_model)
        self.mean_head = nn.Linear(d_model, 1)
        self.time_factor_head = nn.Linear(d_model, time_rank)
        self.time_diag_head = nn.Linear(d_model, 1)
        self.cell_factor_head = nn.Linear(d_model, cell_rank)
        self.cell_diag_head = nn.Linear(d_model, 1)
        self.scale_head = nn.Linear(d_model, 1)
        self.flow_context_head = nn.Sequential(
            nn.Linear(cond_dim + n_cells, flow_context_dim),
            nn.SiLU(),
            nn.Linear(flow_context_dim, flow_context_dim),
        )

        self._init_parameters(init_diag=init_diag, init_scale=init_scale)

    def _init_parameters(self, init_diag: float, init_scale: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(
                    head_name in name
                    for head_name in (
                        "mean_head",
                        "time_factor_head",
                        "time_diag_head",
                        "cell_factor_head",
                        "cell_diag_head",
                        "scale_head",
                        "flow_context_head",
                    )
                ):
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

        nn.init.normal_(self.time_factor_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.time_factor_head.bias, mean=0.0, std=1e-3)
        nn.init.normal_(self.cell_factor_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.cell_factor_head.bias, mean=0.0, std=1e-3)

        nn.init.zeros_(self.time_diag_head.weight)
        nn.init.constant_(
            self.time_diag_head.bias,
            inverse_softplus(max(init_diag - self.diag_floor, 1e-6)),
        )
        nn.init.zeros_(self.cell_diag_head.weight)
        nn.init.constant_(
            self.cell_diag_head.bias,
            inverse_softplus(max(init_diag - self.diag_floor, 1e-6)),
        )
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(
            self.scale_head.bias,
            inverse_softplus(max(init_scale - self.scale_floor, 1e-6)),
        )

        for module in self.flow_context_head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(
        self,
        cond: torch.Tensor,
        prev_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = prev_u.shape[0]
        n_frames, n_cells = self.n_frames, self.n_cells

        prev_rep = prev_u.unsqueeze(1).expand(batch, n_frames, n_cells)
        hidden = self.input_proj(prev_rep.unsqueeze(-1))
        hidden = hidden + self.cond_proj(cond).unsqueeze(1).unsqueeze(1)
        hidden = hidden + self.temporal_pos + self.spatial_pos

        for layer in self.layers:
            h_temp = hidden.permute(0, 2, 1, 3).reshape(batch * n_cells, n_frames, -1)
            h_norm = layer["temp_norm"](h_temp)
            attn_out, _ = layer["temp_attn"](h_norm, h_norm, h_norm)
            h_temp = h_temp + attn_out
            h_temp = h_temp + layer["temp_ff"](layer["temp_ff_norm"](h_temp))
            hidden = h_temp.reshape(batch, n_cells, n_frames, -1).permute(0, 2, 1, 3)

            h_spat = hidden.reshape(batch * n_frames, n_cells, -1)
            h_norm = layer["spat_norm"](h_spat)
            attn_out, _ = layer["spat_attn"](h_norm, h_norm, h_norm)
            h_spat = h_spat + attn_out
            h_spat = h_spat + layer["spat_ff"](layer["spat_ff_norm"](h_spat))
            hidden = h_spat.reshape(batch, n_frames, n_cells, -1)

        hidden = self.out_norm(hidden)
        pooled = hidden.mean(dim=(1, 2))
        time_summary = hidden.mean(dim=2)
        cell_summary = hidden.mean(dim=1)

        mu = prev_rep + self.mean_head(hidden).squeeze(-1)
        time_factor = self.time_factor_head(time_summary)
        time_diag = F.softplus(self.time_diag_head(time_summary).squeeze(-1)) + self.diag_floor
        cell_factor = self.cell_factor_head(cell_summary)
        cell_diag = F.softplus(self.cell_diag_head(cell_summary).squeeze(-1)) + self.diag_floor
        scale = F.softplus(self.scale_head(pooled).squeeze(-1)) + self.scale_floor
        flow_context = self.flow_context_head(torch.cat([cond, prev_u], dim=-1))
        return mu, time_factor, time_diag, cell_factor, cell_diag, scale, flow_context


class ResidualFlowStructuredJointStudentTModel(nn.Module):
    """GRU encoder + structured joint mean/cov + conditional residual flow."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-5,
        base_nu: float = 8.0,
    ):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = ResidualFlowStructuredJointDecoder(**decoder_config)
        self.flow = ConditionalResidualFlow(**flow_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.flow_config = flow_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter
        self.base_nu = float(base_nu)

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        history_norm = normalize_iv(history_01)
        return self.encoder(history_norm)

    def forward_from_history(
        self, history_01: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.encode(history_01)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        return self.decoder(cond, prev_u)

    def normalized_components(
        self,
        factor: torch.Tensor,
        diag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_diag = factor.pow(2).sum(dim=-1) + diag.pow(2) + self.cov_jitter
        avg_var = raw_diag.mean(dim=-1).clamp_min(self.cov_jitter)
        norm = avg_var.sqrt().unsqueeze(-1)
        factor_norm = factor / norm.unsqueeze(-1)
        diag_norm = diag / norm
        return factor_norm, diag_norm

    def covariance_parts(
        self,
        time_factor: torch.Tensor,
        time_diag: torch.Tensor,
        cell_factor: torch.Tensor,
        cell_diag: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_factor_norm, time_diag_norm = self.normalized_components(time_factor, time_diag)
        cell_factor_norm, cell_diag_norm = self.normalized_components(cell_factor, cell_diag)

        cov_t = time_factor_norm @ time_factor_norm.transpose(-1, -2)
        cov_t = cov_t + torch.diag_embed(time_diag_norm.pow(2) + self.cov_jitter)
        cov_c = cell_factor_norm @ cell_factor_norm.transpose(-1, -2)
        cov_c = cov_c + torch.diag_embed(cell_diag_norm.pow(2) + self.cov_jitter)
        scale = scale.clamp_min(self.decoder.scale_floor)
        return cov_t * scale.view(-1, 1, 1), cov_c

    def _base_logprob(self, z: torch.Tensor) -> torch.Tensor:
        nu = z.new_tensor(self.base_nu)
        log_norm = (
            torch.lgamma((nu + 1.0) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * torch.log(nu * z.new_tensor(math.pi))
        )
        log_kernel = -0.5 * (nu + 1.0) * torch.log1p(z.pow(2) / nu)
        return (log_norm + log_kernel).sum(dim=-1)

    def log_prob_future(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        time_factor: torch.Tensor,
        time_diag: torch.Tensor,
        cell_factor: torch.Tensor,
        cell_diag: torch.Tensor,
        scale: torch.Tensor,
        flow_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        batch, n_frames, n_cells = target_u.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)

        diff = target_u - mu
        white_t = torch.linalg.solve_triangular(chol_t, diff, upper=False)
        white = torch.linalg.solve_triangular(chol_c, white_t.transpose(1, 2), upper=False).transpose(1, 2)
        white_flat = white.reshape(batch, n_frames * n_cells)

        z, flow_logdet = self.flow(white_flat, flow_context)

        logdet_t = 2.0 * torch.log(torch.diagonal(chol_t, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_c = 2.0 * torch.log(torch.diagonal(chol_c, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_cov = n_cells * logdet_t + n_frames * logdet_c

        base_logprob = self._base_logprob(z)
        logprob = base_logprob + flow_logdet - 0.5 * logdet_cov

        aux = {
            "cov_t": cov_t,
            "cov_c": cov_c,
            "flow_logdet": flow_logdet,
            "white_std": white_flat.std(dim=-1),
            "z_std": z.std(dim=-1),
        }
        return logprob, aux

    @torch.no_grad()
    def sample_future_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mu, time_factor, time_diag, cell_factor, cell_diag, scale, flow_context = self.forward_from_history(history_01)
        batch, n_frames, n_cells = mu.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)

        base = torch.distributions.StudentT(df=self.base_nu)
        z = base.sample((batch * n_samples, n_frames * n_cells)).to(device=mu.device, dtype=mu.dtype)
        ctx = flow_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
        white_flat, _ = self.flow.inverse(z, ctx)
        white = white_flat.view(batch, n_samples, n_frames, n_cells)

        temp = torch.einsum("bij,bsjk->bsik", chol_t, white)
        noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
        return mu.unsqueeze(1) + noise

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        **kwargs,
    ) -> torch.Tensor:
        history_01 = denormalize_iv(history)
        batch_size = history_01.shape[0]
        chunks = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            samples_u = self.sample_future_u(history_01, n_samples=k)
            samples_01 = unconstrained_to_iv(samples_u, lo=self.support_lo, hi=self.support_hi)
            chunks.append(samples_01.view(batch_size, k, self.decoder.n_frames, 5, 5))
        return torch.cat(chunks, dim=1)


def joint_nll_loss(
    model: ResidualFlowStructuredJointStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    target_u = iv_to_unconstrained(
        future_01,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    mu, time_factor, time_diag, cell_factor, cell_diag, scale, flow_context = model.forward_from_history(history_01)
    logprob, aux = model.log_prob_future(
        target_u=target_u,
        mu=mu,
        time_factor=time_factor,
        time_diag=time_diag,
        cell_factor=cell_factor,
        cell_diag=cell_diag,
        scale=scale,
        flow_context=flow_context,
    )
    pred_01 = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)

    metrics = {
        "joint_nll": (-logprob).mean(),
        "joint_mae": (pred_01 - future_01).abs().mean(),
        "time_eff_rank": effective_rank(aux["cov_t"]).mean(),
        "cell_eff_rank": effective_rank(aux["cov_c"]).mean(),
        "scale_mean": scale.mean(),
        "flow_logdet_mean": aux["flow_logdet"].mean(),
        "white_std_mean": aux["white_std"].mean(),
        "z_std_mean": aux["z_std"].mean(),
    }
    return (-logprob).mean(), metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: ResidualFlowStructuredJointStudentTModel,
    val_loader: DataLoader,
) -> dict:
    model.eval()
    totals = {
        "val_joint_nll": 0.0,
        "val_joint_mae": 0.0,
        "val_time_eff_rank": 0.0,
        "val_cell_eff_rank": 0.0,
        "val_scale_mean": 0.0,
        "val_flow_logdet_mean": 0.0,
        "val_white_std_mean": 0.0,
        "val_z_std_mean": 0.0,
    }
    total_count = 0

    for history_01, future_01 in val_loader:
        loss, metrics = joint_nll_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        totals["val_joint_nll"] += loss.item() * batch_size
        totals["val_joint_mae"] += metrics["joint_mae"].item() * batch_size
        totals["val_time_eff_rank"] += metrics["time_eff_rank"].item() * batch_size
        totals["val_cell_eff_rank"] += metrics["cell_eff_rank"].item() * batch_size
        totals["val_scale_mean"] += metrics["scale_mean"].item() * batch_size
        totals["val_flow_logdet_mean"] += metrics["flow_logdet_mean"].item() * batch_size
        totals["val_white_std_mean"] += metrics["white_std_mean"].item() * batch_size
        totals["val_z_std_mean"] += metrics["z_std_mean"].item() * batch_size
        total_count += batch_size

    if total_count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / total_count for k, v in totals.items()}


@torch.no_grad()
def evaluate_joint_subset(
    model: ResidualFlowStructuredJointStudentTModel,
    val_loader: DataLoader,
    joint_val_samples: int,
    eval_limit: int,
) -> dict:
    model.eval()
    total_cov = 0.0
    total_width = 0.0
    total_mae = 0.0
    total_support_viol = 0.0
    total_count = 0
    all_window_widths = []
    all_vov = []
    all_sample_eff_rank = []

    for history_01, future_01 in val_loader:
        if total_count >= eval_limit:
            break
        if total_count + history_01.shape[0] > eval_limit:
            keep = eval_limit - total_count
            history_01 = history_01[:keep]
            future_01 = future_01[:keep]

        history_norm = normalize_iv(history_01)
        samples = model.sample_batched(history_norm, n_samples=joint_val_samples)

        future_grid = future_01.view(history_01.shape[0], future_01.shape[1], 5, 5)
        lo = samples.quantile(0.05, dim=1)
        hi = samples.quantile(0.95, dim=1)
        median = samples.median(dim=1).values

        coverage = ((future_grid >= lo) & (future_grid <= hi)).float().mean()
        width = (hi - lo).mean()
        mae = (median - future_grid).abs().mean()
        support_viol = (
            (samples < model.support_lo) | (samples > model.support_hi)
        ).float().mean()

        mean_iv = history_01.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)
        window_width = (hi - lo).mean(dim=(1, 2, 3))

        first_sample = samples[:, 0].reshape(history_01.shape[0], future_01.shape[1], -1)
        changes = first_sample[:, 1:] - first_sample[:, :-1]
        flat = changes.reshape(-1, changes.shape[-1]).cpu().numpy()
        corr = np.corrcoef(flat.T)
        all_sample_eff_rank.append(eff_rank_np(corr))

        total_cov += coverage.item() * history_01.shape[0]
        total_width += width.item() * history_01.shape[0]
        total_mae += mae.item() * history_01.shape[0]
        total_support_viol += support_viol.item() * history_01.shape[0]
        total_count += history_01.shape[0]

        all_vov.append(vov.detach().cpu())
        all_window_widths.append(window_width.detach().cpu())

    if total_count == 0:
        return {
            "joint_cov90": float("nan"),
            "joint_width90": float("nan"),
            "joint_mae": float("nan"),
            "joint_support_violation_rate": float("nan"),
            "joint_turb_calm_ratio": float("nan"),
            "joint_sample_eff_rank": float("nan"),
        }

    vov = torch.cat(all_vov)
    widths = torch.cat(all_window_widths)
    q20 = torch.quantile(vov, 0.2)
    q80 = torch.quantile(vov, 0.8)
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    if calm_mask.any() and turb_mask.any():
        turb_calm_ratio = (widths[turb_mask].mean() / widths[calm_mask].mean()).item()
    else:
        turb_calm_ratio = float("nan")

    return {
        "joint_cov90": total_cov / total_count,
        "joint_width90": total_width / total_count,
        "joint_mae": total_mae / total_count,
        "joint_support_violation_rate": total_support_viol / total_count,
        "joint_turb_calm_ratio": turb_calm_ratio,
        "joint_sample_eff_rank": float(np.mean(all_sample_eff_rank)),
    }


def main():
    parser = argparse.ArgumentParser(description="172a: residual flow structured joint Student-t")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--lr_flow", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_flow", type=float, default=0.0)
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
    parser.add_argument("--flow_scale_clip", type=float, default=2.0)
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
    )
    flow_config = dict(
        dim=future_len * 25,
        context_dim=args.flow_context_dim,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        scale_clip=args.flow_scale_clip,
    )
    model = ResidualFlowStructuredJointStudentTModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
    ).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    print(f"\n{'=' * 64}")
    print("172a: Residual flow on top of structured joint base")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Flow params:    {n_flow:,}")
    print(f"  Total params:   {n_enc + n_dec + n_flow:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Time rank={args.time_rank} | Cell rank={args.cell_rank}")
    print(f"  Flow dim={future_len * 25} | hidden={args.flow_hidden_dim} | layers={args.flow_layers}")
    print(f"  Support transform: logit(({args.support_lo}, {args.support_hi}))")
    print("  Objective: exact joint likelihood with residual flow")
    print("  Base structure: scale^2 * (Sigma_time kron Sigma_cell)")
    print(f"  Flow base: iid Student-t nu={args.base_nu}")

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
            {
                "params": model.flow.parameters(),
                "lr": args.lr_flow,
                "weight_decay": args.weight_decay_flow,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        ep_mae = 0.0
        ep_time_rank = 0.0
        ep_cell_rank = 0.0
        ep_scale = 0.0
        ep_flow_logdet = 0.0
        ep_white_std = 0.0
        ep_z_std = 0.0
        nb = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = joint_nll_loss(model, history_01, future_01)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_mae += metrics["joint_mae"].item()
            ep_time_rank += metrics["time_eff_rank"].item()
            ep_cell_rank += metrics["cell_eff_rank"].item()
            ep_scale += metrics["scale_mean"].item()
            ep_flow_logdet += metrics["flow_logdet_mean"].item()
            ep_white_std += metrics["white_std_mean"].item()
            ep_z_std += metrics["z_std_mean"].item()
            nb += 1

        scheduler.step()

        train_metrics = {
            "train_joint_nll": ep_loss / max(nb, 1),
            "train_joint_mae": ep_mae / max(nb, 1),
            "train_time_eff_rank": ep_time_rank / max(nb, 1),
            "train_cell_eff_rank": ep_cell_rank / max(nb, 1),
            "train_scale_mean": ep_scale / max(nb, 1),
            "train_flow_logdet_mean": ep_flow_logdet / max(nb, 1),
            "train_white_std_mean": ep_white_std / max(nb, 1),
            "train_z_std_mean": ep_z_std / max(nb, 1),
        }
        val_metrics = evaluate_teacher_forced(model, val_loader)
        joint_metrics = evaluate_joint_subset(
            model,
            val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )

        elapsed = time.time() - t0
        is_best = val_metrics["val_joint_nll"] < best_val
        if is_best:
            best_val = val_metrics["val_joint_nll"]
            best_metrics = {**val_metrics, **joint_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_joint_nll": best_val,
                    "config": {
                        "type": "residual_flow_structured_joint_student_t_172a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "base_nu": args.base_nu,
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
            **train_metrics,
            **val_metrics,
            **joint_metrics,
        }
        history.append(row)

        print(
            f"Ep {epoch:3d}  "
            f"train_nll={train_metrics['train_joint_nll']:.4f}  "
            f"val_nll={val_metrics['val_joint_nll']:.4f}  "
            f"joint_cov90={joint_metrics['joint_cov90']:.4f}  "
            f"joint_tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"fdet={val_metrics['val_flow_logdet_mean']:.4f}  "
            f"zstd={val_metrics['val_z_std_mean']:.3f}  "
            f"viol={joint_metrics['joint_support_violation_rate']:.4e}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_joint_nll": history[-1]["val_joint_nll"] if history else float("nan"),
        "config": {
            "type": "residual_flow_structured_joint_student_t_172a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "base_nu": args.base_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
        },
        "best_val_joint_nll": best_val,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)

    print(f"\nBest val joint NLL: {best_val:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"joint_cov90={best_metrics['joint_cov90']:.4f}, "
            f"joint_turb_calm_ratio={best_metrics['joint_turb_calm_ratio']:.3f}, "
            f"joint_support_violation_rate={best_metrics['joint_support_violation_rate']:.4e}"
        )


if __name__ == "__main__":
    main()
