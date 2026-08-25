#!/usr/bin/env python
"""
231a: Hybrid recurrent factor-decoupled conditional flow with learnable anchor.

Subclasses 227a's FactorARModel; three new pieces baked into the base:
1. Learnable monotone anchor schedule alpha(t) = sigmoid(theta_0 + theta_1 * t/N).
   Default init theta=(-5, 1) gives alpha(0) ~ 0.007, alpha(29) ~ 0.04. If grads
   push nothing up, the model degenerates to 227a without anchor (matching 230b's
   "do nothing" baseline). If grads push up, that is evidence the regime-coupling
   trap 230b hit is escapable via this parameterization.
2. BPTT-SA state consistency: run a teacher-forced "shadow" GRU alongside the main
   free-running loop; L2-match main's K-mean condition to shadow's condition.
   Cheap because shadow skips the flow decoder. Applied over --state_reg_window
   first steps only.
3. Horizon curriculum (reuse 221a's parser): ramp n_steps 5 -> 15 -> 30 over
   --curriculum_schedule.

Three-way ablation (controlled via CLI flags):
  --learnable_anchor off           : alpha(t) == fixed_anchor_alpha (default 0)
                                     (with --fixed_anchor_alpha 0.50 -> 231a-fixed)
                                     (with --fixed_anchor_alpha 0.00 -> 231a-none)
  --learnable_anchor               : theta is nn.Parameter (231a-learn)

End-to-end native rollout from epoch 1 (no scheduled sampling).
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
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    make_serializable,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
)
from experiments.backfill.block_ar.train_212s_h1_minimal_direct_stochastic_delta_es_vs import (
    variogram_score,
)
from experiments.backfill.block_ar.train_227a_factor_ar import (
    FactorARModel,
    SinusoidalPosEmbed,
    afcrps_per_step,
)


# ---------------------------------------------------------------------------
# Curriculum parser — copied verbatim from 221a
# ---------------------------------------------------------------------------

def parse_curriculum(s: str) -> list[tuple[int, int]]:
    """Parse curriculum schedule string like '0:5,3:15,6:30' into [(epoch, horizon)]."""
    pairs = []
    for part in s.split(","):
        epoch_str, horizon_str = part.split(":")
        pairs.append((int(epoch_str), int(horizon_str)))
    return sorted(pairs, key=lambda x: x[0])


def get_curriculum_horizon(epoch: int, schedule: list[tuple[int, int]]) -> int:
    h = schedule[0][1]
    for ep, horizon in schedule:
        if epoch >= ep:
            h = horizon
    return h


# ---------------------------------------------------------------------------
# 231a model — subclass of 227a FactorARModel
# ---------------------------------------------------------------------------

class HybridRecurrentFactorFlowModel(FactorARModel):
    """227a FactorARModel + learnable anchor schedule + BPTT-SA shadow state.

    Extra __init__ args:
      learnable_anchor: bool        If True, anchor_theta is a trainable Parameter.
      fixed_anchor_alpha: float     Used when learnable_anchor=False. 0 -> no anchor,
                                    0.5 -> 230b regime. Ignored when learnable_anchor=True.
      anchor_theta_init: tuple      Initial (theta_0, theta_1) when learnable. Default
                                    (-5, 1) -> alpha(0) ~ 0.007, alpha(N) ~ 0.04.
      anchor_theta1_max: float      Upper bound on theta_1 via soft clamp. Prevents
                                    runaway alpha during training. Default 5.0.
    """

    def __init__(
        self,
        *,
        learnable_anchor: bool = False,
        fixed_anchor_alpha: float = 0.0,
        anchor_theta_init: tuple[float, float] = (-5.0, 1.0),
        anchor_theta1_max: float = 10.0,
        **base_kwargs,
    ):
        # 227a's use_scale_anchor triggers the blend in forward(); we always want it
        # on for 231a (alpha(t) can still be 0). Clear base's fixed scale_anchor_alpha;
        # we override it per-step via alpha(t).
        base_kwargs["use_scale_anchor"] = True
        base_kwargs["scale_anchor_alpha"] = 0.0  # unused; overridden per-step
        super().__init__(**base_kwargs)

        self.learnable_anchor = learnable_anchor
        self.fixed_anchor_alpha = float(fixed_anchor_alpha)
        self.anchor_theta1_max = float(anchor_theta1_max)
        if learnable_anchor:
            self.anchor_theta = nn.Parameter(
                torch.tensor(list(anchor_theta_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "_anchor_theta_buffer",
                torch.tensor(list(anchor_theta_init), dtype=torch.float32),
                persistent=False,
            )

    # ------------------------------------------------------------------
    # alpha(t) — sigmoid schedule. theta_1 is free; monitor in logs.
    # ------------------------------------------------------------------
    def anchor_alpha(self, t: int, n_steps: int) -> torch.Tensor:
        """Return scalar tensor alpha in [0, 1] at step t of N-step rollout.

        alpha(t) = sigmoid(theta_0 + theta_1 * t / (N - 1))

        At init theta=(-5, 1), alpha(0) ~ 0.007, alpha(29) ~ 0.018 — close
        enough to 0 that the rollout matches the warm-start's behavior when
        gradient has not yet moved theta. If theta_1 is pushed very high
        during training, alpha saturates; upper-bound via --anchor_theta1_max
        if needed (default is loose).
        """
        if self.learnable_anchor:
            theta0 = self.anchor_theta[0]
            # Soft upper bound on theta_1 via clamp_max (keeps grads below limit).
            theta1 = self.anchor_theta[1].clamp(max=self.anchor_theta1_max)
            denom = max(1, n_steps - 1)
            return torch.sigmoid(theta0 + theta1 * (t / denom))
        # fixed alpha (scalar on device)
        return torch.tensor(
            self.fixed_anchor_alpha,
            device=self._anchor_theta_buffer.device,
            dtype=self._anchor_theta_buffer.dtype,
        )

    # ------------------------------------------------------------------
    # forward — overrides 227a to use alpha(t) from schedule.
    # ------------------------------------------------------------------
    def forward(
        self,
        history_01: torch.Tensor,
        n_members: int,
        n_steps: int,
    ) -> torch.Tensor:
        """Inference-only free-run rollout using alpha(t) schedule. Matches 227a
        shape contract: returns (B, K, N, 5, 5).
        """
        B = history_01.shape[0]
        device = history_01.device

        cond, local_scale = self.encode_history(history_01)
        prev = history_01[:, -1].reshape(B, self.n_cells)
        scale_anchor = local_scale.clone()

        BK = B * n_members
        cond = cond.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        local_scale = local_scale.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        prev = prev.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        scale_anchor_bk = scale_anchor.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)

        z_f = torch.randn(BK, self.factor_rank, device=device)
        rho_sq_comp = math.sqrt(1.0 - self.rho ** 2)

        frames = []
        for t in range(n_steps):
            if t > 0:
                z_f = self.rho * z_f + rho_sq_comp * torch.randn_like(z_f)
            z_i = torch.randn(BK, self.n_cells, device=device)
            pos = self.pos_embed(t, BK, device)

            factor_in = torch.cat([prev, cond, z_f, pos], dim=-1)
            f_scores = self.factor_head(factor_in)
            idio_in = torch.cat([prev, cond, z_i, pos], dim=-1)
            i_resid = self.idio_head(idio_in)

            Lambda = self.get_lambda(cond)
            D = self.get_d(cond)
            v = torch.einsum("br,bcr->bc", f_scores, Lambda) + D * i_resid
            if self.noise_skip:
                v = v + torch.tanh(self.noise_skip_proj(z_i))
            if self.cell_spread_enabled:
                cs = F.softplus(self.cell_spread_proj(cond))
                v = v * cs

            delta = torch.sinh(v) * local_scale
            next_iv = (prev + delta).clamp(0.001, 1.0)
            frames.append(next_iv)

            feat, local_scale_raw = self._step_features(prev, next_iv, local_scale)
            cond = self.gru_cell(feat, cond)
            prev = next_iv

            alpha_t = self.anchor_alpha(t, n_steps)
            log_s = ((1.0 - alpha_t) * torch.log(local_scale_raw.clamp_min(self.scale_floor))
                     + alpha_t * torch.log(scale_anchor_bk.clamp_min(self.scale_floor)))
            local_scale = torch.exp(log_s)

        trajectory = torch.stack(frames, dim=0).permute(1, 0, 2).view(
            B, n_members, n_steps, 5, 5
        )
        return trajectory

    # ------------------------------------------------------------------
    # forward_train — returns trajectory AND BPTT-SA state-reg loss.
    # Used only during training; inference uses forward().
    # ------------------------------------------------------------------
    def forward_train(
        self,
        history_01: torch.Tensor,
        future_01: torch.Tensor,
        n_members: int,
        n_steps: int,
        state_reg_window: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Native end-to-end rollout with teacher-forced shadow for state reg.

        Args:
            history_01: (B, H, 5, 5) in [0, 1]
            future_01: (B, N_full, 5, 5) GT frames; only first n_steps used
            n_members: K
            n_steps: rollout length (may be < future_01.shape[1] under curriculum)
            state_reg_window: # of early steps over which state reg is computed

        Returns:
            trajectory: (B, K, n_steps, 5, 5)
            state_reg: scalar loss term (mean squared difference of main K-mean
                condition vs teacher-forced shadow condition, detached)
        """
        B = history_01.shape[0]
        device = history_01.device

        # Shared initial encoding
        cond0, local_scale0 = self.encode_history(history_01)
        prev0 = history_01[:, -1].reshape(B, self.n_cells)
        scale_anchor = local_scale0.clone()

        # Future frames (may be padded/cut to n_steps)
        future_flat = future_01.reshape(B, future_01.shape[1], self.n_cells)

        # Main: K-expanded, free-running
        BK = B * n_members
        cond_m = cond0.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        local_scale_m = local_scale0.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        prev_m = prev0.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        scale_anchor_bk = scale_anchor.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)

        # Shadow: B-only, teacher-forced. Uses same params as main (shared weights).
        cond_s = cond0
        local_scale_s = local_scale0
        prev_s = prev0

        z_f = torch.randn(BK, self.factor_rank, device=device)
        rho_sq_comp = math.sqrt(1.0 - self.rho ** 2)

        frames = []
        state_reg = torch.tensor(0.0, device=device)
        state_reg_count = 0

        for t in range(n_steps):
            if t > 0:
                z_f = self.rho * z_f + rho_sq_comp * torch.randn_like(z_f)
            z_i = torch.randn(BK, self.n_cells, device=device)
            pos = self.pos_embed(t, BK, device)

            # ----- MAIN (free-running) -----
            factor_in = torch.cat([prev_m, cond_m, z_f, pos], dim=-1)
            f_scores = self.factor_head(factor_in)
            idio_in = torch.cat([prev_m, cond_m, z_i, pos], dim=-1)
            i_resid = self.idio_head(idio_in)

            Lambda = self.get_lambda(cond_m)
            D = self.get_d(cond_m)
            v = torch.einsum("br,bcr->bc", f_scores, Lambda) + D * i_resid
            if self.noise_skip:
                v = v + torch.tanh(self.noise_skip_proj(z_i))
            if self.cell_spread_enabled:
                cs = F.softplus(self.cell_spread_proj(cond_m))
                v = v * cs

            delta = torch.sinh(v) * local_scale_m
            next_iv_m = (prev_m + delta).clamp(0.001, 1.0)
            frames.append(next_iv_m)

            feat_m, ls_m_raw = self._step_features(prev_m, next_iv_m, local_scale_m)
            cond_m_new = self.gru_cell(feat_m, cond_m)

            # anchor blend
            alpha_t = self.anchor_alpha(t, n_steps)
            log_s_m = ((1.0 - alpha_t) * torch.log(ls_m_raw.clamp_min(self.scale_floor))
                       + alpha_t * torch.log(scale_anchor_bk.clamp_min(self.scale_floor)))
            local_scale_m = torch.exp(log_s_m)
            cond_m = cond_m_new
            prev_m = next_iv_m

            # ----- SHADOW (teacher-forced, no decoder call) -----
            # Only run shadow while we still contribute to state_reg. After the
            # window it is wasted compute; skip.
            if t < state_reg_window:
                target_t = future_flat[:, t, :]  # (B, 25)
                feat_s, ls_s_raw = self._step_features(prev_s, target_t, local_scale_s)
                cond_s_new = self.gru_cell(feat_s, cond_s)

                log_s_s = ((1.0 - alpha_t) * torch.log(ls_s_raw.clamp_min(self.scale_floor))
                           + alpha_t * torch.log(scale_anchor.clamp_min(self.scale_floor)))
                local_scale_s = torch.exp(log_s_s)
                cond_s = cond_s_new
                prev_s = target_t

                # state reg: K-mean of main cond -> shadow cond (detached)
                cond_m_Bmean = cond_m.view(B, n_members, -1).mean(dim=1)  # (B, 128)
                state_reg = state_reg + F.mse_loss(cond_m_Bmean, cond_s.detach())
                state_reg_count += 1

        trajectory = torch.stack(frames, dim=0).permute(1, 0, 2).view(
            B, n_members, n_steps, 5, 5
        )
        if state_reg_count > 0:
            state_reg = state_reg / state_reg_count
        return trajectory, state_reg


# ---------------------------------------------------------------------------
# Loss — wraps 227a's per-step ES+VS and adds state_reg term.
# ---------------------------------------------------------------------------

def compute_231a_loss(
    trajectory: torch.Tensor,
    future_01: torch.Tensor,
    state_reg: torch.Tensor,
    lambda_vs: float,
    lambda_state: float,
    loss_type: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """per-step ES/afCRPS + VS + state_reg."""
    B, K, N = trajectory.shape[:3]
    total_main = torch.tensor(0.0, device=trajectory.device)
    total_vs = torch.tensor(0.0, device=trajectory.device)

    for t in range(N):
        samples_t = trajectory[:, :, t].reshape(B, K, -1)
        gt_t = future_01[:, t].reshape(B, -1)
        if loss_type == "afcrps":
            total_main = total_main + afcrps_per_step(samples_t, gt_t)
        else:
            total_main = total_main + energy_score(samples_t, gt_t)
        total_vs = total_vs + variogram_score(samples_t, gt_t, p=0.5)

    loss = total_main + lambda_vs * total_vs + lambda_state * state_reg
    metrics = {
        "main": float((total_main / N).detach().item()),
        "vs": float((total_vs / N).detach().item()),
        "state_reg": float(state_reg.detach().item()),
        "loss": float((loss / N).detach().item()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Loader — matches 227a.load_model signature for compatibility with rollout
# utils. model_type == "231a".
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[HybridRecurrentFactorFlowModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = HybridRecurrentFactorFlowModel(
        n_cells=cfg["n_cells"],
        factor_rank=cfg["factor_rank"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        decoder_hidden=cfg["decoder_hidden"],
        rho=cfg["rho"],
        ewma_alpha=cfg["ewma_alpha"],
        scale_floor=cfg["scale_floor"],
        include_scale_feature=cfg["include_scale_feature"],
        pos_embed_dim=cfg["pos_embed_dim"],
        noise_skip=cfg.get("noise_skip", False),
        d_scale=cfg.get("d_scale", 1.0),
        cell_spread=cfg.get("cell_spread", False),
        decoder_layers=cfg.get("decoder_layers", 2),
        no_tanh=cfg.get("no_tanh", False),
        learnable_anchor=cfg.get("learnable_anchor", False),
        fixed_anchor_alpha=cfg.get("fixed_anchor_alpha", 0.0),
        anchor_theta_init=tuple(cfg.get("anchor_theta_init", (-5.0, 1.0))),
        anchor_theta1_max=cfg.get("anchor_theta1_max", 5.0),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


# ---------------------------------------------------------------------------
# Main — training entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="231a: hybrid recurrent factor flow + learnable anchor")
    # data
    p.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    p.add_argument("--pca_init", type=str, default="models/backfill/226a_pca_init.npz")
    p.add_argument("--history_len", type=int, default=30)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--test_start", type=int, default=4511)
    p.add_argument("--val_size", type=int, default=441)
    p.add_argument("--max_train_windows", type=int, default=4010)
    # warm-start (CRITICAL)
    p.add_argument("--warmstart_checkpoint", type=str, required=True,
                   help="Path to 227a-family checkpoint to warm-start from (e.g. 229a@30)")
    # training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_members", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Lower than 227a's 1e-3 because warm-started")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # losses
    p.add_argument("--lambda_vs", type=float, default=0.05)
    p.add_argument("--lambda_state", type=float, default=0.1)
    p.add_argument("--loss_type", type=str, default="es", choices=["es", "afcrps"])
    # architecture (MUST match warm-start)
    p.add_argument("--factor_rank", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--decoder_hidden", type=int, default=256,
                   help="229a-wide default; must match warm-start")
    p.add_argument("--decoder_layers", type=int, default=2)
    p.add_argument("--pos_embed_dim", type=int, default=16)
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--ewma_alpha", type=float, default=0.20)
    p.add_argument("--scale_floor", type=float, default=1e-4)
    p.add_argument("--noise_skip", action="store_true")
    p.add_argument("--d_scale", type=float, default=3.0)
    p.add_argument("--cell_spread", action="store_true")
    p.add_argument("--no_tanh", action="store_true")
    # anchor config (THREE-WAY ABLATION)
    p.add_argument("--learnable_anchor", action="store_true",
                   help="231a-learn: train anchor_theta; else use fixed_anchor_alpha")
    p.add_argument("--fixed_anchor_alpha", type=float, default=0.0,
                   help="231a-none: 0.0; 231a-fixed: 0.50")
    p.add_argument("--anchor_theta_init", type=str, default="-5.0,1.0",
                   help="Initial theta_0,theta_1 when learnable")
    p.add_argument("--anchor_theta1_max", type=float, default=10.0)
    # state reg
    p.add_argument("--state_reg_window", type=int, default=5,
                   help="BPTT-SA applied over first N steps only")
    # curriculum
    p.add_argument("--curriculum_schedule", type=str, default="0:5,3:15,6:30",
                   help="epoch:horizon pairs, comma-separated")
    # checkpointing
    p.add_argument("--checkpoint_every", type=int, default=3)
    # misc
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # parse schedules
    curriculum = parse_curriculum(args.curriculum_schedule)
    anchor_theta_init = tuple(float(x) for x in args.anchor_theta_init.split(","))
    assert len(anchor_theta_init) == 2, "--anchor_theta_init must be 'theta0,theta1'"

    # build model
    model = HybridRecurrentFactorFlowModel(
        n_cells=25,
        factor_rank=args.factor_rank,
        hidden_dim=args.hidden_dim,
        gru_layers=2,
        gru_dropout=0.1,
        decoder_hidden=args.decoder_hidden,
        rho=args.rho,
        ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor,
        include_scale_feature=True,
        pos_embed_dim=args.pos_embed_dim,
        noise_skip=args.noise_skip,
        d_scale=args.d_scale,
        cell_spread=args.cell_spread,
        decoder_layers=args.decoder_layers,
        no_tanh=args.no_tanh,
        learnable_anchor=args.learnable_anchor,
        fixed_anchor_alpha=args.fixed_anchor_alpha,
        anchor_theta_init=anchor_theta_init,
        anchor_theta1_max=args.anchor_theta1_max,
    ).to(device)

    # PCA init (required for factor structure)
    pca = np.load(args.pca_init)
    model.init_from_pca(pca["lambda_init"], pca["d_init"])
    var_ratio = float(pca["explained_variance_ratio"][:int(pca["factor_rank"])].sum())
    print(f"PCA init: rank={pca['factor_rank']}, var={var_ratio:.3f}")

    # warm-start — load compatible 227a keys (skip anchor-specific params)
    ws_payload = torch.load(args.warmstart_checkpoint, map_location=device, weights_only=False)
    ws_state = ws_payload["model_state_dict"]
    # strict=False so 227a checkpoints without anchor_theta still load
    missing, unexpected = model.load_state_dict(ws_state, strict=False)
    # filter expected missing keys (only ours)
    expected_missing = {"anchor_theta"} if args.learnable_anchor else set()
    real_missing = [k for k in missing if k not in expected_missing and not k.endswith("_anchor_theta_buffer")]
    if real_missing:
        print(f"WARN: warm-start missing keys: {real_missing[:5]}...")
    if unexpected:
        print(f"WARN: warm-start unexpected keys: {unexpected[:5]}...")
    print(f"Warm-started from {args.warmstart_checkpoint} "
          f"(epoch {ws_payload.get('epoch', '?')}, val_loss {ws_payload.get('val_loss', '?')})")

    # data
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - args.n_steps
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[:args.max_train_windows]

    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, args.history_len, args.n_steps
    )
    train_future = train_future.view(train_hist.shape[0], args.n_steps, 5, 5)
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, args.history_len, args.n_steps
    )
    val_future = val_future.view(val_hist.shape[0], args.n_steps, 5, 5)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size, shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = {
        "type": "hybrid_factor_ar_231a",
        "n_cells": 25,
        "factor_rank": args.factor_rank,
        "hidden_dim": args.hidden_dim,
        "gru_layers": 2,
        "gru_dropout": 0.1,
        "decoder_hidden": args.decoder_hidden,
        "rho": args.rho,
        "ewma_alpha": args.ewma_alpha,
        "scale_floor": args.scale_floor,
        "include_scale_feature": True,
        "pos_embed_dim": args.pos_embed_dim,
        "lambda_vs": args.lambda_vs,
        "lambda_state": args.lambda_state,
        "n_steps": args.n_steps,
        "n_members": args.n_members,
        "noise_skip": args.noise_skip,
        "d_scale": args.d_scale,
        "cell_spread": args.cell_spread,
        "decoder_layers": args.decoder_layers,
        "loss_type": args.loss_type,
        "no_tanh": args.no_tanh,
        "learnable_anchor": args.learnable_anchor,
        "fixed_anchor_alpha": args.fixed_anchor_alpha,
        "anchor_theta_init": list(anchor_theta_init),
        "anchor_theta1_max": args.anchor_theta1_max,
        "state_reg_window": args.state_reg_window,
        "curriculum_schedule": args.curriculum_schedule,
        "warmstart_checkpoint": args.warmstart_checkpoint,
    }

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n231a Hybrid Recurrent Factor Flow")
    print(f"  learnable_anchor={args.learnable_anchor}, fixed_alpha={args.fixed_anchor_alpha}, theta_init={anchor_theta_init}")
    print(f"  state_reg: lambda={args.lambda_state}, window={args.state_reg_window}")
    print(f"  curriculum: {curriculum}")
    print(f"  factor_rank={args.factor_rank}, decoder_hidden={args.decoder_hidden}, noise_skip={args.noise_skip}, d_scale={args.d_scale}")
    print(f"  n_members={args.n_members}, batch={args.batch_size}, lr={args.lr}")
    print(f"  train={train_hist.shape[0]}, val={val_hist.shape[0]}")
    print(f"  params: {n_params:,} ({trainable:,} trainable)")

    history_log: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        h_cur = get_curriculum_horizon(epoch, curriculum)
        running = {"loss": 0.0, "main": 0.0, "vs": 0.0, "state_reg": 0.0}
        count = 0

        for history_01, future_01 in train_loader:
            # curriculum: only use first h_cur steps of future
            future_cur = future_01[:, :h_cur]
            trajectory, state_reg = model.forward_train(
                history_01=history_01,
                future_01=future_cur,
                n_members=args.n_members,
                n_steps=h_cur,
                state_reg_window=min(args.state_reg_window, h_cur),
            )
            loss, metrics = compute_231a_loss(
                trajectory=trajectory,
                future_01=future_cur,
                state_reg=state_reg,
                lambda_vs=args.lambda_vs,
                lambda_state=args.lambda_state,
                loss_type=args.loss_type,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            b = history_01.shape[0]
            for k in running:
                running[k] += metrics[k] * b
            count += b

        train_metrics = {k: v / count for k, v in running.items()}

        # Validation — use full n_steps (30) regardless of curriculum
        model.eval()
        val_main_sum, val_vs_sum, val_state_sum, val_count = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for v_start in range(0, val_hist.shape[0], args.batch_size):
                v_end = min(v_start + args.batch_size, val_hist.shape[0])
                vh = val_hist[v_start:v_end]
                vf = val_future[v_start:v_end]
                vt, vsr = model.forward_train(
                    history_01=vh,
                    future_01=vf,
                    n_members=args.n_members,
                    n_steps=args.n_steps,
                    state_reg_window=args.state_reg_window,
                )
                _, vm = compute_231a_loss(
                    vt, vf, vsr, args.lambda_vs, args.lambda_state, args.loss_type
                )
                vb = vh.shape[0]
                val_main_sum += vm["main"] * vb
                val_vs_sum += vm["vs"] * vb
                val_state_sum += vm["state_reg"] * vb
                val_count += vb

        val_main = val_main_sum / val_count
        val_vs = val_vs_sum / val_count
        val_state = val_state_sum / val_count
        val_loss = val_main + args.lambda_vs * val_vs + args.lambda_state * val_state

        # Report alpha schedule for learnable variant
        alpha_report = {}
        if args.learnable_anchor:
            with torch.no_grad():
                for t in [0, 4, 14, 29]:
                    alpha_report[f"a{t}"] = float(model.anchor_alpha(t, args.n_steps).item())

        elapsed = time.time() - t0
        record = {
            "epoch": epoch, "elapsed": elapsed, "h_curriculum": h_cur,
            "train_loss": train_metrics["loss"],
            "train_main": train_metrics["main"],
            "train_vs": train_metrics["vs"],
            "train_state_reg": train_metrics["state_reg"],
            "val_main": val_main, "val_vs": val_vs,
            "val_state_reg": val_state, "val_loss": val_loss,
            **alpha_report,
        }
        history_log.append(make_serializable(record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_dir / "best_model.pt")

        if epoch % args.checkpoint_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_dir / f"checkpoint_ep{epoch}.pt")

        msg = (
            f"[{epoch:3d}/{args.epochs}] h={h_cur} "
            f"loss={train_metrics['loss']:.4f} main={train_metrics['main']:.4f} "
            f"vs={train_metrics['vs']:.4f} state={train_metrics['state_reg']:.4f} "
            f"val={val_loss:.4f}"
        )
        if alpha_report:
            msg += f"  alpha=({alpha_report['a0']:.3f},{alpha_report['a14']:.3f},{alpha_report['a29']:.3f})"
        msg += f"  ({elapsed:.1f}s)"
        print(msg)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": args.epochs,
        "val_loss": val_loss,
    }, out_dir / "final_model.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")
    if args.learnable_anchor:
        with torch.no_grad():
            print(f"Final anchor_theta: {model.anchor_theta.tolist()}")
            print(f"Final alpha(0,14,29): "
                  f"{model.anchor_alpha(0, args.n_steps).item():.4f}, "
                  f"{model.anchor_alpha(14, args.n_steps).item():.4f}, "
                  f"{model.anchor_alpha(29, args.n_steps).item():.4f}")


if __name__ == "__main__":
    main()
