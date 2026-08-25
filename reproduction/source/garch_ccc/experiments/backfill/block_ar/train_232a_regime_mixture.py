#!/usr/bin/env python
"""
232a: K=2 regime mixture-of-flows on factor head. Attacks turb/calm gate.

Motivation: 229a's factor head produces a single standardized-innovation law
regardless of regime. Result: turb/calm = 0.957 << 1.15 gate. Adding K=2
factor heads gated by regime features should let one head specialize to calm
(tighter inno distribution) and the other to turbulent (wider).

Key design:
  - K=2 parallel factor_head instances (deepcopy warmstart)
  - gate MLP: (cond, vov_10, slope_proxy) -> softmax(K)
  - entropy bonus on gate (prevent collapse)
  - usage-floor penalty (each component >= 20% of val windows)
  - SOFT mixture during training: expected innovation = sum_k pi_k * f_k(z)
  - HARD routing at inference (argmax) by default, SOFT optional

Inference regime:
  - warmstart from 229a@ep30, scale_anchor=False during training (preserves
    229a's "innovation law learned without anchor" property)
  - flip scale_anchor=True alpha=0.50 at inference (same pattern as 229a prod)
"""

from __future__ import annotations

import argparse
import copy
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
    afcrps_per_step,
)


# ---------------------------------------------------------------------------
# Regime feature extractor — compute vol-of-vol + slope from history window.
# ---------------------------------------------------------------------------

def compute_regime_features(history_01: torch.Tensor, vov_window: int = 10) -> torch.Tensor:
    """From (B, H, 5, 5), compute (B, 2) features: [log_vov, slope_proxy].

    vov = std of daily |ΔIV| over last vov_window days, averaged across cells
    slope_proxy = linear slope of |ΔIV| vs time over last vov_window days
    """
    B, H, _, _ = history_01.shape
    x = history_01.reshape(B, H, -1)  # (B, H, 25)
    deltas = (x[:, 1:] - x[:, :-1]).abs()  # (B, H-1, 25)
    window = deltas[:, -vov_window:]  # (B, W, 25)
    mean_delta_per_day = window.mean(dim=-1)  # (B, W)
    vov = mean_delta_per_day.std(dim=-1).clamp_min(1e-6)  # (B,)
    log_vov = torch.log(vov).unsqueeze(-1)  # (B, 1)
    # slope: fit y = at + b over last vov_window. We just need the coefficient.
    t = torch.arange(vov_window, device=history_01.device, dtype=torch.float32)
    t = t - t.mean()
    y = mean_delta_per_day
    slope = (y * t).sum(dim=-1) / (t * t).sum()  # (B,)
    slope = slope.unsqueeze(-1)  # (B, 1)
    return torch.cat([log_vov, slope], dim=-1)  # (B, 2)


# ---------------------------------------------------------------------------
# Mixture model
# ---------------------------------------------------------------------------

class MixtureFactorARModel(FactorARModel):
    """229a FactorAR + K=2 mixture on factor head."""

    def __init__(
        self,
        *,
        K: int = 2,
        gate_hidden: int = 64,
        vov_feat_dim: int = 2,
        gate_entropy_weight: float = 0.02,
        gate_usage_floor: float = 0.20,
        gate_usage_floor_weight: float = 1.0,
        hard_routing_inference: bool = False,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.K = K
        self.vov_feat_dim = vov_feat_dim
        self.gate_entropy_weight = float(gate_entropy_weight)
        self.gate_usage_floor = float(gate_usage_floor)
        self.gate_usage_floor_weight = float(gate_usage_floor_weight)
        self.hard_routing_inference = bool(hard_routing_inference)

        # Replace single factor_head with K duplicates.
        # CRITICAL: break symmetry at init — add small Gaussian noise to all heads
        # beyond the first, so the mixture gradient can distinguish them. Without
        # this, `out_k * pi_k` gives zero gate-gradient because outs are identical.
        head0 = self.factor_head
        extra_heads = nn.ModuleList()
        for _ in range(K - 1):
            h = copy.deepcopy(head0)
            with torch.no_grad():
                for p in h.parameters():
                    if p.dim() >= 1:
                        p.add_(torch.randn_like(p) * 0.05 * p.std().clamp_min(0.01))
            extra_heads.append(h)
        self.factor_heads = nn.ModuleList([head0, *extra_heads])

        # Gate MLP: input = (cond, vov_features) → K logits.
        # Small weights + bias init gives near-uniform prior; the vov_feat dimension
        # provides the signal that breaks symmetry at data time.
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + vov_feat_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, K),
        )
        with torch.no_grad():
            nn.init.normal_(self.gate_mlp[-1].weight, std=0.1)
            nn.init.zeros_(self.gate_mlp[-1].bias)

    # ------------------------------------------------------------------
    # Core: gated factor forward. Returns (f_scores, gate_probs).
    # f_scores = sum_k pi_k(cond, vov) * f_k(input)   (soft mixture)
    # ------------------------------------------------------------------
    def _mixture_factor(
        self,
        factor_in: torch.Tensor,
        cond: torch.Tensor,
        vov_feat: torch.Tensor,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # gate
        gate_input = torch.cat([cond, vov_feat], dim=-1)  # (BK, H+vov)
        gate_logits = self.gate_mlp(gate_input)  # (BK, K)
        gate = F.softmax(gate_logits, dim=-1)  # (BK, K)

        if hard:
            idx = gate.argmax(dim=-1)  # (BK,)
            # one-hot
            gate_hard = F.one_hot(idx, num_classes=self.K).float()
            # straight-through: use hard for forward, soft for grads (not needed at inference)
            gate_used = gate_hard
        else:
            gate_used = gate

        # Compute each head's output and take weighted sum
        # (BK, factor_rank) per head
        outs = torch.stack(
            [head(factor_in) for head in self.factor_heads], dim=1
        )  # (BK, K, factor_rank)
        f_scores = (outs * gate_used.unsqueeze(-1)).sum(dim=1)  # (BK, factor_rank)
        return f_scores, gate

    # ------------------------------------------------------------------
    # forward — overrides 227a's. Keeps anchor logic (use_scale_anchor flag).
    # Returns trajectory; also stashes gate_usage on self._last_gate_usage.
    # ------------------------------------------------------------------
    def forward(
        self,
        history_01: torch.Tensor,
        n_members: int,
        n_steps: int,
    ) -> torch.Tensor:
        B = history_01.shape[0]
        device = history_01.device

        cond, local_scale = self.encode_history(history_01)
        prev = history_01[:, -1].reshape(B, self.n_cells)
        scale_anchor = local_scale.clone() if self.use_scale_anchor else None

        # static regime features per window (B, 2) (vov + slope from history)
        vov_feat_B = compute_regime_features(history_01, vov_window=10)

        BK = B * n_members
        cond = cond.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        local_scale = local_scale.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        prev = prev.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        vov_feat = vov_feat_B.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)

        if self.use_scale_anchor:
            scale_anchor_bk = (
                scale_anchor.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
            )

        z_f = torch.randn(BK, self.factor_rank, device=device)
        rho_sq_comp = math.sqrt(1.0 - self.rho ** 2)

        frames = []
        # Accumulate WITH gradients for entropy/usage-floor regularizers.
        gate_usage_live_sum = torch.zeros(self.K, device=device)
        gate_entropy_live = torch.tensor(0.0, device=device)
        n_steps_counted = 0
        hard = self.hard_routing_inference and (not self.training)

        for t in range(n_steps):
            if t > 0:
                z_f = self.rho * z_f + rho_sq_comp * torch.randn_like(z_f)
            z_i = torch.randn(BK, self.n_cells, device=device)
            pos = self.pos_embed(t, BK, device)

            factor_in = torch.cat([prev, cond, z_f, pos], dim=-1)
            f_scores, gate = self._mixture_factor(factor_in, cond, vov_feat, hard=hard)
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

            # live gate accumulators (KEEP gradient for regularizers)
            gate_usage_live_sum = gate_usage_live_sum + gate.mean(dim=0)
            # per-sample entropy, averaged over BK
            pent = -(gate.clamp_min(1e-8) * torch.log(gate.clamp_min(1e-8))).sum(dim=-1).mean()
            gate_entropy_live = gate_entropy_live + pent
            n_steps_counted += 1

            feat, local_scale_raw = self._step_features(prev, next_iv, local_scale)
            cond = self.gru_cell(feat, cond)
            prev = next_iv

            if self.use_scale_anchor:
                log_s = (
                    (1.0 - self.scale_anchor_alpha)
                    * torch.log(local_scale_raw.clamp_min(self.scale_floor))
                    + self.scale_anchor_alpha
                    * torch.log(scale_anchor_bk.clamp_min(self.scale_floor))
                )
                local_scale = torch.exp(log_s)
            else:
                local_scale = local_scale_raw

        self._last_gate_usage = (gate_usage_live_sum / max(1, n_steps_counted)).detach()
        # stash live versions for forward_train to use (with gradient)
        self._last_gate_usage_live = gate_usage_live_sum / max(1, n_steps_counted)
        self._last_gate_entropy_live = gate_entropy_live / max(1, n_steps_counted)

        trajectory = (
            torch.stack(frames, dim=0)
            .permute(1, 0, 2)
            .view(B, n_members, n_steps, 5, 5)
        )
        return trajectory

    # ------------------------------------------------------------------
    # forward_train — same but also returns gate regularizers.
    # ------------------------------------------------------------------
    def forward_train(
        self,
        history_01: torch.Tensor,
        n_members: int,
        n_steps: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        trajectory = self.forward(history_01, n_members=n_members, n_steps=n_steps)
        # Use live (differentiable) gate_usage & entropy stashed in forward()
        gate_usage_live = self._last_gate_usage_live  # (K,) WITH grad
        entropy_live = self._last_gate_entropy_live   # per-sample entropy avg, WITH grad

        # CONFIDENT ROUTING: we MINIMIZE per-sample entropy (push each gate toward one-hot).
        # This lets the heads specialize on different regimes per-window.
        entropy_penalty = self.gate_entropy_weight * entropy_live

        # BALANCED USAGE: we also want the AGGREGATE gate usage to have each component
        # used at least `gate_usage_floor` fraction of the time (e.g. 20%). This prevents
        # the confident-routing gradient from collapsing to "always pick component 0".
        shortfalls = (self.gate_usage_floor - gate_usage_live).clamp_min(0.0)
        usage_penalty = (shortfalls ** 2).sum()

        reg_terms = {
            "entropy_penalty": entropy_penalty,
            "usage_penalty": self.gate_usage_floor_weight * usage_penalty,
            "gate_usage": self._last_gate_usage,  # detached for reporting
            "entropy_value": entropy_live.detach(),  # for reporting
        }
        return trajectory, reg_terms


# ---------------------------------------------------------------------------
# Loss — per-step ES + VS plus gate reg.
# ---------------------------------------------------------------------------

def compute_232a_loss(
    trajectory: torch.Tensor,
    future_01: torch.Tensor,
    reg_terms: dict[str, torch.Tensor],
    lambda_vs: float,
    loss_type: str,
) -> tuple[torch.Tensor, dict[str, float]]:
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

    loss = total_main + lambda_vs * total_vs
    # MINIMIZE per-sample entropy (+ sign) to promote confident routing.
    # MINIMIZE usage_penalty (+ sign) to prevent aggregate collapse.
    loss = loss + reg_terms["entropy_penalty"] + reg_terms["usage_penalty"]

    metrics = {
        "main": float((total_main / N).detach().item()),
        "vs": float((total_vs / N).detach().item()),
        "entropy_penalty": float(reg_terms["entropy_penalty"].detach().item()),
        "entropy_value": float(reg_terms["entropy_value"].item()),
        "usage_penalty": float(reg_terms["usage_penalty"].detach().item()),
        "gate_0": float(reg_terms["gate_usage"][0].detach().item()),
        "gate_1": float(reg_terms["gate_usage"][1].detach().item())
        if len(reg_terms["gate_usage"]) > 1 else 0.0,
        "loss": float((loss / N).detach().item()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[MixtureFactorARModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = MixtureFactorARModel(
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
        use_scale_anchor=cfg.get("use_scale_anchor", False),
        scale_anchor_alpha=cfg.get("scale_anchor_alpha", 0.50),
        K=cfg.get("K", 2),
        gate_hidden=cfg.get("gate_hidden", 64),
        vov_feat_dim=cfg.get("vov_feat_dim", 2),
        gate_entropy_weight=cfg.get("gate_entropy_weight", 0.02),
        gate_usage_floor=cfg.get("gate_usage_floor", 0.20),
        gate_usage_floor_weight=cfg.get("gate_usage_floor_weight", 1.0),
        hard_routing_inference=cfg.get("hard_routing_inference", False),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="232a: regime mixture")
    p.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    p.add_argument("--pca_init", type=str, default="models/backfill/226a_pca_init.npz")
    p.add_argument("--history_len", type=int, default=30)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--test_start", type=int, default=4511)
    p.add_argument("--val_size", type=int, default=441)
    p.add_argument("--max_train_windows", type=int, default=4010)
    p.add_argument("--warmstart_checkpoint", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_members", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_vs", type=float, default=0.05)
    p.add_argument("--loss_type", type=str, default="es", choices=["es", "afcrps"])
    p.add_argument("--factor_rank", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--decoder_hidden", type=int, default=256)
    p.add_argument("--decoder_layers", type=int, default=2)
    p.add_argument("--pos_embed_dim", type=int, default=16)
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--ewma_alpha", type=float, default=0.20)
    p.add_argument("--scale_floor", type=float, default=1e-4)
    p.add_argument("--noise_skip", action="store_true")
    p.add_argument("--d_scale", type=float, default=3.0)
    p.add_argument("--no_tanh", action="store_true")
    # mixture
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--gate_hidden", type=int, default=64)
    p.add_argument("--gate_entropy_weight", type=float, default=0.02)
    p.add_argument("--gate_usage_floor", type=float, default=0.20)
    p.add_argument("--gate_usage_floor_weight", type=float, default=1.0)
    p.add_argument("--hard_routing_inference", action="store_true")
    # checkpoint
    p.add_argument("--checkpoint_every", type=int, default=3)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MixtureFactorARModel(
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
        cell_spread=False,
        decoder_layers=args.decoder_layers,
        no_tanh=args.no_tanh,
        use_scale_anchor=False,  # anchor OFF during training (229a regime)
        scale_anchor_alpha=0.50,
        K=args.K,
        gate_hidden=args.gate_hidden,
        vov_feat_dim=2,
        gate_entropy_weight=args.gate_entropy_weight,
        gate_usage_floor=args.gate_usage_floor,
        gate_usage_floor_weight=args.gate_usage_floor_weight,
        hard_routing_inference=args.hard_routing_inference,
    ).to(device)

    # PCA init (required even for fine-tune since factor_heads need sensible loadings)
    pca = np.load(args.pca_init)
    model.init_from_pca(pca["lambda_init"], pca["d_init"])
    print(f"PCA init: rank={pca['factor_rank']}")

    # Warm-start from 229a: single factor_head → all K heads
    ws = torch.load(args.warmstart_checkpoint, map_location=device, weights_only=False)
    ws_state = ws["model_state_dict"]
    # replicate factor_head.{n}.{...} into factor_heads.{0..K-1}.{n}.{...}
    # 229a has factor_head.0.weight, factor_head.0.bias, ... (Sequential)
    # Our model has factor_heads.{k}.0.weight etc and ALSO self.factor_head.{...}
    # For clean load: duplicate the 229a factor_head.{...} keys into our factor_heads.{k}.{...}
    expanded_state = {}
    for key, val in ws_state.items():
        if key.startswith("factor_head."):
            # map to factor_heads.{k}.{rest}
            rest = key[len("factor_head."):]
            for k in range(args.K):
                expanded_state[f"factor_heads.{k}.{rest}"] = val.clone()
            # also keep in factor_head.* (parent still has the attr)
            expanded_state[key] = val
        else:
            expanded_state[key] = val
    missing, unexpected = model.load_state_dict(expanded_state, strict=False)
    # Filter out gate_mlp (expected new), and factor_heads we've already added
    real_missing = [
        k for k in missing
        if not k.startswith("gate_mlp.") and not k.startswith("factor_heads.")
    ]
    if real_missing:
        print(f"WARN warm-start missing: {real_missing[:5]}")
    print(f"Warm-started from {args.warmstart_checkpoint} "
          f"(ep={ws.get('epoch')}, val={ws.get('val_loss'):.4f})")

    # CRITICAL: break symmetry on extra factor_heads AFTER warm-start load.
    # The load duplicated the single 229a factor_head into all K heads identically;
    # without symmetry break, the mixture gradient on the gate is zero (both heads
    # give same output, gate choice doesn't matter). Add small Gaussian noise.
    with torch.no_grad():
        for k in range(1, args.K):
            for p in model.factor_heads[k].parameters():
                if p.dim() >= 1:
                    p.add_(torch.randn_like(p) * 0.05 * p.std().clamp_min(0.01))
    print(f"Symmetry-broken factor_heads[1..{args.K - 1}] with 5% Gaussian noise")

    # Data
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - args.n_steps
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[: args.max_train_windows]

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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    config = {
        "type": "mixture_factor_ar_232a",
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
        "n_steps": args.n_steps,
        "n_members": args.n_members,
        "noise_skip": args.noise_skip,
        "d_scale": args.d_scale,
        "cell_spread": False,
        "decoder_layers": args.decoder_layers,
        "loss_type": args.loss_type,
        "no_tanh": args.no_tanh,
        "use_scale_anchor": False,  # off during training
        "scale_anchor_alpha": 0.50,
        "K": args.K,
        "gate_hidden": args.gate_hidden,
        "vov_feat_dim": 2,
        "gate_entropy_weight": args.gate_entropy_weight,
        "gate_usage_floor": args.gate_usage_floor,
        "gate_usage_floor_weight": args.gate_usage_floor_weight,
        "hard_routing_inference": args.hard_routing_inference,
        "warmstart_checkpoint": args.warmstart_checkpoint,
    }

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n232a Mixture Factor AR")
    print(f"  K={args.K}  gate_hidden={args.gate_hidden}")
    print(f"  entropy_w={args.gate_entropy_weight}  usage_floor={args.gate_usage_floor}")
    print(f"  hard_routing_inference={args.hard_routing_inference}")
    print(f"  decoder_hidden={args.decoder_hidden}  noise_skip={args.noise_skip}  d_scale={args.d_scale}")
    print(f"  n_members={args.n_members}  batch={args.batch_size}  lr={args.lr}")
    print(f"  train={train_hist.shape[0]}  val={val_hist.shape[0]}")
    print(f"  params: {n_params:,}")

    history_log: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        running = {"loss": 0.0, "main": 0.0, "vs": 0.0,
                   "entropy_penalty": 0.0, "entropy_value": 0.0,
                   "usage_penalty": 0.0, "gate_0": 0.0, "gate_1": 0.0}
        count = 0

        for history_01, future_01 in train_loader:
            trajectory, reg_terms = model.forward_train(
                history_01=history_01,
                n_members=args.n_members,
                n_steps=args.n_steps,
            )
            loss, metrics = compute_232a_loss(
                trajectory, future_01, reg_terms, args.lambda_vs, args.loss_type
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

        model.eval()
        val_sum = {"main": 0.0, "vs": 0.0}
        val_count = 0
        with torch.no_grad():
            for v_start in range(0, val_hist.shape[0], args.batch_size):
                v_end = min(v_start + args.batch_size, val_hist.shape[0])
                vh = val_hist[v_start:v_end]
                vf = val_future[v_start:v_end]
                vt, vreg = model.forward_train(
                    history_01=vh, n_members=args.n_members, n_steps=args.n_steps
                )
                _, vm = compute_232a_loss(vt, vf, vreg, args.lambda_vs, args.loss_type)
                vb = vh.shape[0]
                val_sum["main"] += vm["main"] * vb
                val_sum["vs"] += vm["vs"] * vb
                val_count += vb
        val_main = val_sum["main"] / val_count
        val_vs = val_sum["vs"] / val_count
        val_loss = val_main + args.lambda_vs * val_vs

        elapsed = time.time() - t0
        record = {
            "epoch": epoch, "elapsed": elapsed,
            "train_loss": train_metrics["loss"],
            "train_main": train_metrics["main"],
            "train_vs": train_metrics["vs"],
            "train_entropy_value": train_metrics["entropy_value"],
            "train_entropy_penalty": train_metrics["entropy_penalty"],
            "train_usage_penalty": train_metrics["usage_penalty"],
            "train_gate_0": train_metrics["gate_0"],
            "train_gate_1": train_metrics["gate_1"],
            "val_main": val_main, "val_vs": val_vs, "val_loss": val_loss,
        }
        history_log.append(make_serializable(record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "epoch": epoch, "val_loss": val_loss,
            }, out_dir / "best_model.pt")
        if epoch % args.checkpoint_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "epoch": epoch, "val_loss": val_loss,
            }, out_dir / f"checkpoint_ep{epoch}.pt")

        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"loss={train_metrics['loss']:.4f} main={train_metrics['main']:.4f} "
            f"vs={train_metrics['vs']:.4f} "
            f"gate=({train_metrics['gate_0']:.3f},{train_metrics['gate_1']:.3f}) "
            f"H={train_metrics['entropy_value']:.3f} "
            f"up={train_metrics['usage_penalty']:.4f} "
            f"val={val_loss:.4f}  ({elapsed:.1f}s)"
        )

    torch.save({
        "model_state_dict": model.state_dict(), "config": config,
        "epoch": args.epochs, "val_loss": val_loss,
    }, out_dir / "final_model.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)
    print(f"\nSaved to {out_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
