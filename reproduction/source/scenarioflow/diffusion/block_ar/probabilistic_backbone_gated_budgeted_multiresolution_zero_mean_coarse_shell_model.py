from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    load_model as load_277d_model,
)
from diffusion.block_ar.probabilistic_backbone_budgeted_multiresolution_zero_mean_coarse_shell_model import (
    BackboneBudgetedMultiresolutionZeroMeanCoarseShellGenerator,
    BudgetedMultiresolutionZeroMeanCoarseShellCore,
    ProbabilisticBackboneBudgetedMultiresolutionZeroMeanCoarseShellConfig,
)
from diffusion.block_ar.probabilistic_backbone_residual_shell_model import (
    compute_backbone_stats,
    compute_residual_coords,
)
from diffusion.block_ar.probabilistic_backbone_zero_mean_coarse_shell_model import (
    project_residual_controls,
)


@dataclass
class ProbabilisticBackboneGatedBudgetedMultiresolutionZeroMeanCoarseShellConfig(
    ProbabilisticBackboneBudgetedMultiresolutionZeroMeanCoarseShellConfig
):
    fast_knot_max: int = 5
    gate_floor: float = 0.02
    gate_bias_init: float = 0.0


class GatedBudgetedMultiresolutionZeroMeanCoarseShellCore(
    BudgetedMultiresolutionZeroMeanCoarseShellCore
):
    """296g-v0 shell: 296f budgeted shell plus one window-level fast-shell gate."""

    cfg: ProbabilisticBackboneGatedBudgetedMultiresolutionZeroMeanCoarseShellConfig

    def __init__(
        self,
        cfg: ProbabilisticBackboneGatedBudgetedMultiresolutionZeroMeanCoarseShellConfig,
    ):
        super().__init__(cfg)
        self.fast_gate_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 1),
        )
        nn.init.zeros_(self.fast_gate_head[-1].weight)
        nn.init.constant_(self.fast_gate_head[-1].bias, cfg.gate_bias_init)

        fast_mask = torch.tensor(
            [1.0 if k <= cfg.fast_knot_max else 0.0 for k in cfg.knot_positions],
            dtype=torch.float32,
        ).view(1, self.n_knots, 1)
        self.register_buffer("fast_knot_mask", fast_mask)

    def predict_knot_scales_and_gate(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        center_raw, center_coords, center_scales = compute_backbone_stats(
            history_norm,
            center_future_norm,
            eps=self.cfg.change_scale_eps,
            change_coord=self.cfg.change_coord,
        )
        enc = self.encode(history_norm, center_future_norm, center_coords)
        future_enc = enc[:, self.cfg.history_len :]
        knot_idx = torch.tensor(
            [k - 1 for k in self.cfg.knot_positions],
            device=future_enc.device,
            dtype=torch.long,
        )
        knot_states = future_enc.index_select(1, knot_idx)
        raw_knot_scale = F.softplus(self.raw_scale_head(knot_states)) + self.cfg.scale_floor

        pooled = future_enc.mean(dim=1)
        budget = F.softplus(self.budget_head(pooled)) + self.cfg.budget_floor
        gate_raw = torch.sigmoid(self.fast_gate_head(pooled))
        fast_gate = self.cfg.gate_floor + (1.0 - self.cfg.gate_floor) * gate_raw

        gate_multiplier = 1.0 + self.fast_knot_mask * (fast_gate[:, None, :] - 1.0)
        gated_raw_knot_scale = raw_knot_scale * gate_multiplier

        raw_daily_scale = torch.einsum("tk,bkc->btc", self.knot_matrix, gated_raw_knot_scale)
        raw_budget = raw_daily_scale.mean(dim=1).clamp_min(self.cfg.scale_floor)
        renorm = (budget / raw_budget).unsqueeze(1)
        knot_scale = gated_raw_knot_scale * renorm
        daily_scale = raw_daily_scale * renorm
        return knot_scale, center_raw, center_scales, budget, daily_scale, fast_gate

    def predict_knot_scales(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        knot_scale, center_raw, center_scales, budget, daily_scale, _fast_gate = (
            self.predict_knot_scales_and_gate(history_norm, center_future_norm)
        )
        return knot_scale, center_raw, center_scales, budget, daily_scale

    def training_loss(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual_coords = compute_residual_coords(
            history_norm,
            center_future_norm,
            future_norm,
            eps=self.cfg.change_scale_eps,
            change_coord=self.cfg.change_coord,
        )
        target_controls = project_residual_controls(residual_coords, self.knot_pinv)
        knot_scale, center_raw, _center_scales, budget, daily_scale, fast_gate = (
            self.predict_knot_scales_and_gate(history_norm, center_future_norm)
        )
        z = target_controls / knot_scale
        nll = 0.5 * (z.pow(2) + 2.0 * torch.log(knot_scale))
        loss = nll.mean()
        metrics = {
            "total": loss.detach(),
            "control_abs_mean": target_controls.abs().mean().detach(),
            "pred_scale_mean": knot_scale.mean().detach(),
            "pred_scale_daily_mean": daily_scale.mean().detach(),
            "budget_mean": budget.mean().detach(),
            "fast_gate_mean": fast_gate.mean().detach(),
            "center_raw_abs_mean": center_raw.abs().mean().detach(),
            "z_abs_mean": z.abs().mean().detach(),
        }
        return loss, metrics


class BackboneGatedBudgetedMultiresolutionZeroMeanCoarseShellGenerator(
    BackboneBudgetedMultiresolutionZeroMeanCoarseShellGenerator
):
    shell: GatedBudgetedMultiresolutionZeroMeanCoarseShellCore


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[BackboneGatedBudgetedMultiresolutionZeroMeanCoarseShellGenerator, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ProbabilisticBackboneGatedBudgetedMultiresolutionZeroMeanCoarseShellConfig(
        **payload["config"]
    )
    shell = GatedBudgetedMultiresolutionZeroMeanCoarseShellCore(cfg)
    shell.load_state_dict(payload["shell_state_dict"], strict=True)
    backbone_checkpoint_path = payload["backbone_checkpoint_path"]
    backbone, _ = load_277d_model(backbone_checkpoint_path, device)
    model = BackboneGatedBudgetedMultiresolutionZeroMeanCoarseShellGenerator(
        shell.to(device), backbone, backbone_checkpoint_path
    )
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    shell: GatedBudgetedMultiresolutionZeroMeanCoarseShellCore,
    cfg: ProbabilisticBackboneGatedBudgetedMultiresolutionZeroMeanCoarseShellConfig,
    epoch: int,
    best_val: float,
    backbone_checkpoint_path: str,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "shell_state_dict": shell.state_dict(),
            "backbone_checkpoint_path": backbone_checkpoint_path,
        },
        path,
    )
