from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.deterministic_obs_encoded_latent_world_model import (
    load_model as load_289c_model,
)
from diffusion.block_ar.probabilistic_backbone_residual_shell_model import (
    compute_backbone_stats,
    compute_residual_coords,
    flatten_panels,
)
from diffusion.block_ar.probabilistic_backbone_zero_mean_coarse_shell_model import (
    build_knot_interpolation_matrix,
    project_residual_controls,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class ProbabilisticObsWorldProfiledZeroMeanCoarseShellConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    d_model: int = 160
    nhead: int = 5
    num_encoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1

    change_coord: str = "asinh_local_scale"
    change_scale_eps: float = 1e-3

    knot_positions: tuple[int, ...] = (1, 5, 10, 15, 22, 30)
    scale_floor: float = 0.02
    profile_log_amp: float = 0.8
    sample_scale_mult: float = 1.0


class ObsWorldProfiledZeroMeanCoarseShellCore(nn.Module):
    """308a-v0 shell: 296c-style profiled shell around a learned world-model center."""

    def __init__(self, cfg: ProbabilisticObsWorldProfiledZeroMeanCoarseShellConfig):
        super().__init__()
        self.cfg = cfg
        self.n_knots = len(cfg.knot_positions)

        self.history_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.cond_proj = nn.Linear(2 * cfg.n_cells, cfg.d_model)
        self.history_pos = nn.Embedding(cfg.history_len, cfg.d_model)
        self.future_pos = nn.Embedding(cfg.future_len, cfg.d_model)
        self.segment_embed = nn.Embedding(2, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_encoder_layers)
        self.scale_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.n_cells),
        )
        self.profile_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, self.n_knots),
        )

        knot_matrix = build_knot_interpolation_matrix(cfg.future_len, cfg.knot_positions)
        self.register_buffer("knot_matrix", knot_matrix)
        self.register_buffer("knot_pinv", torch.linalg.pinv(knot_matrix))

    def encode(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
        center_coords: torch.Tensor,
    ) -> torch.Tensor:
        history = flatten_panels(history_norm)
        center_future = flatten_panels(center_future_norm)
        pos_h = torch.arange(self.cfg.history_len, device=history.device).unsqueeze(0)
        seg_h = torch.zeros_like(pos_h)
        hist_tokens = (
            self.history_proj(history)
            + self.history_pos(pos_h)
            + self.segment_embed(seg_h)
        )

        center_feats = torch.cat([center_future, center_coords], dim=-1)
        pos_f = torch.arange(self.cfg.future_len, device=history.device).unsqueeze(0)
        seg_f = torch.ones_like(pos_f)
        future_tokens = (
            self.cond_proj(center_feats)
            + self.future_pos(pos_f)
            + self.segment_embed(seg_f)
        )
        seq = torch.cat([hist_tokens, future_tokens], dim=1)
        return self.encoder(seq)

    def predict_knot_scales(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        base_log_scale = self.scale_head(knot_states)
        base_scale = F.softplus(base_log_scale) + self.cfg.scale_floor

        pooled = future_enc.mean(dim=1)
        profile_logits = self.profile_head(pooled)
        profile_scale = torch.exp(
            self.cfg.profile_log_amp * torch.tanh(profile_logits)
        ).unsqueeze(-1)
        knot_scale = base_scale * profile_scale
        return knot_scale, center_raw, center_scales, profile_scale

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
        knot_scale, center_raw, _center_scales, profile_scale = self.predict_knot_scales(
            history_norm, center_future_norm
        )
        z = target_controls / knot_scale
        nll = 0.5 * (z.pow(2) + 2.0 * torch.log(knot_scale))
        loss = nll.mean()

        daily_std = torch.einsum("tk,bkc->btc", self.knot_matrix, knot_scale)
        metrics = {
            "total": loss.detach(),
            "control_abs_mean": target_controls.abs().mean().detach(),
            "pred_scale_mean": knot_scale.mean().detach(),
            "pred_scale_daily_mean": daily_std.mean().detach(),
            "profile_scale_mean": profile_scale.mean().detach(),
            "profile_scale_h1": profile_scale[:, 0].mean().detach(),
            "center_raw_abs_mean": center_raw.abs().mean().detach(),
            "z_abs_mean": z.abs().mean().detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample_with_center(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
        n_samples: int,
        sample_scale_mult: float | None = None,
    ) -> torch.Tensor:
        history_norm = flatten_panels(history_norm)
        center_future_norm = flatten_panels(center_future_norm)
        bsz = history_norm.shape[0]
        scale_mult = (
            self.cfg.sample_scale_mult if sample_scale_mult is None else float(sample_scale_mult)
        )

        knot_scale, center_raw, center_scales, _profile_scale = self.predict_knot_scales(
            history_norm, center_future_norm
        )
        knot_scale = knot_scale * scale_mult

        half = n_samples // 2
        parts: list[torch.Tensor] = []
        if half > 0:
            eps = torch.randn(
                bsz,
                half,
                self.n_knots,
                self.cfg.n_cells,
                device=history_norm.device,
            )
            parts.extend([eps, -eps])
        if n_samples % 2 == 1:
            parts.append(
                torch.zeros(
                    bsz,
                    1,
                    self.n_knots,
                    self.cfg.n_cells,
                    device=history_norm.device,
                )
            )
        noise = torch.cat(parts, dim=1)
        knot_controls = noise * knot_scale.unsqueeze(1)
        daily_coords = torch.einsum("tk,bskc->bstc", self.knot_matrix, knot_controls)
        residual_raw = torch.sinh(daily_coords) * center_scales.unsqueeze(1)
        total_raw = center_raw.unsqueeze(1) + residual_raw

        curr = history_norm[:, -1].unsqueeze(1).expand(-1, n_samples, -1).clone()
        outputs: list[torch.Tensor] = []
        for step in range(self.cfg.future_len):
            next_level = torch.clamp(curr + total_raw[:, :, step], -1.0, 1.0)
            outputs.append(next_level)
            curr = next_level
        future_norm = torch.stack(outputs, dim=2)
        future_01 = denormalize_iv(future_norm)
        if self.cfg.n_cells == 25:
            future_01 = future_01.view(bsz, n_samples, self.cfg.future_len, 5, 5)
        else:
            future_01 = future_01.view(
                bsz, n_samples, self.cfg.future_len, self.cfg.n_cells
            )
        return future_01.contiguous()


class ObsWorldBackboneProfiledZeroMeanCoarseShellGenerator(nn.Module):
    def __init__(
        self,
        shell: ObsWorldProfiledZeroMeanCoarseShellCore,
        backbone: nn.Module,
        backbone_checkpoint_path: str,
    ):
        super().__init__()
        self.shell = shell
        self.backbone = backbone.eval()
        self.backbone_checkpoint_path = backbone_checkpoint_path
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        sample_scale_mult: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.shell.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.shell.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = flatten_panels(history_norm)
        center_future_01 = self.backbone.sample_batched(
            history_norm,
            n_samples=1,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
        ).squeeze(1)
        center_future_norm = normalize_iv(center_future_01)
        return self.shell.sample_with_center(
            history_norm,
            center_future_norm,
            n_samples=n_samples,
            sample_scale_mult=sample_scale_mult,
        )


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ObsWorldBackboneProfiledZeroMeanCoarseShellGenerator, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ProbabilisticObsWorldProfiledZeroMeanCoarseShellConfig(**payload["config"])
    shell = ObsWorldProfiledZeroMeanCoarseShellCore(cfg)
    shell.load_state_dict(payload["shell_state_dict"], strict=True)
    backbone_checkpoint_path = payload["backbone_checkpoint_path"]
    backbone, _ = load_289c_model(backbone_checkpoint_path, device)
    model = ObsWorldBackboneProfiledZeroMeanCoarseShellGenerator(
        shell.to(device), backbone, backbone_checkpoint_path
    )
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    shell: ObsWorldProfiledZeroMeanCoarseShellCore,
    cfg: ProbabilisticObsWorldProfiledZeroMeanCoarseShellConfig,
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
