from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.deterministic_obs_encoded_latent_world_model import (
    load_model as load_289c_model,
)
from diffusion.block_ar.joint_token_logit_transition_flow_matching import (
    JointTokenLogitTransitionFMConfig,
    JointTokenLogitTransitionFlowMatching,
    _flow_time_features,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


class CenterConditionedResidualPathVelocity(nn.Module):
    def __init__(self, cfg: JointTokenLogitTransitionFMConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(3, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.prefix_embed = nn.Parameter(torch.zeros(2, cfg.token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.token_heads,
            dim_feedforward=cfg.token_ff,
            dropout=cfg.model_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=cfg.token_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def forward(
        self,
        residual_t: torch.Tensor,
        center_transitions: torch.Tensor,
        last_logit: torch.Tensor,
        context: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, horizon, n_cells = residual_t.shape
        h_idx = torch.arange(horizon, device=residual_t.device)
        c_idx = torch.arange(n_cells, device=residual_t.device)
        pos = (
            self.horizon_embed(h_idx)[:, None, :]
            + self.cell_embed(c_idx)[None, :, :]
        ).reshape(horizon * n_cells, self.cfg.token_dim)
        total_transitions = center_transitions + residual_t
        total_state_logits = last_logit[:, None, :] + total_transitions.cumsum(dim=1)
        values = torch.stack(
            [residual_t, center_transitions, total_state_logits], dim=-1
        ).reshape(bsz, horizon * n_cells, 3)
        token = self.value_proj(values) + pos[None, :, :]
        ctx_token = self.context_proj(context) + self.prefix_embed[0][None, :]
        time_token = self.flow_time_proj(
            _flow_time_features(flow_t, self.cfg.flow_time_dim)
        ) + self.prefix_embed[1][None, :]
        hidden = torch.cat([torch.stack([ctx_token, time_token], dim=1), token], dim=1)
        hidden = self.mixer(hidden)
        vel = self.out(hidden[:, 2:, :]).squeeze(-1)
        return vel.view(bsz, horizon, n_cells)


class CenterConditionedResidualLogitTransitionFlowMatching(
    JointTokenLogitTransitionFlowMatching
):
    """309a-v0: vanilla joint residual flow around a learned deterministic center path."""

    def __init__(self, cfg: JointTokenLogitTransitionFMConfig):
        super().__init__(cfg)
        self.velocity = CenterConditionedResidualPathVelocity(cfg)

    def center_transitions(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
    ) -> torch.Tensor:
        return self.target_transitions(history_norm, center_future_norm)

    def target_residual(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> torch.Tensor:
        target = self.target_transitions(history_norm, future_norm)
        center = self.center_transitions(history_norm, center_future_norm)
        return target - center

    def predict_velocity(
        self,
        residual_t: torch.Tensor,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        center_future_norm = self._flatten(center_future_norm)
        center = self.center_transitions(history_norm, center_future_norm)
        last_logit = self.history_last_logit(history_norm)
        return self.velocity(residual_t, center, last_logit, context, t)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        center_future_norm = self._flatten(center_future_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        center = self.center_transitions(history_norm, center_future_norm)
        residual = self.target_residual(history_norm, center_future_norm, future_norm)
        noise = torch.randn_like(residual)
        bsz = history_norm.shape[0]
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        residual_t = (1.0 - t)[:, None, None] * noise + t[:, None, None] * residual
        target_velocity = residual - noise
        pred_velocity = self.predict_velocity(
            residual_t, history_norm, center_future_norm, context, t
        )
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        total_transitions = center + residual
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "center_transition_std": center.std(unbiased=False).detach(),
            "center_transition_abs": center.abs().mean().detach(),
            "residual_std": residual.std(unbiased=False).detach(),
            "residual_abs": residual.abs().mean().detach(),
            "total_transition_std": total_transitions.std(unbiased=False).detach(),
        }
        return fm_loss, metrics

    @torch.no_grad()
    def sample_with_center(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
        n_samples: int = 50,
        temperature: float | None = None,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        center_future_norm = self._flatten(center_future_norm)
        context = self.encode_history(history_norm)
        center = self.center_transitions(history_norm, center_future_norm)
        bsz = history_norm.shape[0]
        chunk_size = max(
            1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk))
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            residual = temp * torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=context.device,
                dtype=context.dtype,
            )
            hist = history_norm.repeat_interleave(k, dim=0)
            cen_future = center_future_norm.repeat_interleave(k, dim=0)
            ctx = context.repeat_interleave(k, dim=0)
            cen = center.repeat_interleave(k, dim=0)
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=context.device,
                    dtype=context.dtype,
                )
                residual = residual + dt * self.predict_velocity(
                    residual, hist, cen_future, ctx, t
                )
            future_01 = self.transitions_to_future(hist, cen + residual)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(
                    bsz, k, self.cfg.future_len, 5, 5
                )
            else:
                future_01 = future_01.view(
                    bsz, k, self.cfg.future_len, self.cfg.n_cells
                )
            outs.append(future_01)
        return torch.cat(outs, dim=1)


class ObsWorldBackboneCenterResidualFlowGenerator(nn.Module):
    def __init__(
        self,
        flow: CenterConditionedResidualLogitTransitionFlowMatching,
        backbone: nn.Module,
        backbone_checkpoint_path: str,
    ):
        super().__init__()
        self.flow = flow
        self.backbone = backbone.eval()
        self.backbone_checkpoint_path = backbone_checkpoint_path
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.flow.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.flow.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        center_future_01 = self.backbone.sample_batched(
            history_norm,
            n_samples=1,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
        ).squeeze(1)
        center_future_norm = normalize_iv(center_future_01)
        return self.flow.sample_with_center(
            history_norm,
            center_future_norm,
            n_samples=n_samples,
            temperature=temperature,
            chunk_size=chunk_size,
        )


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ObsWorldBackboneCenterResidualFlowGenerator, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = JointTokenLogitTransitionFMConfig(**payload["config"])
    flow = CenterConditionedResidualLogitTransitionFlowMatching(cfg)
    flow.load_state_dict(payload["model_state_dict"], strict=True)
    backbone_checkpoint_path = payload["backbone_checkpoint_path"]
    backbone, _ = load_289c_model(backbone_checkpoint_path, device)
    model = ObsWorldBackboneCenterResidualFlowGenerator(
        flow.to(device), backbone, backbone_checkpoint_path
    )
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: CenterConditionedResidualLogitTransitionFlowMatching,
    cfg: JointTokenLogitTransitionFMConfig,
    epoch: int,
    best_val: float,
    backbone_checkpoint_path: str,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
            "backbone_checkpoint_path": backbone_checkpoint_path,
        },
        path,
    )
