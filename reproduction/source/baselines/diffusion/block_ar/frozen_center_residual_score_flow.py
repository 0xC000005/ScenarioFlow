from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
    load_model as load_empirical_score_ar_model,
)
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.joint_token_logit_transition_flow_matching import _flow_time_features
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


@dataclass
class FrozenCenterResidualScoreFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    context_dim: int = 128
    history_hidden: int = 128
    center_hidden: int = 128
    encoder_dropout: float = 0.1
    token_dim: int = 128
    token_layers: int = 3
    token_heads: int = 4
    token_ff: int = 256
    model_dropout: float = 0.1
    flow_time_dim: int = 32
    flow_steps: int = 24
    sample_temperature: float = 1.0
    transport_strength: float = 1.0
    max_sample_chunk: int = 16
    center_samples: int = 8
    center_chunk_size: int = 4


class FrozenCenterResidualVelocity(nn.Module):
    def __init__(self, cfg: FrozenCenterResidualScoreFMConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(4, cfg.token_dim)
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
        center_scores: torch.Tensor,
        context: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, horizon, n_cells = residual_t.shape
        if horizon != self.cfg.future_len or n_cells != self.cfg.n_cells:
            raise ValueError(
                f"Expected residual path (*,{self.cfg.future_len},{self.cfg.n_cells}), "
                f"got {tuple(residual_t.shape)}"
            )
        h_idx = torch.arange(horizon, device=residual_t.device)
        c_idx = torch.arange(n_cells, device=residual_t.device)
        pos = (
            self.horizon_embed(h_idx)[:, None, :]
            + self.cell_embed(c_idx)[None, :, :]
        ).reshape(horizon * n_cells, self.cfg.token_dim)

        center_prev = torch.cat([center_scores[:, :1], center_scores[:, :-1]], dim=1)
        center_change = center_scores - center_prev
        total_scores = center_scores + residual_t
        values = torch.stack(
            [residual_t, center_scores, total_scores, center_change],
            dim=-1,
        ).reshape(bsz, horizon * n_cells, 4)

        token = self.value_proj(values) + pos[None, :, :]
        ctx_token = self.context_proj(context) + self.prefix_embed[0][None, :]
        time_token = (
            self.flow_time_proj(_flow_time_features(flow_t, self.cfg.flow_time_dim))
            + self.prefix_embed[1][None, :]
        )
        hidden = torch.cat([torch.stack([ctx_token, time_token], dim=1), token], dim=1)
        hidden = self.mixer(hidden)
        velocity = self.out(hidden[:, 2:, :]).squeeze(-1)
        return velocity.view(bsz, horizon, n_cells)


class FrozenCenterResidualScoreFlow(nn.Module):
    """470a: vanilla residual flow around a frozen learned empirical-score center."""

    def __init__(self, cfg: FrozenCenterResidualScoreFMConfig):
        super().__init__()
        self.cfg = cfg
        hist_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            gru_hidden_dim=cfg.history_hidden,
            bottleneck_dim=cfg.context_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=0.0,
        )
        center_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            gru_hidden_dim=cfg.center_hidden,
            bottleneck_dim=cfg.context_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=0.0,
        )
        self.history_encoder = GRUEncoder(hist_cfg)
        self.center_encoder = GRUEncoder(center_cfg)
        self.context_fuse = nn.Sequential(
            nn.LayerNorm(2 * cfg.context_dim),
            nn.Linear(2 * cfg.context_dim, cfg.context_dim),
            nn.GELU(),
            nn.Linear(cfg.context_dim, cfg.context_dim),
        )
        self.velocity = FrozenCenterResidualVelocity(cfg)

    def encode_condition(
        self,
        history_scores: torch.Tensor,
        center_scores: torch.Tensor,
    ) -> torch.Tensor:
        hist_ctx = self.history_encoder(history_scores)
        center_ctx = self.center_encoder(center_scores)
        return self.context_fuse(torch.cat([hist_ctx, center_ctx], dim=-1))

    def training_loss(
        self,
        history_scores: torch.Tensor,
        center_scores: torch.Tensor,
        future_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual = future_scores - center_scores
        noise = torch.randn_like(residual)
        bsz = residual.shape[0]
        t = torch.rand(bsz, device=residual.device, dtype=residual.dtype)
        residual_t = (1.0 - t)[:, None, None] * noise + t[:, None, None] * residual
        target_velocity = residual - noise
        context = self.encode_condition(history_scores, center_scores)
        pred_velocity = self.velocity(residual_t, center_scores, context, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        total_scores = center_scores + residual
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "center_score_std": center_scores.std(unbiased=False).detach(),
            "residual_score_std": residual.std(unbiased=False).detach(),
            "residual_score_abs": residual.abs().mean().detach(),
            "future_score_std": total_scores.std(unbiased=False).detach(),
        }
        return fm_loss, metrics

    @torch.no_grad()
    def sample_residual_scores(
        self,
        history_scores: torch.Tensor,
        center_scores: torch.Tensor,
        n_samples: int = 50,
        temperature: float | None = None,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        bsz = history_scores.shape[0]
        chunk_size = max(
            1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk))
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        context = self.encode_condition(history_scores, center_scores)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            residual = temp * torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=history_scores.device,
                dtype=history_scores.dtype,
            )
            center = center_scores.repeat_interleave(k, dim=0)
            ctx = context.repeat_interleave(k, dim=0)
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=history_scores.device,
                    dtype=history_scores.dtype,
                )
                residual = residual + dt * self.velocity(residual, center, ctx, t)
            outs.append(residual.view(bsz, k, self.cfg.future_len, self.cfg.n_cells))
        return torch.cat(outs, dim=1)


class FrozenCenterResidualScenarioGenerator(nn.Module):
    def __init__(
        self,
        residual_flow: FrozenCenterResidualScoreFlow,
        base_model: nn.Module,
        base_checkpoint_path: str,
    ):
        super().__init__()
        self.residual_flow = residual_flow
        self.base_model = base_model.eval()
        self.base_checkpoint_path = base_checkpoint_path
        self.cfg = residual_flow.cfg
        for param in self.base_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def estimate_center_scores(
        self,
        history_norm: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        center_samples = self.base_model.sample_batched(
            history_norm,
            n_samples=int(self.cfg.center_samples),
            n_steps=n_steps,
            chunk_size=int(self.cfg.center_chunk_size),
            history_is_normalized=True,
        )
        center_01 = center_samples.median(dim=1).values
        center_norm = normalize_iv(center_01).view(center_01.shape[0], n_steps, -1)
        return self.base_model.target_future_scores(center_norm)

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
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = history_norm.view(history_norm.shape[0], history_norm.shape[1], -1)
        history_scores = self.base_model.history_scores(history_norm)
        center_scores = self.estimate_center_scores(history_norm, n_steps=n_steps)
        residual = self.residual_flow.sample_residual_scores(
            history_scores=history_scores,
            center_scores=center_scores,
            n_samples=n_samples,
            temperature=temperature,
            chunk_size=chunk_size,
        )
        future_scores = center_scores[:, None, :, :] + residual
        flat = future_scores.reshape(
            future_scores.shape[0] * future_scores.shape[1],
            n_steps,
            self.cfg.n_cells,
        )
        future_01 = self.base_model._scores_to_values(flat)
        if self.cfg.n_cells == 25:
            return future_01.view(future_scores.shape[0], future_scores.shape[1], n_steps, 5, 5)
        side = int(math.sqrt(self.cfg.n_cells))
        if side * side == self.cfg.n_cells:
            return future_01.view(
                future_scores.shape[0],
                future_scores.shape[1],
                n_steps,
                side,
                side,
            )
        return future_01.view(
            future_scores.shape[0],
            future_scores.shape[1],
            n_steps,
            self.cfg.n_cells,
        )


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[FrozenCenterResidualScenarioGenerator, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = FrozenCenterResidualScoreFMConfig(**payload["config"])
    residual_flow = FrozenCenterResidualScoreFlow(cfg)
    residual_flow.load_state_dict(payload["model_state_dict"], strict=True)
    base_checkpoint_path = payload["base_checkpoint_path"]
    base_model, _base_payload = load_empirical_score_ar_model(base_checkpoint_path, device)
    model = FrozenCenterResidualScenarioGenerator(
        residual_flow=residual_flow.to(device),
        base_model=base_model,
        base_checkpoint_path=base_checkpoint_path,
    )
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: FrozenCenterResidualScoreFlow,
    cfg: FrozenCenterResidualScoreFMConfig,
    epoch: int,
    best_val: float,
    base_checkpoint_path: str,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
            "base_checkpoint_path": base_checkpoint_path,
        },
        path,
    )
