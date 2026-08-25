from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.joint_state_token_logit_transition_flow_matching import (
    StateAwareJointTokenLogitTransitionFlowMatching,
)
from diffusion.block_ar.joint_token_logit_transition_flow_matching import (
    JointTokenLogitTransitionFMConfig,
    _flow_time_features,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


@dataclass
class LatentStateJointTokenLogitTransitionFMConfig(JointTokenLogitTransitionFMConfig):
    latent_tokens: int = 8
    latent_loss_weight: float = 1.0


class LatentStateJointTokenPathVelocity(nn.Module):
    def __init__(self, cfg: LatentStateJointTokenLogitTransitionFMConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(2, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.latent_embed = nn.Parameter(torch.zeros(cfg.latent_tokens, cfg.token_dim))
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
        self.path_out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )
        self.latent_out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, cfg.token_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        state_logits: torch.Tensor,
        latent_t: torch.Tensor,
        context: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, horizon, n_cells = x_t.shape
        h_idx = torch.arange(horizon, device=x_t.device)
        c_idx = torch.arange(n_cells, device=x_t.device)
        pos = (
            self.horizon_embed(h_idx)[:, None, :]
            + self.cell_embed(c_idx)[None, :, :]
        ).reshape(horizon * n_cells, self.cfg.token_dim)
        values = torch.stack([x_t, state_logits], dim=-1).reshape(bsz, horizon * n_cells, 2)
        path_token = self.value_proj(values) + pos[None, :, :]
        latent_token = latent_t + self.latent_embed[None, :, :]
        ctx_token = self.context_proj(context) + self.prefix_embed[0][None, :]
        time_token = self.flow_time_proj(_flow_time_features(flow_t, self.cfg.flow_time_dim)) + self.prefix_embed[1][None, :]
        hidden = torch.cat(
            [torch.stack([ctx_token, time_token], dim=1), latent_token, path_token],
            dim=1,
        )
        hidden = self.mixer(hidden)
        latent_hidden = hidden[:, 2 : 2 + self.cfg.latent_tokens, :]
        path_hidden = hidden[:, 2 + self.cfg.latent_tokens :, :]
        path_vel = self.path_out(path_hidden).squeeze(-1).view(bsz, horizon, n_cells)
        latent_vel = self.latent_out(latent_hidden)
        return path_vel, latent_vel


class LatentStateJointTokenLogitTransitionFlowMatching(StateAwareJointTokenLogitTransitionFlowMatching):
    """305a-v0: state-aware joint path flow with stochastic latent prefix tokens."""

    def __init__(self, cfg: LatentStateJointTokenLogitTransitionFMConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.velocity = LatentStateJointTokenPathVelocity(cfg)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        latent_t: torch.Tensor,
        history_norm: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_logits = self.implied_state_logits(history_norm, x_t)
        return self.velocity(x_t, state_logits, latent_t, context, t)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        x1 = self.target_transitions(history_norm, future_norm)
        x0 = torch.randn_like(x1)
        bsz = history_norm.shape[0]
        latent0 = torch.randn(
            bsz,
            self.cfg.latent_tokens,
            self.cfg.token_dim,
            device=history_norm.device,
            dtype=history_norm.dtype,
        )
        latent1 = torch.zeros_like(latent0)
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        latent_t = (1.0 - t)[:, None, None] * latent0 + t[:, None, None] * latent1
        target_path_velocity = x1 - x0
        target_latent_velocity = latent1 - latent0
        pred_path_velocity, pred_latent_velocity = self.predict_velocity(
            x_t,
            latent_t,
            history_norm,
            context,
            t,
        )
        path_loss = F.mse_loss(pred_path_velocity, target_path_velocity)
        latent_loss = F.mse_loss(pred_latent_velocity, target_latent_velocity)
        total = path_loss + float(self.cfg.latent_loss_weight) * latent_loss
        state_logits = self.implied_state_logits(history_norm, x_t)
        metrics = {
            "total": total.detach(),
            "path_loss": path_loss.detach(),
            "latent_loss": latent_loss.detach(),
            "transition_std": x1.std(unbiased=False).detach(),
            "transition_abs": x1.abs().mean().detach(),
            "state_logit_abs": state_logits.abs().mean().detach(),
            "target_velocity_std": target_path_velocity.std(unbiased=False).detach(),
        }
        return total, metrics

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
        history_norm = self._flatten(history_norm)
        context = self.encode_history(history_norm)
        bsz = history_norm.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            x = temp * torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=context.device,
                dtype=context.dtype,
            )
            latent = temp * torch.randn(
                bsz * k,
                self.cfg.latent_tokens,
                self.cfg.token_dim,
                device=context.device,
                dtype=context.dtype,
            )
            hist = history_norm.repeat_interleave(k, dim=0)
            ctx = context.repeat_interleave(k, dim=0)
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=context.device,
                    dtype=context.dtype,
                )
                path_vel, latent_vel = self.predict_velocity(x, latent, hist, ctx, t)
                x = x + dt * path_vel
                latent = latent + dt * latent_vel
            future_01 = self.transitions_to_future(hist, x)
            outs.append(future_01.view(bsz, k, self.cfg.future_len, 5, 5))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[LatentStateJointTokenLogitTransitionFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LatentStateJointTokenLogitTransitionFMConfig(**payload["config"])
    model = LatentStateJointTokenLogitTransitionFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: LatentStateJointTokenLogitTransitionFlowMatching,
    cfg: LatentStateJointTokenLogitTransitionFMConfig,
    epoch: int,
    best_val: float,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
