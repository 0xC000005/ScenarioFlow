from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.joint_token_logit_transition_flow_matching import (
    JointTokenLogitTransitionFMConfig,
    _flow_time_features,
)
from diffusion.block_ar.latent_bottleneck_future_logit_path_flow import (
    FuturePathEncoder,
    LatentTokenVelocity,
)
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class RecurrentLatentStateFutureLogitPathFlowConfig(JointTokenLogitTransitionFMConfig):
    latent_tokens: int = 12
    latent_dim: int = 48
    standardize_logits: bool = True
    logit_std_floor: float = 1e-3
    recon_weight: float = 1.0


class CellMixerBlock(nn.Module):
    def __init__(self, cfg: RecurrentLatentStateFutureLogitPathFlowConfig):
        super().__init__()
        self.cell_norm = nn.LayerNorm(cfg.n_cells)
        self.cell_mlp = nn.Sequential(
            nn.Linear(cfg.n_cells, 2 * cfg.n_cells),
            nn.GELU(),
            nn.Dropout(cfg.model_dropout),
            nn.Linear(2 * cfg.n_cells, cfg.n_cells),
        )
        self.channel_norm = nn.LayerNorm(cfg.token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.token_ff),
            nn.GELU(),
            nn.Dropout(cfg.model_dropout),
            nn.Linear(cfg.token_ff, cfg.token_dim),
        )
        self.dropout = nn.Dropout(cfg.model_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, cell, channel)
        y = self.cell_norm(x.transpose(1, 2))
        y = self.cell_mlp(y).transpose(1, 2)
        x = x + self.dropout(y)
        x = x + self.dropout(self.channel_mlp(self.channel_norm(x)))
        return x


class RecurrentLatentStateDecoder(nn.Module):
    def __init__(self, cfg: RecurrentLatentStateFutureLogitPathFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.state_proj = nn.Linear(1, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.token_dim)
        self.latent_pos = nn.Embedding(cfg.latent_tokens, cfg.token_dim)
        self.cross_attn = nn.MultiheadAttention(
            cfg.token_dim,
            cfg.token_heads,
            dropout=cfg.model_dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(cfg.token_dim)
        self.latent_norm = nn.LayerNorm(cfg.token_dim)
        self.blocks = nn.ModuleList([CellMixerBlock(cfg) for _ in range(cfg.token_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def latent_tokens(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        n_latent = latent.shape[1]
        l_idx = torch.arange(n_latent, device=latent.device)
        ctx = self.context_proj(context)
        z = self.latent_proj(latent)
        z = z + self.latent_pos(l_idx)[None, :, :]
        z = z + ctx[:, None, :]
        return self.latent_norm(z)

    def step(
        self,
        current_coord: torch.Tensor,
        latent_tokens: torch.Tensor,
        context: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        bsz, n_cells = current_coord.shape
        c_idx = torch.arange(n_cells, device=current_coord.device)
        ctx = self.context_proj(context)
        step = torch.full((bsz,), step_idx, device=current_coord.device, dtype=torch.long)
        x = self.state_proj(current_coord.unsqueeze(-1))
        x = x + self.cell_embed(c_idx)[None, :, :]
        x = x + self.horizon_embed(step)[:, None, :]
        x = x + ctx[:, None, :]
        attn, _ = self.cross_attn(
            self.query_norm(x),
            latent_tokens,
            latent_tokens,
            need_weights=False,
        )
        x = x + attn
        for block in self.blocks:
            x = block(x)
        return self.out(x).squeeze(-1)

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        initial_coord: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.latent_tokens(latent, context)
        current = initial_coord
        frames: list[torch.Tensor] = []
        for step_idx in range(self.cfg.future_len):
            delta = self.step(current, memory, context, step_idx)
            current = current + delta
            frames.append(current)
        return torch.stack(frames, dim=1)


class RecurrentLatentStateFutureLogitPathFlow(nn.Module):
    """328a-v0: learned latent bottleneck with recurrent state-feedback decoder."""

    def __init__(self, cfg: RecurrentLatentStateFutureLogitPathFlowConfig):
        super().__init__()
        self.cfg = cfg
        hist_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            gru_hidden_dim=cfg.history_hidden,
            bottleneck_dim=cfg.context_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=0.0,
        )
        self.history_encoder = GRUEncoder(hist_cfg)
        self.future_encoder = FuturePathEncoder(cfg)
        self.decoder = RecurrentLatentStateDecoder(cfg)
        self.latent_velocity = LatentTokenVelocity(cfg)
        self.register_buffer("cell_logit_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_logit_std", torch.ones(cfg.n_cells))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def set_logit_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.cfg.n_cells,) or std.shape != (self.cfg.n_cells,):
            raise ValueError("Expected per-cell logit stats with shape (n_cells,)")
        self.cell_logit_mean.copy_(mean.to(self.cell_logit_mean))
        self.cell_logit_std.copy_(
            std.clamp_min(self.cfg.logit_std_floor).to(self.cell_logit_std)
        )

    def _to_model_coord(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return logits
        return (logits - self.cell_logit_mean) / self.cell_logit_std

    def _from_model_coord(self, coord: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return coord
        return coord * self.cell_logit_std + self.cell_logit_mean

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten(history_norm))

    def target_future_coord(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        future_01 = denormalize_iv(future_norm)
        return self._to_model_coord(iv_to_logit(future_01, self.cfg.logit_eps))

    def history_last_coord(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        last_01 = denormalize_iv(history_norm[:, -1])
        return self._to_model_coord(iv_to_logit(last_01, self.cfg.logit_eps))

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        target = self.target_future_coord(future_norm)
        context = self.encode_history(history_norm)
        z1 = self.future_encoder(target, context)
        initial = self.history_last_coord(history_norm)
        recon = self.decoder(z1, context, initial)
        recon_mse = F.mse_loss(recon, target)

        z_target = z1.detach()
        z0 = torch.randn_like(z_target)
        bsz = history_norm.shape[0]
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        z_t = (1.0 - t)[:, None, None] * z0 + t[:, None, None] * z_target
        target_velocity = z_target - z0
        pred_velocity = self.latent_velocity(z_t, context, t)
        latent_fm = F.mse_loss(pred_velocity, target_velocity)
        total = float(self.cfg.recon_weight) * recon_mse + latent_fm
        recon_delta = torch.cat(
            [recon[:, :1] - initial[:, None, :], recon[:, 1:] - recon[:, :-1]],
            dim=1,
        )
        target_delta = torch.cat(
            [target[:, :1] - initial[:, None, :], target[:, 1:] - target[:, :-1]],
            dim=1,
        )
        metrics = {
            "total": total.detach(),
            "recon_mse": recon_mse.detach(),
            "latent_fm": latent_fm.detach(),
            "target_std": target.std(unbiased=False).detach(),
            "recon_std": recon.std(unbiased=False).detach(),
            "target_delta_std": target_delta.std(unbiased=False).detach(),
            "recon_delta_std": recon_delta.std(unbiased=False).detach(),
            "latent_std": z_target.std(unbiased=False).detach(),
            "latent_abs": z_target.abs().mean().detach(),
        }
        return total, metrics

    def sample_latent(
        self,
        context: torch.Tensor,
        n_samples: int,
        temperature: float | None = None,
    ) -> torch.Tensor:
        bsz = context.shape[0]
        k = int(n_samples)
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        z = temp * torch.randn(
            bsz * k,
            self.cfg.latent_tokens,
            self.cfg.latent_dim,
            device=context.device,
            dtype=context.dtype,
        )
        ctx = context.repeat_interleave(k, dim=0)
        dt = 1.0 / float(self.cfg.flow_steps)
        for step in range(self.cfg.flow_steps):
            t = torch.full(
                (bsz * k,),
                (step + 0.5) * dt,
                device=context.device,
                dtype=context.dtype,
            )
            z = z + dt * self.latent_velocity(z, ctx, t)
        return z

    def generate_model_coord(self, history_norm: torch.Tensor, n_samples: int) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        context = self.encode_history(history_norm)
        z = self.sample_latent(context, n_samples)
        ctx = context.repeat_interleave(int(n_samples), dim=0)
        initial = self.history_last_coord(history_norm).repeat_interleave(int(n_samples), dim=0)
        coord = self.decoder(z, ctx, initial)
        return coord.view(
            history_norm.shape[0],
            int(n_samples),
            self.cfg.future_len,
            self.cfg.n_cells,
        )

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 16,
        history_is_normalized: bool = True,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        chunk_size = max(1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)))
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            coord = self.generate_model_coord(history_norm, k)
            future_01 = logit_to_iv(self._from_model_coord(coord))
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(
                    history_norm.shape[0],
                    k,
                    self.cfg.future_len,
                    5,
                    5,
                )
            else:
                future_01 = future_01.view(
                    history_norm.shape[0],
                    k,
                    self.cfg.future_len,
                    self.cfg.n_cells,
                )
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[RecurrentLatentStateFutureLogitPathFlow, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = RecurrentLatentStateFutureLogitPathFlowConfig(**payload["config"])
    model = RecurrentLatentStateFutureLogitPathFlow(cfg)
    result = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = {"cell_logit_mean", "cell_logit_std"}
    missing = set(result.missing_keys) - allowed_missing
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    if cfg.standardize_logits and allowed_missing.intersection(result.missing_keys):
        raise RuntimeError("Standardized-logit checkpoint is missing saved logit stats")
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: RecurrentLatentStateFutureLogitPathFlow,
    cfg: RecurrentLatentStateFutureLogitPathFlowConfig,
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
