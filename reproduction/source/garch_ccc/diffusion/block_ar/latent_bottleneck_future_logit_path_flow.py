from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.axial_mixer_future_logit_path_flow_matching import AxialMixerBlock
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.joint_token_logit_transition_flow_matching import (
    JointTokenLogitTransitionFMConfig,
    _flow_time_features,
)
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class LatentBottleneckFutureLogitPathFlowConfig(JointTokenLogitTransitionFMConfig):
    latent_tokens: int = 8
    latent_dim: int = 32
    target_mode: str = "level"
    standardize_logits: bool = True
    logit_std_floor: float = 1e-3
    recon_weight: float = 1.0


class FuturePathEncoder(nn.Module):
    def __init__(self, cfg: LatentBottleneckFutureLogitPathFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(1, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.blocks = nn.ModuleList([AxialMixerBlock(cfg) for _ in range(cfg.token_layers)])
        self.latent_query = nn.Parameter(torch.zeros(cfg.latent_tokens, cfg.token_dim))
        self.query_norm = nn.LayerNorm(cfg.token_dim)
        self.token_norm = nn.LayerNorm(cfg.token_dim)
        self.pool_attn = nn.MultiheadAttention(
            cfg.token_dim,
            cfg.token_heads,
            dropout=cfg.model_dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, cfg.latent_dim),
        )

    def forward(self, future_coord: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, horizon, n_cells = future_coord.shape
        h_idx = torch.arange(horizon, device=future_coord.device)
        c_idx = torch.arange(n_cells, device=future_coord.device)
        x = self.value_proj(future_coord.unsqueeze(-1))
        x = x + self.horizon_embed(h_idx)[None, :, None, :]
        x = x + self.cell_embed(c_idx)[None, None, :, :]
        x = x + self.context_proj(context)[:, None, None, :]
        for block in self.blocks:
            x = block(x)
        tokens = x.reshape(bsz, horizon * n_cells, self.cfg.token_dim)
        query = self.latent_query[None, :, :] + self.context_proj(context)[:, None, :]
        pooled, _ = self.pool_attn(
            self.query_norm(query),
            self.token_norm(tokens),
            self.token_norm(tokens),
            need_weights=False,
        )
        return self.out(pooled)


class FuturePathDecoder(nn.Module):
    def __init__(self, cfg: LatentBottleneckFutureLogitPathFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.latent_pos = nn.Embedding(cfg.latent_tokens, cfg.token_dim)
        self.cross_attn = nn.MultiheadAttention(
            cfg.token_dim,
            cfg.token_heads,
            dropout=cfg.model_dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(cfg.token_dim)
        self.latent_norm = nn.LayerNorm(cfg.token_dim)
        self.blocks = nn.ModuleList([AxialMixerBlock(cfg) for _ in range(cfg.token_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, n_latent, _ = latent.shape
        h_idx = torch.arange(self.cfg.future_len, device=latent.device)
        c_idx = torch.arange(self.cfg.n_cells, device=latent.device)
        l_idx = torch.arange(n_latent, device=latent.device)
        ctx = self.context_proj(context)

        latent_tokens = self.latent_proj(latent)
        latent_tokens = latent_tokens + self.latent_pos(l_idx)[None, :, :]
        latent_tokens = latent_tokens + ctx[:, None, :]
        latent_tokens = self.latent_norm(latent_tokens)

        query = self.horizon_embed(h_idx)[:, None, :] + self.cell_embed(c_idx)[None, :, :]
        query = query.reshape(self.cfg.future_len * self.cfg.n_cells, self.cfg.token_dim)
        query = query[None, :, :] + ctx[:, None, :]
        attn, _ = self.cross_attn(
            self.query_norm(query),
            latent_tokens,
            latent_tokens,
            need_weights=False,
        )
        x = (query + attn).view(
            bsz,
            self.cfg.future_len,
            self.cfg.n_cells,
            self.cfg.token_dim,
        )
        for block in self.blocks:
            x = block(x)
        return self.out(x).squeeze(-1)


class LatentTokenVelocity(nn.Module):
    def __init__(self, cfg: LatentBottleneckFutureLogitPathFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(cfg.latent_dim, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.latent_pos = nn.Embedding(cfg.latent_tokens, cfg.token_dim)
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
            nn.Linear(cfg.token_dim, cfg.latent_dim),
        )

    def forward(
        self,
        latent_t: torch.Tensor,
        context: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_latent, _ = latent_t.shape
        l_idx = torch.arange(n_latent, device=latent_t.device)
        x = self.value_proj(latent_t)
        x = x + self.latent_pos(l_idx)[None, :, :]
        x = x + self.context_proj(context)[:, None, :]
        x = x + self.flow_time_proj(_flow_time_features(flow_t, self.cfg.flow_time_dim))[
            :, None, :
        ]
        return self.out(self.mixer(x))


class LatentBottleneckFutureLogitPathFlow(nn.Module):
    """327a-v0: learned future-path bottleneck plus vanilla conditional latent flow."""

    def __init__(self, cfg: LatentBottleneckFutureLogitPathFlowConfig):
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
        self.decoder = FuturePathDecoder(cfg)
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

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        future_01 = denormalize_iv(future_norm)
        return self._to_model_coord(iv_to_logit(future_01, self.cfg.logit_eps))

    def history_last_coord(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        last_01 = denormalize_iv(history_norm[:, -1])
        return self._to_model_coord(iv_to_logit(last_01, self.cfg.logit_eps))

    def target_model_coord(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> torch.Tensor:
        future_coord = self.target_future_logits(future_norm)
        if self.cfg.target_mode == "level":
            return future_coord
        if self.cfg.target_mode == "transition":
            last_coord = self.history_last_coord(history_norm)
            return torch.cat(
                [
                    future_coord[:, :1] - last_coord[:, None, :],
                    future_coord[:, 1:] - future_coord[:, :-1],
                ],
                dim=1,
            )
        raise ValueError(f"Unknown target_mode={self.cfg.target_mode!r}")

    def model_coord_to_future_iv(
        self,
        history_norm: torch.Tensor,
        coord: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.target_mode == "level":
            future_coord = coord
        elif self.cfg.target_mode == "transition":
            current = self.history_last_coord(history_norm)
            if coord.ndim == 4:
                current = current[:, None, :]
            frames: list[torch.Tensor] = []
            for step in range(coord.shape[-2]):
                current = current + coord[..., step, :]
                frames.append(current)
            future_coord = torch.stack(frames, dim=-2)
        else:
            raise ValueError(f"Unknown target_mode={self.cfg.target_mode!r}")
        return logit_to_iv(self._from_model_coord(future_coord))

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        target = self.target_model_coord(history_norm, future_norm)
        context = self.encode_history(history_norm)
        z1 = self.future_encoder(target, context)
        recon = self.decoder(z1, context)
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
        metrics = {
            "total": total.detach(),
            "recon_mse": recon_mse.detach(),
            "latent_fm": latent_fm.detach(),
            "target_std": target.std(unbiased=False).detach(),
            "recon_std": recon.std(unbiased=False).detach(),
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
        coord = self.decoder(z, ctx)
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
            future_01 = self.model_coord_to_future_iv(history_norm, coord)
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
) -> tuple[LatentBottleneckFutureLogitPathFlow, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LatentBottleneckFutureLogitPathFlowConfig(**payload["config"])
    model = LatentBottleneckFutureLogitPathFlow(cfg)
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
    model: LatentBottleneckFutureLogitPathFlow,
    cfg: LatentBottleneckFutureLogitPathFlowConfig,
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
