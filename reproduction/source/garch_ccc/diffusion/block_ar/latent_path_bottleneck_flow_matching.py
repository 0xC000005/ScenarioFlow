from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.dual_timescale_low_rank_temporal import TemporalConvTower
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)]
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


def _time_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        torch.linspace(
            0.0,
            -torch.log(torch.tensor(10000.0, device=device)),
            half,
            device=device,
        )
    )
    angles = t[:, None] * freqs[None, :] * 2.0 * torch.pi
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


@dataclass
class LatentPathBottleneckFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 128
    latent_dim: int = 32
    latent_tokens: int = 5
    history_hidden: int = 64
    future_hidden: int = 96
    encoder_dropout: float = 0.1

    decoder_hidden: int = 128
    decoder_layers: int = 2
    decoder_dropout: float = 0.1
    pos_dim: int = 32

    velocity_hidden: int = 128
    velocity_layers: int = 3
    velocity_dropout: float = 0.1
    time_dim: int = 32
    token_kernel: int = 3
    token_dilation: int = 1

    flow_steps: int = 16


class FuturePathEncoder(nn.Module):
    def __init__(self, cfg: LatentPathBottleneckFMConfig):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=cfg.n_cells,
            hidden_size=cfg.future_hidden,
            batch_first=True,
            num_layers=1,
        )
        self.token_proj = _mlp(cfg.future_hidden, cfg.latent_dim, cfg.future_hidden, 1, cfg.encoder_dropout)
        self.norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, future_norm: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(future_norm)
        pooled = F.adaptive_avg_pool1d(hidden.transpose(1, 2), self.cfg.latent_tokens).transpose(1, 2)
        return self.norm(self.token_proj(pooled))


class FuturePathDecoder(nn.Module):
    def __init__(self, cfg: LatentPathBottleneckFMConfig):
        super().__init__()
        self.cfg = cfg
        self.pos_embed = nn.Embedding(cfg.future_len, cfg.pos_dim)
        token_in = cfg.context_dim + cfg.latent_dim + cfg.pos_dim
        self.token_proj = _mlp(token_in, cfg.decoder_hidden, cfg.decoder_hidden, 1, cfg.decoder_dropout)
        self.init_proj = nn.Linear(cfg.context_dim, cfg.decoder_hidden)
        self.gru = nn.GRU(
            input_size=cfg.decoder_hidden,
            hidden_size=cfg.decoder_hidden,
            batch_first=True,
            num_layers=cfg.decoder_layers,
            dropout=cfg.decoder_dropout if cfg.decoder_layers > 1 else 0.0,
        )
        self.out = nn.Sequential(
            nn.Linear(cfg.decoder_hidden, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.n_cells),
            nn.Tanh(),
        )

    def forward(self, context: torch.Tensor, latent_tokens: torch.Tensor) -> torch.Tensor:
        bsz = context.shape[0]
        token_series = F.interpolate(
            latent_tokens.transpose(1, 2),
            size=self.cfg.future_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        pos_idx = torch.arange(self.cfg.future_len, device=context.device)
        pos = self.pos_embed(pos_idx)[None, :, :].expand(bsz, -1, -1)
        ctx = context[:, None, :].expand(-1, self.cfg.future_len, -1)
        tokens = self.token_proj(torch.cat([ctx, token_series, pos], dim=-1))
        h0 = self.init_proj(context).unsqueeze(0).expand(self.cfg.decoder_layers, -1, -1).contiguous()
        hidden, _ = self.gru(tokens, h0)
        return self.out(hidden)


class TokenSequenceVelocity(nn.Module):
    def __init__(self, cfg: LatentPathBottleneckFMConfig):
        super().__init__()
        self.cfg = cfg
        self.pos_embed = nn.Embedding(cfg.latent_tokens, cfg.pos_dim)
        in_dim = cfg.latent_dim + cfg.context_dim + cfg.time_dim + cfg.pos_dim
        self.token_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.velocity_hidden),
            nn.GELU(),
            nn.Linear(cfg.velocity_hidden, cfg.velocity_hidden),
        )
        self.backbone = TemporalConvTower(
            channels=cfg.velocity_hidden,
            layers=cfg.velocity_layers,
            kernel_size=cfg.token_kernel,
            dilation=cfg.token_dilation,
            dropout=cfg.velocity_dropout,
        )
        self.out = nn.Sequential(
            nn.Linear(cfg.velocity_hidden, cfg.velocity_hidden),
            nn.GELU(),
            nn.Linear(cfg.velocity_hidden, cfg.latent_dim),
        )

    def forward(self, latent_t: torch.Tensor, context: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        bsz, k, _ = latent_t.shape
        pos_idx = torch.arange(k, device=latent_t.device)
        pos = self.pos_embed(pos_idx)[None, :, :].expand(bsz, -1, -1)
        ctx = context[:, None, :].expand(-1, k, -1)
        t = time_feat[:, None, :].expand(-1, k, -1)
        tokens = self.token_proj(torch.cat([latent_t, ctx, t, pos], dim=-1))
        hidden = self.backbone(tokens)
        return self.out(hidden)


class LatentPathBottleneckFM(nn.Module):
    """266d-v0: temporal bottleneck with vanilla conditional flow matching."""

    def __init__(self, cfg: LatentPathBottleneckFMConfig):
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
        self.velocity = TokenSequenceVelocity(cfg)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten(history_norm))

    def encode_future(self, future_norm: torch.Tensor) -> torch.Tensor:
        return self.future_encoder(self._flatten(future_norm))

    def decode_future(self, context: torch.Tensor, latent_tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder(context, latent_tokens)

    def predict_velocity(self, latent_t: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_feat = _time_features(t, self.cfg.time_dim)
        return self.velocity(latent_t, context, t_feat)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
        recon_weight: float = 1.0,
        fm_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        latent_target = self.encode_future(future_norm)
        recon = self.decode_future(context, latent_target)
        recon_loss = F.smooth_l1_loss(recon, future_norm)

        bsz = history_norm.shape[0]
        latent_base = torch.randn_like(latent_target)
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        latent_t = (1.0 - t)[:, None, None] * latent_base + t[:, None, None] * latent_target
        target_velocity = latent_target - latent_base
        pred_velocity = self.predict_velocity(latent_t, context, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)

        total = recon_weight * recon_loss + fm_weight * fm_loss
        metrics = {
            "total": total.detach(),
            "recon_loss": recon_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "latent_std": latent_target.std(unbiased=False).detach(),
            "recon_abs": (recon - future_norm).abs().mean().detach(),
        }
        return total, metrics

    @torch.no_grad()
    def sample_latents(self, context: torch.Tensor, n_samples: int) -> torch.Tensor:
        bsz = context.shape[0]
        device = context.device
        latent = torch.randn(
            bsz * n_samples,
            self.cfg.latent_tokens,
            self.cfg.latent_dim,
            device=device,
            dtype=context.dtype,
        )
        ctx = context.repeat_interleave(n_samples, dim=0)
        dt = 1.0 / float(self.cfg.flow_steps)
        for step in range(self.cfg.flow_steps):
            t = torch.full(
                (bsz * n_samples,),
                (step + 0.5) * dt,
                device=device,
                dtype=context.dtype,
            )
            velocity = self.predict_velocity(latent, ctx, t)
            latent = latent + dt * velocity
        return latent.view(bsz, n_samples, self.cfg.latent_tokens, self.cfg.latent_dim)

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        context = self.encode_history(history_norm)
        bsz = context.shape[0]
        outs: list[torch.Tensor] = []
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            latent = self.sample_latents(context, n_samples=k)
            flat_context = context.repeat_interleave(k, dim=0)
            flat_latent = latent.reshape(bsz * k, self.cfg.latent_tokens, self.cfg.latent_dim)
            future_norm = self.decode_future(flat_context, flat_latent)
            future_01 = denormalize_iv(future_norm)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[LatentPathBottleneckFM, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LatentPathBottleneckFMConfig(**payload["config"])
    model = LatentPathBottleneckFM(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: LatentPathBottleneckFM,
    cfg: LatentPathBottleneckFMConfig,
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
