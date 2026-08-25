from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

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


@dataclass
class LatentPathBottleneckDiffusionConfig:
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

    denoiser_hidden: int = 256
    denoiser_layers: int = 3
    denoiser_dropout: float = 0.1
    timestep_dim: int = 32

    diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class FuturePathEncoder(nn.Module):
    def __init__(self, cfg: LatentPathBottleneckDiffusionConfig):
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
    def __init__(self, cfg: LatentPathBottleneckDiffusionConfig):
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


class LatentPathBottleneckDiffusion(nn.Module):
    """266b-v0: temporally structured bottleneck with vanilla latent diffusion."""

    def __init__(self, cfg: LatentPathBottleneckDiffusionConfig):
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

        self.time_embed = nn.Embedding(cfg.diffusion_steps, cfg.timestep_dim)
        flat_dim = cfg.latent_tokens * cfg.latent_dim
        self.denoiser = _mlp(
            flat_dim + cfg.context_dim + cfg.timestep_dim,
            flat_dim,
            cfg.denoiser_hidden,
            cfg.denoiser_layers,
            cfg.denoiser_dropout,
        )

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("posterior_var", posterior_var.clamp_min(1e-8))

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

    def predict_noise(self, latent_noisy: torch.Tensor, context: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        bsz = latent_noisy.shape[0]
        t_emb = self.time_embed(t_idx)
        flat = latent_noisy.reshape(bsz, -1)
        pred = self.denoiser(torch.cat([flat, context, t_emb], dim=-1))
        return pred.view(bsz, self.cfg.latent_tokens, self.cfg.latent_dim)

    def q_sample(self, latent_clean: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self.sqrt_alpha_bars[t_idx][:, None, None] * latent_clean
            + self.sqrt_one_minus_alpha_bars[t_idx][:, None, None] * noise
        )

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
        recon_weight: float = 1.0,
        diffusion_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        latent_clean = self.encode_future(future_norm)
        recon = self.decode_future(context, latent_clean)
        recon_loss = F.smooth_l1_loss(recon, future_norm)

        bsz = history_norm.shape[0]
        t_idx = torch.randint(0, self.cfg.diffusion_steps, (bsz,), device=history_norm.device)
        noise = torch.randn_like(latent_clean)
        latent_noisy = self.q_sample(latent_clean, t_idx, noise)
        pred_noise = self.predict_noise(latent_noisy, context, t_idx)
        diffusion_loss = F.mse_loss(pred_noise, noise)

        total = recon_weight * recon_loss + diffusion_weight * diffusion_loss
        metrics = {
            "total": total.detach(),
            "recon_loss": recon_loss.detach(),
            "diffusion_loss": diffusion_loss.detach(),
            "latent_std": latent_clean.std(unbiased=False).detach(),
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
        for step in reversed(range(self.cfg.diffusion_steps)):
            t_idx = torch.full((bsz * n_samples,), step, device=device, dtype=torch.long)
            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]
            pred_noise = self.predict_noise(latent, ctx, t_idx)
            coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mean = (latent - coef * pred_noise) / torch.sqrt(alpha_t)
            if step > 0:
                latent = mean + torch.sqrt(self.posterior_var[step]) * torch.randn_like(latent)
            else:
                latent = mean
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


def load_model(checkpoint_path: str, device: torch.device) -> tuple[LatentPathBottleneckDiffusion, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LatentPathBottleneckDiffusionConfig(**payload["config"])
    model = LatentPathBottleneckDiffusion(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: LatentPathBottleneckDiffusion,
    cfg: LatentPathBottleneckDiffusionConfig,
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
