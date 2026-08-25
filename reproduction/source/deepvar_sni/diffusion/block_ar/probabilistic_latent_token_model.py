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
class ProbabilisticLatentTokenConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 128
    latent_dim: int = 32
    latent_tokens: int = 5
    history_hidden: int = 64
    future_hidden: int = 96
    encoder_dropout: float = 0.1

    posterior_hidden: int = 128
    prior_hidden: int = 128
    decoder_hidden: int = 128
    decoder_layers: int = 2
    decoder_dropout: float = 0.1
    pos_dim: int = 32

    kl_weight: float = 1.0


class FutureTokenEncoder(nn.Module):
    def __init__(self, cfg: ProbabilisticLatentTokenConfig):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=cfg.n_cells,
            hidden_size=cfg.future_hidden,
            batch_first=True,
            num_layers=1,
        )
        self.token_proj = _mlp(cfg.future_hidden, cfg.future_hidden, cfg.future_hidden, 1, cfg.encoder_dropout)
        self.norm = nn.LayerNorm(cfg.future_hidden)

    def forward(self, future_norm: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(future_norm)
        pooled = F.adaptive_avg_pool1d(hidden.transpose(1, 2), self.cfg.latent_tokens).transpose(1, 2)
        return self.norm(self.token_proj(pooled))


class PriorHead(nn.Module):
    def __init__(self, cfg: ProbabilisticLatentTokenConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(cfg.context_dim, cfg.latent_tokens * cfg.latent_dim * 2, cfg.prior_hidden, 2, cfg.encoder_dropout)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.net(context).view(context.shape[0], self.cfg.latent_tokens, self.cfg.latent_dim, 2)
        mu = stats[..., 0]
        logvar = stats[..., 1].clamp(-6.0, 4.0)
        return mu, logvar


class PosteriorHead(nn.Module):
    def __init__(self, cfg: ProbabilisticLatentTokenConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.future_hidden + cfg.context_dim
        self.net = _mlp(in_dim, cfg.latent_dim * 2, cfg.posterior_hidden, 2, cfg.encoder_dropout)

    def forward(self, future_tokens: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = context[:, None, :].expand(-1, future_tokens.shape[1], -1)
        stats = self.net(torch.cat([future_tokens, ctx], dim=-1)).view(
            future_tokens.shape[0], future_tokens.shape[1], self.cfg.latent_dim, 2
        )
        mu = stats[..., 0]
        logvar = stats[..., 1].clamp(-6.0, 4.0)
        return mu, logvar


class FuturePathDecoder(nn.Module):
    def __init__(self, cfg: ProbabilisticLatentTokenConfig):
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


class ProbabilisticLatentTokenModel(nn.Module):
    """267a-v0: clean probabilistic latent-token conditional scenario generator."""

    def __init__(self, cfg: ProbabilisticLatentTokenConfig):
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
        self.future_encoder = FutureTokenEncoder(cfg)
        self.prior_head = PriorHead(cfg)
        self.posterior_head = PosteriorHead(cfg)
        self.decoder = FuturePathDecoder(cfg)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten(history_norm))

    def encode_future_tokens(self, future_norm: torch.Tensor) -> torch.Tensor:
        return self.future_encoder(self._flatten(future_norm))

    def decode_future(self, context: torch.Tensor, latent_tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder(context, latent_tokens)

    @staticmethod
    def _sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    @staticmethod
    def _kl_diag_gaussian(
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mu: torch.Tensor,
        p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)
        kl = 0.5 * (
            p_logvar - q_logvar
            + (q_var + (q_mu - p_mu).pow(2)) / p_var
            - 1.0
        )
        return kl.mean()

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
        recon_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        future_tokens = self.encode_future_tokens(future_norm)
        p_mu, p_logvar = self.prior_head(context)
        q_mu, q_logvar = self.posterior_head(future_tokens, context)
        latent = self._sample_gaussian(q_mu, q_logvar)
        recon = self.decode_future(context, latent)
        recon_loss = F.smooth_l1_loss(recon, future_norm)
        kl_loss = self._kl_diag_gaussian(q_mu, q_logvar, p_mu, p_logvar)
        total = recon_weight * recon_loss + self.cfg.kl_weight * kl_loss
        metrics = {
            "total": total.detach(),
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "prior_std": torch.exp(0.5 * p_logvar).mean().detach(),
            "post_std": torch.exp(0.5 * q_logvar).mean().detach(),
        }
        return total, metrics

    @torch.no_grad()
    def sample_latents(self, context: torch.Tensor, n_samples: int) -> torch.Tensor:
        p_mu, p_logvar = self.prior_head(context)
        bsz = context.shape[0]
        mu = p_mu[:, None, :, :].expand(-1, n_samples, -1, -1)
        logvar = p_logvar[:, None, :, :].expand(-1, n_samples, -1, -1)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

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


def load_model(checkpoint_path: str, device: torch.device) -> tuple[ProbabilisticLatentTokenModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ProbabilisticLatentTokenConfig(**payload["config"])
    model = ProbabilisticLatentTokenModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ProbabilisticLatentTokenModel,
    cfg: ProbabilisticLatentTokenConfig,
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
