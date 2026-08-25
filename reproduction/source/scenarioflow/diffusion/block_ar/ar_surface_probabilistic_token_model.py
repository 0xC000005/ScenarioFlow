from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods.extend([nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)])
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


@dataclass
class ARSurfaceProbabilisticTokenConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    n_tokens: int = 4
    token_dim: int = 16
    encoder_hidden: int = 128
    encoder_layers: int = 2
    encoder_dropout: float = 0.1

    prior_layers: int = 2
    prior_heads: int = 4
    prior_ff_mult: int = 4
    prior_dropout: float = 0.1

    decoder_hidden: int = 128
    decoder_layers: int = 2
    decoder_dropout: float = 0.1

    recon_weight: float = 1.0
    kl_weight: float = 1e-3


class TokenPosterior(nn.Module):
    def __init__(self, cfg: ARSurfaceProbabilisticTokenConfig):
        super().__init__()
        latent_dim = cfg.n_tokens * cfg.token_dim
        self.net = _mlp(
            in_dim=cfg.n_cells,
            out_dim=2 * latent_dim,
            hidden=cfg.encoder_hidden,
            layers=cfg.encoder_layers,
            dropout=cfg.encoder_dropout,
        )
        self.cfg = cfg

    def forward(self, level_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(level_norm)
        mean, logvar = params.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-8.0, max=4.0)
        mean = mean.view(level_norm.shape[0], self.cfg.n_tokens, self.cfg.token_dim)
        logvar = logvar.view(level_norm.shape[0], self.cfg.n_tokens, self.cfg.token_dim)
        return mean, logvar


class TokenPrior(nn.Module):
    def __init__(self, cfg: ARSurfaceProbabilisticTokenConfig):
        super().__init__()
        self.cfg = cfg
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_tokens, cfg.token_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.prior_heads,
            dim_feedforward=cfg.prior_ff_mult * cfg.token_dim,
            dropout=cfg.prior_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.prior_layers)
        self.out_proj = nn.Linear(cfg.token_dim, 2 * cfg.token_dim)

    def forward(self, z_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(z_prev + self.pos)
        params = self.out_proj(h)
        mean, logvar = params.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-8.0, max=4.0)
        return mean, logvar


class SurfaceDecoder(nn.Module):
    def __init__(self, cfg: ARSurfaceProbabilisticTokenConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            in_dim=cfg.n_tokens * cfg.token_dim,
            out_dim=cfg.n_cells,
            hidden=cfg.decoder_hidden,
            layers=cfg.decoder_layers,
            dropout=cfg.decoder_dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        flat = z.reshape(z.shape[0], -1)
        return self.net(flat)


class ARSurfaceProbabilisticTokenModel(nn.Module):
    """274a-v0: probabilistic latent-token state-space model with explicit observation model."""

    def __init__(self, cfg: ARSurfaceProbabilisticTokenConfig):
        super().__init__()
        self.cfg = cfg
        self.posterior = TokenPosterior(cfg)
        self.prior = TokenPrior(cfg)
        self.decoder = SurfaceDecoder(cfg)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    @staticmethod
    def _sample(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    @staticmethod
    def _kl_gaussian(
        q_mean: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mean: torch.Tensor,
        p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)
        kl = 0.5 * (
            p_logvar - q_logvar
            + (q_var + (q_mean - p_mean).pow(2)) / p_var
            - 1.0
        )
        return kl.mean()

    def step_losses(
        self,
        prev_level_norm: torch.Tensor,
        next_level_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prev_mean, prev_logvar = self.posterior(prev_level_norm)
        next_mean, next_logvar = self.posterior(next_level_norm)
        z_prev = self._sample(prev_mean, prev_logvar)
        z_next = self._sample(next_mean, next_logvar)
        prior_mean, prior_logvar = self.prior(z_prev)

        recon_prev = self.decoder(z_prev)
        recon_next = self.decoder(z_next)
        recon_loss = 0.5 * (
            F.smooth_l1_loss(recon_prev, prev_level_norm)
            + F.smooth_l1_loss(recon_next, next_level_norm)
        )
        kl_loss = self._kl_gaussian(next_mean, next_logvar, prior_mean, prior_logvar)
        prior_std = torch.exp(0.5 * prior_logvar).mean()
        post_std = torch.exp(0.5 * next_logvar).mean()
        return recon_loss, kl_loss, prior_std, post_std

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        prev_level = history_norm[:, -1, :]
        recon_losses = []
        kl_losses = []
        prior_stds = []
        post_stds = []
        for step in range(self.cfg.future_len):
            next_level = future_norm[:, step, :]
            recon_loss, kl_loss, prior_std, post_std = self.step_losses(prev_level, next_level)
            recon_losses.append(recon_loss)
            kl_losses.append(kl_loss)
            prior_stds.append(prior_std)
            post_stds.append(post_std)
            prev_level = next_level
        mean_recon = torch.stack(recon_losses).mean()
        mean_kl = torch.stack(kl_losses).mean()
        total = self.cfg.recon_weight * mean_recon + self.cfg.kl_weight * mean_kl
        metrics = {
            "total": total.detach(),
            "recon_loss": mean_recon.detach(),
            "kl_loss": mean_kl.detach(),
            "prior_std": torch.stack(prior_stds).mean().detach(),
            "post_std": torch.stack(post_stds).mean().detach(),
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
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        last_level = history_norm[:, -1, :]
        last_mean, _ = self.posterior(last_level)

        bsz = history_norm.shape[0]
        outs: list[torch.Tensor] = []
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            z_prev = last_mean.repeat_interleave(k, dim=0)
            steps: list[torch.Tensor] = []
            for _step in range(self.cfg.future_len):
                prior_mean, prior_logvar = self.prior(z_prev)
                z_prev = self._sample(prior_mean, prior_logvar)
                next_level = self.decoder(z_prev)
                steps.append(next_level)
            future_norm = torch.stack(steps, dim=1)
            future_01 = denormalize_iv(future_norm)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ARSurfaceProbabilisticTokenModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ARSurfaceProbabilisticTokenConfig(**payload["config"])
    model = ARSurfaceProbabilisticTokenModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ARSurfaceProbabilisticTokenModel,
    cfg: ARSurfaceProbabilisticTokenConfig,
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
