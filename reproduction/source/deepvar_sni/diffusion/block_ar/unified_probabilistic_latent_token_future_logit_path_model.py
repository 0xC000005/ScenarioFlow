from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.unified_latent_token_future_logit_path_flow_matching import (
    FutureLogitPathDecoder,
    FutureLogitPathLatentEncoder,
    UnifiedLatentTokenFutureLogitPathFMConfig,
)
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class UnifiedProbabilisticLatentTokenFutureLogitPathConfig(
    UnifiedLatentTokenFutureLogitPathFMConfig
):
    prior_hidden: int = 128
    posterior_hidden: int = 128
    kl_weight: float = 1.0


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)]
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


class PriorHead(nn.Module):
    def __init__(self, cfg: UnifiedProbabilisticLatentTokenFutureLogitPathConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            cfg.context_dim,
            cfg.latent_tokens * cfg.latent_dim * 2,
            cfg.prior_hidden,
            2,
            cfg.encoder_dropout,
        )

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.net(context).view(
            context.shape[0], self.cfg.latent_tokens, self.cfg.latent_dim, 2
        )
        mu = stats[..., 0]
        logvar = stats[..., 1].clamp(-6.0, 4.0)
        return mu, logvar


class PosteriorHead(nn.Module):
    def __init__(self, cfg: UnifiedProbabilisticLatentTokenFutureLogitPathConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.latent_dim + cfg.context_dim
        self.net = _mlp(
            in_dim,
            cfg.latent_dim * 2,
            cfg.posterior_hidden,
            2,
            cfg.encoder_dropout,
        )

    def forward(
        self,
        future_latents: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = context[:, None, :].expand(-1, future_latents.shape[1], -1)
        stats = self.net(torch.cat([future_latents, ctx], dim=-1)).view(
            future_latents.shape[0],
            future_latents.shape[1],
            self.cfg.latent_dim,
            2,
        )
        mu = stats[..., 0]
        logvar = stats[..., 1].clamp(-6.0, 4.0)
        return mu, logvar


class UnifiedProbabilisticLatentTokenFutureLogitPathModel(nn.Module):
    """313b-v0: unified future-logit path model with stochastic shared latent tokens."""

    def __init__(self, cfg: UnifiedProbabilisticLatentTokenFutureLogitPathConfig):
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
        self.future_encoder = FutureLogitPathLatentEncoder(cfg)
        self.prior_head = PriorHead(cfg)
        self.posterior_head = PosteriorHead(cfg)
        self.decoder = FutureLogitPathDecoder(cfg)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten(history_norm))

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        future_01 = denormalize_iv(future_norm)
        return iv_to_logit(future_01, self.cfg.logit_eps)

    def encode_future_latents(self, future_logits: torch.Tensor) -> torch.Tensor:
        return self.future_encoder(future_logits)

    def decode_future_logits(
        self,
        context: torch.Tensor,
        latent_tokens: torch.Tensor,
    ) -> torch.Tensor:
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
            p_logvar
            - q_logvar
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
        future_logits = self.target_future_logits(future_norm)
        future_latents = self.encode_future_latents(future_logits)
        p_mu, p_logvar = self.prior_head(context)
        q_mu, q_logvar = self.posterior_head(future_latents, context)
        latent = self._sample_gaussian(q_mu, q_logvar)
        recon_logits = self.decode_future_logits(context, latent)
        recon_loss = F.smooth_l1_loss(recon_logits, future_logits)
        kl_loss = self._kl_diag_gaussian(q_mu, q_logvar, p_mu, p_logvar)
        total = recon_weight * recon_loss + self.cfg.kl_weight * kl_loss
        metrics = {
            "total": total.detach(),
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "prior_std": torch.exp(0.5 * p_logvar).mean().detach(),
            "post_std": torch.exp(0.5 * q_logvar).mean().detach(),
            "future_logit_std": future_logits.std(unbiased=False).detach(),
            "recon_logit_abs": (recon_logits - future_logits).abs().mean().detach(),
        }
        return total, metrics

    @torch.no_grad()
    def sample_latents(self, context: torch.Tensor, n_samples: int) -> torch.Tensor:
        p_mu, p_logvar = self.prior_head(context)
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
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        context = self.encode_history(history_norm)
        bsz = context.shape[0]
        outs: list[torch.Tensor] = []
        chunk_size = max(
            1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk))
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            latent = self.sample_latents(context, n_samples=k)
            flat_context = context.repeat_interleave(k, dim=0)
            flat_latent = latent.reshape(
                bsz * k, self.cfg.latent_tokens, self.cfg.latent_dim
            )
            if temp != 1.0:
                flat_latent = flat_latent * temp
            future_logits = self.decode_future_logits(flat_context, flat_latent)
            future_01 = logit_to_iv(future_logits)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(
                    bsz, k, self.cfg.future_len, self.cfg.n_cells
                )
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[UnifiedProbabilisticLatentTokenFutureLogitPathModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = UnifiedProbabilisticLatentTokenFutureLogitPathConfig(**payload["config"])
    model = UnifiedProbabilisticLatentTokenFutureLogitPathModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: UnifiedProbabilisticLatentTokenFutureLogitPathModel,
    cfg: UnifiedProbabilisticLatentTokenFutureLogitPathConfig,
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
