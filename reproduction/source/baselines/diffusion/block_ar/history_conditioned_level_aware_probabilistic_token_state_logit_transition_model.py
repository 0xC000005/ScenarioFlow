from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
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
class HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 96
    history_hidden: int = 96
    encoder_dropout: float = 0.1

    n_tokens: int = 4
    token_dim: int = 24

    posterior_hidden: int = 128
    posterior_layers: int = 2

    prior_layers: int = 2
    prior_heads: int = 4
    prior_ff_mult: int = 4
    prior_dropout: float = 0.1

    decoder_hidden: int = 128
    decoder_layers: int = 2
    decoder_dropout: float = 0.1

    logit_eps: float = 1e-4
    recon_weight: float = 1.0
    kl_weight: float = 1e-3


class TokenPosterior(nn.Module):
    def __init__(self, cfg: HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionConfig):
        super().__init__()
        self.cfg = cfg
        latent_dim = cfg.n_tokens * cfg.token_dim
        self.net = _mlp(
            in_dim=2 * cfg.n_cells + cfg.context_dim,
            out_dim=2 * latent_dim,
            hidden=cfg.posterior_hidden,
            layers=cfg.posterior_layers,
            dropout=cfg.encoder_dropout,
        )

    def forward(
        self,
        transitions: torch.Tensor,
        current_level: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if transitions.ndim == 2:
            inputs = torch.cat([transitions, current_level, context], dim=-1)
            params = self.net(inputs)
            mean, logvar = params.chunk(2, dim=-1)
            mean = mean.view(transitions.shape[0], self.cfg.n_tokens, self.cfg.token_dim)
            logvar = logvar.view(transitions.shape[0], self.cfg.n_tokens, self.cfg.token_dim)
            return mean, logvar.clamp(-8.0, 4.0)

        bsz, steps, _ = transitions.shape
        ctx = context[:, None, :].expand(-1, steps, -1)
        flat_inputs = torch.cat([transitions, current_level, ctx], dim=-1).reshape(
            bsz * steps, 2 * self.cfg.n_cells + self.cfg.context_dim
        )
        params = self.net(flat_inputs)
        mean, logvar = params.chunk(2, dim=-1)
        mean = mean.view(bsz, steps, self.cfg.n_tokens, self.cfg.token_dim)
        logvar = logvar.view(bsz, steps, self.cfg.n_tokens, self.cfg.token_dim)
        return mean, logvar.clamp(-8.0, 4.0)


class TokenPrior(nn.Module):
    def __init__(self, cfg: HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionConfig):
        super().__init__()
        self.cfg = cfg
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_tokens, cfg.token_dim))
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.level_proj = nn.Linear(cfg.n_cells, cfg.token_dim)
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

    def forward(
        self,
        z_prev: torch.Tensor,
        current_level: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.context_proj(context)[:, None, :]
        lvl = self.level_proj(current_level)[:, None, :]
        hidden = torch.cat([ctx, lvl, z_prev + self.pos], dim=1)
        hidden = self.encoder(hidden)
        params = self.out_proj(hidden[:, 2:, :])
        mean, logvar = params.chunk(2, dim=-1)
        return mean, logvar.clamp(-8.0, 4.0)


class TokenDecoder(nn.Module):
    def __init__(self, cfg: HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            in_dim=cfg.n_tokens * cfg.token_dim + cfg.n_cells,
            out_dim=cfg.n_cells,
            hidden=cfg.decoder_hidden,
            layers=cfg.decoder_layers,
            dropout=cfg.decoder_dropout,
        )

    def forward(self, z: torch.Tensor, current_level: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([z.reshape(z.shape[0], -1), current_level], dim=-1)
        return self.net(flat)


class HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionModel(nn.Module):
    """314c-v0: level-aware transition-emission token-state model."""

    def __init__(self, cfg: HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionConfig):
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
        self.posterior = TokenPosterior(cfg)
        self.prior = TokenPrior(cfg)
        self.decoder = TokenDecoder(cfg)

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
            p_logvar
            - q_logvar
            + (q_var + (q_mean - p_mean).pow(2)) / p_var
            - 1.0
        )
        return kl.mean()

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten(history_norm))

    def to_logits(self, levels_norm: torch.Tensor) -> torch.Tensor:
        levels_norm = self._flatten(levels_norm)
        levels_01 = denormalize_iv(levels_norm)
        return iv_to_logit(levels_01, self.cfg.logit_eps)

    def future_state_views(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_logits = self.to_logits(history_norm)
        future_logits = self.to_logits(future_norm)
        last_logit = history_logits[:, -1, :]
        prev_logit = history_logits[:, -2, :]
        last_transition = last_logit - prev_logit
        current_levels = torch.cat([last_logit[:, None, :], future_logits[:, :-1, :]], dim=1)
        future_trans = torch.cat(
            [
                future_logits[:, :1, :] - last_logit[:, None, :],
                future_logits[:, 1:, :] - future_logits[:, :-1, :],
            ],
            dim=1,
        )
        return last_transition, current_levels, future_trans

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        last_transition, current_levels, future_trans = self.future_state_views(history_norm, future_norm)
        last_level = self.to_logits(history_norm[:, -1:, :]).squeeze(1)

        init_mean, init_logvar = self.posterior(last_transition, last_level, context)
        post_mean, post_logvar = self.posterior(future_trans, current_levels, context)

        recon_losses = []
        kl_losses = []
        prior_stds = []
        post_stds = []
        recon_abs = []
        for step in range(self.cfg.future_len):
            current_level = current_levels[:, step, :]
            if step == 0:
                prev_mean = init_mean
                prev_logvar = init_logvar
            else:
                prev_mean = post_mean[:, step - 1]
                prev_logvar = post_logvar[:, step - 1]
            next_mean = post_mean[:, step]
            next_logvar = post_logvar[:, step]

            z_prev = self._sample(prev_mean, prev_logvar)
            z_next = self._sample(next_mean, next_logvar)
            prior_mean, prior_logvar = self.prior(z_prev, current_level, context)
            recon_trans = self.decoder(z_next, current_level)

            recon_losses.append(F.smooth_l1_loss(recon_trans, future_trans[:, step, :]))
            kl_losses.append(
                self._kl_gaussian(next_mean, next_logvar, prior_mean, prior_logvar)
            )
            prior_stds.append(torch.exp(0.5 * prior_logvar).mean())
            post_stds.append(torch.exp(0.5 * next_logvar).mean())
            recon_abs.append((recon_trans - future_trans[:, step, :]).abs().mean())

        mean_recon = torch.stack(recon_losses).mean()
        mean_kl = torch.stack(kl_losses).mean()
        total = self.cfg.recon_weight * mean_recon + self.cfg.kl_weight * mean_kl
        metrics = {
            "total": total.detach(),
            "recon_loss": mean_recon.detach(),
            "kl_loss": mean_kl.detach(),
            "prior_std": torch.stack(prior_stds).mean().detach(),
            "post_std": torch.stack(post_stds).mean().detach(),
            "transition_std": future_trans.std(unbiased=False).detach(),
            "recon_transition_abs": torch.stack(recon_abs).mean().detach(),
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
        context = self.encode_history(history_norm)
        history_logits = self.to_logits(history_norm)
        last_logit = history_logits[:, -1, :]
        prev_logit = history_logits[:, -2, :]
        last_transition = last_logit - prev_logit
        init_mean, _ = self.posterior(last_transition, last_logit, context)

        bsz = history_norm.shape[0]
        outs: list[torch.Tensor] = []
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            z_prev = init_mean.repeat_interleave(k, dim=0)
            ctx = context.repeat_interleave(k, dim=0)
            current_level = last_logit.repeat_interleave(k, dim=0)
            steps: list[torch.Tensor] = []
            for _step in range(self.cfg.future_len):
                prior_mean, prior_logvar = self.prior(z_prev, current_level, ctx)
                z_prev = self._sample(prior_mean, prior_logvar)
                delta = self.decoder(z_prev, current_level)
                current_level = current_level + delta
                steps.append(current_level)
            future_logits = torch.stack(steps, dim=1)
            future_01 = logit_to_iv(future_logits)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionConfig(**payload["config"])
    model = HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionModel,
    cfg: HistoryConditionedLevelAwareProbabilisticTokenStateLogitTransitionConfig,
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
