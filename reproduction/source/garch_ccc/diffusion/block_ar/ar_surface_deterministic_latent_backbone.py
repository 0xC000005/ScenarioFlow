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
class ARSurfaceDeterministicLatentConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    latent_dim: int = 32
    encoder_hidden: int = 128
    encoder_layers: int = 2
    encoder_dropout: float = 0.1

    decoder_hidden: int = 128
    decoder_layers: int = 2
    decoder_dropout: float = 0.1

    latent_dropout: float = 0.1

    level_weight: float = 1.0
    latent_weight: float = 0.5
    recon_weight: float = 0.5


class ARSurfaceDeterministicLatentBackbone(nn.Module):
    """275b-v0: deterministic latent-state backbone with explicit observation model."""

    def __init__(self, cfg: ARSurfaceDeterministicLatentConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = _mlp(
            in_dim=cfg.n_cells,
            out_dim=cfg.latent_dim,
            hidden=cfg.encoder_hidden,
            layers=cfg.encoder_layers,
            dropout=cfg.encoder_dropout,
        )
        self.history_rnn = nn.GRU(
            input_size=cfg.latent_dim,
            hidden_size=cfg.latent_dim,
            num_layers=1,
            batch_first=True,
        )
        self.transition_cell = nn.GRUCell(cfg.latent_dim, cfg.latent_dim)
        self.transition_proj = _mlp(
            in_dim=cfg.latent_dim,
            out_dim=cfg.latent_dim,
            hidden=cfg.latent_dim,
            layers=2,
            dropout=cfg.latent_dropout,
        )
        self.decoder = _mlp(
            in_dim=cfg.latent_dim,
            out_dim=cfg.n_cells,
            hidden=cfg.decoder_hidden,
            layers=cfg.decoder_layers,
            dropout=cfg.decoder_dropout,
        )

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def encode_surface(self, level_norm: torch.Tensor) -> torch.Tensor:
        return self.encoder(level_norm)

    def decode_surface(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.decoder(latent), -1.0, 1.0)

    def encode_history_state(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hist = self._flatten(history_norm)
        z_hist = self.encode_surface(hist.reshape(-1, self.cfg.n_cells)).view(hist.shape[0], hist.shape[1], -1)
        _, h_n = self.history_rnn(z_hist)
        return z_hist[:, -1], h_n[-1]

    def rollout(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_prev, hidden = self.encode_history_state(history_norm)
        pred_levels = []
        pred_latents = []
        for _ in range(self.cfg.future_len):
            hidden = self.transition_cell(z_prev, hidden)
            z_next = self.transition_proj(hidden)
            next_level = self.decode_surface(z_next)
            pred_levels.append(next_level)
            pred_latents.append(z_next)
            z_prev = z_next
        return torch.stack(pred_levels, dim=1), torch.stack(pred_latents, dim=1)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future = self._flatten(future_norm)
        pred_levels, pred_latents = self.rollout(history_norm)

        target_latents = self.encode_surface(future.reshape(-1, self.cfg.n_cells)).view(
            future.shape[0], future.shape[1], -1
        )
        recon_future = self.decode_surface(target_latents.reshape(-1, self.cfg.latent_dim)).view_as(future)

        level_loss = F.smooth_l1_loss(pred_levels, future)
        latent_loss = F.smooth_l1_loss(pred_latents, target_latents)
        recon_loss = F.smooth_l1_loss(recon_future, future)
        total = (
            self.cfg.level_weight * level_loss
            + self.cfg.latent_weight * latent_loss
            + self.cfg.recon_weight * recon_loss
        )
        metrics = {
            "total": total.detach(),
            "level_loss": level_loss.detach(),
            "latent_loss": latent_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "pred_level_std": pred_levels.std(unbiased=False).detach(),
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
        pred_levels, _ = self.rollout(history_norm)
        future_01 = denormalize_iv(pred_levels)
        if self.cfg.n_cells == 25:
            future_01 = future_01.view(future_01.shape[0], self.cfg.future_len, 5, 5)
        else:
            future_01 = future_01.view(future_01.shape[0], self.cfg.future_len, self.cfg.n_cells)
        return future_01.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ARSurfaceDeterministicLatentBackbone, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ARSurfaceDeterministicLatentConfig(**payload["config"])
    model = ARSurfaceDeterministicLatentBackbone(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ARSurfaceDeterministicLatentBackbone,
    cfg: ARSurfaceDeterministicLatentConfig,
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
