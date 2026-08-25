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


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods.extend([nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)])
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


@dataclass
class ARLatentBottleneckFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    latent_dim: int = 24
    history_hidden: int = 64
    encoder_dropout: float = 0.1

    transition_hidden: int = 128
    transition_layers: int = 3
    transition_dropout: float = 0.1
    time_dim: int = 32
    step_dim: int = 16

    decoder_hidden: int = 128
    decoder_layers: int = 2
    decoder_dropout: float = 0.1

    flow_steps: int = 24
    recon_weight: float = 1.0
    fm_weight: float = 1.0


class LatentTransitionVelocity(nn.Module):
    def __init__(self, cfg: ARLatentBottleneckFMConfig):
        super().__init__()
        self.cfg = cfg
        self.step_embed = nn.Embedding(cfg.future_len, cfg.step_dim)
        in_dim = cfg.latent_dim + cfg.latent_dim + cfg.time_dim + cfg.step_dim
        self.net = _mlp(
            in_dim=in_dim,
            out_dim=cfg.latent_dim,
            hidden=cfg.transition_hidden,
            layers=cfg.transition_layers,
            dropout=cfg.transition_dropout,
        )

    def forward(
        self,
        delta_t: torch.Tensor,
        latent_prev: torch.Tensor,
        t: torch.Tensor,
        step_idx: torch.Tensor,
    ) -> torch.Tensor:
        time = _time_features(t, self.cfg.time_dim)
        step = self.step_embed(step_idx)
        return self.net(torch.cat([delta_t, latent_prev, time, step], dim=-1))


class ARLatentBottleneckFlowMatching(nn.Module):
    """270a-v0: autoregressive latent-bottleneck next-change flow matching."""

    def __init__(self, cfg: ARLatentBottleneckFMConfig):
        super().__init__()
        self.cfg = cfg
        enc_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            gru_hidden_dim=cfg.history_hidden,
            bottleneck_dim=cfg.latent_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=0.0,
        )
        self.encoder = GRUEncoder(enc_cfg)
        self.latent_norm = nn.LayerNorm(cfg.latent_dim)
        self.transition = LatentTransitionVelocity(cfg)
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

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.latent_norm(self.encoder(self._flatten(history_norm)))

    def decode_next_level(self, latent_next: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_next)

    def step_losses(
        self,
        history_norm: torch.Tensor,
        next_level_norm: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_prev = self.encode_history(history_norm)
        next_history = torch.cat([history_norm[:, 1:, :], next_level_norm[:, None, :]], dim=1)
        latent_next = self.encode_history(next_history)
        target_delta = latent_next - latent_prev

        x0 = torch.randn_like(target_delta)
        bsz = history_norm.shape[0]
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        x_t = (1.0 - t)[:, None] * x0 + t[:, None] * target_delta
        target_velocity = target_delta - x0
        step = torch.full((bsz,), step_idx, device=history_norm.device, dtype=torch.long)
        pred_velocity = self.transition(x_t, latent_prev, t, step)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)

        decoded_next = self.decode_next_level(latent_next)
        recon_loss = F.smooth_l1_loss(decoded_next, next_level_norm)
        target_std = target_velocity.std(unbiased=False)
        return fm_loss, recon_loss, target_std

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        buf = history_norm
        fm_losses = []
        recon_losses = []
        target_stds = []
        for step in range(self.cfg.future_len):
            next_level = future_norm[:, step, :]
            fm_loss, recon_loss, target_std = self.step_losses(buf, next_level, step)
            fm_losses.append(fm_loss)
            recon_losses.append(recon_loss)
            target_stds.append(target_std)
            buf = torch.cat([buf[:, 1:, :], next_level[:, None, :]], dim=1)
        mean_fm = torch.stack(fm_losses).mean()
        mean_recon = torch.stack(recon_losses).mean()
        total = self.cfg.fm_weight * mean_fm + self.cfg.recon_weight * mean_recon
        metrics = {
            "total": total.detach(),
            "fm_loss": mean_fm.detach(),
            "recon_loss": mean_recon.detach(),
            "target_std": torch.stack(target_stds).mean().detach(),
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
        bsz = history_norm.shape[0]
        outs: list[torch.Tensor] = []
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        dt = 1.0 / float(self.cfg.flow_steps)

        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            hist = history_norm.repeat_interleave(k, dim=0).clone()
            steps: list[torch.Tensor] = []
            for step in range(self.cfg.future_len):
                latent_prev = self.encode_history(hist)
                delta = torch.randn(bsz * k, self.cfg.latent_dim, device=hist.device, dtype=hist.dtype)
                step_idx = torch.full((bsz * k,), step, device=hist.device, dtype=torch.long)
                for fm_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (fm_step + 0.5) * dt,
                        device=hist.device,
                        dtype=hist.dtype,
                    )
                    v = self.transition(delta, latent_prev, t, step_idx)
                    delta = delta + dt * v
                latent_next = latent_prev + delta
                next_level = self.decode_next_level(latent_next)
                steps.append(next_level)
                hist = torch.cat([hist[:, 1:, :], next_level[:, None, :]], dim=1)
            future_norm = torch.stack(steps, dim=1)
            future_01 = denormalize_iv(future_norm)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[ARLatentBottleneckFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ARLatentBottleneckFMConfig(**payload["config"])
    model = ARLatentBottleneckFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ARLatentBottleneckFlowMatching,
    cfg: ARLatentBottleneckFMConfig,
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
