from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

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
class ARSurfaceTokenFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    n_tokens: int = 4
    token_dim: int = 16
    encoder_hidden: int = 128
    encoder_layers: int = 2
    encoder_dropout: float = 0.1

    transition_layers: int = 2
    transition_heads: int = 4
    transition_ff_mult: int = 4
    transition_dropout: float = 0.1
    time_dim: int = 32

    decoder_hidden: int = 128
    decoder_layers: int = 2
    decoder_dropout: float = 0.1

    flow_steps: int = 24
    recon_weight: float = 1.0
    fm_weight: float = 1.0


class TokenVelocity(nn.Module):
    def __init__(self, cfg: ARSurfaceTokenFMConfig):
        super().__init__()
        self.cfg = cfg
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_tokens, cfg.token_dim))
        self.in_proj = nn.Linear(2 * cfg.token_dim + cfg.time_dim, cfg.token_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.transition_heads,
            dim_feedforward=cfg.transition_ff_mult * cfg.token_dim,
            dropout=cfg.transition_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.transition_layers)
        self.out_proj = nn.Linear(cfg.token_dim, cfg.token_dim)

    def forward(
        self,
        delta_t: torch.Tensor,
        tokens_prev: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        time = _time_features(t, self.cfg.time_dim)[:, None, :].expand(-1, self.cfg.n_tokens, -1)
        x = torch.cat([delta_t, tokens_prev, time], dim=-1)
        h = self.in_proj(x) + self.pos
        h = self.encoder(h)
        return self.out_proj(h)


class ARSurfaceTokenFlowMatching(nn.Module):
    """273a-v0: token latent state-space FM with explicit observation model."""

    def __init__(self, cfg: ARSurfaceTokenFMConfig):
        super().__init__()
        self.cfg = cfg
        latent_dim = cfg.n_tokens * cfg.token_dim
        self.encoder = _mlp(
            in_dim=cfg.n_cells,
            out_dim=latent_dim,
            hidden=cfg.encoder_hidden,
            layers=cfg.encoder_layers,
            dropout=cfg.encoder_dropout,
        )
        self.transition = TokenVelocity(cfg)
        self.decoder = _mlp(
            in_dim=latent_dim,
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
        flat = self.encoder(level_norm)
        return flat.view(level_norm.shape[0], self.cfg.n_tokens, self.cfg.token_dim)

    def decode_surface(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        flat = latent_tokens.reshape(latent_tokens.shape[0], -1)
        return self.decoder(flat)

    def step_losses(
        self,
        prev_level_norm: torch.Tensor,
        next_level_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_prev = self.encode_surface(prev_level_norm)
        z_next = self.encode_surface(next_level_norm)
        target_delta = z_next - z_prev

        x0 = torch.randn_like(target_delta)
        bsz = prev_level_norm.shape[0]
        t = torch.rand(bsz, device=prev_level_norm.device, dtype=prev_level_norm.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * target_delta
        target_velocity = target_delta - x0
        pred_velocity = self.transition(x_t, z_prev, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)

        recon_prev = self.decode_surface(z_prev)
        recon_next = self.decode_surface(z_next)
        recon_loss = 0.5 * (
            F.smooth_l1_loss(recon_prev, prev_level_norm) +
            F.smooth_l1_loss(recon_next, next_level_norm)
        )
        target_std = target_velocity.std(unbiased=False)
        return fm_loss, recon_loss, target_std

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        prev_level = history_norm[:, -1, :]
        fm_losses = []
        recon_losses = []
        target_stds = []
        for step in range(self.cfg.future_len):
            next_level = future_norm[:, step, :]
            fm_loss, recon_loss, target_std = self.step_losses(prev_level, next_level)
            fm_losses.append(fm_loss)
            recon_losses.append(recon_loss)
            target_stds.append(target_std)
            prev_level = next_level
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
            prev_level = history_norm[:, -1, :].repeat_interleave(k, dim=0).clone()
            steps: list[torch.Tensor] = []
            for _step in range(self.cfg.future_len):
                z_prev = self.encode_surface(prev_level)
                delta = torch.randn_like(z_prev)
                for fm_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (fm_step + 0.5) * dt,
                        device=prev_level.device,
                        dtype=prev_level.dtype,
                    )
                    v = self.transition(delta, z_prev, t)
                    delta = delta + dt * v
                z_next = z_prev + delta
                next_level = self.decode_surface(z_next)
                steps.append(next_level)
                prev_level = next_level
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
) -> tuple[ARSurfaceTokenFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ARSurfaceTokenFMConfig(**payload["config"])
    model = ARSurfaceTokenFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ARSurfaceTokenFlowMatching,
    cfg: ARSurfaceTokenFMConfig,
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
