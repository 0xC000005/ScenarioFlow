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
class ARSurfaceLatentMemoryFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    latent_dim: int = 16
    encoder_hidden: int = 128
    encoder_layers: int = 2
    encoder_dropout: float = 0.1

    memory_dim: int = 32
    transition_hidden: int = 128
    transition_layers: int = 3
    transition_dropout: float = 0.1
    time_dim: int = 32

    decoder_hidden: int = 128
    decoder_layers: int = 2
    decoder_dropout: float = 0.1

    flow_steps: int = 24
    recon_weight: float = 1.0
    fm_weight: float = 1.0


class LatentMemoryVelocity(nn.Module):
    def __init__(self, cfg: ARSurfaceLatentMemoryFMConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            in_dim=cfg.latent_dim + cfg.latent_dim + cfg.memory_dim + cfg.time_dim,
            out_dim=cfg.latent_dim,
            hidden=cfg.transition_hidden,
            layers=cfg.transition_layers,
            dropout=cfg.transition_dropout,
        )

    def forward(
        self,
        delta_t: torch.Tensor,
        z_prev: torch.Tensor,
        memory: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([delta_t, z_prev, memory, _time_features(t, self.cfg.time_dim)], dim=-1))


class ARSurfaceLatentMemoryFlowMatching(nn.Module):
    """272b-v0: AR latent state-space FM with recurrent latent memory."""

    def __init__(self, cfg: ARSurfaceLatentMemoryFMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = _mlp(
            in_dim=cfg.n_cells,
            out_dim=cfg.latent_dim,
            hidden=cfg.encoder_hidden,
            layers=cfg.encoder_layers,
            dropout=cfg.encoder_dropout,
        )
        self.memory_cell = nn.GRUCell(cfg.latent_dim, cfg.memory_dim)
        self.transition = LatentMemoryVelocity(cfg)
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
        return self.decoder(latent)

    def init_memory(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history_norm = self._flatten(history_norm)
        bsz = history_norm.shape[0]
        memory = torch.zeros(bsz, self.cfg.memory_dim, device=history_norm.device, dtype=history_norm.dtype)
        z_prev = None
        for step in range(history_norm.shape[1]):
            z = self.encode_surface(history_norm[:, step, :])
            memory = self.memory_cell(z, memory)
            z_prev = z
        assert z_prev is not None
        return z_prev, memory

    def step_losses(
        self,
        z_prev: torch.Tensor,
        memory: torch.Tensor,
        prev_level_norm: torch.Tensor,
        next_level_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_next = self.encode_surface(next_level_norm)
        target_delta = z_next - z_prev

        x0 = torch.randn_like(target_delta)
        bsz = prev_level_norm.shape[0]
        t = torch.rand(bsz, device=prev_level_norm.device, dtype=prev_level_norm.dtype)
        x_t = (1.0 - t)[:, None] * x0 + t[:, None] * target_delta
        target_velocity = target_delta - x0
        pred_velocity = self.transition(x_t, z_prev, memory, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)

        recon_prev = self.decode_surface(z_prev)
        recon_next = self.decode_surface(z_next)
        recon_loss = 0.5 * (
            F.smooth_l1_loss(recon_prev, prev_level_norm) +
            F.smooth_l1_loss(recon_next, next_level_norm)
        )
        next_memory = self.memory_cell(z_next, memory)
        target_std = target_velocity.std(unbiased=False)
        return fm_loss, recon_loss, target_std, z_next, next_memory

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        z_prev, memory = self.init_memory(history_norm)
        prev_level = history_norm[:, -1, :]
        fm_losses = []
        recon_losses = []
        target_stds = []
        for step in range(self.cfg.future_len):
            next_level = future_norm[:, step, :]
            fm_loss, recon_loss, target_std, z_next, memory = self.step_losses(
                z_prev, memory, prev_level, next_level
            )
            fm_losses.append(fm_loss)
            recon_losses.append(recon_loss)
            target_stds.append(target_std)
            z_prev = z_next
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
            hist = history_norm.repeat_interleave(k, dim=0).clone()
            z_prev, memory = self.init_memory(hist)
            prev_level = hist[:, -1, :]
            steps: list[torch.Tensor] = []
            for _step in range(self.cfg.future_len):
                delta = torch.randn_like(z_prev)
                for fm_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (fm_step + 0.5) * dt,
                        device=prev_level.device,
                        dtype=prev_level.dtype,
                    )
                    v = self.transition(delta, z_prev, memory, t)
                    delta = delta + dt * v
                z_next = z_prev + delta
                next_level = self.decode_surface(z_next)
                steps.append(next_level)
                memory = self.memory_cell(z_next, memory)
                z_prev = z_next
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
) -> tuple[ARSurfaceLatentMemoryFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ARSurfaceLatentMemoryFMConfig(**payload["config"])
    model = ARSurfaceLatentMemoryFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ARSurfaceLatentMemoryFlowMatching,
    cfg: ARSurfaceLatentMemoryFMConfig,
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
