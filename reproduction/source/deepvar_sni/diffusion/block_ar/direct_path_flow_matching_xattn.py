from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.dual_timescale_low_rank_temporal import TemporalConvTower
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


@dataclass
class DirectPathCrossAttnFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    model_hidden: int = 192
    model_layers: int = 4
    model_dropout: float = 0.1
    time_dim: int = 32
    pos_dim: int = 32
    kernel_size: int = 3
    dilation: int = 1
    n_heads: int = 4

    flow_steps: int = 24


class DirectPathCrossAttnVelocity(nn.Module):
    def __init__(self, cfg: DirectPathCrossAttnFMConfig):
        super().__init__()
        self.cfg = cfg
        self.hist_pos = nn.Embedding(cfg.history_len, cfg.pos_dim)
        self.future_pos = nn.Embedding(cfg.future_len, cfg.pos_dim)

        self.hist_proj = nn.Sequential(
            nn.Linear(cfg.n_cells + cfg.pos_dim, cfg.model_hidden),
            nn.GELU(),
            nn.Linear(cfg.model_hidden, cfg.model_hidden),
        )
        self.future_proj = nn.Sequential(
            nn.Linear(cfg.n_cells + cfg.time_dim + cfg.pos_dim, cfg.model_hidden),
            nn.GELU(),
            nn.Linear(cfg.model_hidden, cfg.model_hidden),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.model_hidden,
            num_heads=cfg.n_heads,
            dropout=cfg.model_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(cfg.model_hidden)
        self.ff = nn.Sequential(
            nn.Linear(cfg.model_hidden, cfg.model_hidden * 2),
            nn.GELU(),
            nn.Dropout(cfg.model_dropout),
            nn.Linear(cfg.model_hidden * 2, cfg.model_hidden),
        )
        self.ff_norm = nn.LayerNorm(cfg.model_hidden)
        self.backbone = TemporalConvTower(
            channels=cfg.model_hidden,
            layers=cfg.model_layers,
            kernel_size=cfg.kernel_size,
            dilation=cfg.dilation,
            dropout=cfg.model_dropout,
        )
        self.out = nn.Sequential(
            nn.Linear(cfg.model_hidden, cfg.model_hidden),
            nn.GELU(),
            nn.Linear(cfg.model_hidden, cfg.n_cells),
        )

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        bsz, hist_len, _ = history_norm.shape
        pos_idx = torch.arange(hist_len, device=history_norm.device)
        pos = self.hist_pos(pos_idx)[None, :, :].expand(bsz, -1, -1)
        return self.hist_proj(torch.cat([history_norm, pos], dim=-1))

    def forward(self, x_t: torch.Tensor, history_norm: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, horizon, _ = x_t.shape
        pos_idx = torch.arange(horizon, device=x_t.device)
        pos = self.future_pos(pos_idx)[None, :, :].expand(bsz, -1, -1)
        time = _time_features(t, self.cfg.time_dim)[:, None, :].expand(-1, horizon, -1)
        future = self.future_proj(torch.cat([x_t, time, pos], dim=-1))
        hist = self.encode_history(history_norm)

        attn_out, _ = self.cross_attn(future, hist, hist, need_weights=False)
        hidden = self.attn_norm(future + attn_out)
        hidden = self.ff_norm(hidden + self.ff(hidden))
        hidden = self.backbone(hidden)
        return self.out(hidden)


class DirectPathCrossAttnFlowMatching(nn.Module):
    """268b-v0: direct conditional path-space FM with sequence-aware history conditioning."""

    def __init__(self, cfg: DirectPathCrossAttnFMConfig):
        super().__init__()
        self.cfg = cfg
        self.velocity = DirectPathCrossAttnVelocity(cfg)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        x1 = future_norm
        x0 = torch.randn_like(x1)
        bsz = history_norm.shape[0]
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.velocity(x_t, history_norm, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "target_std": target_velocity.std(unbiased=False).detach(),
        }
        return fm_loss, metrics

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
            x = torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=history_norm.device,
                dtype=history_norm.dtype,
            )
            hist = history_norm.repeat_interleave(k, dim=0)
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=history_norm.device,
                    dtype=history_norm.dtype,
                )
                v = self.velocity(x, hist, t)
                x = x + dt * v
            future_01 = denormalize_iv(x)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DirectPathCrossAttnFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = DirectPathCrossAttnFMConfig(**payload["config"])
    model = DirectPathCrossAttnFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: DirectPathCrossAttnFlowMatching,
    cfg: DirectPathCrossAttnFMConfig,
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
