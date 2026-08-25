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


@dataclass
class ARChangeFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 128
    history_hidden: int = 64
    encoder_dropout: float = 0.1

    model_hidden: int = 192
    model_layers: int = 3
    model_dropout: float = 0.1
    time_dim: int = 32
    step_dim: int = 16
    flow_steps: int = 24


class NextChangeVelocity(nn.Module):
    def __init__(self, cfg: ARChangeFMConfig):
        super().__init__()
        self.cfg = cfg
        self.step_embed = nn.Embedding(cfg.future_len, cfg.step_dim)
        in_dim = cfg.n_cells + cfg.context_dim + cfg.time_dim + cfg.step_dim
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(cfg.model_layers):
            layers.extend(
                [
                    nn.Linear(last, cfg.model_hidden),
                    nn.GELU(),
                    nn.Dropout(cfg.model_dropout),
                ]
            )
            last = cfg.model_hidden
        layers.append(nn.Linear(last, cfg.n_cells))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        step_idx: torch.Tensor,
    ) -> torch.Tensor:
        time = _time_features(t, self.cfg.time_dim)
        step = self.step_embed(step_idx)
        inp = torch.cat([x_t, context, time, step], dim=-1)
        return self.net(inp)


class ARChangeFlowMatching(nn.Module):
    """269a-v0: autoregressive next-change flow matching with evolving generated state."""

    def __init__(self, cfg: ARChangeFMConfig):
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
        self.velocity = NextChangeVelocity(cfg)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten(history_norm))

    def one_step_loss(
        self,
        history_norm: torch.Tensor,
        next_level_norm: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_history(history_norm)
        target_change = next_level_norm - history_norm[:, -1, :]
        x0 = torch.randn_like(target_change)
        bsz = history_norm.shape[0]
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        x_t = (1.0 - t)[:, None] * x0 + t[:, None] * target_change
        target_velocity = target_change - x0
        step = torch.full((bsz,), step_idx, device=history_norm.device, dtype=torch.long)
        pred_velocity = self.velocity(x_t, context, t, step)
        return F.mse_loss(pred_velocity, target_velocity), target_velocity.std(unbiased=False)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        buf = history_norm
        losses = []
        target_stds = []
        for step in range(self.cfg.future_len):
            next_level = future_norm[:, step, :]
            loss_step, target_std = self.one_step_loss(buf, next_level, step)
            losses.append(loss_step)
            target_stds.append(target_std)
            buf = torch.cat([buf[:, 1:, :], next_level[:, None, :]], dim=1)
        total = torch.stack(losses).mean()
        metrics = {
            "total": total.detach(),
            "fm_loss": total.detach(),
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
                context = self.encode_history(hist)
                x = torch.randn(bsz * k, self.cfg.n_cells, device=hist.device, dtype=hist.dtype)
                step_idx = torch.full((bsz * k,), step, device=hist.device, dtype=torch.long)
                for fm_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (fm_step + 0.5) * dt,
                        device=hist.device,
                        dtype=hist.dtype,
                    )
                    v = self.velocity(x, context, t, step_idx)
                    x = x + dt * v
                next_level = hist[:, -1, :] + x
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


def load_model(checkpoint_path: str, device: torch.device) -> tuple[ARChangeFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ARChangeFMConfig(**payload["config"])
    model = ARChangeFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ARChangeFlowMatching,
    cfg: ARChangeFMConfig,
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
