from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.dual_timescale_low_rank_temporal import TemporalConvTower
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
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
class LogitTransitionFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 160
    history_hidden: int = 96
    encoder_dropout: float = 0.1

    model_hidden: int = 256
    model_layers: int = 5
    model_dropout: float = 0.1
    time_dim: int = 32
    pos_dim: int = 32
    kernel_size: int = 3
    dilation: int = 1

    logit_eps: float = 1e-4
    flow_steps: int = 32
    sample_temperature: float = 1.0


class LogitTransitionVelocity(nn.Module):
    def __init__(self, cfg: LogitTransitionFMConfig):
        super().__init__()
        self.cfg = cfg
        self.pos_embed = nn.Embedding(cfg.future_len, cfg.pos_dim)
        in_dim = cfg.n_cells + cfg.context_dim + cfg.time_dim + cfg.pos_dim
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.model_hidden),
            nn.GELU(),
            nn.Linear(cfg.model_hidden, cfg.model_hidden),
        )
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

    def forward(self, x_t: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, horizon, _ = x_t.shape
        pos_idx = torch.arange(horizon, device=x_t.device)
        pos = self.pos_embed(pos_idx)[None, :, :].expand(bsz, -1, -1)
        ctx = context[:, None, :].expand(-1, horizon, -1)
        time = _time_features(t, self.cfg.time_dim)[:, None, :].expand(-1, horizon, -1)
        hidden = self.in_proj(torch.cat([x_t, ctx, time, pos], dim=-1))
        hidden = self.backbone(hidden)
        return self.out(hidden)


class LogitTransitionFlowMatching(nn.Module):
    """301a-v0: vanilla flow over rolling support-valid logit transitions."""

    def __init__(self, cfg: LogitTransitionFMConfig):
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
        self.velocity = LogitTransitionVelocity(cfg)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten(history_norm))

    def target_transitions(self, history_norm: torch.Tensor, future_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        future_logits = iv_to_logit(denormalize_iv(future_norm), self.cfg.logit_eps)
        last_logit = iv_to_logit(denormalize_iv(history_norm[:, -1]), self.cfg.logit_eps)
        prev_logits = torch.cat([last_logit[:, None], future_logits[:, :-1]], dim=1)
        return future_logits - prev_logits

    def transitions_to_future(self, history_norm: torch.Tensor, transitions: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        curr_logit = iv_to_logit(denormalize_iv(history_norm[:, -1]), self.cfg.logit_eps)
        futures: list[torch.Tensor] = []
        for step in range(self.cfg.future_len):
            curr_logit = curr_logit + transitions[:, step]
            futures.append(logit_to_iv(curr_logit))
        return torch.stack(futures, dim=1)

    def predict_velocity(self, x_t: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.velocity(x_t, context, t)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        context = self.encode_history(history_norm)
        x1 = self.target_transitions(history_norm, future_norm)
        x0 = torch.randn_like(x1)
        bsz = history_norm.shape[0]
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.predict_velocity(x_t, context, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "transition_std": x1.std(unbiased=False).detach(),
            "transition_abs": x1.abs().mean().detach(),
            "target_velocity_std": target_velocity.std(unbiased=False).detach(),
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
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        context = self.encode_history(history_norm)
        bsz = history_norm.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)

        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            x = temp * torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=context.device,
                dtype=context.dtype,
            )
            ctx = context.repeat_interleave(k, dim=0)
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=context.device,
                    dtype=context.dtype,
                )
                x = x + dt * self.predict_velocity(x, ctx, t)
            hist = history_norm.repeat_interleave(k, dim=0)
            future_01 = self.transitions_to_future(hist, x)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[LogitTransitionFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LogitTransitionFMConfig(**payload["config"])
    model = LogitTransitionFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: LogitTransitionFlowMatching,
    cfg: LogitTransitionFMConfig,
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
