from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.joint_token_logit_transition_flow_matching import _flow_time_features
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class ConditionalMaskedPathFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    token_dim: int = 128
    token_layers: int = 6
    token_ff: int = 256
    model_dropout: float = 0.1
    global_mixer: bool = False
    transition_features: bool = False
    flow_time_dim: int = 32
    logit_eps: float = 1e-4
    standardize_logits: bool = True
    logit_std_floor: float = 1e-3
    flow_steps: int = 32
    sample_temperature: float = 1.0
    max_sample_chunk: int = 16


class MaskedPathAxialBlock(nn.Module):
    """Efficient mixing over time, cells, and channels for a 60x25 path grid."""

    def __init__(
        self,
        seq_len: int,
        n_cells: int,
        token_dim: int,
        token_ff: int,
        dropout: float,
        global_mixer: bool,
    ):
        super().__init__()
        self.time_norm = nn.LayerNorm(seq_len)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, 2 * seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * seq_len, seq_len),
        )
        self.cell_norm = nn.LayerNorm(n_cells)
        self.cell_mlp = nn.Sequential(
            nn.Linear(n_cells, 2 * n_cells),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * n_cells, n_cells),
        )
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, token_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_ff, token_dim),
        )
        self.use_global_mixer = bool(global_mixer)
        if self.use_global_mixer:
            self.global_norm = nn.LayerNorm(token_dim)
            self.global_mlp = nn.Sequential(
                nn.Linear(token_dim, token_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(token_ff, token_dim),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, cell, channel)
        y = self.time_norm(x.permute(0, 2, 3, 1))
        x = x + self.dropout(self.time_mlp(y).permute(0, 3, 1, 2))

        y = self.cell_norm(x.permute(0, 1, 3, 2))
        x = x + self.dropout(self.cell_mlp(y).permute(0, 1, 3, 2))

        x = x + self.dropout(self.channel_mlp(self.channel_norm(x)))
        if self.use_global_mixer:
            global_state = self.global_norm(x.mean(dim=(1, 2)))
            x = x + self.dropout(self.global_mlp(global_state))[:, None, None, :]
        return x


class ConditionalMaskedPathVelocity(nn.Module):
    def __init__(self, cfg: ConditionalMaskedPathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.history_len + cfg.future_len
        self.value_proj = nn.Linear(2 if cfg.transition_features else 1, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.time_embed = nn.Embedding(self.seq_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.type_embed = nn.Embedding(2, cfg.token_dim)
        self.blocks = nn.ModuleList(
            [
                MaskedPathAxialBlock(
                    seq_len=self.seq_len,
                    n_cells=cfg.n_cells,
                    token_dim=cfg.token_dim,
                    token_ff=cfg.token_ff,
                    dropout=cfg.model_dropout,
                    global_mixer=cfg.global_mixer,
                )
                for _ in range(cfg.token_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def forward(self, path_t: torch.Tensor, flow_t: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_cells = path_t.shape
        if seq_len != self.seq_len or n_cells != self.cfg.n_cells:
            raise ValueError(
                f"Expected path shape (*,{self.seq_len},{self.cfg.n_cells}), "
                f"got {tuple(path_t.shape)}"
            )
        time_ids = torch.arange(seq_len, device=path_t.device)
        cell_ids = torch.arange(n_cells, device=path_t.device)
        type_ids = torch.cat(
            [
                torch.zeros(self.cfg.history_len, device=path_t.device, dtype=torch.long),
                torch.ones(self.cfg.future_len, device=path_t.device, dtype=torch.long),
            ],
            dim=0,
        )
        if self.cfg.transition_features:
            prev = torch.cat([path_t[:, :1], path_t[:, :-1]], dim=1)
            values = torch.stack([path_t, path_t - prev], dim=-1)
        else:
            values = path_t[..., None]
        x = self.value_proj(values)
        x = x + self.time_embed(time_ids)[None, :, None, :]
        x = x + self.cell_embed(cell_ids)[None, None, :, :]
        x = x + self.type_embed(type_ids)[None, :, None, :]
        x = x + self.flow_time_proj(_flow_time_features(flow_t, self.cfg.flow_time_dim))[
            :, None, None, :
        ]
        for block in self.blocks:
            x = block(x)
        future = x[:, self.cfg.history_len :, :, :]
        return self.out(future).squeeze(-1)


class ConditionalMaskedPathFlowMatching(nn.Module):
    """333a: vanilla full-future rectified flow conditioned by clean history tokens."""

    def __init__(self, cfg: ConditionalMaskedPathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.velocity = ConditionalMaskedPathVelocity(cfg)
        self.register_buffer("cell_logit_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_logit_std", torch.ones(cfg.n_cells))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def set_logit_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.cfg.n_cells,) or std.shape != (self.cfg.n_cells,):
            raise ValueError("Expected per-cell logit stats with shape (n_cells,)")
        self.cell_logit_mean.copy_(mean.to(self.cell_logit_mean))
        self.cell_logit_std.copy_(
            std.clamp_min(self.cfg.logit_std_floor).to(self.cell_logit_std)
        )

    def _to_model_coord(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return logits
        return (logits - self.cell_logit_mean) / self.cell_logit_std

    def _from_model_coord(self, coord: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return coord
        return coord * self.cell_logit_std + self.cell_logit_mean

    def history_logits(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        logits = iv_to_logit(denormalize_iv(history_norm), self.cfg.logit_eps)
        return self._to_model_coord(logits)

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        logits = iv_to_logit(denormalize_iv(future_norm), self.cfg.logit_eps)
        return self._to_model_coord(logits)

    def implied_transitions(
        self,
        history_logits: torch.Tensor,
        future_logits: torch.Tensor,
    ) -> torch.Tensor:
        last = history_logits[:, -1]
        return torch.cat(
            [
                future_logits[:, :1] - last[:, None, :],
                future_logits[:, 1:] - future_logits[:, :-1],
            ],
            dim=1,
        )

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_logits = self.history_logits(history_norm)
        x1 = self.target_future_logits(future_norm)
        x0 = torch.randn_like(x1)
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        path_t = torch.cat([history_logits, x_t], dim=1)
        target_velocity = x1 - x0
        pred_velocity = self.velocity(path_t, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        trans = self.implied_transitions(history_logits, x1)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "future_logit_std": x1.std(unbiased=False).detach(),
            "future_logit_abs": x1.abs().mean().detach(),
            "implied_transition_std": trans.std(unbiased=False).detach(),
            "implied_transition_abs": trans.abs().mean().detach(),
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
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(f"Expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_logits = self.history_logits(history_norm)
        bsz = history_logits.shape[0]
        chunk_size = max(
            1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk))
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            x = temp * torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=history_logits.device,
                dtype=history_logits.dtype,
            )
            hist = history_logits.repeat_interleave(k, dim=0)
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=history_logits.device,
                    dtype=history_logits.dtype,
                )
                path_t = torch.cat([hist, x], dim=1)
                x = x + dt * self.velocity(path_t, t)
            future_01 = logit_to_iv(self._from_model_coord(x[:, :n_steps]))
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, n_steps, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, n_steps, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        **kwargs: object,
    ) -> torch.Tensor:
        history_norm = normalize_iv(history_01)
        return self.sample_batched(
            history_norm,
            n_samples=n_samples,
            n_steps=1,
            history_is_normalized=True,
            **kwargs,
        )[:, :, 0]


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ConditionalMaskedPathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ConditionalMaskedPathFMConfig(**payload["config"])
    model = ConditionalMaskedPathFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ConditionalMaskedPathFlowMatching,
    cfg: ConditionalMaskedPathFMConfig,
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
