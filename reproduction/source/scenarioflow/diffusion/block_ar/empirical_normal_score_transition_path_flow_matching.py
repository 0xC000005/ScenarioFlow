from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.conditional_masked_path_flow_matching import MaskedPathAxialBlock
from diffusion.block_ar.joint_token_logit_transition_flow_matching import _flow_time_features
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class EmpiricalNormalScoreTransitionPathFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    token_dim: int = 128
    token_layers: int = 4
    token_ff: int = 256
    model_dropout: float = 0.1
    global_mixer: bool = True
    flow_time_dim: int = 32
    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    flow_steps: int = 32
    sample_temperature: float = 1.0
    max_sample_chunk: int = 16


class EmpiricalNormalScoreTransitionPathVelocity(nn.Module):
    """Velocity over future score transitions, conditioned on noised score path."""

    def __init__(self, cfg: EmpiricalNormalScoreTransitionPathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.history_len + cfg.future_len
        self.value_proj = nn.Linear(2, cfg.token_dim)
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
        prev = torch.cat([path_t[:, :1], path_t[:, :-1]], dim=1)
        values = torch.stack([path_t, path_t - prev], dim=-1)
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


class EmpiricalNormalScoreTransitionPathFlowMatching(nn.Module):
    """Full future transition-path flow in empirical normal-score coordinates."""

    def __init__(self, cfg: EmpiricalNormalScoreTransitionPathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.velocity = EmpiricalNormalScoreTransitionPathVelocity(cfg)
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("level_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def set_empirical_quantiles(
        self,
        level_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if level_quantiles.shape != expected:
            raise ValueError(f"Expected level quantiles with shape {expected}")
        self.level_quantiles.copy_(level_quantiles.to(self.level_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("Expected quantile_levels with shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("Empirical quantiles must be set before training or sampling")

    def _values_to_scores(self, values_01: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        values_01 = self._flatten(values_01)
        levels = self.quantile_levels.to(device=values_01.device, dtype=values_01.dtype)
        table = self.level_quantiles.to(device=values_01.device, dtype=values_01.dtype)
        cols: list[torch.Tensor] = []
        for cell in range(self.cfg.n_cells):
            q = table[cell]
            flat = values_01[..., cell].reshape(-1)
            idx = torch.searchsorted(q.contiguous(), flat.contiguous(), right=False)
            idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
            idx_lo = idx_hi - 1
            q_lo = q[idx_lo]
            q_hi = q[idx_hi]
            u_lo = levels[idx_lo]
            u_hi = levels[idx_hi]
            alpha = (flat - q_lo) / (q_hi - q_lo).clamp_min(1e-12)
            u = u_lo + alpha.clamp(0.0, 1.0) * (u_hi - u_lo)
            u = torch.where(flat <= q[0], levels[0], u)
            u = torch.where(flat >= q[-1], levels[-1], u)
            eps = float(self.cfg.cdf_eps)
            z = torch.special.ndtri(u.clamp(eps, 1.0 - eps))
            cols.append(z.view(values_01.shape[:-1]))
        return torch.stack(cols, dim=-1)

    def _scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = self.level_quantiles.to(device=scores.device, dtype=scores.dtype)
        eps = float(self.cfg.cdf_eps)
        u_all = (0.5 * (1.0 + torch.erf(scores / math.sqrt(2.0)))).clamp(eps, 1.0 - eps)
        cols: list[torch.Tensor] = []
        for cell in range(self.cfg.n_cells):
            q = table[cell]
            flat = u_all[..., cell].reshape(-1)
            idx = torch.searchsorted(levels.contiguous(), flat.contiguous(), right=False)
            idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
            idx_lo = idx_hi - 1
            u_lo = levels[idx_lo]
            u_hi = levels[idx_hi]
            q_lo = q[idx_lo]
            q_hi = q[idx_hi]
            alpha = (flat - u_lo) / (u_hi - u_lo).clamp_min(1e-12)
            x = q_lo + alpha.clamp(0.0, 1.0) * (q_hi - q_lo)
            x = torch.where(flat <= levels[0], q[0], x)
            x = torch.where(flat >= levels[-1], q[-1], x)
            cols.append(x.view(scores.shape[:-1]))
        return torch.stack(cols, dim=-1).clamp(0.0, 1.0)

    def history_scores(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_01 = denormalize_iv(self._flatten(history_norm))
        return self._values_to_scores(history_01)

    def target_future_scores(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_01 = denormalize_iv(self._flatten(future_norm))
        return self._values_to_scores(future_01)

    @staticmethod
    def score_transitions(
        history_scores: torch.Tensor,
        future_scores: torch.Tensor,
    ) -> torch.Tensor:
        prev = torch.cat([history_scores[:, -1:], future_scores[:, :-1]], dim=1)
        return future_scores - prev

    def path_from_transitions(
        self,
        history_scores: torch.Tensor,
        transitions: torch.Tensor,
    ) -> torch.Tensor:
        future_scores = history_scores[:, -1:] + transitions.cumsum(dim=1)
        return torch.cat([history_scores, future_scores], dim=1)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_z = self.history_scores(history_norm)
        future_z = self.target_future_scores(future_norm)
        x1 = self.score_transitions(history_z, future_z)
        x0 = torch.randn_like(x1)
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        path_t = self.path_from_transitions(history_z, x_t)
        target_velocity = x1 - x0
        pred_velocity = self.velocity(path_t, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "future_score_std": future_z.std(unbiased=False).detach(),
            "transition_std": x1.std(unbiased=False).detach(),
            "transition_abs": x1.abs().mean().detach(),
            "path_t_std": path_t[:, self.cfg.history_len :].std(unbiased=False).detach(),
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
        history_z = self.history_scores(history_norm)
        bsz = history_z.shape[0]
        chunk_size = max(
            1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk))
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            hist = history_z.repeat_interleave(k, dim=0)
            x = temp * torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=history_z.device,
                dtype=history_z.dtype,
            )
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=history_z.device,
                    dtype=history_z.dtype,
                )
                path_t = self.path_from_transitions(hist, x)
                x = x + dt * self.velocity(path_t, t)
            future_scores = hist[:, -1:] + x.cumsum(dim=1)
            future_01 = self._scores_to_values(future_scores[:, :n_steps])
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
        return self.sample_batched(
            normalize_iv(history_01),
            n_samples=n_samples,
            n_steps=1,
            history_is_normalized=True,
            **kwargs,
        )[:, :, 0]


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[EmpiricalNormalScoreTransitionPathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScoreTransitionPathFMConfig(**payload["config"])
    model = EmpiricalNormalScoreTransitionPathFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScoreTransitionPathFlowMatching,
    cfg: EmpiricalNormalScoreTransitionPathFMConfig,
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
