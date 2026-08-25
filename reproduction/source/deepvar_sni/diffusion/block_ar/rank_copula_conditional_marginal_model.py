from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn as nn

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class RankCopulaConditionalMarginalConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 192
    history_hidden: int = 128
    encoder_dropout: float = 0.1
    marginal_hidden: int = 256

    logit_eps: float = 1e-4
    logit_std_floor: float = 1e-3
    sample_temperature: float = 1.0
    base_model_type: str = "340c"
    base_checkpoint: str = "models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt"
    quantile_levels: tuple[float, ...] = (
        0.01,
        0.025,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.65,
        0.80,
        0.90,
        0.95,
        0.975,
        0.99,
    )


def _validate_quantile_levels(levels: Sequence[float]) -> None:
    if len(levels) < 3:
        raise ValueError("quantile_levels must contain at least three levels")
    prev = 0.0
    for level in levels:
        value = float(level)
        if not 0.0 < value < 1.0:
            raise ValueError("quantile levels must be strictly inside (0, 1)")
        if value <= prev:
            raise ValueError("quantile levels must be strictly increasing")
        prev = value


class RankCopulaConditionalMarginalModel(nn.Module):
    """492a: frozen 392a rank copula plus learned conditional marginal quantiles."""

    def __init__(
        self,
        cfg: RankCopulaConditionalMarginalConfig,
        base_model: nn.Module | None = None,
    ):
        super().__init__()
        _validate_quantile_levels(cfg.quantile_levels)
        self.cfg = cfg
        hist_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            gru_hidden_dim=cfg.history_hidden,
            bottleneck_dim=cfg.context_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=0.0,
        )
        self.history_encoder = GRUEncoder(hist_cfg)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.marginal_hidden)
        self.day_embed = nn.Embedding(cfg.future_len, cfg.marginal_hidden)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.marginal_hidden)
        self.last_level_proj = nn.Linear(1, cfg.marginal_hidden)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.encoder_dropout),
            nn.Linear(cfg.marginal_hidden, cfg.marginal_hidden),
            nn.GELU(),
            nn.Dropout(cfg.encoder_dropout),
            nn.Linear(cfg.marginal_hidden, len(cfg.quantile_levels)),
        )
        token_idx = torch.arange(cfg.future_len * cfg.n_cells)
        self.register_buffer("token_day", token_idx // cfg.n_cells)
        self.register_buffer("token_cell", token_idx % cfg.n_cells)
        self.register_buffer("cell_logit_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_logit_std", torch.ones(cfg.n_cells))
        self.__dict__["base_model"] = base_model

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

    def _to_logits(self, levels_norm: torch.Tensor) -> torch.Tensor:
        levels_norm = self._flatten(levels_norm)
        levels_01 = denormalize_iv(levels_norm)
        return iv_to_logit(levels_01, self.cfg.logit_eps)

    def _to_coord(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits - self.cell_logit_mean) / self.cell_logit_std

    def _from_coord(self, coord: torch.Tensor) -> torch.Tensor:
        return coord * self.cell_logit_std + self.cell_logit_mean

    def _token_state(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        context = self.history_encoder(history_norm)
        history_coord = self._to_coord(self._to_logits(history_norm))
        last_cell = history_coord[:, -1, self.token_cell].unsqueeze(-1)
        token_state = self.context_proj(context)[:, None]
        token_state = token_state + self.day_embed(self.token_day)[None]
        token_state = token_state + self.cell_embed(self.token_cell)[None]
        token_state = token_state + self.last_level_proj(last_cell)
        return token_state

    def marginal_quantiles(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        raw = self.head(self._token_state(history_norm)).view(
            history_norm.shape[0],
            self.cfg.future_len,
            self.cfg.n_cells,
            len(self.cfg.quantile_levels),
        )
        return torch.sort(raw, dim=-1).values

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        target = self._to_coord(self._to_logits(future_norm))
        quantiles = self.marginal_quantiles(history_norm)
        loss = self.quantile_loss(target, quantiles, self.cfg.quantile_levels)
        levels = self.cfg.quantile_levels
        median_idx = min(range(len(levels)), key=lambda idx: abs(levels[idx] - 0.5))
        lo_idx = min(range(len(levels)), key=lambda idx: abs(levels[idx] - 0.05))
        hi_idx = min(range(len(levels)), key=lambda idx: abs(levels[idx] - 0.95))
        metrics = {
            "pinball": loss.detach(),
            "target_std": target.std(unbiased=False).detach(),
            "q90_width": (quantiles[..., hi_idx] - quantiles[..., lo_idx]).mean().detach(),
            "median_abs_err": (quantiles[..., median_idx] - target).abs().mean().detach(),
        }
        return loss, metrics

    @staticmethod
    def quantile_loss(
        target: torch.Tensor,
        quantiles: torch.Tensor,
        levels: Sequence[float],
    ) -> torch.Tensor:
        level_tensor = torch.tensor(levels, device=target.device, dtype=target.dtype)
        err = target.unsqueeze(-1) - quantiles
        return torch.maximum(level_tensor * err, (level_tensor - 1.0) * err).mean()

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        temperature: float | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        if self.base_model is None:
            raise RuntimeError("RankCopulaConditionalMarginalModel requires a base model")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        quantiles = self.marginal_quantiles(history_norm)
        base_samples = self.base_model.sample_batched(
            history_norm,
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
            **kwargs,
        )
        base_flat = base_samples.view(
            base_samples.shape[0],
            base_samples.shape[1],
            self.cfg.future_len,
            self.cfg.n_cells,
        )
        # The samplewise rank copula is the only dependence object reused from 392a.
        ranks = torch.argsort(torch.argsort(base_flat, dim=1), dim=1).to(base_flat)
        u = (ranks + 0.5) / float(n_samples)
        future_coord = self._interp_quantiles(u, quantiles, self.cfg.quantile_levels)
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        if temp != 1.0:
            median = self._interp_quantiles(
                torch.full_like(u, 0.5),
                quantiles,
                self.cfg.quantile_levels,
            )
            future_coord = median + temp * (future_coord - median)
        future_01 = logit_to_iv(self._from_coord(future_coord))
        if self.cfg.n_cells == 25:
            return future_01.view(
                history_norm.shape[0],
                n_samples,
                self.cfg.future_len,
                5,
                5,
            )
        return future_01

    @staticmethod
    def _interp_quantiles(
        u: torch.Tensor,
        quantiles: torch.Tensor,
        levels: Sequence[float],
    ) -> torch.Tensor:
        level_tensor = torch.tensor(levels, device=u.device, dtype=u.dtype)
        q = quantiles[:, None]
        slope_low = (q[..., 1] - q[..., 0]) / (
            level_tensor[1] - level_tensor[0]
        )
        out = q[..., 0] + (u - level_tensor[0]) * slope_low
        for idx in range(len(levels) - 1):
            lo = level_tensor[idx]
            hi = level_tensor[idx + 1]
            weight = (u - lo) / (hi - lo)
            value = q[..., idx] + weight * (q[..., idx + 1] - q[..., idx])
            out = torch.where((u >= lo) & (u <= hi), value, out)
        slope_high = (q[..., -1] - q[..., -2]) / (
            level_tensor[-1] - level_tensor[-2]
        )
        high = q[..., -1] + (u - level_tensor[-1]) * slope_high
        return torch.where(u > level_tensor[-1], high, out)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[RankCopulaConditionalMarginalModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = RankCopulaConditionalMarginalConfig(**payload["config"])
    from experiments.backfill.block_ar._rollout_220_utils import load_one_day_kernel

    base_model, _ = load_one_day_kernel(cfg.base_model_type, cfg.base_checkpoint, device)
    model = RankCopulaConditionalMarginalModel(cfg, base_model=base_model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    model.base_model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: RankCopulaConditionalMarginalModel,
    cfg: RankCopulaConditionalMarginalConfig,
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
