from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.future_scalar_ar_mixture_density_model import (
    FutureScalarARMixtureDensityModel,
    load_model as load_scalar_ar_model,
)
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class ConditionalMarginalCopulaConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 192
    history_hidden: int = 128
    encoder_dropout: float = 0.1
    marginal_hidden: int = 256

    logit_eps: float = 1e-4
    logit_std_floor: float = 1e-3
    scale_floor: float = 1e-3
    sample_temperature: float = 1.0
    max_sample_chunk: int = 8
    base_checkpoint: str = "models/backfill/321c_v0_s42/best_model.pt"
    marginal_family: str = "gaussian"
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


@dataclass
class EmpiricalMarginalTransportConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    logit_eps: float = 1e-4
    max_sample_chunk: int = 8
    base_checkpoint: str = "models/backfill/321c_v0_s42/best_model.pt"
    n_quantiles: int = 101


@dataclass
class StateConditionalEmpiricalTransportConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    logit_eps: float = 1e-4
    max_sample_chunk: int = 8
    base_checkpoint: str = "models/backfill/321c_v0_s42/best_model.pt"
    n_quantiles: int = 101
    n_level_bins: int = 5
    n_vol_bins: int = 3


class ConditionalMarginalCopulaModel(nn.Module):
    """323a: conditional scalar marginals plus an empirical neural copula.

    The marginal law is learned directly from history. Dependence is supplied by
    the rank ordering of a clean scalar AR model, so sampling uses the identity
    "joint law = marginals + copula" rather than an additive correction path.
    """

    def __init__(
        self,
        cfg: ConditionalMarginalCopulaConfig,
        base_model: FutureScalarARMixtureDensityModel | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.marginal_family not in {"gaussian", "quantile"}:
            raise ValueError("marginal_family must be 'gaussian' or 'quantile'")
        if cfg.marginal_family == "quantile":
            _validate_quantile_levels(cfg.quantile_levels)
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
        head_dim = 2 if cfg.marginal_family == "gaussian" else len(cfg.quantile_levels)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.encoder_dropout),
            nn.Linear(cfg.marginal_hidden, cfg.marginal_hidden),
            nn.GELU(),
            nn.Dropout(cfg.encoder_dropout),
            nn.Linear(cfg.marginal_hidden, head_dim),
        )
        token_idx = torch.arange(cfg.future_len * cfg.n_cells)
        self.register_buffer("token_day", token_idx // cfg.n_cells)
        self.register_buffer("token_cell", token_idx % cfg.n_cells)
        self.register_buffer("cell_logit_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_logit_std", torch.ones(cfg.n_cells))
        # Keep the copula supplier out of this module's state_dict. The 323a
        # checkpoint stores its path and only learns the conditional marginals.
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

    def to_logits(self, levels_norm: torch.Tensor) -> torch.Tensor:
        levels_norm = self._flatten(levels_norm)
        levels_01 = denormalize_iv(levels_norm)
        return iv_to_logit(levels_01, self.cfg.logit_eps)

    def _to_model_coord(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits - self.cell_logit_mean) / self.cell_logit_std

    def _from_model_coord(self, coord: torch.Tensor) -> torch.Tensor:
        return coord * self.cell_logit_std + self.cell_logit_mean

    def marginal_params(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.marginal_family != "gaussian":
            raise RuntimeError("marginal_params is only valid for gaussian marginals")
        history_norm = self._flatten(history_norm)
        token_state = self._token_state(history_norm)
        raw = self.head(token_state)
        mean = raw[..., 0].view(history_norm.shape[0], self.cfg.future_len, self.cfg.n_cells)
        scale = (F.softplus(raw[..., 1]) + self.cfg.scale_floor).view_as(mean)
        return mean, scale

    def marginal_quantiles(self, history_norm: torch.Tensor) -> torch.Tensor:
        if self.cfg.marginal_family != "quantile":
            raise RuntimeError("marginal_quantiles is only valid for quantile marginals")
        history_norm = self._flatten(history_norm)
        token_state = self._token_state(history_norm)
        raw = self.head(token_state).view(
            history_norm.shape[0],
            self.cfg.future_len,
            self.cfg.n_cells,
            len(self.cfg.quantile_levels),
        )
        return torch.sort(raw, dim=-1).values

    def _token_state(self, history_norm: torch.Tensor) -> torch.Tensor:
        context = self.history_encoder(history_norm)
        history_coord = self._to_model_coord(self.to_logits(history_norm))
        last_cell = history_coord[:, -1, self.token_cell].unsqueeze(-1)
        token_state = self.context_proj(context)[:, None]
        token_state = token_state + self.day_embed(self.token_day)[None]
        token_state = token_state + self.cell_embed(self.token_cell)[None]
        token_state = token_state + self.last_level_proj(last_cell)
        return token_state

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        target = self._to_model_coord(self.to_logits(future_norm))
        if self.cfg.marginal_family == "gaussian":
            mean, scale = self.marginal_params(history_norm)
            z = (target - mean) / scale
            nll_grid = 0.5 * z.pow(2) + torch.log(scale) + 0.5 * math.log(2.0 * math.pi)
            loss = nll_grid.mean()
            scale_metric = scale.mean()
            mean_abs_err = (mean - target).abs().mean()
        else:
            quantiles = self.marginal_quantiles(history_norm)
            loss = self.quantile_loss(target, quantiles, self.cfg.quantile_levels)
            q_levels = self.cfg.quantile_levels
            median_idx = min(range(len(q_levels)), key=lambda idx: abs(q_levels[idx] - 0.5))
            lo_idx = min(range(len(q_levels)), key=lambda idx: abs(q_levels[idx] - 0.05))
            hi_idx = min(range(len(q_levels)), key=lambda idx: abs(q_levels[idx] - 0.95))
            scale_metric = (quantiles[..., hi_idx] - quantiles[..., lo_idx]).mean()
            mean_abs_err = (quantiles[..., median_idx] - target).abs().mean()
        metrics = {
            "nll": loss.detach(),
            "target_std": target.std(unbiased=False).detach(),
            "scale_mean": scale_metric.detach(),
            "mean_abs_err": mean_abs_err.detach(),
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
        loss = torch.maximum(level_tensor * err, (level_tensor - 1.0) * err)
        return loss.mean()

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
            raise RuntimeError("ConditionalMarginalCopulaModel requires a base copula model")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        mean = scale = quantiles = None
        if self.cfg.marginal_family == "gaussian":
            mean, scale = self.marginal_params(history_norm)
        else:
            quantiles = self.marginal_quantiles(history_norm)
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
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
        base_coord = self._to_model_coord(
            iv_to_logit(base_flat, self.cfg.logit_eps)
        )
        ranks = torch.argsort(torch.argsort(base_coord, dim=1), dim=1).to(base_coord)
        u = (ranks + 0.5) / float(n_samples)
        eps = 1.0 / (2.0 * float(n_samples) + 2.0)
        z = torch.special.ndtri(u.clamp(eps, 1.0 - eps))
        if self.cfg.marginal_family == "gaussian":
            future_coord = mean[:, None] + temp * scale[:, None] * z
        else:
            if quantiles is None:
                raise RuntimeError("quantile marginals were not computed")
            if temp != 1.0:
                median = self._interp_quantiles(
                    torch.full_like(u, 0.5),
                    quantiles,
                    self.cfg.quantile_levels,
                )
                future_coord = median + temp * (
                    self._interp_quantiles(u, quantiles, self.cfg.quantile_levels) - median
                )
            else:
                future_coord = self._interp_quantiles(
                    u,
                    quantiles,
                    self.cfg.quantile_levels,
                )
        future_01 = logit_to_iv(self._from_model_coord(future_coord))
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
        out = q[..., 0] + (u - level_tensor[0]) * (
            (q[..., 1] - q[..., 0]) / (level_tensor[1] - level_tensor[0])
        )
        for idx in range(len(levels) - 1):
            lo = level_tensor[idx]
            hi = level_tensor[idx + 1]
            weight = (u - lo) / (hi - lo)
            value = q[..., idx] + weight * (q[..., idx + 1] - q[..., idx])
            mask = (u >= lo) & (u <= hi)
            out = torch.where(mask, value, out)
        high = q[..., -1] + (u - level_tensor[-1]) * (
            (q[..., -1] - q[..., -2]) / (level_tensor[-1] - level_tensor[-2])
        )
        out = torch.where(u > level_tensor[-1], high, out)
        return out


class EmpiricalMarginalTransportCopulaModel(nn.Module):
    """323c: empirical marginal transport applied to a neural copula sampler."""

    def __init__(
        self,
        cfg: EmpiricalMarginalTransportConfig,
        source_quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor,
        base_model: FutureScalarARMixtureDensityModel | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        expected = (cfg.future_len, cfg.n_cells, cfg.n_quantiles)
        if tuple(source_quantiles.shape) != expected or tuple(target_quantiles.shape) != expected:
            raise ValueError(f"Expected quantile tensors with shape {expected}")
        if tuple(quantile_levels.shape) != (cfg.n_quantiles,):
            raise ValueError("Expected quantile_levels with shape (n_quantiles,)")
        self.register_buffer("source_quantiles", source_quantiles.float())
        self.register_buffer("target_quantiles", target_quantiles.float())
        self.register_buffer("quantile_levels", quantile_levels.float())
        self.__dict__["base_model"] = base_model

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        if self.base_model is None:
            raise RuntimeError("EmpiricalMarginalTransportCopulaModel requires a base model")
        history_norm = history if history_is_normalized else normalize_iv(history)
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
        base_logits = iv_to_logit(base_flat, self.cfg.logit_eps)
        transported = self._transport(base_logits)
        future_01 = logit_to_iv(transported)
        if self.cfg.n_cells == 25:
            return future_01.view(
                history.shape[0],
                n_samples,
                self.cfg.future_len,
                5,
                5,
            )
        return future_01

    def _transport(self, x: torch.Tensor) -> torch.Tensor:
        src = self.source_quantiles[None, None]
        tgt = self.target_quantiles[None, None]
        low_denom = (src[..., 1] - src[..., 0]).abs().clamp_min(1e-6)
        low = tgt[..., 0] + (x - src[..., 0]) * (tgt[..., 1] - tgt[..., 0]) / low_denom
        out = low
        for idx in range(self.cfg.n_quantiles - 1):
            src_lo = src[..., idx]
            src_hi = src[..., idx + 1]
            tgt_lo = tgt[..., idx]
            tgt_hi = tgt[..., idx + 1]
            denom = (src_hi - src_lo).abs().clamp_min(1e-6)
            weight = (x - src_lo) / denom
            value = tgt_lo + weight * (tgt_hi - tgt_lo)
            mask = (x >= src_lo) & (x <= src_hi)
            out = torch.where(mask, value, out)
        high_denom = (src[..., -1] - src[..., -2]).abs().clamp_min(1e-6)
        high = tgt[..., -1] + (x - src[..., -1]) * (
            tgt[..., -1] - tgt[..., -2]
        ) / high_denom
        out = torch.where(x > src[..., -1], high, out)
        return out


class StateConditionalEmpiricalTransportCopulaModel(nn.Module):
    """323d: empirical marginal transport conditioned on current state."""

    def __init__(
        self,
        cfg: StateConditionalEmpiricalTransportConfig,
        source_quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor,
        level_edges: torch.Tensor,
        vol_edges: torch.Tensor,
        base_model: FutureScalarARMixtureDensityModel | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        n_bins = cfg.n_level_bins * cfg.n_vol_bins
        expected_q = (cfg.n_cells, n_bins, cfg.future_len, cfg.n_quantiles)
        if tuple(source_quantiles.shape) != expected_q or tuple(target_quantiles.shape) != expected_q:
            raise ValueError(f"Expected conditional quantile tensors with shape {expected_q}")
        if tuple(level_edges.shape) != (cfg.n_cells, cfg.n_level_bins - 1):
            raise ValueError("Unexpected level_edges shape")
        if tuple(vol_edges.shape) != (cfg.n_vol_bins - 1,):
            raise ValueError("Unexpected vol_edges shape")
        self.register_buffer("source_quantiles", source_quantiles.float())
        self.register_buffer("target_quantiles", target_quantiles.float())
        self.register_buffer("quantile_levels", quantile_levels.float())
        self.register_buffer("level_edges", level_edges.float())
        self.register_buffer("vol_edges", vol_edges.float())
        self.__dict__["base_model"] = base_model

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        if self.base_model is None:
            raise RuntimeError("StateConditionalEmpiricalTransportCopulaModel requires a base model")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = history_norm.view(history_norm.shape[0], history_norm.shape[1], -1)
        bin_idx = self._state_bins(history_norm)
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
        base_logits = iv_to_logit(base_flat, self.cfg.logit_eps)
        source, target = self._select_maps(bin_idx)
        transported = self._transport_selected(base_logits, source, target)
        future_01 = logit_to_iv(transported)
        if self.cfg.n_cells == 25:
            return future_01.view(
                history_norm.shape[0],
                n_samples,
                self.cfg.future_len,
                5,
                5,
            )
        return future_01

    def _state_bins(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_01 = denormalize_iv(history_norm)
        history_logits = iv_to_logit(history_01, self.cfg.logit_eps)
        last_logits = history_logits[:, -1]
        level_bins = torch.empty_like(last_logits, dtype=torch.long)
        for cell in range(self.cfg.n_cells):
            level_bins[:, cell] = torch.bucketize(last_logits[:, cell], self.level_edges[cell])
        daily = history_01[:, 1:] - history_01[:, :-1]
        realized_var = daily.pow(2).mean(dim=(1, 2))
        vol_bins = torch.bucketize(realized_var, self.vol_edges)
        return level_bins * self.cfg.n_vol_bins + vol_bins[:, None]

    def _select_maps(self, bin_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = bin_idx.shape[0]
        source = torch.empty(
            bsz,
            self.cfg.future_len,
            self.cfg.n_cells,
            self.cfg.n_quantiles,
            device=bin_idx.device,
        )
        target = torch.empty_like(source)
        src = self.source_quantiles.to(bin_idx.device)
        tgt = self.target_quantiles.to(bin_idx.device)
        for cell in range(self.cfg.n_cells):
            source[:, :, cell] = src[cell, bin_idx[:, cell]]
            target[:, :, cell] = tgt[cell, bin_idx[:, cell]]
        return source, target

    @staticmethod
    def _transport_selected(
        x: torch.Tensor,
        source_quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
    ) -> torch.Tensor:
        src = source_quantiles[:, None]
        tgt = target_quantiles[:, None]
        low_denom = (src[..., 1] - src[..., 0]).abs().clamp_min(1e-6)
        out = tgt[..., 0] + (x - src[..., 0]) * (tgt[..., 1] - tgt[..., 0]) / low_denom
        for idx in range(source_quantiles.shape[-1] - 1):
            src_lo = src[..., idx]
            src_hi = src[..., idx + 1]
            tgt_lo = tgt[..., idx]
            tgt_hi = tgt[..., idx + 1]
            denom = (src_hi - src_lo).abs().clamp_min(1e-6)
            weight = (x - src_lo) / denom
            value = tgt_lo + weight * (tgt_hi - tgt_lo)
            mask = (x >= src_lo) & (x <= src_hi)
            out = torch.where(mask, value, out)
        high_denom = (src[..., -1] - src[..., -2]).abs().clamp_min(1e-6)
        high = tgt[..., -1] + (x - src[..., -1]) * (
            tgt[..., -1] - tgt[..., -2]
        ) / high_denom
        return torch.where(x > src[..., -1], high, out)


def _validate_quantile_levels(levels: Sequence[float]) -> None:
    if len(levels) < 3:
        raise ValueError("quantile_levels must contain at least three levels")
    prev = 0.0
    for level in levels:
        if not 0.0 < float(level) < 1.0:
            raise ValueError("quantile levels must be strictly inside (0, 1)")
        if float(level) <= prev:
            raise ValueError("quantile levels must be strictly increasing")
        prev = float(level)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("model_class") == "empirical_transport":
        cfg = EmpiricalMarginalTransportConfig(**payload["config"])
        base_model, _ = load_scalar_ar_model(cfg.base_checkpoint, device)
        model = EmpiricalMarginalTransportCopulaModel(
            cfg,
            source_quantiles=payload["source_quantiles"],
            target_quantiles=payload["target_quantiles"],
            quantile_levels=payload["quantile_levels"],
            base_model=base_model,
        )
        model.to(device).eval()
        model.base_model.to(device).eval()
        return model, payload
    if payload.get("model_class") == "state_conditional_transport":
        cfg = StateConditionalEmpiricalTransportConfig(**payload["config"])
        base_model, _ = load_scalar_ar_model(cfg.base_checkpoint, device)
        model = StateConditionalEmpiricalTransportCopulaModel(
            cfg,
            source_quantiles=payload["source_quantiles"],
            target_quantiles=payload["target_quantiles"],
            quantile_levels=payload["quantile_levels"],
            level_edges=payload["level_edges"],
            vol_edges=payload["vol_edges"],
            base_model=base_model,
        )
        model.to(device).eval()
        model.base_model.to(device).eval()
        return model, payload
    cfg = ConditionalMarginalCopulaConfig(**payload["config"])
    base_model, _ = load_scalar_ar_model(cfg.base_checkpoint, device)
    model = ConditionalMarginalCopulaModel(cfg, base_model=base_model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    model.base_model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ConditionalMarginalCopulaModel,
    cfg: ConditionalMarginalCopulaConfig,
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


def save_transport_checkpoint(
    path: str,
    cfg: EmpiricalMarginalTransportConfig,
    source_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    quantile_levels: torch.Tensor,
) -> None:
    torch.save(
        {
            "model_class": "empirical_transport",
            "config": asdict(cfg),
            "epoch": 0,
            "best_val": 0.0,
            "source_quantiles": source_quantiles.cpu(),
            "target_quantiles": target_quantiles.cpu(),
            "quantile_levels": quantile_levels.cpu(),
        },
        path,
    )


def save_state_conditional_transport_checkpoint(
    path: str,
    cfg: StateConditionalEmpiricalTransportConfig,
    source_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    quantile_levels: torch.Tensor,
    level_edges: torch.Tensor,
    vol_edges: torch.Tensor,
) -> None:
    torch.save(
        {
            "model_class": "state_conditional_transport",
            "config": asdict(cfg),
            "epoch": 0,
            "best_val": 0.0,
            "source_quantiles": source_quantiles.cpu(),
            "target_quantiles": target_quantiles.cpu(),
            "quantile_levels": quantile_levels.cpu(),
            "level_edges": level_edges.cpu(),
            "vol_edges": vol_edges.cpu(),
        },
        path,
    )
