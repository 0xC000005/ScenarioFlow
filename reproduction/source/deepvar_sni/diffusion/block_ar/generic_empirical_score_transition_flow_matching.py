from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.causal_future_memory_transition_flow_matching import (
    CausalFutureMemoryTransitionFMConfig,
    MemoryConditionedTokenTransitionVelocity,
)


@dataclass
class GenericEmpiricalScoreTransitionFMConfig(CausalFutureMemoryTransitionFMConfig):
    """Generic empirical-score AR transition flow for financial state panels."""

    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    prefix_feature_mode: str = "basic"
    path_source_corr: float = 0.0
    path_source_ar: float = 0.0
    conditional_source_scale: bool = False
    source_scale_min: float = 0.25
    source_scale_max: float = 4.0


class GenericEmpiricalScoreTransitionFlowMatching(nn.Module):
    """510a-style AR transition flow over an arbitrary state panel.

    The model treats all variables as one exchangeable panel dimension. It learns
    daily empirical-score increments and rolls them forward autoregressively.
    """

    def __init__(self, cfg: GenericEmpiricalScoreTransitionFMConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.prefix_feature_mode not in {"basic", "scale"}:
            raise ValueError("prefix_feature_mode must be 'basic' or 'scale'")
        feature_mult = 4 if cfg.prefix_feature_mode == "scale" else 2
        self.feature_proj = nn.Linear(feature_mult * cfg.n_cells, cfg.memory_dim)
        self.pos_embed = nn.Embedding(cfg.history_len + cfg.future_len, cfg.memory_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.memory_dim,
            nhead=cfg.memory_heads,
            dim_feedforward=cfg.memory_ff,
            dropout=cfg.model_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory = nn.TransformerEncoder(layer, num_layers=cfg.memory_layers)
        self.memory_norm = nn.LayerNorm(cfg.memory_dim)
        self.velocity = MemoryConditionedTokenTransitionVelocity(cfg)
        if cfg.conditional_source_scale:
            self.source_log_scale = nn.Sequential(
                nn.LayerNorm(cfg.memory_dim),
                nn.Linear(cfg.memory_dim, cfg.n_cells),
            )
            nn.init.zeros_(self.source_log_scale[-1].weight)
            nn.init.zeros_(self.source_log_scale[-1].bias)
        else:
            self.source_log_scale = None
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("value_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))

    def set_empirical_quantiles(
        self,
        value_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if value_quantiles.shape != expected:
            raise ValueError(f"expected value_quantiles shape {expected}, got {tuple(value_quantiles.shape)}")
        self.value_quantiles.copy_(value_quantiles.to(self.value_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("quantile_levels must have shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("empirical quantiles must be set before use")

    def values_to_scores(self, values: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        if values.ndim != 3 or values.shape[-1] != self.cfg.n_cells:
            raise ValueError(
                f"values must have shape (batch,time,{self.cfg.n_cells}), got {tuple(values.shape)}"
            )
        levels = self.quantile_levels.to(device=values.device, dtype=values.dtype)
        table = self.value_quantiles.to(device=values.device, dtype=values.dtype)
        cols: list[torch.Tensor] = []
        for var in range(self.cfg.n_cells):
            q = table[var]
            flat = values[..., var].reshape(-1)
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
            score = torch.special.ndtri(u.clamp(eps, 1.0 - eps))
            cols.append(score.view(values.shape[:-1]))
        return torch.stack(cols, dim=-1)

    def scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        if scores.ndim != 3 and scores.ndim != 2:
            raise ValueError("scores must have shape (batch,time,vars) or (batch,vars)")
        if scores.shape[-1] != self.cfg.n_cells:
            raise ValueError(f"last dimension must be {self.cfg.n_cells}")
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = self.value_quantiles.to(device=scores.device, dtype=scores.dtype)
        eps = float(self.cfg.cdf_eps)
        u_all = (0.5 * (1.0 + torch.erf(scores / math.sqrt(2.0)))).clamp(eps, 1.0 - eps)
        cols: list[torch.Tensor] = []
        for var in range(self.cfg.n_cells):
            q = table[var]
            flat = u_all[..., var].reshape(-1)
            idx = torch.searchsorted(levels.contiguous(), flat.contiguous(), right=False)
            idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
            idx_lo = idx_hi - 1
            u_lo = levels[idx_lo]
            u_hi = levels[idx_hi]
            q_lo = q[idx_lo]
            q_hi = q[idx_hi]
            alpha = (flat - u_lo) / (u_hi - u_lo).clamp_min(1e-12)
            value = q_lo + alpha.clamp(0.0, 1.0) * (q_hi - q_lo)
            value = torch.where(flat <= levels[0], q[0], value)
            value = torch.where(flat >= levels[-1], q[-1], value)
            cols.append(value.view(scores.shape[:-1]))
        return torch.stack(cols, dim=-1)

    def _score_features(self, scores: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(scores)
        deltas[:, 1:] = scores[:, 1:] - scores[:, :-1]
        if self.cfg.prefix_feature_mode == "scale":
            return torch.cat([scores, deltas, deltas.abs(), deltas.square()], dim=-1)
        return torch.cat([scores, deltas], dim=-1)

    def _encode_prefix_scores(self, prefix_scores: torch.Tensor) -> torch.Tensor:
        seq_len = prefix_scores.shape[1]
        if seq_len > self.cfg.history_len + self.cfg.future_len:
            raise ValueError(
                f"prefix length {seq_len} exceeds {self.cfg.history_len + self.cfg.future_len}"
            )
        pos = torch.arange(seq_len, device=prefix_scores.device)
        x = self.feature_proj(self._score_features(prefix_scores))
        x = x + self.pos_embed(pos)[None, :, :]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=prefix_scores.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.memory_norm(self.memory(x, mask=mask))

    @staticmethod
    def _ar1_source_noise(shape: torch.Size, rho: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base = torch.randn(shape, device=device, dtype=dtype)
        rho = float(max(0.0, min(0.999, rho)))
        if rho <= 0.0 or shape[1] <= 1:
            return base
        innovation_scale = math.sqrt(max(1.0 - rho * rho, 1e-8))
        states = [base[:, 0]]
        state = base[:, 0]
        for step in range(1, shape[1]):
            state = rho * state + innovation_scale * base[:, step]
            states.append(state)
        return torch.stack(states, dim=1)

    def _source_noise_like(self, target: torch.Tensor) -> torch.Tensor:
        temporal_noise = self._ar1_source_noise(
            target.shape,
            float(self.cfg.path_source_ar),
            target.device,
            target.dtype,
        )
        rho = float(max(0.0, min(0.999, self.cfg.path_source_corr)))
        if rho <= 0.0:
            return temporal_noise
        path_noise = torch.randn(
            target.shape[0],
            1,
            target.shape[2],
            device=target.device,
            dtype=target.dtype,
        )
        return math.sqrt(rho) * path_noise + math.sqrt(1.0 - rho) * temporal_noise

    def _conditional_source_scale(self, memory_state: torch.Tensor) -> torch.Tensor | None:
        if self.source_log_scale is None:
            return None
        lo = math.log(float(self.cfg.source_scale_min))
        hi = math.log(float(self.cfg.source_scale_max))
        return torch.exp(self.source_log_scale(memory_state).clamp(lo, hi))

    def training_loss(
        self,
        history_values: torch.Tensor,
        future_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_scores = self.values_to_scores(history_values)
        future_scores = self.values_to_scores(future_values)
        prefix_scores = torch.cat([history_scores, future_scores[:, :-1]], dim=1)
        hidden = self._encode_prefix_scores(prefix_scores)
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        current_scores = prefix_scores[:, start : start + self.cfg.future_len]
        x1 = future_scores - current_scores
        x0 = self._source_noise_like(x1)
        source_scale = self._conditional_source_scale(memory_states)
        if source_scale is not None:
            x0 = x0 * source_scale
        bsz, horizon, n_vars = x1.shape
        t = torch.rand(bsz, horizon, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t[..., None]) * x0 + t[..., None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.velocity(
            x_t.reshape(bsz * horizon, n_vars),
            current_scores.reshape(bsz * horizon, n_vars),
            memory_states.reshape(bsz * horizon, self.cfg.memory_dim),
            t.reshape(bsz * horizon),
        ).view_as(x1)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "transition_std": x1.std(unbiased=False).detach(),
            "transition_abs": x1.abs().mean().detach(),
            "target_velocity_std": target_velocity.std(unbiased=False).detach(),
            "memory_abs": memory_states.abs().mean().detach(),
        }
        if source_scale is not None:
            metrics.update(
                {
                    "source_scale_mean": source_scale.mean().detach(),
                    "source_scale_std": source_scale.std(unbiased=False).detach(),
                    "source_scale_min": source_scale.min().detach(),
                    "source_scale_max": source_scale.max().detach(),
                }
            )
        return fm_loss, metrics

    @torch.no_grad()
    def sample_batched(
        self,
        history_values: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(f"expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        history_scores = self.values_to_scores(history_values)
        bsz = history_scores.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, int(n_samples), chunk_size):
            k = min(chunk_size, int(n_samples) - start)
            prefix = (
                history_scores.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            rho = float(max(0.0, min(0.999, self.cfg.path_source_corr)))
            ar_rho = float(max(0.0, min(0.999, self.cfg.path_source_ar)))
            path_source = None
            if rho > 0.0:
                path_source = torch.randn(
                    bsz * k,
                    self.cfg.n_cells,
                    device=history_scores.device,
                    dtype=history_scores.dtype,
                )
            temporal_source = self._ar1_source_noise(
                torch.Size((bsz * k, n_steps, self.cfg.n_cells)),
                ar_rho,
                history_scores.device,
                history_scores.dtype,
            )
            frames: list[torch.Tensor] = []
            for step in range(n_steps):
                memory_state = self._encode_prefix_scores(prefix)[:, -1]
                current_score = prefix[:, -1]
                if path_source is None:
                    x = temp * temporal_source[:, step]
                else:
                    x = temp * (
                        math.sqrt(rho) * path_source
                        + math.sqrt(1.0 - rho) * temporal_source[:, step]
                    )
                source_scale = self._conditional_source_scale(memory_state)
                if source_scale is not None:
                    x = x * source_scale
                for flow_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (flow_step + 0.5) * dt,
                        device=history_scores.device,
                        dtype=history_scores.dtype,
                    )
                    x = x + dt * self.velocity(x, current_score, memory_state, t)
                next_score = current_score + x
                next_values = self.scores_to_values(next_score)
                frames.append(next_values.view(bsz, k, self.cfg.n_cells))
                prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GenericEmpiricalScoreTransitionFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GenericEmpiricalScoreTransitionFMConfig(**payload["config"])
    model = GenericEmpiricalScoreTransitionFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: GenericEmpiricalScoreTransitionFlowMatching,
    cfg: GenericEmpiricalScoreTransitionFMConfig,
    epoch: int,
    best_val: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": asdict(cfg),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
