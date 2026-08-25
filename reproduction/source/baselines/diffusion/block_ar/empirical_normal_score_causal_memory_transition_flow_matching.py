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
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class EmpiricalNormalScoreCausalMemoryTransitionFMConfig(
    CausalFutureMemoryTransitionFMConfig
):
    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    prefix_feature_mode: str = "basic"
    conditional_noise_scale: bool = False
    noise_scale_min: float = 0.25
    noise_scale_max: float = 4.0
    path_source_corr: float = 0.0
    path_source_ar: float = 0.0
    factor_dim: int = 0


class EmpiricalNormalScoreCausalMemoryTransitionFlowMatching(nn.Module):
    """340a: causal-memory AR transition FM in shared empirical normal-score coordinates."""

    def __init__(self, cfg: EmpiricalNormalScoreCausalMemoryTransitionFMConfig):
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
        if cfg.factor_dim < 0:
            raise ValueError("factor_dim must be non-negative")
        if cfg.factor_dim > 0:
            self.factor_encoder = nn.GRU(
                input_size=cfg.factor_dim,
                hidden_size=cfg.memory_dim,
                batch_first=True,
            )
            self.factor_context_norm = nn.LayerNorm(cfg.memory_dim)
            self.factor_context_scale = nn.Parameter(torch.zeros(cfg.memory_dim))
        if cfg.conditional_noise_scale:
            self.noise_log_scale = nn.Sequential(
                nn.LayerNorm(cfg.memory_dim),
                nn.Linear(cfg.memory_dim, cfg.n_cells),
            )
            nn.init.zeros_(self.noise_log_scale[-1].weight)
            nn.init.zeros_(self.noise_log_scale[-1].bias)
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
            cols.append(torch.special.ndtri(u.clamp(eps, 1.0 - eps)).view(values_01.shape[:-1]))
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

    def _score_features_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(scores)
        deltas[:, 1:] = scores[:, 1:] - scores[:, :-1]
        if self.cfg.prefix_feature_mode == "scale":
            return torch.cat([scores, deltas, deltas.abs(), deltas.square()], dim=-1)
        return torch.cat([scores, deltas], dim=-1)

    def _factor_context(
        self,
        factor_history: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.cfg.factor_dim <= 0:
            return None
        if factor_history is None:
            raise ValueError("factor_history is required when factor_dim > 0")
        if factor_history.ndim != 3:
            raise ValueError(
                "factor_history must have shape (batch, history_len, factor_dim)"
            )
        if factor_history.shape[1] != self.cfg.history_len:
            raise ValueError(
                f"Expected factor_history history_len={self.cfg.history_len}, "
                f"got {factor_history.shape[1]}"
            )
        if factor_history.shape[2] != self.cfg.factor_dim:
            raise ValueError(
                f"Expected factor_dim={self.cfg.factor_dim}, got {factor_history.shape[2]}"
            )
        factor_history = factor_history.to(device=device, dtype=dtype)
        _, hidden = self.factor_encoder(factor_history)
        return self.factor_context_norm(hidden[-1])

    def _encode_prefix_scores(
        self,
        prefix_scores: torch.Tensor,
        factor_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = prefix_scores.shape[1]
        if seq_len > self.cfg.history_len + self.cfg.future_len:
            raise ValueError(
                f"Prefix length {seq_len} exceeds configured max "
                f"{self.cfg.history_len + self.cfg.future_len}"
            )
        pos = torch.arange(seq_len, device=prefix_scores.device)
        x = self.feature_proj(self._score_features_from_scores(prefix_scores))
        x = x + self.pos_embed(pos)[None, :, :]
        if factor_context is not None:
            if factor_context.shape != (prefix_scores.shape[0], self.cfg.memory_dim):
                raise ValueError(
                    "factor_context must have shape (batch, memory_dim), got "
                    f"{tuple(factor_context.shape)}"
                )
            x = x + factor_context[:, None, :] * self.factor_context_scale.view(1, 1, -1)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=prefix_scores.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.memory_norm(self.memory(x, mask=mask))

    def history_scores(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_01 = denormalize_iv(self._flatten(history_norm))
        return self._values_to_scores(history_01)

    def target_future_scores(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_01 = denormalize_iv(self._flatten(future_norm))
        return self._values_to_scores(future_01)

    def teacher_forced_memory(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
        factor_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_scores = self.history_scores(history_norm)
        future_scores = self.target_future_scores(future_norm)
        prefix_scores = torch.cat([history_scores, future_scores[:, :-1]], dim=1)
        factor_context = self._factor_context(
            factor_history,
            device=prefix_scores.device,
            dtype=prefix_scores.dtype,
        )
        hidden = self._encode_prefix_scores(prefix_scores, factor_context=factor_context)
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        current_scores = prefix_scores[:, start : start + self.cfg.future_len]
        return memory_states, current_scores, future_scores

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        current_score: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity(x_t, current_score, memory_state, t)

    def _conditional_noise_scale(self, memory_state: torch.Tensor) -> torch.Tensor | None:
        if not self.cfg.conditional_noise_scale:
            return None
        log_scale = self.noise_log_scale(memory_state)
        lo = math.log(float(self.cfg.noise_scale_min))
        hi = math.log(float(self.cfg.noise_scale_max))
        return torch.exp(log_scale.clamp(lo, hi))

    @staticmethod
    def _ar1_source_noise(shape: torch.Size, rho: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if len(shape) != 3:
            raise ValueError("AR(1) source noise expects shape (batch,horizon,cells)")
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
        ar_rho = float(max(0.0, min(0.999, self.cfg.path_source_ar)))
        temporal_noise = self._ar1_source_noise(
            target.shape,
            ar_rho,
            target.device,
            target.dtype,
        )
        rho = float(max(0.0, min(0.999, self.cfg.path_source_corr)))
        if rho <= 0.0:
            return temporal_noise
        if target.ndim != 3:
            raise ValueError("Persistent source noise expects shape (batch,horizon,cells)")
        path_noise = torch.randn(
            target.shape[0],
            1,
            target.shape[2],
            device=target.device,
            dtype=target.dtype,
        )
        return math.sqrt(rho) * path_noise + math.sqrt(1.0 - rho) * temporal_noise

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
        factor_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_scores = self.history_scores(history_norm)
        future_scores = self.target_future_scores(future_norm)
        prefix_scores = torch.cat([history_scores, future_scores[:, :-1]], dim=1)
        factor_context = self._factor_context(
            factor_history,
            device=prefix_scores.device,
            dtype=prefix_scores.dtype,
        )
        hidden = self._encode_prefix_scores(
            prefix_scores,
            factor_context=factor_context,
        )
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        current_scores = prefix_scores[:, start : start + self.cfg.future_len]
        x1 = future_scores - current_scores
        x0 = self._source_noise_like(x1)
        noise_scale = self._conditional_noise_scale(memory_states)
        if noise_scale is not None:
            x0 = x0 * noise_scale
        bsz, horizon, n_cells = x1.shape
        t = torch.rand(bsz, horizon, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t[..., None]) * x0 + t[..., None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.predict_velocity(
            x_t.reshape(bsz * horizon, n_cells),
            current_scores.reshape(bsz * horizon, n_cells),
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
            "path_source_corr": torch.tensor(
                float(self.cfg.path_source_corr), device=x1.device, dtype=x1.dtype
            ),
            "path_source_ar": torch.tensor(
                float(self.cfg.path_source_ar), device=x1.device, dtype=x1.dtype
            ),
        }
        if factor_context is not None:
            metrics["factor_context_abs"] = factor_context.abs().mean().detach()
            metrics["factor_context_scale_abs"] = self.factor_context_scale.abs().mean().detach()
        if noise_scale is not None:
            metrics.update(
                {
                    "noise_scale_mean": noise_scale.mean().detach(),
                    "noise_scale_std": noise_scale.std(unbiased=False).detach(),
                    "noise_scale_min": noise_scale.min().detach(),
                    "noise_scale_max": noise_scale.max().detach(),
                }
            )
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
        factor_history: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(f"Expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_scores = self.history_scores(history_norm)
        factor_context = self._factor_context(
            factor_history,
            device=history_scores.device,
            dtype=history_scores.dtype,
        )
        bsz = history_scores.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)

        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            prefix = (
                history_scores.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            chunk_factor_context = None
            if factor_context is not None:
                chunk_factor_context = (
                    factor_context.unsqueeze(1)
                    .expand(bsz, k, self.cfg.memory_dim)
                    .reshape(bsz * k, self.cfg.memory_dim)
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
            for _step in range(n_steps):
                memory_state = self._encode_prefix_scores(
                    prefix,
                    factor_context=chunk_factor_context,
                )[:, -1]
                current_score = prefix[:, -1]
                if path_source is None:
                    x = temp * temporal_source[:, _step]
                else:
                    x = temp * (
                        math.sqrt(rho) * path_source
                        + math.sqrt(1.0 - rho) * temporal_source[:, _step]
                    )
                noise_scale = self._conditional_noise_scale(memory_state)
                if noise_scale is not None:
                    x = x * noise_scale
                for flow_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (flow_step + 0.5) * dt,
                        device=history_scores.device,
                        dtype=history_scores.dtype,
                    )
                    x = x + dt * self.predict_velocity(x, current_score, memory_state, t)
                next_score = current_score + x
                next_iv = self._scores_to_values(next_score)
                frames.append(next_iv.view(bsz, k, 5, 5))
                prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
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
) -> tuple[EmpiricalNormalScoreCausalMemoryTransitionFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScoreCausalMemoryTransitionFMConfig(**payload["config"])
    model = EmpiricalNormalScoreCausalMemoryTransitionFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    cfg: EmpiricalNormalScoreCausalMemoryTransitionFMConfig,
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
