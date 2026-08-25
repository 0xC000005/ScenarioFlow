from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.causal_future_memory_transition_flow_matching import (
    CausalFutureMemoryTransitionFMConfig,
)
from diffusion.block_ar.recurrent_logit_transition_flow_matching import _time_features


@dataclass
class GenericMixedCoordinatePathFMConfig(CausalFutureMemoryTransitionFMConfig):
    """Full-future path flow in fixed mixed generated coordinates."""

    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    prefix_feature_mode: str = "scale"
    level_score_channels: list[int] = field(default_factory=list)
    conditional_source_affine: bool = False
    source_affine_mode: str = "full"
    source_scale_min: float = 0.5
    source_scale_max: float = 2.0
    source_loc_clip: float = 3.0
    terminal_path_loss_weight: float = 0.0
    terminal_tail_weight: float = 0.0
    terminal_tail_threshold: float = 1.5
    path_energy_loss_weight: float = 0.0
    path_energy_samples: int = 0
    path_energy_tail_weight: float = 0.0
    path_energy_tail_threshold: float = 1.5


class GenericMixedCoordinatePathFlowMatching(nn.Module):
    """One-shot full-path flow for a generic financial state panel.

    The model conditions on history level/increment scores and generates the
    entire future tensor jointly. Channels listed in ``level_score_channels``
    use cumulative level-score deltas from the current state; other channels use
    encoded increment scores at each future step.
    """

    def __init__(self, cfg: GenericMixedCoordinatePathFMConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.prefix_feature_mode not in {"basic", "scale"}:
            raise ValueError("prefix_feature_mode must be 'basic' or 'scale'")
        feature_mult = 4 if cfg.prefix_feature_mode == "scale" else 2
        self.history_proj = nn.Linear(feature_mult * cfg.n_cells, cfg.memory_dim)
        self.history_pos = nn.Embedding(cfg.history_len, cfg.memory_dim)
        history_layer = nn.TransformerEncoderLayer(
            d_model=cfg.memory_dim,
            nhead=cfg.memory_heads,
            dim_feedforward=cfg.memory_ff,
            dropout=cfg.model_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            history_layer,
            num_layers=cfg.memory_layers,
        )
        self.history_norm = nn.LayerNorm(cfg.memory_dim)
        if cfg.conditional_source_affine:
            if cfg.source_affine_mode == "full":
                source_out_dim = 2 * cfg.future_len * cfg.n_cells
            elif cfg.source_affine_mode == "horizon_scalar":
                source_out_dim = 2 * cfg.future_len
            elif cfg.source_affine_mode == "global_scalar":
                source_out_dim = 2
            else:
                raise ValueError(
                    "source_affine_mode must be 'full', 'horizon_scalar', "
                    "or 'global_scalar'"
                )
            self.source_affine = nn.Sequential(
                nn.LayerNorm(cfg.memory_dim),
                nn.Linear(cfg.memory_dim, source_out_dim),
            )
            nn.init.zeros_(self.source_affine[-1].weight)
            nn.init.zeros_(self.source_affine[-1].bias)
        else:
            self.source_affine = None

        self.future_value_proj = nn.Linear(cfg.n_cells, cfg.token_dim)
        self.future_pos = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.memory_dim, cfg.token_dim)
        self.time_proj = nn.Linear(cfg.time_dim, cfg.token_dim)
        future_layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.token_heads,
            dim_feedforward=cfg.token_ff,
            dropout=cfg.model_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.future_denoiser = nn.TransformerEncoder(
            future_layer,
            num_layers=cfg.token_layers,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, cfg.n_cells),
        )

        mask = torch.zeros(cfg.n_cells, dtype=torch.bool)
        if cfg.level_score_channels:
            idx = torch.tensor(cfg.level_score_channels, dtype=torch.long)
            if idx.min().item() < 0 or idx.max().item() >= cfg.n_cells:
                raise ValueError("level_score_channels contains out-of-range indices")
            mask[idx] = True
        self.register_buffer("level_score_mask", mask)
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer(
            "level_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles)
        )
        self.register_buffer(
            "increment_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles)
        )
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))

    def set_empirical_quantiles(
        self,
        level_quantiles: torch.Tensor,
        increment_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if level_quantiles.shape != expected:
            raise ValueError(
                f"level_quantiles must have shape {expected}, got {tuple(level_quantiles.shape)}"
            )
        if increment_quantiles.shape != expected:
            raise ValueError(
                f"increment_quantiles must have shape {expected}, got {tuple(increment_quantiles.shape)}"
            )
        self.level_quantiles.copy_(level_quantiles.to(self.level_quantiles))
        self.increment_quantiles.copy_(increment_quantiles.to(self.increment_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("quantile_levels must have shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("empirical quantiles must be set before use")

    def _values_to_scores(
        self, values: torch.Tensor, table: torch.Tensor
    ) -> torch.Tensor:
        self._check_quantiles()
        if values.shape[-1] != self.cfg.n_cells:
            raise ValueError(
                f"last dimension must be {self.cfg.n_cells}, got {values.shape[-1]}"
            )
        levels = self.quantile_levels.to(device=values.device, dtype=values.dtype)
        table = table.to(device=values.device, dtype=values.dtype)
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

    def _scores_to_values(
        self, scores: torch.Tensor, table: torch.Tensor
    ) -> torch.Tensor:
        self._check_quantiles()
        if scores.shape[-1] != self.cfg.n_cells:
            raise ValueError(
                f"last dimension must be {self.cfg.n_cells}, got {scores.shape[-1]}"
            )
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = table.to(device=scores.device, dtype=scores.dtype)
        eps = float(self.cfg.cdf_eps)
        u_all = (0.5 * (1.0 + torch.erf(scores / math.sqrt(2.0)))).clamp(
            eps,
            1.0 - eps,
        )
        cols: list[torch.Tensor] = []
        for var in range(self.cfg.n_cells):
            q = table[var]
            flat = u_all[..., var].reshape(-1)
            idx = torch.searchsorted(
                levels.contiguous(), flat.contiguous(), right=False
            )
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

    def level_values_to_scores(self, values: torch.Tensor) -> torch.Tensor:
        return self._values_to_scores(values, self.level_quantiles)

    def increment_values_to_scores(self, values: torch.Tensor) -> torch.Tensor:
        return self._values_to_scores(values, self.increment_quantiles)

    def _level_scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        return self._scores_to_values(scores, self.level_quantiles)

    def increment_scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        return self._scores_to_values(scores, self.increment_quantiles)

    def _history_features(
        self,
        level_scores: torch.Tensor,
        increment_scores: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.prefix_feature_mode == "scale":
            return torch.cat(
                [
                    level_scores,
                    increment_scores,
                    increment_scores.abs(),
                    increment_scores.square(),
                ],
                dim=-1,
            )
        return torch.cat([level_scores, increment_scores], dim=-1)

    def encode_history(
        self,
        history_level_values: torch.Tensor,
        history_increment_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        level_scores = self.level_values_to_scores(history_level_values)
        increment_scores = self.increment_values_to_scores(history_increment_values)
        seq_len = int(level_scores.shape[1])
        if seq_len != self.cfg.history_len:
            raise ValueError(
                f"expected history_len={self.cfg.history_len}, got {seq_len}"
            )
        pos = torch.arange(seq_len, device=level_scores.device)
        x = self.history_proj(self._history_features(level_scores, increment_scores))
        x = x + self.history_pos(pos)[None, :, :]
        hidden = self.history_norm(self.history_encoder(x))
        return hidden[:, -1], level_scores[:, -1]

    def conditional_source_affine(
        self,
        context: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.source_affine is None:
            return None, None
        if self.cfg.source_affine_mode == "full":
            raw = self.source_affine(context).view(
                context.shape[0],
                self.cfg.future_len,
                self.cfg.n_cells,
                2,
            )
            loc_raw = raw[:, :horizon, :, 0]
            log_scale_raw = raw[:, :horizon, :, 1]
        elif self.cfg.source_affine_mode == "horizon_scalar":
            raw = self.source_affine(context).view(
                context.shape[0],
                self.cfg.future_len,
                2,
            )
            loc_raw = raw[:, :horizon, 0].unsqueeze(-1)
            log_scale_raw = raw[:, :horizon, 1].unsqueeze(-1)
        elif self.cfg.source_affine_mode == "global_scalar":
            raw = self.source_affine(context)
            loc_raw = raw[:, 0].view(context.shape[0], 1, 1)
            log_scale_raw = raw[:, 1].view(context.shape[0], 1, 1)
        else:
            raise ValueError(
                f"unknown source_affine_mode={self.cfg.source_affine_mode!r}"
            )
        loc_clip = float(self.cfg.source_loc_clip)
        loc = loc_raw.clamp(-loc_clip, loc_clip) if loc_clip > 0 else loc_raw
        lo = math.log(float(self.cfg.source_scale_min))
        hi = math.log(float(self.cfg.source_scale_max))
        scale = torch.exp(log_scale_raw.clamp(lo, hi))
        if loc.shape[-1] == 1:
            loc = loc.expand(context.shape[0], horizon, self.cfg.n_cells)
        if scale.shape[-1] == 1:
            scale = scale.expand(context.shape[0], horizon, self.cfg.n_cells)
        return loc, scale

    def target_mixed_coordinates(
        self,
        history_level_values: torch.Tensor,
        future_level_values: torch.Tensor,
        future_increment_values: torch.Tensor,
    ) -> torch.Tensor:
        history_level_scores = self.level_values_to_scores(history_level_values)
        future_level_scores = self.level_values_to_scores(future_level_values)
        future_increment_scores = self.increment_values_to_scores(
            future_increment_values
        )
        level_delta_scores = future_level_scores - history_level_scores[:, -1:, :]
        mask = self.level_score_mask.to(device=future_level_scores.device)[
            None, None, :
        ]
        return torch.where(mask, level_delta_scores, future_increment_scores)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, horizon, _n_cells = x_t.shape
        pos = torch.arange(horizon, device=x_t.device)
        token = self.future_value_proj(x_t)
        token = token + self.future_pos(pos)[None, :, :]
        token = token + self.context_proj(context)[:, None, :]
        token = token + self.time_proj(_time_features(t, self.cfg.time_dim))[:, None, :]
        hidden = self.future_denoiser(token)
        return self.out(hidden)

    def training_loss(
        self,
        history_level_values: torch.Tensor,
        history_increment_values: torch.Tensor,
        future_level_values: torch.Tensor,
        future_increment_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context, _current_level_score = self.encode_history(
            history_level_values,
            history_increment_values,
        )
        x1 = self.target_mixed_coordinates(
            history_level_values,
            future_level_values,
            future_increment_values,
        )
        x0 = torch.randn_like(x1)
        source_loc, source_scale = self.conditional_source_affine(
            context,
            int(x1.shape[1]),
        )
        if source_loc is not None and source_scale is not None:
            x0 = source_loc + source_scale * x0
        bsz = int(x1.shape[0])
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.predict_velocity(x_t, context, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        loss = fm_loss
        metrics = {
            "total": loss.detach(),
            "fm_loss": fm_loss.detach(),
            "target_mixed_coord_std": x1.std(unbiased=False).detach(),
            "target_mixed_coord_abs": x1.abs().mean().detach(),
            "target_velocity_std": target_velocity.std(unbiased=False).detach(),
            "context_abs": context.abs().mean().detach(),
        }
        if float(self.cfg.terminal_path_loss_weight) > 0.0:
            x1_pred = x_t + (1.0 - t[:, None, None]) * pred_velocity
            terminal_error = F.smooth_l1_loss(x1_pred, x1, reduction="none")
            if float(self.cfg.terminal_tail_weight) > 0.0:
                tail_gate = (x1.abs() > float(self.cfg.terminal_tail_threshold)).to(
                    terminal_error
                )
                weights = 1.0 + float(self.cfg.terminal_tail_weight) * tail_gate
                terminal_loss = (terminal_error * weights).mean()
                metrics["terminal_tail_rate"] = tail_gate.mean().detach()
            else:
                terminal_loss = terminal_error.mean()
            loss = loss + float(self.cfg.terminal_path_loss_weight) * terminal_loss
            metrics["terminal_path_loss"] = terminal_loss.detach()
            metrics["total"] = loss.detach()
        if (
            float(self.cfg.path_energy_loss_weight) > 0.0
            and int(self.cfg.path_energy_samples) >= 2
        ):
            k = int(self.cfg.path_energy_samples)
            bsz, horizon, n_cells = x1.shape
            source = torch.randn(
                bsz * k,
                horizon,
                n_cells,
                device=x1.device,
                dtype=x1.dtype,
            )
            if source_loc is not None and source_scale is not None:
                source = (
                    source_loc[:, None]
                    .expand(bsz, k, horizon, n_cells)
                    .reshape(bsz * k, horizon, n_cells)
                    + source_scale[:, None]
                    .expand(bsz, k, horizon, n_cells)
                    .reshape(bsz * k, horizon, n_cells)
                    * source
                )
            context_energy = (
                context[:, None]
                .expand(bsz, k, self.cfg.memory_dim)
                .reshape(bsz * k, self.cfg.memory_dim)
            )
            t0 = torch.zeros(bsz * k, device=x1.device, dtype=x1.dtype)
            samples = source + self.predict_velocity(source, context_energy, t0)
            samples = samples.view(bsz, k, horizon, n_cells)
            target = x1[:, None, :, :]
            if float(self.cfg.path_energy_tail_weight) > 0.0:
                tail_gate = (x1.abs() > float(self.cfg.path_energy_tail_threshold)).to(
                    x1
                )
                dim_weight = torch.sqrt(
                    1.0 + float(self.cfg.path_energy_tail_weight) * tail_gate
                )[:, None, :, :]
                samples_for_score = samples * dim_weight
                target_for_score = target * dim_weight
                metrics["path_energy_tail_rate"] = tail_gate.mean().detach()
            else:
                samples_for_score = samples
                target_for_score = target
            sample_flat = samples_for_score.reshape(bsz, k, horizon * n_cells)
            target_flat = target_for_score.reshape(bsz, 1, horizon * n_cells)
            target_dist = torch.linalg.vector_norm(
                sample_flat - target_flat,
                dim=-1,
            ).mean()
            pair_dist = torch.cdist(sample_flat, sample_flat, p=2).mean()
            path_energy = target_dist - 0.5 * pair_dist
            loss = loss + float(self.cfg.path_energy_loss_weight) * path_energy
            metrics["path_energy_loss"] = path_energy.detach()
            metrics["path_energy_target_dist"] = target_dist.detach()
            metrics["path_energy_pair_dist"] = pair_dist.detach()
            metrics["total"] = loss.detach()
        if source_loc is not None and source_scale is not None:
            metrics.update(
                {
                    "source_loc_abs": source_loc.abs().mean().detach(),
                    "source_scale_mean": source_scale.mean().detach(),
                    "source_scale_std": source_scale.std(unbiased=False).detach(),
                    "source_scale_min": source_scale.min().detach(),
                    "source_scale_max": source_scale.max().detach(),
                }
            )
        return loss, metrics

    def mixed_coordinates_to_increment_values(
        self,
        x: torch.Tensor,
        history_level_values: torch.Tensor,
        current_level_score: torch.Tensor,
    ) -> torch.Tensor:
        bsz = int(history_level_values.shape[0])
        n_samples = int(x.shape[1])
        horizon = int(x.shape[2])
        last_level_value = (
            history_level_values[:, -1, :]
            .unsqueeze(1)
            .expand(bsz, n_samples, self.cfg.n_cells)
            .reshape(bsz * n_samples, self.cfg.n_cells)
        )
        current_score = (
            current_level_score.unsqueeze(1)
            .expand(bsz, n_samples, self.cfg.n_cells)
            .reshape(bsz * n_samples, self.cfg.n_cells)
        )
        flat_x = x.reshape(bsz * n_samples, horizon, self.cfg.n_cells)
        level_path_scores = current_score[:, None, :] + flat_x
        level_path_values = self._level_scores_to_values(level_path_scores)
        increment_values = self.increment_scores_to_values(flat_x)
        increment_level_values = last_level_value[:, None, :] + torch.cumsum(
            increment_values,
            dim=1,
        )
        mask = self.level_score_mask.to(device=x.device)[None, None, :]
        path_values = torch.where(mask, level_path_values, increment_level_values)
        prev_values = torch.cat(
            [last_level_value[:, None, :], path_values[:, :-1]], dim=1
        )
        increments = path_values - prev_values
        return increments.view(bsz, n_samples, horizon, self.cfg.n_cells)

    @torch.no_grad()
    def sample_batched(
        self,
        history_level_values: torch.Tensor,
        history_increment_values: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(
                f"expected n_steps in [1,{self.cfg.future_len}], got {n_steps}"
            )
        context, current_level_score = self.encode_history(
            history_level_values,
            history_increment_values,
        )
        bsz = int(history_level_values.shape[0])
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(
            self.cfg.sample_temperature if temperature is None else temperature
        )
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, int(n_samples), chunk_size):
            k = min(chunk_size, int(n_samples) - start)
            ctx = (
                context.unsqueeze(1)
                .expand(bsz, k, self.cfg.memory_dim)
                .reshape(bsz * k, self.cfg.memory_dim)
            )
            noise = torch.randn(
                bsz * k,
                int(n_steps),
                self.cfg.n_cells,
                device=history_level_values.device,
                dtype=history_level_values.dtype,
            )
            source_loc, source_scale = self.conditional_source_affine(
                context,
                int(n_steps),
            )
            if source_loc is not None and source_scale is not None:
                loc = (
                    source_loc.unsqueeze(1)
                    .expand(bsz, k, int(n_steps), self.cfg.n_cells)
                    .reshape(bsz * k, int(n_steps), self.cfg.n_cells)
                )
                scale = (
                    source_scale.unsqueeze(1)
                    .expand(bsz, k, int(n_steps), self.cfg.n_cells)
                    .reshape(bsz * k, int(n_steps), self.cfg.n_cells)
                )
                x = loc + temp * scale * noise
            else:
                x = temp * noise
            for flow_step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (flow_step + 0.5) * dt,
                    device=history_level_values.device,
                    dtype=history_level_values.dtype,
                )
                x = x + dt * self.predict_velocity(x, ctx, t)
            x = x.view(bsz, k, int(n_steps), self.cfg.n_cells)
            outs.append(
                self.mixed_coordinates_to_increment_values(
                    x,
                    history_level_values,
                    current_level_score,
                )
            )
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GenericMixedCoordinatePathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GenericMixedCoordinatePathFMConfig(**payload["config"])
    model = GenericMixedCoordinatePathFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: GenericMixedCoordinatePathFlowMatching,
    cfg: GenericMixedCoordinatePathFMConfig,
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
