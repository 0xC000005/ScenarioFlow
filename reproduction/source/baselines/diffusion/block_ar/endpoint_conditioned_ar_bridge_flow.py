from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class EndpointConditionedARBridgeConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    history_hidden: int = 160
    history_layers: int = 2
    model_hidden: int = 384
    model_layers: int = 4
    time_dim: int = 32
    dropout: float = 0.05

    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    prefix_feature_mode: str = "scale"
    endpoint_flow_steps: int = 32
    bridge_flow_steps: int = 24
    sample_temperature: float = 1.0
    endpoint_temperature: float = 1.0
    max_sample_chunk: int = 8
    endpoint_loss_weight: float = 1.0
    bridge_loss_weight: float = 1.0


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(1, n_layers - 1)):
        layers.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            half,
            device=t.device,
            dtype=t.dtype,
        )
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class EndpointConditionedARBridgeFlow(nn.Module):
    """481a: endpoint law plus endpoint-conditioned one-step bridge flow."""

    def __init__(self, cfg: EndpointConditionedARBridgeConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.prefix_feature_mode not in {"basic", "scale"}:
            raise ValueError("prefix_feature_mode must be 'basic' or 'scale'")
        feature_mult = 4 if cfg.prefix_feature_mode == "scale" else 2
        self.history_gru = nn.GRU(
            input_size=feature_mult * cfg.n_cells,
            hidden_size=cfg.history_hidden,
            num_layers=cfg.history_layers,
            dropout=cfg.dropout if cfg.history_layers > 1 else 0.0,
            batch_first=True,
        )
        self.history_norm = nn.LayerNorm(cfg.history_hidden)

        self.endpoint_velocity = _make_mlp(
            cfg.n_cells + cfg.history_hidden + cfg.time_dim,
            cfg.model_hidden,
            cfg.n_cells,
            cfg.model_layers,
            cfg.dropout,
        )
        bridge_in = (
            cfg.n_cells  # noised transition
            + cfg.n_cells  # current score
            + cfg.n_cells  # endpoint score
            + cfg.n_cells  # endpoint gap
            + 1  # remaining fraction
            + cfg.history_hidden
            + cfg.time_dim
        )
        self.bridge_velocity = _make_mlp(
            bridge_in,
            cfg.model_hidden,
            cfg.n_cells,
            cfg.model_layers,
            cfg.dropout,
        )

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
            raise RuntimeError("Empirical quantiles must be set before use")

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

    def history_scores(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_01 = denormalize_iv(self._flatten(history_norm))
        return self._values_to_scores(history_01)

    def target_future_scores(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_01 = denormalize_iv(self._flatten(future_norm))
        return self._values_to_scores(future_01)

    def _score_features_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(scores)
        deltas[:, 1:] = scores[:, 1:] - scores[:, :-1]
        if self.cfg.prefix_feature_mode == "scale":
            return torch.cat([scores, deltas, deltas.abs(), deltas.square()], dim=-1)
        return torch.cat([scores, deltas], dim=-1)

    def encode_history_scores(self, history_scores: torch.Tensor) -> torch.Tensor:
        features = self._score_features_from_scores(history_scores)
        out, _hidden = self.history_gru(features)
        return self.history_norm(out[:, -1])

    def _endpoint_velocity(
        self,
        x_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.endpoint_velocity(
            torch.cat([x_t, context, _sinusoidal_time_embedding(t, self.cfg.time_dim)], dim=-1)
        )

    def _bridge_velocity(
        self,
        x_t: torch.Tensor,
        current_score: torch.Tensor,
        endpoint_score: torch.Tensor,
        context: torch.Tensor,
        remaining_frac: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        gap = endpoint_score - current_score
        return self.bridge_velocity(
            torch.cat(
                [
                    x_t,
                    current_score,
                    endpoint_score,
                    gap,
                    remaining_frac[:, None],
                    context,
                    _sinusoidal_time_embedding(t, self.cfg.time_dim),
                ],
                dim=-1,
            )
        )

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_scores = self.history_scores(history_norm)
        future_scores = self.target_future_scores(future_norm)
        context = self.encode_history_scores(history_scores)

        endpoint = future_scores[:, -1]
        endpoint_x0 = torch.randn_like(endpoint)
        endpoint_t = torch.rand(endpoint.shape[0], device=endpoint.device, dtype=endpoint.dtype)
        endpoint_xt = (1.0 - endpoint_t[:, None]) * endpoint_x0 + endpoint_t[:, None] * endpoint
        endpoint_target_v = endpoint - endpoint_x0
        endpoint_pred_v = self._endpoint_velocity(endpoint_xt, context, endpoint_t)
        endpoint_loss = F.mse_loss(endpoint_pred_v, endpoint_target_v)

        current = torch.cat([history_scores[:, -1:, :], future_scores[:, :-1, :]], dim=1)
        target_delta = future_scores - current
        bsz, horizon, n_cells = target_delta.shape
        bridge_x0 = torch.randn_like(target_delta)
        bridge_t = torch.rand(bsz, horizon, device=target_delta.device, dtype=target_delta.dtype)
        bridge_xt = (1.0 - bridge_t[..., None]) * bridge_x0 + bridge_t[..., None] * target_delta
        bridge_target_v = target_delta - bridge_x0
        remaining = (
            torch.arange(horizon, 0, -1, device=target_delta.device, dtype=target_delta.dtype)
            / float(horizon)
        )
        remaining = remaining[None, :].expand(bsz, horizon)
        endpoint_rep = endpoint[:, None, :].expand(bsz, horizon, n_cells)
        context_rep = context[:, None, :].expand(bsz, horizon, self.cfg.history_hidden)
        bridge_pred_v = self._bridge_velocity(
            bridge_xt.reshape(bsz * horizon, n_cells),
            current.reshape(bsz * horizon, n_cells),
            endpoint_rep.reshape(bsz * horizon, n_cells),
            context_rep.reshape(bsz * horizon, self.cfg.history_hidden),
            remaining.reshape(bsz * horizon),
            bridge_t.reshape(bsz * horizon),
        ).view_as(target_delta)
        bridge_loss = F.mse_loss(bridge_pred_v, bridge_target_v)

        loss = (
            float(self.cfg.endpoint_loss_weight) * endpoint_loss
            + float(self.cfg.bridge_loss_weight) * bridge_loss
        )
        metrics = {
            "total": loss.detach(),
            "endpoint_loss": endpoint_loss.detach(),
            "bridge_loss": bridge_loss.detach(),
            "endpoint_score_std": endpoint.std(unbiased=False).detach(),
            "delta_std": target_delta.std(unbiased=False).detach(),
            "delta_abs": target_delta.abs().mean().detach(),
            "context_abs": context.abs().mean().detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def _sample_endpoint(
        self,
        context: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        endpoint = temperature * torch.randn(
            context.shape[0],
            self.cfg.n_cells,
            device=context.device,
            dtype=context.dtype,
        )
        dt = 1.0 / float(self.cfg.endpoint_flow_steps)
        for step in range(self.cfg.endpoint_flow_steps):
            t = torch.full(
                (context.shape[0],),
                (step + 0.5) * dt,
                device=context.device,
                dtype=context.dtype,
            )
            endpoint = endpoint + dt * self._endpoint_velocity(endpoint, context, t)
        return endpoint

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
        history_scores = self.history_scores(history_norm)
        base_context = self.encode_history_scores(history_scores)
        bsz = history_scores.shape[0]
        chunk_size = max(
            1,
            min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)),
        )
        bridge_temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        endpoint_temp = float(self.cfg.endpoint_temperature)
        bridge_dt = 1.0 / float(self.cfg.bridge_flow_steps)

        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            context = base_context.repeat_interleave(k, dim=0)
            current = (
                history_scores[:, -1, :]
                .unsqueeze(1)
                .expand(bsz, k, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.n_cells)
                .clone()
            )
            endpoint = self._sample_endpoint(context, endpoint_temp)
            frames: list[torch.Tensor] = []
            for step_idx in range(n_steps):
                remaining_value = float(n_steps - step_idx) / float(n_steps)
                remaining = torch.full(
                    (bsz * k,),
                    remaining_value,
                    device=current.device,
                    dtype=current.dtype,
                )
                delta = bridge_temp * torch.randn_like(current)
                for flow_step in range(self.cfg.bridge_flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (flow_step + 0.5) * bridge_dt,
                        device=current.device,
                        dtype=current.dtype,
                    )
                    delta = delta + bridge_dt * self._bridge_velocity(
                        delta,
                        current,
                        endpoint,
                        context,
                        remaining,
                        t,
                    )
                current = current + delta
                next_iv = self._scores_to_values(current)
                frames.append(next_iv.view(bsz, k, 5, 5))
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
) -> tuple[EndpointConditionedARBridgeFlow, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EndpointConditionedARBridgeConfig(**payload["config"])
    model = EndpointConditionedARBridgeFlow(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EndpointConditionedARBridgeFlow,
    cfg: EndpointConditionedARBridgeConfig,
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
