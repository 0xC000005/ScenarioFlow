from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.conditional_masked_path_flow_matching import MaskedPathAxialBlock
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.joint_token_logit_transition_flow_matching import _flow_time_features
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class EmpiricalNormalScorePathFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    token_dim: int = 128
    token_layers: int = 4
    token_ff: int = 256
    model_dropout: float = 0.1
    mixer_type: str = "axial"
    context_dim: int = 128
    history_hidden: int = 128
    encoder_dropout: float = 0.1
    token_heads: int = 4
    global_mixer: bool = True
    transition_features: bool = True
    flow_time_dim: int = 32
    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    flow_steps: int = 32
    sample_temperature: float = 1.0
    max_sample_chunk: int = 16
    path_source_corr: float = 0.0


class EmpiricalNormalScorePathVelocity(nn.Module):
    def __init__(self, cfg: EmpiricalNormalScorePathFMConfig):
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


class EmpiricalNormalScoreTransformerPathVelocity(nn.Module):
    def __init__(self, cfg: EmpiricalNormalScorePathFMConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = 2 if cfg.transition_features else 1
        self.value_proj = nn.Linear(input_dim, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.prefix_embed = nn.Parameter(torch.zeros(2, cfg.token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.token_heads,
            dim_feedforward=cfg.token_ff,
            dropout=cfg.model_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=cfg.token_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        context: torch.Tensor,
        flow_t: torch.Tensor,
        last_history: torch.Tensor,
    ) -> torch.Tensor:
        bsz, horizon, n_cells = x_t.shape
        if horizon != self.cfg.future_len or n_cells != self.cfg.n_cells:
            raise ValueError(
                f"Expected future path shape (*,{self.cfg.future_len},{self.cfg.n_cells}), "
                f"got {tuple(x_t.shape)}"
            )
        if self.cfg.transition_features:
            prev = torch.cat([last_history[:, None, :], x_t[:, :-1]], dim=1)
            values = torch.stack([x_t, x_t - prev], dim=-1)
        else:
            values = x_t[..., None]
        h_idx = torch.arange(horizon, device=x_t.device)
        c_idx = torch.arange(n_cells, device=x_t.device)
        pos = (
            self.horizon_embed(h_idx)[:, None, :]
            + self.cell_embed(c_idx)[None, :, :]
        ).reshape(horizon * n_cells, self.cfg.token_dim)
        token = self.value_proj(values.reshape(bsz, horizon * n_cells, -1)) + pos[None]
        ctx_token = self.context_proj(context) + self.prefix_embed[0][None]
        time_token = (
            self.flow_time_proj(_flow_time_features(flow_t, self.cfg.flow_time_dim))
            + self.prefix_embed[1][None]
        )
        hidden = torch.cat([torch.stack([ctx_token, time_token], dim=1), token], dim=1)
        hidden = self.mixer(hidden)
        return self.out(hidden[:, 2:, :]).squeeze(-1).view(bsz, horizon, n_cells)


class _MixLinearBlock(nn.Module):
    def __init__(self, cfg: EmpiricalNormalScorePathFMConfig):
        super().__init__()
        self.time_norm = nn.LayerNorm(cfg.token_dim)
        self.time_mixer = nn.Linear(cfg.future_len, cfg.future_len)
        self.cell_norm = nn.LayerNorm(cfg.token_dim)
        self.cell_mixer = nn.Linear(cfg.n_cells, cfg.n_cells)
        self.channel_norm = nn.LayerNorm(cfg.token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.token_ff),
            nn.GELU(),
            nn.Dropout(cfg.model_dropout),
            nn.Linear(cfg.token_ff, cfg.token_dim),
        )
        self.dropout = nn.Dropout(cfg.model_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, future, cells, channels]
        y = self.time_norm(x).permute(0, 2, 3, 1)
        y = self.time_mixer(y).permute(0, 3, 1, 2)
        x = x + self.dropout(y)
        y = self.cell_norm(x).permute(0, 1, 3, 2)
        y = self.cell_mixer(y).permute(0, 1, 3, 2)
        x = x + self.dropout(y)
        return x + self.dropout(self.channel_mlp(self.channel_norm(x)))


class EmpiricalNormalScoreMixLinearPathVelocity(nn.Module):
    """Small separable linear-mixer velocity for direct future-path flow."""

    def __init__(self, cfg: EmpiricalNormalScorePathFMConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = 5 if cfg.transition_features else 4
        self.history_level_proj = nn.Linear(cfg.history_len, cfg.future_len)
        self.history_delta_proj = nn.Linear(cfg.history_len, cfg.future_len)
        self.input_proj = nn.Linear(input_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.blocks = nn.ModuleList([_MixLinearBlock(cfg) for _ in range(cfg.token_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def _history_features(self, history_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hist_by_cell = history_z.transpose(1, 2)
        deltas = torch.zeros_like(history_z)
        deltas[:, 1:] = history_z[:, 1:] - history_z[:, :-1]
        delta_by_cell = deltas.transpose(1, 2)
        level = self.history_level_proj(hist_by_cell).transpose(1, 2)
        delta = self.history_delta_proj(delta_by_cell).transpose(1, 2)
        return level, delta

    def forward(
        self,
        x_t: torch.Tensor,
        history_z: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, horizon, n_cells = x_t.shape
        if horizon != self.cfg.future_len or n_cells != self.cfg.n_cells:
            raise ValueError(
                f"Expected future path shape (*,{self.cfg.future_len},{self.cfg.n_cells}), "
                f"got {tuple(x_t.shape)}"
            )
        hist_level, hist_delta = self._history_features(history_z)
        last = history_z[:, -1:, :].expand(bsz, horizon, n_cells)
        if self.cfg.transition_features:
            prev = torch.cat([history_z[:, -1:], x_t[:, :-1]], dim=1)
            values = torch.stack([x_t, x_t - prev, hist_level, hist_delta, last], dim=-1)
        else:
            values = torch.stack([x_t, hist_level, hist_delta, last], dim=-1)
        h_idx = torch.arange(horizon, device=x_t.device)
        c_idx = torch.arange(n_cells, device=x_t.device)
        x = self.input_proj(values)
        x = x + self.horizon_embed(h_idx)[None, :, None, :]
        x = x + self.cell_embed(c_idx)[None, None, :, :]
        x = x + self.flow_time_proj(_flow_time_features(flow_t, self.cfg.flow_time_dim))[
            :, None, None, :
        ]
        for block in self.blocks:
            x = block(x)
        return self.out(x).squeeze(-1)


class EmpiricalNormalScorePathFlowMatching(nn.Module):
    """339a: full-path rectified flow in empirical normal-score coordinates."""

    def __init__(self, cfg: EmpiricalNormalScorePathFMConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.mixer_type not in {"axial", "transformer", "mixlinear"}:
            raise ValueError("mixer_type must be 'axial', 'transformer', or 'mixlinear'")
        self.history_encoder: GRUEncoder | None = None
        if cfg.mixer_type == "transformer":
            hist_cfg = EncoderConfig(
                input_dim=cfg.n_cells,
                gru_hidden_dim=cfg.history_hidden,
                bottleneck_dim=cfg.context_dim,
                dropout=cfg.encoder_dropout,
                cond_aug_sigma=0.0,
            )
            self.history_encoder = GRUEncoder(hist_cfg)
            self.velocity = EmpiricalNormalScoreTransformerPathVelocity(cfg)
        elif cfg.mixer_type == "mixlinear":
            self.velocity = EmpiricalNormalScoreMixLinearPathVelocity(cfg)
        else:
            self.velocity = EmpiricalNormalScorePathVelocity(cfg)
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("history_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("future_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def set_empirical_quantiles(
        self,
        history_quantiles: torch.Tensor,
        future_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if history_quantiles.shape != expected or future_quantiles.shape != expected:
            raise ValueError(f"Expected quantile tensors with shape {expected}")
        self.history_quantiles.copy_(history_quantiles.to(self.history_quantiles))
        self.future_quantiles.copy_(future_quantiles.to(self.future_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("Expected quantile_levels with shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("Empirical quantiles must be set before training or sampling")

    def _values_to_scores(self, values_01: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        values_01 = self._flatten(values_01)
        levels = self.quantile_levels.to(device=values_01.device, dtype=values_01.dtype)
        table = table.to(device=values_01.device, dtype=values_01.dtype)
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

    def _scores_to_values(self, scores: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = table.to(device=scores.device, dtype=scores.dtype)
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
        return self._values_to_scores(history_01, self.history_quantiles)

    def target_future_scores(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_01 = denormalize_iv(self._flatten(future_norm))
        return self._values_to_scores(future_01, self.future_quantiles)

    def encode_history_scores(self, history_z: torch.Tensor) -> torch.Tensor:
        if self.history_encoder is None:
            raise RuntimeError("History encoder is only available for transformer mixer")
        return self.history_encoder(history_z)

    def _source_noise_like(self, target: torch.Tensor) -> torch.Tensor:
        rho = float(max(0.0, min(0.999, self.cfg.path_source_corr)))
        local = torch.randn_like(target)
        if rho <= 0.0:
            return local
        common = torch.randn(
            target.shape[0],
            target.shape[1],
            1,
            device=target.device,
            dtype=target.dtype,
        ).expand_as(target)
        return math.sqrt(rho) * common + math.sqrt(1.0 - rho) * local

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_z = self.history_scores(history_norm)
        x1 = self.target_future_scores(future_norm)
        x0 = self._source_noise_like(x1)
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        target_velocity = x1 - x0
        if self.cfg.mixer_type == "transformer":
            context = self.encode_history_scores(history_z)
            pred_velocity = self.velocity(x_t, context, t, history_z[:, -1])
        elif self.cfg.mixer_type == "mixlinear":
            pred_velocity = self.velocity(x_t, history_z, t)
        else:
            path_t = torch.cat([history_z, x_t], dim=1)
            pred_velocity = self.velocity(path_t, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        transitions = x1 - torch.cat([history_z[:, -1:], x1[:, :-1]], dim=1)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "future_score_std": x1.std(unbiased=False).detach(),
            "future_score_abs": x1.abs().mean().detach(),
            "implied_transition_std": transitions.std(unbiased=False).detach(),
            "implied_transition_abs": transitions.abs().mean().detach(),
            "path_source_corr": torch.tensor(
                float(self.cfg.path_source_corr), device=x1.device, dtype=x1.dtype
            ),
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
        context = None
        if self.cfg.mixer_type == "transformer":
            context = self.encode_history_scores(history_z)
        bsz = history_z.shape[0]
        chunk_size = max(
            1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk))
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            x = temp * self._source_noise_like(
                torch.empty(
                    bsz * k,
                    self.cfg.future_len,
                    self.cfg.n_cells,
                    device=history_z.device,
                    dtype=history_z.dtype,
                )
            )
            hist = history_z.repeat_interleave(k, dim=0)
            ctx = context.repeat_interleave(k, dim=0) if context is not None else None
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=history_z.device,
                    dtype=history_z.dtype,
                )
                if self.cfg.mixer_type == "transformer":
                    if ctx is None:
                        raise RuntimeError("Missing transformer context")
                    x = x + dt * self.velocity(x, ctx, t, hist[:, -1])
                elif self.cfg.mixer_type == "mixlinear":
                    x = x + dt * self.velocity(x, hist, t)
                else:
                    x = x + dt * self.velocity(torch.cat([hist, x], dim=1), t)
            future_01 = self._scores_to_values(x[:, :n_steps], self.future_quantiles)
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
) -> tuple[EmpiricalNormalScorePathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScorePathFMConfig(**payload["config"])
    model = EmpiricalNormalScorePathFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScorePathFlowMatching,
    cfg: EmpiricalNormalScorePathFMConfig,
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
