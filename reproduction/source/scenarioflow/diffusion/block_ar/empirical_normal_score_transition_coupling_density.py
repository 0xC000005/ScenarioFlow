from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.empirical_normal_score_path_coupling_density import CouplingLayer
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class EmpiricalNormalScoreTransitionCouplingConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    memory_dim: int = 128
    memory_layers: int = 3
    memory_heads: int = 4
    memory_ff: int = 256
    model_dropout: float = 0.1
    prefix_feature_mode: str = "basic"

    coupling_layers: int = 6
    coupling_hidden: int = 256
    coupling_dropout: float = 0.1
    scale_clip: float = 2.0
    use_location: bool = False
    location_hidden: int = 256
    location_mode: str = "free"
    base_distribution: str = "normal"
    student_df_init: float = 8.0
    student_df_min: float = 2.1

    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    sample_temperature: float = 1.0
    max_sample_chunk: int = 8
    target_mode: str = "transition"


class EmpiricalNormalScoreTransitionCouplingDensity(nn.Module):
    """347a: causal-memory AR transition density in empirical normal-score coordinates."""

    def __init__(self, cfg: EmpiricalNormalScoreTransitionCouplingConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.prefix_feature_mode not in {"basic", "scale"}:
            raise ValueError("prefix_feature_mode must be 'basic' or 'scale'")
        if cfg.base_distribution not in {"normal", "student_t"}:
            raise ValueError("base_distribution must be 'normal' or 'student_t'")
        if cfg.target_mode not in {"transition", "level"}:
            raise ValueError("target_mode must be 'transition' or 'level'")
        if cfg.location_mode not in {"free", "stationary"}:
            raise ValueError("location_mode must be 'free' or 'stationary'")
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

        context_dim = cfg.memory_dim + cfg.n_cells
        base = torch.arange(cfg.n_cells)
        self.layers = nn.ModuleList(
            [
                CouplingLayer(
                    path_dim=cfg.n_cells,
                    context_dim=context_dim,
                    hidden_dim=cfg.coupling_hidden,
                    dropout=cfg.coupling_dropout,
                    scale_clip=cfg.scale_clip,
                    mask=((base + idx) % 2 == 0).float(),
                )
                for idx in range(cfg.coupling_layers)
            ]
        )
        self.location_head: nn.Module | None = None
        if cfg.use_location:
            location_out = 2 * cfg.n_cells if cfg.location_mode == "stationary" else cfg.n_cells
            self.location_head = nn.Sequential(
                nn.Linear(context_dim, cfg.location_hidden),
                nn.GELU(),
                nn.Dropout(cfg.coupling_dropout),
                nn.Linear(cfg.location_hidden, location_out),
            )
            last = self.location_head[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.zero_()
                    last.bias.zero_()
                    if cfg.location_mode == "stationary":
                        last.bias[cfg.n_cells :].fill_(math.log(0.1 / 0.9))
        self.student_df_raw: nn.Parameter | None = None
        if cfg.base_distribution == "student_t":
            df_offset = max(float(cfg.student_df_init) - float(cfg.student_df_min), 1e-4)
            raw = math.log(math.expm1(df_offset))
            self.student_df_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
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

    def _encode_prefix_scores(self, prefix_scores: torch.Tensor) -> torch.Tensor:
        seq_len = prefix_scores.shape[1]
        if seq_len > self.cfg.history_len + self.cfg.future_len:
            raise ValueError(
                f"Prefix length {seq_len} exceeds configured max "
                f"{self.cfg.history_len + self.cfg.future_len}"
            )
        pos = torch.arange(seq_len, device=prefix_scores.device)
        x = self.feature_proj(self._score_features_from_scores(prefix_scores))
        x = x + self.pos_embed(pos)[None, :, :]
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

    def teacher_forced_states(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history_scores = self.history_scores(history_norm)
        future_scores = self.target_future_scores(future_norm)
        prefix_scores = torch.cat([history_scores, future_scores[:, :-1]], dim=1)
        hidden = self._encode_prefix_scores(prefix_scores)
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        current_scores = prefix_scores[:, start : start + self.cfg.future_len]
        transitions = future_scores - current_scores
        target = transitions if self.cfg.target_mode == "transition" else future_scores
        return memory_states, current_scores, target, transitions

    @staticmethod
    def _context(memory_state: torch.Tensor, current_score: torch.Tensor) -> torch.Tensor:
        return torch.cat([memory_state, current_score], dim=-1)

    def transition_location(
        self,
        memory_state: torch.Tensor,
        current_score: torch.Tensor,
    ) -> torch.Tensor:
        if self.location_head is None:
            return torch.zeros_like(current_score)
        raw = self.location_head(self._context(memory_state, current_score))
        if self.cfg.location_mode == "free":
            return raw
        anchor, alpha_raw = raw.chunk(2, dim=-1)
        alpha = torch.sigmoid(alpha_raw)
        return alpha * (anchor - current_score)

    def student_df(self) -> torch.Tensor:
        if self.student_df_raw is None:
            return torch.tensor(float("inf"), device=self.level_quantiles.device)
        return float(self.cfg.student_df_min) + F.softplus(self.student_df_raw)

    def base_nll(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.base_distribution == "normal":
            return 0.5 * z.square().sum(dim=-1) + 0.5 * z.shape[-1] * math.log(2.0 * math.pi)
        df = self.student_df().to(device=z.device, dtype=z.dtype)
        dist = torch.distributions.StudentT(df)
        return -dist.log_prob(z).sum(dim=-1)

    def sample_base_like(self, ref: torch.Tensor, temperature: float) -> torch.Tensor:
        if self.cfg.base_distribution == "normal":
            return temperature * torch.randn_like(ref)
        df = self.student_df().to(device=ref.device, dtype=ref.dtype)
        dist = torch.distributions.StudentT(df)
        return temperature * dist.sample(ref.shape)

    def forward_to_base(
        self,
        transition: torch.Tensor,
        memory_state: torch.Tensor,
        current_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = transition
        context = self._context(memory_state, current_score)
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            z, ld = layer(z, context)
            log_det = log_det + ld
        return z, log_det

    def inverse_from_base(
        self,
        base: torch.Tensor,
        memory_state: torch.Tensor,
        current_score: torch.Tensor,
    ) -> torch.Tensor:
        x = base
        context = self._context(memory_state, current_score)
        for layer in reversed(self.layers):
            x = layer.inverse(x, context)
        return x

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory_states, current_scores, target, transitions = self.teacher_forced_states(
            history_norm,
            future_norm,
        )
        bsz, horizon, n_cells = target.shape
        flat_target = target.reshape(bsz * horizon, n_cells)
        flat_memory = memory_states.reshape(bsz * horizon, self.cfg.memory_dim)
        flat_current = current_scores.reshape(bsz * horizon, n_cells)
        location = self.transition_location(flat_memory, flat_current)
        residual = flat_target - location
        z, log_det = self.forward_to_base(residual, flat_memory, flat_current)
        nll = (self.base_nll(z) - log_det) / float(n_cells)
        loss = nll.mean()
        metrics = {
            "total": loss.detach(),
            "nll": loss.detach(),
            "transition_std": transitions.std(unbiased=False).detach(),
            "transition_abs": transitions.abs().mean().detach(),
            "base_std": z.std(unbiased=False).detach(),
            "base_abs": z.abs().mean().detach(),
            "log_det_per_dim": (log_det / float(n_cells)).mean().detach(),
            "location_abs": location.abs().mean().detach(),
            "location_std": location.std(unbiased=False).detach(),
            "base_df": self.student_df().detach(),
            "memory_abs": memory_states.abs().mean().detach(),
        }
        return loss, metrics

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
        bsz = history_scores.shape[0]
        chunk_size = max(
            1,
            min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)),
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)

        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            prefix = (
                history_scores.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            frames: list[torch.Tensor] = []
            for _step in range(n_steps):
                memory_state = self._encode_prefix_scores(prefix)[:, -1]
                current_score = prefix[:, -1]
                base = self.sample_base_like(current_score, temp)
                location = self.transition_location(memory_state, current_score)
                target = location + self.inverse_from_base(
                    base,
                    memory_state,
                    current_score,
                )
                if self.cfg.target_mode == "transition":
                    next_score = current_score + target
                else:
                    next_score = target
                next_iv = self._scores_to_values(next_score)
                if self.cfg.n_cells == 25:
                    frames.append(next_iv.view(bsz, k, 5, 5))
                else:
                    frames.append(next_iv.view(bsz, k, self.cfg.n_cells))
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
) -> tuple[EmpiricalNormalScoreTransitionCouplingDensity, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScoreTransitionCouplingConfig(**payload["config"])
    model = EmpiricalNormalScoreTransitionCouplingDensity(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScoreTransitionCouplingDensity,
    cfg: EmpiricalNormalScoreTransitionCouplingConfig,
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
