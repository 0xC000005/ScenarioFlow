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
class EmpiricalNormalScoreDayVectorDensityConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    memory_dim: int = 128
    memory_layers: int = 3
    memory_heads: int = 4
    memory_ff: int = 256
    dropout: float = 0.1

    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    prefix_feature_mode: str = "basic"
    chol_diag_floor: float = 1e-3
    sample_temperature: float = 1.0
    max_sample_chunk: int = 8


class EmpiricalNormalScoreDayVectorDensity(nn.Module):
    """460a: causal day-vector transition density in empirical normal-score space.

    The model factorizes the future law by day:
    p(Y_1:T | H) = prod_t p(S_t - S_{t-1} | H, S_<t)
    where S are per-cell empirical normal scores. Each conditional factor is a
    full-rank 25-dimensional Gaussian, so cross-cell geometry is learned jointly.
    """

    def __init__(self, cfg: EmpiricalNormalScoreDayVectorDensityConfig):
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
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory = nn.TransformerEncoder(layer, num_layers=cfg.memory_layers)
        self.memory_norm = nn.LayerNorm(cfg.memory_dim)

        tril_size = cfg.n_cells * (cfg.n_cells + 1) // 2
        self.param_head = nn.Sequential(
            nn.LayerNorm(cfg.memory_dim + cfg.n_cells),
            nn.Linear(cfg.memory_dim + cfg.n_cells, cfg.memory_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.memory_ff, cfg.n_cells + tril_size),
        )
        tril_idx = torch.tril_indices(cfg.n_cells, cfg.n_cells)
        self.register_buffer("tril_row", tril_idx[0])
        self.register_buffer("tril_col", tril_idx[1])
        self.register_buffer("tril_is_diag", tril_idx[0] == tril_idx[1])

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

    def teacher_forced_states(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_scores = self.history_scores(history_norm)
        future_scores = self.target_future_scores(future_norm)
        prefix_scores = torch.cat([history_scores, future_scores[:, :-1]], dim=1)
        hidden = self._encode_prefix_scores(prefix_scores)
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        current_scores = prefix_scores[:, start : start + self.cfg.future_len]
        transitions = future_scores - current_scores
        return memory_states, current_scores, transitions

    def _density_params(
        self,
        memory_state: torch.Tensor,
        current_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.param_head(torch.cat([memory_state, current_score], dim=-1))
        loc = raw[..., : self.cfg.n_cells]
        raw_tril = raw[..., self.cfg.n_cells :]
        leading = raw_tril.shape[:-1]
        chol = raw_tril.new_zeros(*leading, self.cfg.n_cells, self.cfg.n_cells)
        tril_values = raw_tril.clone()
        diag_values = F.softplus(tril_values[..., self.tril_is_diag])
        diag_values = diag_values + float(self.cfg.chol_diag_floor)
        tril_values[..., self.tril_is_diag] = diag_values
        chol[..., self.tril_row, self.tril_col] = tril_values
        return loc, chol

    def transition_nll(
        self,
        target_transition: torch.Tensor,
        loc: torch.Tensor,
        chol: torch.Tensor,
    ) -> torch.Tensor:
        diff = target_transition - loc
        whitened = torch.linalg.solve_triangular(
            chol,
            diff.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        quad = whitened.square().sum(dim=-1)
        diag = torch.diagonal(chol, dim1=-2, dim2=-1)
        logdet = torch.log(diag).sum(dim=-1)
        const = 0.5 * self.cfg.n_cells * math.log(2.0 * math.pi)
        return 0.5 * quad + logdet + const

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory_states, current_scores, transitions = self.teacher_forced_states(
            history_norm,
            future_norm,
        )
        loc, chol = self._density_params(memory_states, current_scores)
        nll_grid = self.transition_nll(transitions, loc, chol)
        nll = nll_grid.mean()
        diag = torch.diagonal(chol, dim1=-2, dim2=-1)
        metrics = {
            "total": nll.detach(),
            "nll": nll.detach(),
            "transition_std": transitions.std(unbiased=False).detach(),
            "transition_abs": transitions.abs().mean().detach(),
            "loc_abs": loc.abs().mean().detach(),
            "diag_mean": diag.mean().detach(),
            "diag_min": diag.min().detach(),
            "mean_abs_err": (loc - transitions).abs().mean().detach(),
        }
        return nll, metrics

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
                loc, chol = self._density_params(memory_state, current_score)
                z = torch.randn_like(current_score)
                transition = loc + temp * torch.matmul(chol, z.unsqueeze(-1)).squeeze(-1)
                next_score = current_score + transition
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
) -> tuple[EmpiricalNormalScoreDayVectorDensity, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScoreDayVectorDensityConfig(**payload["config"])
    model = EmpiricalNormalScoreDayVectorDensity(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScoreDayVectorDensity,
    cfg: EmpiricalNormalScoreDayVectorDensityConfig,
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
