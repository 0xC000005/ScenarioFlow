from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.causal_future_memory_transition_flow_matching import (
    CausalFutureMemoryTransitionFMConfig,
)
from diffusion.block_ar.joint_token_logit_transition_flow_matching import _flow_time_features
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class EmpiricalNormalScoreBlockCausalFMConfig(CausalFutureMemoryTransitionFMConfig):
    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    block_len: int = 5


class MemoryConditionedBlockVelocity(nn.Module):
    def __init__(self, cfg: EmpiricalNormalScoreBlockCausalFMConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(3, cfg.token_dim)
        self.memory_proj = nn.Linear(cfg.memory_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.time_dim, cfg.token_dim)
        self.block_pos_embed = nn.Embedding(cfg.block_len, cfg.token_dim)
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
        current_score: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, block_len, n_cells = x_t.shape
        if block_len != self.cfg.block_len or n_cells != self.cfg.n_cells:
            raise ValueError(
                f"Expected block shape (*,{self.cfg.block_len},{self.cfg.n_cells}), "
                f"got {tuple(x_t.shape)}"
            )
        prev_disp = torch.cat([torch.zeros_like(x_t[:, :1]), x_t[:, :-1]], dim=1)
        step_delta = x_t - prev_disp
        current = current_score[:, None, :].expand(-1, block_len, -1)
        values = torch.stack([x_t, step_delta, current], dim=-1)

        h_idx = torch.arange(block_len, device=x_t.device)
        c_idx = torch.arange(n_cells, device=x_t.device)
        pos = (
            self.block_pos_embed(h_idx)[:, None, :]
            + self.cell_embed(c_idx)[None, :, :]
        ).reshape(block_len * n_cells, self.cfg.token_dim)
        token = self.value_proj(values.reshape(bsz, block_len * n_cells, 3))
        token = token + pos[None, :, :]

        memory_token = self.memory_proj(memory_state) + self.prefix_embed[0][None, :]
        time_token = (
            self.flow_time_proj(_flow_time_features(t, self.cfg.time_dim))
            + self.prefix_embed[1][None, :]
        )
        hidden = torch.cat([torch.stack([memory_token, time_token], dim=1), token], dim=1)
        hidden = self.mixer(hidden)[:, 2:, :]
        return self.out(hidden).squeeze(-1).view(bsz, block_len, n_cells)


class EmpiricalNormalScoreBlockCausalFlowMatching(nn.Module):
    """341a: block-causal path FM in shared empirical normal-score coordinates."""

    def __init__(self, cfg: EmpiricalNormalScoreBlockCausalFMConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.future_len % cfg.block_len != 0:
            raise ValueError("future_len must be divisible by block_len for 341a")
        self.feature_proj = nn.Linear(2 * cfg.n_cells, cfg.memory_dim)
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
        self.velocity = MemoryConditionedBlockVelocity(cfg)
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

    def _score_features_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(scores)
        deltas[:, 1:] = scores[:, 1:] - scores[:, :-1]
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

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_scores = self.history_scores(history_norm)
        future_scores = self.target_future_scores(future_norm)
        bsz = history_scores.shape[0]
        block_losses: list[torch.Tensor] = []
        target_velocity_stds: list[torch.Tensor] = []
        block_transition_abs: list[torch.Tensor] = []

        for block_start in range(0, self.cfg.future_len, self.cfg.block_len):
            if block_start == 0:
                prefix = history_scores
            else:
                prefix = torch.cat([history_scores, future_scores[:, :block_start]], dim=1)
            memory_state = self._encode_prefix_scores(prefix)[:, -1]
            current_score = prefix[:, -1]
            block_scores = future_scores[:, block_start : block_start + self.cfg.block_len]
            x1 = block_scores - current_score[:, None, :]
            x0 = torch.randn_like(x1)
            t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)
            x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
            target_velocity = x1 - x0
            pred_velocity = self.velocity(x_t, current_score, memory_state, t)
            block_losses.append(F.mse_loss(pred_velocity, target_velocity))
            target_velocity_stds.append(target_velocity.std(unbiased=False).detach())
            block_transition_abs.append(x1.abs().mean().detach())

        total = torch.stack(block_losses).mean()
        daily_transitions = future_scores - torch.cat(
            [history_scores[:, -1:], future_scores[:, :-1]], dim=1
        )
        metrics = {
            "total": total.detach(),
            "fm_loss": total.detach(),
            "daily_transition_std": daily_transitions.std(unbiased=False).detach(),
            "daily_transition_abs": daily_transitions.abs().mean().detach(),
            "block_transition_abs": torch.stack(block_transition_abs).mean().detach(),
            "target_velocity_std": torch.stack(target_velocity_stds).mean().detach(),
        }
        return total, metrics

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
            frames: list[torch.Tensor] = []
            remaining = n_steps
            while remaining > 0:
                memory_state = self._encode_prefix_scores(prefix)[:, -1]
                current_score = prefix[:, -1]
                x = temp * torch.randn(
                    bsz * k,
                    self.cfg.block_len,
                    self.cfg.n_cells,
                    device=history_scores.device,
                    dtype=history_scores.dtype,
                )
                for flow_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (flow_step + 0.5) * dt,
                        device=history_scores.device,
                        dtype=history_scores.dtype,
                    )
                    x = x + dt * self.velocity(x, current_score, memory_state, t)
                block_scores = current_score[:, None, :] + x
                take = min(remaining, self.cfg.block_len)
                block_scores = block_scores[:, :take, :]
                block_iv = self._scores_to_values(block_scores).view(bsz, k, take, 5, 5)
                frames.extend([block_iv[:, :, i] for i in range(take)])
                prefix = torch.cat([prefix, block_scores], dim=1)
                remaining -= take
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
) -> tuple[EmpiricalNormalScoreBlockCausalFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScoreBlockCausalFMConfig(**payload["config"])
    model = EmpiricalNormalScoreBlockCausalFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScoreBlockCausalFlowMatching,
    cfg: EmpiricalNormalScoreBlockCausalFMConfig,
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
