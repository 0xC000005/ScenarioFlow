from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.joint_token_logit_transition_flow_matching import (
    JointTokenLogitTransitionFMConfig,
    JointTokenLogitTransitionFlowMatching,
    _flow_time_features,
)
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class UnifiedGlobalTokenStateAwareFutureLogitPathFMConfig(JointTokenLogitTransitionFMConfig):
    n_global_tokens: int = 4
    standardize_logits: bool = False
    logit_std_floor: float = 1e-3


class UnifiedGlobalTokenStateAwareFutureLogitPathVelocity(nn.Module):
    def __init__(self, cfg: UnifiedGlobalTokenStateAwareFutureLogitPathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(2, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.prefix_embed = nn.Parameter(torch.zeros(2, cfg.token_dim))
        self.global_embed = nn.Parameter(0.02 * torch.randn(cfg.n_global_tokens, cfg.token_dim))
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
        future_logits_t: torch.Tensor,
        implied_transitions_t: torch.Tensor,
        context: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, horizon, n_cells = future_logits_t.shape
        h_idx = torch.arange(horizon, device=future_logits_t.device)
        c_idx = torch.arange(n_cells, device=future_logits_t.device)
        pos = (
            self.horizon_embed(h_idx)[:, None, :]
            + self.cell_embed(c_idx)[None, :, :]
        ).reshape(horizon * n_cells, self.cfg.token_dim)
        values = torch.stack(
            [future_logits_t, implied_transitions_t], dim=-1
        ).reshape(bsz, horizon * n_cells, 2)
        token = self.value_proj(values) + pos[None, :, :]
        ctx_token = self.context_proj(context) + self.prefix_embed[0][None, :]
        time_token = self.flow_time_proj(
            _flow_time_features(flow_t, self.cfg.flow_time_dim)
        ) + self.prefix_embed[1][None, :]
        global_tokens = self.global_embed[None, :, :] + ctx_token[:, None, :] + time_token[:, None, :]
        hidden = torch.cat(
            [torch.stack([ctx_token, time_token], dim=1), global_tokens, token],
            dim=1,
        )
        hidden = self.mixer(hidden)
        prefix_len = 2 + self.cfg.n_global_tokens
        vel = self.out(hidden[:, prefix_len:, :]).squeeze(-1)
        return vel.view(bsz, horizon, n_cells)


class UnifiedGlobalTokenStateAwareFutureLogitPathFlowMatching(
    JointTokenLogitTransitionFlowMatching
):
    """316a-v0: direct future-logit path flow with generic global mixer tokens."""

    def __init__(self, cfg: UnifiedGlobalTokenStateAwareFutureLogitPathFMConfig):
        super().__init__(cfg)
        if cfg.n_global_tokens <= 0:
            raise ValueError("n_global_tokens must be positive")
        self.velocity = UnifiedGlobalTokenStateAwareFutureLogitPathVelocity(cfg)
        self.register_buffer("cell_logit_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_logit_std", torch.ones(cfg.n_cells))

    def set_logit_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.cfg.n_cells,) or std.shape != (self.cfg.n_cells,):
            raise ValueError("Expected per-cell logit stats with shape (n_cells,)")
        self.cell_logit_mean.copy_(mean.to(self.cell_logit_mean))
        self.cell_logit_std.copy_(
            std.clamp_min(self.cfg.logit_std_floor).to(self.cell_logit_std)
        )

    def _to_model_coord(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return logits
        return (logits - self.cell_logit_mean) / self.cell_logit_std

    def _from_model_coord(self, coord: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return coord
        return coord * self.cell_logit_std + self.cell_logit_mean

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        future_01 = denormalize_iv(future_norm)
        return self._to_model_coord(iv_to_logit(future_01, self.cfg.logit_eps))

    def implied_transitions(
        self,
        history_norm: torch.Tensor,
        future_logits: torch.Tensor,
    ) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        last_logit = self._to_model_coord(self.history_last_logit(history_norm))
        return torch.cat(
            [
                future_logits[:, :1] - last_logit[:, None, :],
                future_logits[:, 1:] - future_logits[:, :-1],
            ],
            dim=1,
        )

    def predict_velocity(
        self,
        future_logits_t: torch.Tensor,
        history_norm: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        trans_t = self.implied_transitions(history_norm, future_logits_t)
        return self.velocity(future_logits_t, trans_t, context, t)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        x1 = self.target_future_logits(future_norm)
        x0 = torch.randn_like(x1)
        bsz = history_norm.shape[0]
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.predict_velocity(x_t, history_norm, context, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        trans = self.implied_transitions(history_norm, x1)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "future_logit_std": x1.std(unbiased=False).detach(),
            "future_logit_abs": x1.abs().mean().detach(),
            "implied_transition_std": trans.std(unbiased=False).detach(),
            "implied_transition_abs": trans.abs().mean().detach(),
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
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        context = self.encode_history(history_norm)
        bsz = history_norm.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            x = temp * torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=context.device,
                dtype=context.dtype,
            )
            hist = history_norm.repeat_interleave(k, dim=0)
            ctx = context.repeat_interleave(k, dim=0)
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=context.device,
                    dtype=context.dtype,
                )
                x = x + dt * self.predict_velocity(x, hist, ctx, t)
            future_01 = logit_to_iv(self._from_model_coord(x))
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(
                    bsz, k, self.cfg.future_len, self.cfg.n_cells
                )
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[UnifiedGlobalTokenStateAwareFutureLogitPathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = UnifiedGlobalTokenStateAwareFutureLogitPathFMConfig(**payload["config"])
    model = UnifiedGlobalTokenStateAwareFutureLogitPathFlowMatching(cfg)
    result = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = {"cell_logit_mean", "cell_logit_std"}
    missing = set(result.missing_keys) - allowed_missing
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    if cfg.standardize_logits and allowed_missing.intersection(result.missing_keys):
        raise RuntimeError("Standardized-logit checkpoint is missing saved logit stats")
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: UnifiedGlobalTokenStateAwareFutureLogitPathFlowMatching,
    cfg: UnifiedGlobalTokenStateAwareFutureLogitPathFMConfig,
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
