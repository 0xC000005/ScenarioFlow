from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.joint_token_logit_transition_flow_matching import (
    JointTokenLogitTransitionFMConfig,
    _flow_time_features,
)
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


class UnifiedPooledHistoryMemoryStateAwareFutureLogitPathVelocity(nn.Module):
    def __init__(self, cfg: JointTokenLogitTransitionFMConfig):
        super().__init__()
        self.cfg = cfg
        self.future_value_proj = nn.Linear(2, cfg.token_dim)
        self.history_value_proj = nn.Linear(2 * cfg.n_cells, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.history_time_embed = nn.Embedding(cfg.history_len, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.summary_attn = nn.Linear(cfg.token_dim, 1)
        self.token_type_embed = nn.Parameter(torch.zeros(4, cfg.token_dim))
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
        history_tokens: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, horizon, n_cells = future_logits_t.shape
        h_idx = torch.arange(horizon, device=future_logits_t.device)
        c_idx = torch.arange(n_cells, device=future_logits_t.device)
        future_pos = (
            self.horizon_embed(h_idx)[:, None, :]
            + self.cell_embed(c_idx)[None, :, :]
        ).reshape(horizon * n_cells, self.cfg.token_dim)
        future_values = torch.stack(
            [future_logits_t, implied_transitions_t], dim=-1
        ).reshape(bsz, horizon * n_cells, 2)
        future_token = (
            self.future_value_proj(future_values)
            + future_pos[None, :, :]
            + self.token_type_embed[2][None, None, :]
        )

        hist_idx = torch.arange(history_tokens.shape[1], device=history_tokens.device)
        hist_token = (
            self.history_value_proj(history_tokens)
            + self.history_time_embed(hist_idx)[None, :, :]
            + self.token_type_embed[1][None, None, :]
        )
        summary_weights = torch.softmax(self.summary_attn(hist_token).squeeze(-1), dim=1)
        summary_token = (
            (summary_weights[:, :, None] * hist_token).sum(dim=1, keepdim=True)
            + self.token_type_embed[3][None, None, :]
        )

        time_token = (
            self.flow_time_proj(_flow_time_features(flow_t, self.cfg.flow_time_dim))
            + self.token_type_embed[0][None, :]
        )
        hidden = torch.cat(
            [time_token[:, None, :], summary_token, hist_token, future_token], dim=1
        )
        hidden = self.mixer(hidden)
        vel = self.out(hidden[:, 2 + history_tokens.shape[1] :, :]).squeeze(-1)
        return vel.view(bsz, horizon, n_cells)


class UnifiedPooledHistoryMemoryStateAwareFutureLogitPathFlowMatching(nn.Module):
    """312c-v0: unified future-logit path flow with pooled global history and memory."""

    def __init__(self, cfg: JointTokenLogitTransitionFMConfig):
        super().__init__()
        self.cfg = cfg
        self.velocity = UnifiedPooledHistoryMemoryStateAwareFutureLogitPathVelocity(cfg)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        future_01 = denormalize_iv(future_norm)
        return iv_to_logit(future_01, self.cfg.logit_eps)

    def prepare_history_state(
        self,
        history_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_norm = self._flatten(history_norm)
        history_01 = denormalize_iv(history_norm)
        history_logits = iv_to_logit(history_01, self.cfg.logit_eps)
        prev_logits = torch.cat([history_logits[:, :1], history_logits[:, :-1]], dim=1)
        history_transitions = history_logits - prev_logits
        history_tokens = torch.cat([history_logits, history_transitions], dim=-1)
        return history_tokens, history_logits[:, -1]

    @staticmethod
    def implied_transitions_from_last_logit(
        last_logit: torch.Tensor,
        future_logits: torch.Tensor,
    ) -> torch.Tensor:
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
        history_tokens: torch.Tensor,
        last_logit: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        trans_t = self.implied_transitions_from_last_logit(last_logit, future_logits_t)
        return self.velocity(future_logits_t, trans_t, history_tokens, t)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        history_tokens, last_logit = self.prepare_history_state(history_norm)
        x1 = self.target_future_logits(future_norm)
        x0 = torch.randn_like(x1)
        bsz = history_norm.shape[0]
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.predict_velocity(x_t, history_tokens, last_logit, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        trans = self.implied_transitions_from_last_logit(last_logit, x1)
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
        history_tokens, last_logit = self.prepare_history_state(history_norm)
        bsz = history_norm.shape[0]
        chunk_size = max(
            1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk))
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            x = temp * torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=history_tokens.device,
                dtype=history_tokens.dtype,
            )
            hist_tokens = history_tokens.repeat_interleave(k, dim=0)
            last = last_logit.repeat_interleave(k, dim=0)
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=history_tokens.device,
                    dtype=history_tokens.dtype,
                )
                x = x + dt * self.predict_velocity(x, hist_tokens, last, t)
            future_01 = logit_to_iv(x)
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
) -> tuple[UnifiedPooledHistoryMemoryStateAwareFutureLogitPathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = JointTokenLogitTransitionFMConfig(**payload["config"])
    model = UnifiedPooledHistoryMemoryStateAwareFutureLogitPathFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: UnifiedPooledHistoryMemoryStateAwareFutureLogitPathFlowMatching,
    cfg: JointTokenLogitTransitionFMConfig,
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
