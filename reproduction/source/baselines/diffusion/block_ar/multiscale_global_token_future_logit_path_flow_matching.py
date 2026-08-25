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
class MultiscaleGlobalTokenFutureLogitPathFMConfig(JointTokenLogitTransitionFMConfig):
    knot_positions: tuple[int, ...] = (0, 3, 7, 13, 20, 29)
    n_global_tokens: int = 4


class MultiscaleGlobalTokenFutureLogitPathVelocity(nn.Module):
    def __init__(self, cfg: MultiscaleGlobalTokenFutureLogitPathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(2, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.prefix_embed = nn.Parameter(torch.zeros(2, cfg.token_dim))
        self.global_embed = nn.Parameter(0.02 * torch.randn(cfg.n_global_tokens, cfg.token_dim))
        self.token_type_embed = nn.Parameter(torch.zeros(2, cfg.token_dim))
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
        knot_mask: torch.Tensor,
        context: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        flat_mask = knot_mask.reshape(horizon * n_cells)
        type_emb = torch.where(
            flat_mask[:, None],
            self.token_type_embed[0][None, :],
            self.token_type_embed[1][None, :],
        )
        token = token + type_emb[None, :, :]

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
        vel = self.out(hidden[:, prefix_len:, :]).squeeze(-1).view(bsz, horizon, n_cells)
        return vel[:, self.cfg.knot_positions, :], vel[:, ~knot_mask[:, 0], :]


class MultiscaleGlobalTokenFutureLogitPathFlowMatching(JointTokenLogitTransitionFlowMatching):
    """315b-v0: multiscale path flow with generic global mixer tokens for shared coupling."""

    def __init__(self, cfg: MultiscaleGlobalTokenFutureLogitPathFMConfig):
        super().__init__(cfg)
        self.cfg = cfg
        knot_positions = tuple(int(x) for x in cfg.knot_positions)
        if len(knot_positions) < 2:
            raise ValueError("Need at least two knot positions")
        if knot_positions[0] != 0 or knot_positions[-1] != cfg.future_len - 1:
            raise ValueError("knot_positions must include the first and last future horizon")
        if sorted(knot_positions) != list(knot_positions):
            raise ValueError("knot_positions must be sorted")
        if cfg.n_global_tokens <= 0:
            raise ValueError("n_global_tokens must be positive")
        self.velocity = MultiscaleGlobalTokenFutureLogitPathVelocity(cfg)

        knot_mask = torch.zeros(cfg.future_len, dtype=torch.bool)
        knot_mask[list(knot_positions)] = True
        self.register_buffer("knot_mask_h", knot_mask, persistent=False)
        non_knot_positions = [i for i in range(cfg.future_len) if i not in knot_positions]
        self.register_buffer(
            "non_knot_positions",
            torch.tensor(non_knot_positions, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "knot_positions_tensor",
            torch.tensor(knot_positions, dtype=torch.long),
            persistent=False,
        )
        knot_mask_full = knot_mask[:, None].expand(cfg.future_len, cfg.n_cells)
        self.register_buffer("knot_mask_full", knot_mask_full, persistent=False)

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        future_01 = denormalize_iv(future_norm)
        return iv_to_logit(future_01, self.cfg.logit_eps)

    def interpolate_knots(self, coarse_knots: torch.Tensor) -> torch.Tensor:
        bsz, _, n_cells = coarse_knots.shape
        scaffold = torch.empty(
            bsz,
            self.cfg.future_len,
            n_cells,
            device=coarse_knots.device,
            dtype=coarse_knots.dtype,
        )
        knot_pos = list(self.cfg.knot_positions)
        for i in range(len(knot_pos) - 1):
            start = knot_pos[i]
            end = knot_pos[i + 1]
            left = coarse_knots[:, i : i + 1, :]
            right = coarse_knots[:, i + 1 : i + 2, :]
            alpha = torch.linspace(
                0.0,
                1.0,
                end - start + 1,
                device=coarse_knots.device,
                dtype=coarse_knots.dtype,
            ).view(1, -1, 1)
            scaffold[:, start : end + 1, :] = (1.0 - alpha) * left + alpha * right
        return scaffold

    def decompose_future_logits(
        self,
        future_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse = future_logits[:, self.knot_positions_tensor, :]
        scaffold = self.interpolate_knots(coarse)
        residual = future_logits[:, self.non_knot_positions, :] - scaffold[:, self.non_knot_positions, :]
        return coarse, residual

    def compose_future_logits(
        self,
        coarse: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        scaffold = self.interpolate_knots(coarse)
        future_logits = scaffold.clone()
        future_logits[:, self.non_knot_positions, :] = (
            future_logits[:, self.non_knot_positions, :] + residual
        )
        return future_logits

    def implied_transitions(
        self,
        history_norm: torch.Tensor,
        future_logits: torch.Tensor,
    ) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        last_logit = self.history_last_logit(history_norm)
        return torch.cat(
            [
                future_logits[:, :1] - last_logit[:, None, :],
                future_logits[:, 1:] - future_logits[:, :-1],
            ],
            dim=1,
        )

    def predict_velocity(
        self,
        coarse_t: torch.Tensor,
        residual_t: torch.Tensor,
        history_norm: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        future_logits_t = self.compose_future_logits(coarse_t, residual_t)
        trans_t = self.implied_transitions(history_norm, future_logits_t)
        return self.velocity(
            future_logits_t,
            trans_t,
            self.knot_mask_full,
            context,
            t,
        )

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        future_logits = self.target_future_logits(future_norm)
        coarse_1, residual_1 = self.decompose_future_logits(future_logits)
        coarse_0 = torch.randn_like(coarse_1)
        residual_0 = torch.randn_like(residual_1)

        bsz = history_norm.shape[0]
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        coarse_t = (1.0 - t)[:, None, None] * coarse_0 + t[:, None, None] * coarse_1
        residual_t = (1.0 - t)[:, None, None] * residual_0 + t[:, None, None] * residual_1
        coarse_target_velocity = coarse_1 - coarse_0
        residual_target_velocity = residual_1 - residual_0
        pred_coarse_velocity, pred_residual_velocity = self.predict_velocity(
            coarse_t, residual_t, history_norm, context, t
        )

        target_velocity = torch.cat([coarse_target_velocity, residual_target_velocity], dim=1)
        pred_velocity = torch.cat([pred_coarse_velocity, pred_residual_velocity], dim=1)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        trans = self.implied_transitions(history_norm, future_logits)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "coarse_std": coarse_1.std(unbiased=False).detach(),
            "residual_std": residual_1.std(unbiased=False).detach(),
            "future_logit_std": future_logits.std(unbiased=False).detach(),
            "implied_transition_std": trans.std(unbiased=False).detach(),
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
            coarse = temp * torch.randn(
                bsz * k,
                len(self.cfg.knot_positions),
                self.cfg.n_cells,
                device=context.device,
                dtype=context.dtype,
            )
            residual = temp * torch.randn(
                bsz * k,
                int(self.non_knot_positions.numel()),
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
                coarse_v, residual_v = self.predict_velocity(coarse, residual, hist, ctx, t)
                coarse = coarse + dt * coarse_v
                residual = residual + dt * residual_v
            future_logits = self.compose_future_logits(coarse, residual)
            future_01 = logit_to_iv(future_logits)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[MultiscaleGlobalTokenFutureLogitPathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = MultiscaleGlobalTokenFutureLogitPathFMConfig(**payload["config"])
    model = MultiscaleGlobalTokenFutureLogitPathFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: MultiscaleGlobalTokenFutureLogitPathFlowMatching,
    cfg: MultiscaleGlobalTokenFutureLogitPathFMConfig,
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
