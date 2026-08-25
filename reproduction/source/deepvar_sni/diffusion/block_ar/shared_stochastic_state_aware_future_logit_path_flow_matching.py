from __future__ import annotations

import math
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
class SharedStochasticStateAwareFutureLogitPathFMConfig(
    JointTokenLogitTransitionFMConfig
):
    shared_noise_tokens: int = 4
    shared_noise_dim: int = 16
    ot_source_coupling: bool = False


class SharedStochasticFutureLogitPathVelocity(nn.Module):
    def __init__(self, cfg: SharedStochasticStateAwareFutureLogitPathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(2, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.shared_noise_proj = nn.Linear(cfg.shared_noise_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.shared_pos = nn.Embedding(cfg.shared_noise_tokens, cfg.token_dim)
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
        future_logits_t: torch.Tensor,
        implied_transitions_t: torch.Tensor,
        context: torch.Tensor,
        flow_t: torch.Tensor,
        shared_noise: torch.Tensor,
    ) -> torch.Tensor:
        bsz, horizon, n_cells = future_logits_t.shape
        h_idx = torch.arange(horizon, device=future_logits_t.device)
        c_idx = torch.arange(n_cells, device=future_logits_t.device)
        pos = (
            self.horizon_embed(h_idx)[:, None, :] + self.cell_embed(c_idx)[None, :, :]
        ).reshape(horizon * n_cells, self.cfg.token_dim)
        values = torch.stack([future_logits_t, implied_transitions_t], dim=-1).reshape(
            bsz, horizon * n_cells, 2
        )
        token = self.value_proj(values) + pos[None, :, :]

        ctx_token = self.context_proj(context) + self.prefix_embed[0][None, :]
        time_token = (
            self.flow_time_proj(_flow_time_features(flow_t, self.cfg.flow_time_dim))
            + self.prefix_embed[1][None, :]
        )

        s_idx = torch.arange(shared_noise.shape[1], device=shared_noise.device)
        shared_token = (
            self.shared_noise_proj(shared_noise) + self.shared_pos(s_idx)[None, :, :]
        )
        hidden = torch.cat(
            [torch.stack([ctx_token, time_token], dim=1), shared_token, token],
            dim=1,
        )
        hidden = self.mixer(hidden)
        prefix_len = 2 + shared_noise.shape[1]
        vel = self.out(hidden[:, prefix_len:, :]).squeeze(-1)
        return vel.view(bsz, horizon, n_cells)


class SharedStochasticStateAwareFutureLogitPathFlowMatching(
    JointTokenLogitTransitionFlowMatching
):
    """317a-v0: direct future-logit path FM with sampled shared base-noise channels."""

    def __init__(self, cfg: SharedStochasticStateAwareFutureLogitPathFMConfig):
        super().__init__(cfg)
        if cfg.shared_noise_tokens <= 0:
            raise ValueError("shared_noise_tokens must be positive")
        if cfg.shared_noise_dim <= 0:
            raise ValueError("shared_noise_dim must be positive")
        self.velocity = SharedStochasticFutureLogitPathVelocity(cfg)
        latent_dim = cfg.shared_noise_tokens * cfg.shared_noise_dim
        basis = torch.randn(latent_dim, cfg.future_len * cfg.n_cells) / math.sqrt(
            float(latent_dim)
        )
        self.register_buffer("shared_noise_basis", basis)

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        future_01 = denormalize_iv(future_norm)
        return iv_to_logit(future_01, self.cfg.logit_eps)

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

    def sample_base_noise(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_noise = torch.randn(
            batch_size,
            self.cfg.shared_noise_tokens,
            self.cfg.shared_noise_dim,
            device=device,
            dtype=dtype,
        )
        shared_field = (
            shared_noise.reshape(batch_size, -1)
            @ self.shared_noise_basis.to(device=device, dtype=dtype)
        ).view(batch_size, self.cfg.future_len, self.cfg.n_cells)
        local_noise = torch.randn(
            batch_size,
            self.cfg.future_len,
            self.cfg.n_cells,
            device=device,
            dtype=dtype,
        )
        base_noise = (local_noise + shared_field) / math.sqrt(2.0)
        return base_noise, shared_noise, shared_field

    @staticmethod
    def _source_permutation_from_ot(
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        from scipy.optimize import linear_sum_assignment

        cost = torch.cdist(
            source.detach().flatten(1).float(),
            target.detach().flatten(1).float(),
        )
        source_idx, target_idx = linear_sum_assignment(cost.cpu().numpy())
        source_for_target = torch.empty(
            target.shape[0],
            dtype=torch.long,
            device=target.device,
        )
        source_for_target[
            torch.as_tensor(target_idx, dtype=torch.long, device=target.device)
        ] = torch.as_tensor(source_idx, dtype=torch.long, device=target.device)
        return source_for_target

    def predict_velocity(
        self,
        future_logits_t: torch.Tensor,
        history_norm: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        shared_noise: torch.Tensor,
    ) -> torch.Tensor:
        trans_t = self.implied_transitions(history_norm, future_logits_t)
        return self.velocity(future_logits_t, trans_t, context, t, shared_noise)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        x1 = self.target_future_logits(future_norm)
        x0, shared_noise, shared_field = self.sample_base_noise(
            history_norm.shape[0], x1.device, x1.dtype
        )
        if self.cfg.ot_source_coupling and history_norm.shape[0] > 1:
            source_perm = self._source_permutation_from_ot(x0, x1)
            x0 = x0[source_perm]
            shared_noise = shared_noise[source_perm]
            shared_field = shared_field[source_perm]
        bsz = history_norm.shape[0]
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.predict_velocity(
            x_t, history_norm, context, t, shared_noise
        )
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        trans = self.implied_transitions(history_norm, x1)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "future_logit_std": x1.std(unbiased=False).detach(),
            "future_logit_abs": x1.abs().mean().detach(),
            "implied_transition_std": trans.std(unbiased=False).detach(),
            "implied_transition_abs": trans.abs().mean().detach(),
            "base_noise_std": x0.std(unbiased=False).detach(),
            "shared_field_std": shared_field.std(unbiased=False).detach(),
        }
        if self.cfg.ot_source_coupling:
            metrics["ot_source_cost"] = (
                x0.detach().flatten(1).float() - x1.detach().flatten(1).float()
            ).pow(2).mean(dim=1).sqrt().mean().detach()
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
        chunk_size = max(
            1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk))
        )
        temp = float(
            self.cfg.sample_temperature if temperature is None else temperature
        )
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            x, shared_noise, _ = self.sample_base_noise(
                bsz * k, context.device, context.dtype
            )
            if temp != 1.0:
                x = x * temp
                shared_noise = shared_noise * temp
            hist = history_norm.repeat_interleave(k, dim=0)
            ctx = context.repeat_interleave(k, dim=0)
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=context.device,
                    dtype=context.dtype,
                )
                x = x + dt * self.predict_velocity(x, hist, ctx, t, shared_noise)
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
) -> tuple[SharedStochasticStateAwareFutureLogitPathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = SharedStochasticStateAwareFutureLogitPathFMConfig(**payload["config"])
    model = SharedStochasticStateAwareFutureLogitPathFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: SharedStochasticStateAwareFutureLogitPathFlowMatching,
    cfg: SharedStochasticStateAwareFutureLogitPathFMConfig,
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
