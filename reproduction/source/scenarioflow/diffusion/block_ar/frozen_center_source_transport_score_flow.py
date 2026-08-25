from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
    load_model as load_empirical_score_ar_model,
)
from diffusion.block_ar.frozen_center_residual_score_flow import (
    FrozenCenterResidualScoreFMConfig,
    FrozenCenterResidualScoreFlow,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


def source_transport_loss(
    model: FrozenCenterResidualScoreFlow,
    history_scores: torch.Tensor,
    center_scores: torch.Tensor,
    source_residual_scores: torch.Tensor,
    future_scores: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_residual = future_scores - center_scores
    bsz = target_residual.shape[0]
    t = torch.rand(bsz, device=target_residual.device, dtype=target_residual.dtype)
    residual_t = (
        (1.0 - t)[:, None, None] * source_residual_scores
        + t[:, None, None] * target_residual
    )
    target_velocity = target_residual - source_residual_scores
    context = model.encode_condition(history_scores, center_scores)
    pred_velocity = model.velocity(residual_t, center_scores, context, t)
    fm_loss = F.mse_loss(pred_velocity, target_velocity)
    metrics = {
        "total": fm_loss.detach(),
        "fm_loss": fm_loss.detach(),
        "center_score_std": center_scores.std(unbiased=False).detach(),
        "source_residual_std": source_residual_scores.std(unbiased=False).detach(),
        "target_residual_std": target_residual.std(unbiased=False).detach(),
        "transport_delta_std": target_velocity.std(unbiased=False).detach(),
        "transport_delta_abs": target_velocity.abs().mean().detach(),
    }
    return fm_loss, metrics


@torch.no_grad()
def transport_residual_scores(
    model: FrozenCenterResidualScoreFlow,
    history_scores: torch.Tensor,
    center_scores: torch.Tensor,
    source_residual_scores: torch.Tensor,
    chunk_size: int,
    transport_strength: float | None = None,
) -> torch.Tensor:
    bsz, n_samples, horizon, n_cells = source_residual_scores.shape
    if horizon != model.cfg.future_len or n_cells != model.cfg.n_cells:
        raise ValueError(
            f"Expected source residual shape (*,*,{model.cfg.future_len},{model.cfg.n_cells}), "
            f"got {tuple(source_residual_scores.shape)}"
        )
    chunk_size = max(
        1, min(int(chunk_size), int(n_samples), int(model.cfg.max_sample_chunk))
    )
    strength = float(
        model.cfg.transport_strength
        if transport_strength is None
        else transport_strength
    )
    strength = max(0.0, min(1.0, strength))
    if strength == 0.0:
        return source_residual_scores
    n_flow_steps = max(1, int(math.ceil(float(model.cfg.flow_steps) * strength)))
    dt = strength / float(n_flow_steps)
    context = model.encode_condition(history_scores, center_scores)
    outs: list[torch.Tensor] = []
    for start in range(0, n_samples, chunk_size):
        k = min(chunk_size, n_samples - start)
        residual = source_residual_scores[:, start : start + k].reshape(
            bsz * k,
            horizon,
            n_cells,
        )
        center = center_scores.repeat_interleave(k, dim=0)
        ctx = context.repeat_interleave(k, dim=0)
        for step in range(n_flow_steps):
            t = torch.full(
                (bsz * k,),
                (step + 0.5) * dt,
                device=history_scores.device,
                dtype=history_scores.dtype,
            )
            residual = residual + dt * model.velocity(residual, center, ctx, t)
        outs.append(residual.view(bsz, k, horizon, n_cells))
    return torch.cat(outs, dim=1)


class FrozenCenterSourceTransportScenarioGenerator(nn.Module):
    def __init__(
        self,
        residual_flow: FrozenCenterResidualScoreFlow,
        base_model: nn.Module,
        base_checkpoint_path: str,
    ):
        super().__init__()
        self.residual_flow = residual_flow
        self.base_model = base_model.eval()
        self.base_checkpoint_path = base_checkpoint_path
        self.cfg = residual_flow.cfg
        for param in self.base_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def _base_source_scores(
        self,
        history_norm: torch.Tensor,
        n_samples: int,
        n_steps: int,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_01 = self.base_model.sample_batched(
            history_norm,
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
        )
        center_01 = source_01.median(dim=1).values
        center_norm = normalize_iv(center_01).view(center_01.shape[0], n_steps, -1)
        center_scores = self.base_model.target_future_scores(center_norm)
        flat = source_01.reshape(source_01.shape[0] * source_01.shape[1], n_steps, 5, 5)
        source_norm = normalize_iv(flat).view(flat.shape[0], n_steps, -1)
        source_scores = self.base_model.target_future_scores(source_norm).view(
            source_01.shape[0],
            source_01.shape[1],
            n_steps,
            self.cfg.n_cells,
        )
        return center_scores, source_scores

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
        history_norm = history_norm.view(history_norm.shape[0], history_norm.shape[1], -1)
        history_scores = self.base_model.history_scores(history_norm)
        center_scores, source_scores = self._base_source_scores(
            history_norm=history_norm,
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        source_residual = temp * (source_scores - center_scores[:, None, :, :])
        transported = transport_residual_scores(
            model=self.residual_flow,
            history_scores=history_scores,
            center_scores=center_scores,
            source_residual_scores=source_residual,
            chunk_size=chunk_size,
            transport_strength=self.cfg.transport_strength,
        )
        future_scores = center_scores[:, None, :, :] + transported
        flat = future_scores.reshape(
            future_scores.shape[0] * future_scores.shape[1],
            n_steps,
            self.cfg.n_cells,
        )
        future_01 = self.base_model._scores_to_values(flat)
        if self.cfg.n_cells == 25:
            return future_01.view(future_scores.shape[0], future_scores.shape[1], n_steps, 5, 5)
        side = int(math.sqrt(self.cfg.n_cells))
        if side * side == self.cfg.n_cells:
            return future_01.view(
                future_scores.shape[0],
                future_scores.shape[1],
                n_steps,
                side,
                side,
            )
        return future_01.view(
            future_scores.shape[0],
            future_scores.shape[1],
            n_steps,
            self.cfg.n_cells,
        )


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[FrozenCenterSourceTransportScenarioGenerator, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = FrozenCenterResidualScoreFMConfig(**payload["config"])
    residual_flow = FrozenCenterResidualScoreFlow(cfg)
    residual_flow.load_state_dict(payload["model_state_dict"], strict=True)
    base_checkpoint_path = payload["base_checkpoint_path"]
    base_model, _base_payload = load_empirical_score_ar_model(base_checkpoint_path, device)
    model = FrozenCenterSourceTransportScenarioGenerator(
        residual_flow=residual_flow.to(device),
        base_model=base_model,
        base_checkpoint_path=base_checkpoint_path,
    )
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: FrozenCenterResidualScoreFlow,
    cfg: FrozenCenterResidualScoreFMConfig,
    epoch: int,
    best_val: float,
    base_checkpoint_path: str,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
            "base_checkpoint_path": base_checkpoint_path,
        },
        path,
    )
