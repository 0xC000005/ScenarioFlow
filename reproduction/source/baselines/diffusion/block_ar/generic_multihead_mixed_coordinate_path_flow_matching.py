from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.generic_mixed_coordinate_path_flow_matching import (
    GenericMixedCoordinatePathFMConfig,
    GenericMixedCoordinatePathFlowMatching,
)
from diffusion.block_ar.recurrent_logit_transition_flow_matching import _time_features


@dataclass
class GenericMultiHeadMixedCoordinatePathFMConfig(GenericMixedCoordinatePathFMConfig):
    """Mixed-coordinate path flow with shared backbone and typed readout heads."""

    iv_count: int = 25
    head_hidden: int = 128


class _TypedOutputHead(nn.Module):
    def __init__(self, token_dim: int, head_hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class GenericMultiHeadMixedCoordinatePathFlowMatching(
    GenericMixedCoordinatePathFlowMatching
):
    """One stochastic path model with separate IV/factor observation heads.

    The history encoder, future denoiser, flow time, and sampled source path are
    shared. The only specialization is in how typed variables are embedded into
    the shared future tokens and decoded from shared hidden states.
    """

    cfg: GenericMultiHeadMixedCoordinatePathFMConfig

    def __init__(self, cfg: GenericMultiHeadMixedCoordinatePathFMConfig):
        if int(cfg.iv_count) < 1 or int(cfg.iv_count) > int(cfg.n_cells):
            raise ValueError("iv_count must be in [1, n_cells]")
        super().__init__(cfg)
        self.iv_count = int(cfg.iv_count)
        self.factor_count = int(cfg.n_cells) - self.iv_count
        self.iv_value_proj = nn.Linear(self.iv_count, cfg.token_dim)
        self.factor_value_proj = (
            nn.Linear(self.factor_count, cfg.token_dim)
            if self.factor_count > 0
            else None
        )
        self.iv_head = _TypedOutputHead(
            cfg.token_dim,
            int(cfg.head_hidden),
            self.iv_count,
        )
        self.factor_head = (
            _TypedOutputHead(cfg.token_dim, int(cfg.head_hidden), self.factor_count)
            if self.factor_count > 0
            else None
        )

    def _future_value_embedding(self, x_t: torch.Tensor) -> torch.Tensor:
        iv_hidden = self.iv_value_proj(x_t[..., : self.iv_count])
        if self.factor_count <= 0 or self.factor_value_proj is None:
            return iv_hidden
        return iv_hidden + self.factor_value_proj(x_t[..., self.iv_count :])

    def _typed_output(self, hidden: torch.Tensor) -> torch.Tensor:
        iv_out = self.iv_head(hidden)
        if self.factor_count <= 0 or self.factor_head is None:
            return iv_out
        factor_out = self.factor_head(hidden)
        return torch.cat([iv_out, factor_out], dim=-1)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        horizon = int(x_t.shape[1])
        pos = torch.arange(horizon, device=x_t.device)
        token = self._future_value_embedding(x_t)
        token = token + self.future_pos(pos)[None, :, :]
        token = token + self.context_proj(context)[:, None, :]
        token = token + self.time_proj(_time_features(t, self.cfg.time_dim))[:, None, :]
        hidden = self.future_denoiser(token)
        return self._typed_output(hidden)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GenericMultiHeadMixedCoordinatePathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GenericMultiHeadMixedCoordinatePathFMConfig(**payload["config"])
    model = GenericMultiHeadMixedCoordinatePathFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: GenericMultiHeadMixedCoordinatePathFlowMatching,
    cfg: GenericMultiHeadMixedCoordinatePathFMConfig,
    epoch: int,
    best_val: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": asdict(cfg),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
