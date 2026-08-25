from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.generic_state_conditioned_mixed_coordinate_flow_matching import (
    GenericStateConditionedMixedCoordinateFMConfig,
    GenericStateConditionedMixedCoordinateFlowMatching,
)
from diffusion.block_ar.recurrent_logit_transition_flow_matching import _time_features


@dataclass
class GenericMultiHeadStateConditionedMixedCoordinateFMConfig(
    GenericStateConditionedMixedCoordinateFMConfig
):
    """State-conditioned mixed-coordinate AR flow with typed readout heads."""

    iv_count: int = 25
    head_hidden: int = 128


class _TypedTokenOutputHead(nn.Module):
    def __init__(self, token_dim: int, head_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


class MultiHeadMemoryConditionedTransitionVelocity(nn.Module):
    """Shared transition mixer with separate IV and factor token readouts."""

    def __init__(self, cfg: GenericMultiHeadStateConditionedMixedCoordinateFMConfig):
        super().__init__()
        if int(cfg.iv_count) < 1 or int(cfg.iv_count) > int(cfg.n_cells):
            raise ValueError("iv_count must be in [1, n_cells]")
        self.cfg = cfg
        self.iv_count = int(cfg.iv_count)
        self.factor_count = int(cfg.n_cells) - self.iv_count
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.memory_proj = nn.Linear(cfg.memory_dim, cfg.token_dim)
        self.time_proj = nn.Linear(cfg.time_dim, cfg.token_dim)
        self.value_proj = nn.Linear(2, cfg.token_dim)
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
        self.iv_head = _TypedTokenOutputHead(cfg.token_dim, int(cfg.head_hidden))
        self.factor_head = (
            _TypedTokenOutputHead(cfg.token_dim, int(cfg.head_hidden))
            if self.factor_count > 0
            else None
        )

    def forward(
        self,
        x_t: torch.Tensor,
        current_score: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_cells = x_t.shape
        cell_ids = torch.arange(n_cells, device=x_t.device)
        cell = self.cell_embed(cell_ids)[None, :, :].expand(bsz, -1, -1)
        values = torch.stack([x_t, current_score], dim=-1)
        token = self.value_proj(values) + cell
        memory = self.memory_proj(memory_state)[:, None, :]
        time = self.time_proj(_time_features(t, self.cfg.time_dim))[:, None, :]
        if self.cfg.conditioning_mode == "additive":
            hidden = self.mixer(token + memory + time)
        elif self.cfg.conditioning_mode == "prefix":
            hidden = torch.cat([memory, time, token], dim=1)
            hidden = self.mixer(hidden)[:, 2:]
        else:
            raise ValueError(f"Unknown conditioning_mode={self.cfg.conditioning_mode!r}")

        iv_out = self.iv_head(hidden[:, : self.iv_count])
        if self.factor_count <= 0 or self.factor_head is None:
            return iv_out
        factor_out = self.factor_head(hidden[:, self.iv_count :])
        return torch.cat([iv_out, factor_out], dim=1)


class GenericMultiHeadStateConditionedMixedCoordinateFlowMatching(
    GenericStateConditionedMixedCoordinateFlowMatching
):
    """One AR transition law with a shared source and typed observation heads.

    This is the AR analogue of the 652a multihead path-flow readout. It keeps the
    same history encoder, stochastic source, flow time, and token mixer for all
    channels. The only specialization is the final readout used for IV tokens
    versus non-IV factor tokens.
    """

    cfg: GenericMultiHeadStateConditionedMixedCoordinateFMConfig

    def __init__(self, cfg: GenericMultiHeadStateConditionedMixedCoordinateFMConfig):
        if int(cfg.iv_count) < 1 or int(cfg.iv_count) > int(cfg.n_cells):
            raise ValueError("iv_count must be in [1, n_cells]")
        super().__init__(cfg)
        self.velocity = MultiHeadMemoryConditionedTransitionVelocity(cfg)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GenericMultiHeadStateConditionedMixedCoordinateFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GenericMultiHeadStateConditionedMixedCoordinateFMConfig(**payload["config"])
    model = GenericMultiHeadStateConditionedMixedCoordinateFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: GenericMultiHeadStateConditionedMixedCoordinateFlowMatching,
    cfg: GenericMultiHeadStateConditionedMixedCoordinateFMConfig,
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
