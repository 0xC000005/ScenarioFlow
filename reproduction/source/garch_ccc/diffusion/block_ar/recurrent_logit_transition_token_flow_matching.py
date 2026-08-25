from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.recurrent_logit_transition_flow_matching import (
    RecurrentLogitTransitionFMConfig,
    RecurrentLogitTransitionFlowMatching,
    _time_features,
)


@dataclass
class RecurrentLogitTransitionTokenFMConfig(RecurrentLogitTransitionFMConfig):
    token_dim: int = 128
    token_layers: int = 3
    token_heads: int = 4
    token_ff: int = 256


class TokenTransitionVelocity(nn.Module):
    def __init__(self, cfg: RecurrentLogitTransitionTokenFMConfig):
        super().__init__()
        self.cfg = cfg
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.state_proj = nn.Linear(cfg.hidden_dim, cfg.token_dim)
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
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        current_logit: torch.Tensor,
        state_top: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_cells = x_t.shape
        cell_ids = torch.arange(n_cells, device=x_t.device)
        cell = self.cell_embed(cell_ids)[None, :, :].expand(bsz, -1, -1)
        values = torch.stack([x_t, current_logit], dim=-1)
        token = self.value_proj(values)
        state = self.state_proj(state_top)[:, None, :]
        time = self.time_proj(_time_features(t, self.cfg.time_dim))[:, None, :]
        hidden = token + cell + state + time
        hidden = self.mixer(hidden)
        return self.out(hidden).squeeze(-1)


class RecurrentLogitTransitionTokenFlowMatching(RecurrentLogitTransitionFlowMatching):
    """303b-v0: recurrent logit-transition flow with generic token-mixing velocity."""

    def __init__(self, cfg: RecurrentLogitTransitionTokenFMConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.velocity = TokenTransitionVelocity(cfg)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[RecurrentLogitTransitionTokenFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = RecurrentLogitTransitionTokenFMConfig(**payload["config"])
    model = RecurrentLogitTransitionTokenFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: RecurrentLogitTransitionTokenFlowMatching,
    cfg: RecurrentLogitTransitionTokenFMConfig,
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
