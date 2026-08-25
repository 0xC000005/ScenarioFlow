from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.recurrent_logit_transition_flow_matching import (
    RecurrentLogitTransitionFlowMatching,
    _time_features,
)
from diffusion.block_ar.recurrent_logit_transition_token_flow_matching import (
    RecurrentLogitTransitionTokenFMConfig,
)


@dataclass
class RecurrentLogitTransitionPrefixTokenFMConfig(RecurrentLogitTransitionTokenFMConfig):
    n_prefix_tokens: int = 2


class PrefixTokenTransitionVelocity(nn.Module):
    def __init__(self, cfg: RecurrentLogitTransitionPrefixTokenFMConfig):
        super().__init__()
        self.cfg = cfg
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.prefix_embed = nn.Parameter(torch.zeros(cfg.n_prefix_tokens, cfg.token_dim))
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
        values = torch.stack([x_t, current_logit], dim=-1)
        cell_ids = torch.arange(n_cells, device=x_t.device)
        cell_tokens = self.value_proj(values) + self.cell_embed(cell_ids)[None, :, :]

        state_token = self.state_proj(state_top) + self.prefix_embed[0][None, :]
        time_token = self.time_proj(_time_features(t, self.cfg.time_dim)) + self.prefix_embed[1][None, :]
        prefix = torch.stack([state_token, time_token], dim=1)

        hidden = torch.cat([prefix, cell_tokens], dim=1)
        hidden = self.mixer(hidden)
        cell_hidden = hidden[:, prefix.shape[1] :, :]
        return self.out(cell_hidden).squeeze(-1)


class RecurrentLogitTransitionPrefixTokenFlowMatching(RecurrentLogitTransitionFlowMatching):
    """303c-v0: recurrent token flow with explicit state/time conditioning tokens."""

    def __init__(self, cfg: RecurrentLogitTransitionPrefixTokenFMConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.velocity = PrefixTokenTransitionVelocity(cfg)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[RecurrentLogitTransitionPrefixTokenFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = RecurrentLogitTransitionPrefixTokenFMConfig(**payload["config"])
    model = RecurrentLogitTransitionPrefixTokenFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: RecurrentLogitTransitionPrefixTokenFlowMatching,
    cfg: RecurrentLogitTransitionPrefixTokenFMConfig,
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
