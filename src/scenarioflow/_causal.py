"""Private checkpoint-compatible causal token primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _time_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        torch.linspace(
            0.0,
            -torch.log(torch.tensor(10000.0, device=device)),
            half,
            device=device,
        )
    )
    angles = t[:, None] * freqs[None, :] * 2.0 * torch.pi
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


@dataclass
class RecurrentLogitTransitionFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    hidden_dim: int = 128
    gru_layers: int = 2
    gru_dropout: float = 0.1
    model_hidden: int = 256
    model_layers: int = 4
    model_dropout: float = 0.1
    time_dim: int = 32
    logit_eps: float = 1e-4
    flow_steps: int = 32
    sample_temperature: float = 1.0


@dataclass
class RecurrentLogitTransitionTokenFMConfig(RecurrentLogitTransitionFMConfig):
    token_dim: int = 128
    token_layers: int = 3
    token_heads: int = 4
    token_ff: int = 256


@dataclass
class CausalFutureMemoryTransitionFMConfig(RecurrentLogitTransitionTokenFMConfig):
    memory_dim: int = 128
    memory_layers: int = 3
    memory_heads: int = 4
    memory_ff: int = 256
    conditioning_mode: str = "additive"
    standardize_logits: bool = False
    logit_std_floor: float = 1e-3


class MemoryConditionedTokenTransitionVelocity(nn.Module):
    def __init__(self, cfg: CausalFutureMemoryTransitionFMConfig):
        super().__init__()
        self.cfg = cfg
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
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def hidden_tokens(
        self,
        x_t: torch.Tensor,
        current_logit: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_cells = x_t.shape
        cell_ids = torch.arange(n_cells, device=x_t.device)
        cell = self.cell_embed(cell_ids)[None, :, :].expand(batch_size, -1, -1)
        values = torch.stack([x_t, current_logit], dim=-1)
        token = self.value_proj(values) + cell
        memory = self.memory_proj(memory_state)[:, None, :]
        time = self.time_proj(_time_features(t, self.cfg.time_dim))[:, None, :]
        if self.cfg.conditioning_mode == "additive":
            hidden = self.mixer(token + memory + time)
        elif self.cfg.conditioning_mode == "prefix":
            hidden = self.mixer(torch.cat([memory, time, token], dim=1))[:, 2:]
        else:
            raise ValueError(
                f"Unknown conditioning_mode={self.cfg.conditioning_mode!r}"
            )
        return hidden

    def forward(
        self,
        x_t: torch.Tensor,
        current_logit: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.out(
            self.hidden_tokens(x_t, current_logit, memory_state, t)
        ).squeeze(-1)


class NarrativeConditioningAdapter(nn.Module):
    """Zero-initialized optional additive context used by compatible checkpoints."""

    def __init__(self, narrative_dim: int, memory_dim: int, hidden: int = 512) -> None:
        super().__init__()
        if hidden < memory_dim:
            raise AssertionError(
                f"hidden ({hidden}) must be >= memory_dim ({memory_dim})"
            )
        self.narrative_dim = int(narrative_dim)
        self.memory_dim = int(memory_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(narrative_dim),
            nn.Linear(narrative_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, memory_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, narrative_emb: torch.Tensor, present: torch.Tensor | None = None
    ) -> torch.Tensor:
        context = self.net(narrative_emb)
        if present is None:
            return context
        return context * present.to(context.dtype).reshape(-1, 1)
