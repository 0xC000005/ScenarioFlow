from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class LearnedRetrievalChangeAnchorConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    embed_dim: int = 96
    rnn_hidden: int = 128
    rnn_layers: int = 2
    dropout: float = 0.1
    temperature: float = 0.07
    metric: str = "cosine"


class PathEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        rnn_hidden: int,
        rnn_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, rnn_hidden)
        self.rnn = nn.GRU(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(rnn_hidden),
            nn.Linear(rnn_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.in_proj(x))
        _, last = self.rnn(h)
        z = self.out_proj(last[-1])
        return F.normalize(z, dim=-1)


class LearnedRetrievalChangeAnchorBackbone(nn.Module):
    """277c-v0: learned cross-view retrieval for anchored future changes."""

    def __init__(
        self,
        cfg: LearnedRetrievalChangeAnchorConfig,
        library_future_embeddings: torch.Tensor | None = None,
        library_last_level_01: torch.Tensor | None = None,
        library_future_01: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.history_encoder = PathEncoder(
            input_dim=cfg.n_cells,
            embed_dim=cfg.embed_dim,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            dropout=cfg.dropout,
        )
        self.future_encoder = PathEncoder(
            input_dim=cfg.n_cells,
            embed_dim=cfg.embed_dim,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            dropout=cfg.dropout,
        )
        if library_future_embeddings is not None:
            self.register_buffer("library_future_embeddings", F.normalize(library_future_embeddings, dim=-1).contiguous())
            if library_last_level_01 is None or library_future_01 is None:
                raise ValueError("Need library levels and future paths when embeddings are provided.")
            if library_last_level_01.ndim == 2:
                library_last_level_01 = library_last_level_01.view(library_last_level_01.shape[0], 5, 5)
            if library_future_01.ndim == 3:
                library_future_01 = library_future_01.view(library_future_01.shape[0], library_future_01.shape[1], 5, 5)
            self.register_buffer("library_last_level_01", library_last_level_01.contiguous())
            self.register_buffer("library_future_01", library_future_01.contiguous())
        else:
            self.library_future_embeddings = None
            self.library_last_level_01 = None
            self.library_future_01 = None

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history_norm)

    def encode_future_changes(self, future_changes_01: torch.Tensor) -> torch.Tensor:
        if future_changes_01.ndim == 4:
            future_changes_01 = future_changes_01.view(future_changes_01.shape[0], future_changes_01.shape[1], -1)
        return self.future_encoder(future_changes_01)

    def contrastive_logits(
        self,
        history_norm: torch.Tensor,
        future_changes_01: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_z = self.encode_history(history_norm)
        fut_z = self.encode_future_changes(future_changes_01)
        logits = hist_z @ fut_z.transpose(0, 1)
        logits = logits / self.cfg.temperature
        return logits, hist_z, fut_z

    @torch.no_grad()
    def nearest_indices(self, history_norm: torch.Tensor) -> torch.Tensor:
        if self.library_future_embeddings is None:
            raise RuntimeError("Retrieval library is not loaded.")
        query_z = self.encode_history(history_norm)
        scores = query_z @ self.library_future_embeddings.transpose(0, 1)
        return scores.argmax(dim=-1)

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        if history_norm.ndim == 4:
            history_norm = history_norm.view(history_norm.shape[0], history_norm.shape[1], -1)
        history_01 = denormalize_iv(history_norm).view(history_norm.shape[0], history_norm.shape[1], 5, 5)

        idx = self.nearest_indices(history_norm)
        retrieved_future = self.library_future_01[idx]
        retrieved_prev = torch.cat(
            [self.library_last_level_01[idx].unsqueeze(1), retrieved_future[:, :-1]], dim=1
        )
        deltas = retrieved_future - retrieved_prev
        query_last = history_01[:, -1]
        anchored_future = torch.clamp(query_last.unsqueeze(1) + torch.cumsum(deltas, dim=1), 0.0, 1.0)
        return anchored_future.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[LearnedRetrievalChangeAnchorBackbone, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LearnedRetrievalChangeAnchorConfig(**payload["config"])
    model = LearnedRetrievalChangeAnchorBackbone(
        cfg,
        library_future_embeddings=payload["library_future_embeddings"].to(device),
        library_last_level_01=payload["library_last_level_01"].to(device),
        library_future_01=payload["library_future_01"].to(device),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    cfg: LearnedRetrievalChangeAnchorConfig,
    model: LearnedRetrievalChangeAnchorBackbone,
    library_future_embeddings: torch.Tensor,
    library_last_level_01: torch.Tensor,
    library_future_01: torch.Tensor,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "library_future_embeddings": library_future_embeddings.cpu(),
        "library_last_level_01": library_last_level_01.cpu(),
        "library_future_01": library_future_01.cpu(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
