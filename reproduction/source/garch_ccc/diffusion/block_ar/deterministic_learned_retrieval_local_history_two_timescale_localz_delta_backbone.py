from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    LearnedRetrievalRichTargetConfig,
    PathEncoder,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


class DeterministicLearnedRetrievalLocalHistoryTwoTimescaleLocalZDeltaBackbone(torch.nn.Module):
    """287d-v0: 287c with query/history features in the same local-history z-coordinate."""

    def __init__(
        self,
        cfg: LearnedRetrievalRichTargetConfig,
        library_future_fast_embeddings: torch.Tensor | None = None,
        library_future_slow_embeddings: torch.Tensor | None = None,
        library_future_delta_z_01: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.history_fast_encoder = PathEncoder(
            input_dim=cfg.n_cells,
            embed_dim=cfg.embed_dim,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            dropout=cfg.dropout,
        )
        self.history_slow_encoder = PathEncoder(
            input_dim=cfg.n_cells,
            embed_dim=cfg.embed_dim,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            dropout=cfg.dropout,
        )
        self.future_fast_encoder = PathEncoder(
            input_dim=cfg.n_cells,
            embed_dim=cfg.embed_dim,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            dropout=cfg.dropout,
        )
        self.future_slow_encoder = PathEncoder(
            input_dim=cfg.n_cells,
            embed_dim=cfg.embed_dim,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            dropout=cfg.dropout,
        )
        if library_future_fast_embeddings is not None:
            if library_future_slow_embeddings is None or library_future_delta_z_01 is None:
                raise ValueError("Need fast/slow library embeddings and delta support when embeddings are provided.")
            self.register_buffer(
                "library_future_fast_embeddings",
                F.normalize(library_future_fast_embeddings, dim=-1).contiguous(),
            )
            self.register_buffer(
                "library_future_slow_embeddings",
                F.normalize(library_future_slow_embeddings, dim=-1).contiguous(),
            )
            if library_future_delta_z_01.ndim == 3:
                library_future_delta_z_01 = library_future_delta_z_01.view(
                    library_future_delta_z_01.shape[0], library_future_delta_z_01.shape[1], 5, 5
                )
            self.register_buffer("library_future_delta_z_01", library_future_delta_z_01.contiguous())
        else:
            self.library_future_fast_embeddings = None
            self.library_future_slow_embeddings = None
            self.library_future_delta_z_01 = None

    def encode_history(
        self,
        history_fast: torch.Tensor,
        history_slow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.history_fast_encoder(history_fast), self.history_slow_encoder(history_slow)

    def encode_future_representation(
        self,
        future_fast: torch.Tensor,
        future_slow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if future_fast.ndim == 4:
            future_fast = future_fast.view(future_fast.shape[0], future_fast.shape[1], -1)
        if future_slow.ndim == 4:
            future_slow = future_slow.view(future_slow.shape[0], future_slow.shape[1], -1)
        return self.future_fast_encoder(future_fast), self.future_slow_encoder(future_slow)

    def contrastive_logits(
        self,
        history_fast: torch.Tensor,
        history_slow: torch.Tensor,
        future_fast: torch.Tensor,
        future_slow: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        hist_fast_z, hist_slow_z = self.encode_history(history_fast, history_slow)
        fut_fast_z, fut_slow_z = self.encode_future_representation(future_fast, future_slow)
        logits = hist_fast_z @ fut_fast_z.transpose(0, 1)
        logits = logits + hist_slow_z @ fut_slow_z.transpose(0, 1)
        logits = logits / self.cfg.temperature
        return logits, (hist_fast_z, hist_slow_z), (fut_fast_z, fut_slow_z)

    @torch.no_grad()
    def nearest_indices(
        self,
        history_fast: torch.Tensor,
        history_slow: torch.Tensor,
    ) -> torch.Tensor:
        if self.library_future_fast_embeddings is None or self.library_future_slow_embeddings is None:
            raise RuntimeError("Retrieval library is not loaded.")
        query_fast_z, query_slow_z = self.encode_history(history_fast, history_slow)
        scores = query_fast_z @ self.library_future_fast_embeddings.transpose(0, 1)
        scores = scores + query_slow_z @ self.library_future_slow_embeddings.transpose(0, 1)
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
        history_fast, history_slow = build_local_history_fast_slow(history_01)

        query_mean = history_01.mean(dim=1)
        query_std = history_01.std(dim=1, unbiased=False).clamp_min(1e-4)
        query_last_z = (history_01[:, -1] - query_mean) / query_std

        idx = self.nearest_indices(history_fast, history_slow)
        delta_z = self.library_future_delta_z_01[idx]
        future_z = query_last_z.unsqueeze(1) + torch.cumsum(delta_z, dim=1)
        future = torch.clamp(
            query_mean.unsqueeze(1) + query_std.unsqueeze(1) * future_z,
            0.0,
            1.0,
        )
        return future.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).contiguous()


def build_local_history_fast_slow(history_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if history_01.ndim == 4:
        history_flat = history_01.view(history_01.shape[0], history_01.shape[1], -1)
    else:
        history_flat = history_01
    hist_mean = history_flat.mean(dim=1, keepdim=True)
    hist_std = history_flat.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
    history_z = (history_flat - hist_mean) / hist_std
    prev = torch.cat([history_z[:, :1], history_z[:, :-1]], dim=1)
    fast = history_z - prev
    slow = history_z - history_z[:, :1]
    return fast, slow


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[DeterministicLearnedRetrievalLocalHistoryTwoTimescaleLocalZDeltaBackbone, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LearnedRetrievalRichTargetConfig(**payload["config"])
    model = DeterministicLearnedRetrievalLocalHistoryTwoTimescaleLocalZDeltaBackbone(
        cfg,
        library_future_fast_embeddings=payload["library_future_fast_embeddings"].to(device),
        library_future_slow_embeddings=payload["library_future_slow_embeddings"].to(device),
        library_future_delta_z_01=payload["library_future_delta_z_01"].to(device),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    cfg: LearnedRetrievalRichTargetConfig,
    model: DeterministicLearnedRetrievalLocalHistoryTwoTimescaleLocalZDeltaBackbone,
    library_future_fast_embeddings: torch.Tensor,
    library_future_slow_embeddings: torch.Tensor,
    library_future_delta_z_01: torch.Tensor,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "library_future_fast_embeddings": library_future_fast_embeddings.cpu(),
        "library_future_slow_embeddings": library_future_slow_embeddings.cpu(),
        "library_future_delta_z_01": library_future_delta_z_01.cpu(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
