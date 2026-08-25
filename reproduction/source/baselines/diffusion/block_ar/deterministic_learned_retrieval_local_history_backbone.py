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


class DeterministicLearnedRetrievalLocalHistoryBackbone(torch.nn.Module):
    """287a-v0: learned retrieval in a local-history future coordinate."""

    def __init__(
        self,
        cfg: LearnedRetrievalRichTargetConfig,
        library_future_embeddings: torch.Tensor | None = None,
        library_future_z_01: torch.Tensor | None = None,
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
            input_dim=2 * cfg.n_cells,
            embed_dim=cfg.embed_dim,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            dropout=cfg.dropout,
        )
        if library_future_embeddings is not None:
            self.register_buffer("library_future_embeddings", F.normalize(library_future_embeddings, dim=-1).contiguous())
            if library_future_z_01 is None:
                raise ValueError("Need library_future_z_01 when embeddings are provided.")
            if library_future_z_01.ndim == 3:
                library_future_z_01 = library_future_z_01.view(
                    library_future_z_01.shape[0], library_future_z_01.shape[1], 5, 5
                )
            self.register_buffer("library_future_z_01", library_future_z_01.contiguous())
        else:
            self.library_future_embeddings = None
            self.library_future_z_01 = None

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history_norm)

    def encode_future_representation(self, future_repr: torch.Tensor) -> torch.Tensor:
        if future_repr.ndim == 4:
            future_repr = future_repr.view(future_repr.shape[0], future_repr.shape[1], -1)
        return self.future_encoder(future_repr)

    def contrastive_logits(
        self,
        history_norm: torch.Tensor,
        future_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_z = self.encode_history(history_norm)
        fut_z = self.encode_future_representation(future_repr)
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
        query_mean = history_01.mean(dim=1).unsqueeze(1)
        query_std = history_01.std(dim=1, unbiased=False).clamp_min(1e-4).unsqueeze(1)

        idx = self.nearest_indices(history_norm)
        retrieved_future_z = self.library_future_z_01[idx]
        future = torch.clamp(query_mean + query_std * retrieved_future_z, 0.0, 1.0)
        return future.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[DeterministicLearnedRetrievalLocalHistoryBackbone, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LearnedRetrievalRichTargetConfig(**payload["config"])
    model = DeterministicLearnedRetrievalLocalHistoryBackbone(
        cfg,
        library_future_embeddings=payload["library_future_embeddings"].to(device),
        library_future_z_01=payload["library_future_z_01"].to(device),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    cfg: LearnedRetrievalRichTargetConfig,
    model: DeterministicLearnedRetrievalLocalHistoryBackbone,
    library_future_embeddings: torch.Tensor,
    library_future_z_01: torch.Tensor,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "library_future_embeddings": library_future_embeddings.cpu(),
        "library_future_z_01": library_future_z_01.cpu(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
