from __future__ import annotations

from dataclasses import dataclass

import torch

from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    LearnedRetrievalRichTargetBackbone,
    LearnedRetrievalRichTargetConfig,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class LearnedRetrievalMedoidConfig:
    top_k: int = 8


class LearnedRetrievalMedoidBackbone(torch.nn.Module):
    """277e-v0: use top-k learned retrieval, then choose the medoid future."""

    def __init__(
        self,
        base_model: LearnedRetrievalRichTargetBackbone,
        top_k: int,
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg = LearnedRetrievalMedoidConfig(top_k=top_k)

    @property
    def library_future_embeddings(self) -> torch.Tensor:
        return self.base_model.library_future_embeddings

    @property
    def library_last_level_01(self) -> torch.Tensor:
        return self.base_model.library_last_level_01

    @property
    def library_future_01(self) -> torch.Tensor:
        return self.base_model.library_future_01

    @torch.no_grad()
    def nearest_indices(self, history_norm: torch.Tensor) -> torch.Tensor:
        query_z = self.base_model.encode_history(history_norm)
        scores = query_z @ self.library_future_embeddings.transpose(0, 1)
        k = min(self.cfg.top_k, self.library_future_embeddings.shape[0])
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        candidate_emb = self.library_future_embeddings[top_idx]  # [B,K,D]
        affinity = candidate_emb @ candidate_emb.transpose(1, 2)  # [B,K,K]
        medoid_pos = affinity.mean(dim=-1).argmax(dim=-1)
        batch = torch.arange(top_idx.shape[0], device=top_idx.device)
        return top_idx[batch, medoid_pos]

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


def load_model(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_cfg = LearnedRetrievalRichTargetConfig(**payload["base_config"])
    base_model = LearnedRetrievalRichTargetBackbone(
        base_cfg,
        library_future_embeddings=payload["library_future_embeddings"].to(device),
        library_last_level_01=payload["library_last_level_01"].to(device),
        library_future_01=payload["library_future_01"].to(device),
    )
    base_model.load_state_dict(payload["model_state_dict"], strict=False)
    base_model.to(device).eval()
    model = LearnedRetrievalMedoidBackbone(base_model, top_k=payload.get("top_k", 8))
    model.to(device).eval()
    return model, payload
