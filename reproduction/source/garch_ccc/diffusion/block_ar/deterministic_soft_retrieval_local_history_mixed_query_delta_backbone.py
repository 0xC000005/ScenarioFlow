from __future__ import annotations

import torch

from diffusion.block_ar.deterministic_learned_retrieval_local_history_two_timescale_delta_backbone import (
    build_history_slow,
)
from diffusion.block_ar.deterministic_learned_retrieval_local_history_two_timescale_localz_delta_backbone import (
    build_local_history_fast_slow,
)
from diffusion.block_ar.deterministic_learned_retrieval_local_history_two_timescale_mixed_query_delta_backbone import (
    DeterministicLearnedRetrievalLocalHistoryTwoTimescaleMixedQueryDeltaBackbone,
)
from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    LearnedRetrievalRichTargetConfig,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


class DeterministicSoftRetrievalLocalHistoryMixedQueryDeltaBackbone(
    DeterministicLearnedRetrievalLocalHistoryTwoTimescaleMixedQueryDeltaBackbone
):
    """288a-v0: deterministic top-k soft replay on top of the 287e mixed query key."""

    def __init__(self, cfg: LearnedRetrievalRichTargetConfig, top_k: int = 8, **kwargs):
        super().__init__(cfg, **kwargs)
        self.top_k = top_k

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
        history_fast, _ = build_local_history_fast_slow(history_01)
        history_slow = build_history_slow(history_norm)

        query_fast_z, query_slow_z = self.encode_history(history_fast, history_slow)
        scores = query_fast_z @ self.library_future_fast_embeddings.transpose(0, 1)
        scores = scores + query_slow_z @ self.library_future_slow_embeddings.transpose(0, 1)
        top_k = min(self.top_k, scores.shape[1])
        top_scores, top_idx = scores.topk(k=top_k, dim=1)
        weights = torch.softmax(top_scores, dim=1)

        delta_bank = self.library_future_delta_z_01[top_idx]
        weighted_delta_z = (weights[:, :, None, None, None] * delta_bank).sum(dim=1)

        query_mean = history_01.mean(dim=1)
        query_std = history_01.std(dim=1, unbiased=False).clamp_min(1e-4)
        query_last_z = (history_01[:, -1] - query_mean) / query_std
        future_z = query_last_z.unsqueeze(1) + torch.cumsum(weighted_delta_z, dim=1)
        future = torch.clamp(
            query_mean.unsqueeze(1) + query_std.unsqueeze(1) * future_z,
            0.0,
            1.0,
        )
        return future.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[DeterministicSoftRetrievalLocalHistoryMixedQueryDeltaBackbone, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LearnedRetrievalRichTargetConfig(**payload["config"])
    model = DeterministicSoftRetrievalLocalHistoryMixedQueryDeltaBackbone(
        cfg,
        library_future_fast_embeddings=payload["library_future_fast_embeddings"].to(device),
        library_future_slow_embeddings=payload["library_future_slow_embeddings"].to(device),
        library_future_delta_z_01=payload["library_future_delta_z_01"].to(device),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload
