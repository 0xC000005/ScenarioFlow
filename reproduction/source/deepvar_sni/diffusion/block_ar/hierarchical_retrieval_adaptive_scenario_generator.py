from __future__ import annotations

from dataclasses import dataclass

import math
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
class HierarchicalAdaptiveRetrievalScenarioConfig:
    top_k: int = 16
    base_temperature: float = 0.2


class HierarchicalAdaptiveRetrievalScenarioGenerator(torch.nn.Module):
    """278b-v0: adaptive retrieval-temperature scenario sampling from top-k futures."""

    def __init__(
        self,
        base_model: LearnedRetrievalRichTargetBackbone,
        top_k: int,
        base_temperature: float,
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg = HierarchicalAdaptiveRetrievalScenarioConfig(top_k=top_k, base_temperature=base_temperature)

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
    def topk_candidates(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_z = self.base_model.encode_history(history_norm)
        scores = query_z @ self.library_future_embeddings.transpose(0, 1)
        k = min(self.cfg.top_k, self.library_future_embeddings.shape[0])
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        return top_idx, top_scores

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
        top_idx, top_scores = self.topk_candidates(history_norm)

        base_probs = torch.softmax(top_scores / self.cfg.base_temperature, dim=-1)
        entropy = -(base_probs * (base_probs.clamp_min(1e-8)).log()).sum(dim=-1)
        entropy_norm = entropy / math.log(top_scores.shape[-1])
        adaptive_temp = self.cfg.base_temperature * (0.5 + entropy_norm).clamp_min(0.5)
        probs = torch.softmax(top_scores / adaptive_temp.unsqueeze(-1), dim=-1)

        sampled_pos = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        batch = torch.arange(history_norm.shape[0], device=history_norm.device).unsqueeze(-1)
        chosen_idx = top_idx[batch, sampled_pos]
        retrieved_future = self.library_future_01[chosen_idx]
        retrieved_prev = torch.cat(
            [self.library_last_level_01[chosen_idx].unsqueeze(2), retrieved_future[:, :, :-1]], dim=2
        )
        deltas = retrieved_future - retrieved_prev
        query_last = history_01[:, -1].unsqueeze(1)
        anchored_future = torch.clamp(query_last.unsqueeze(2) + torch.cumsum(deltas, dim=2), 0.0, 1.0)
        return anchored_future.contiguous()


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
    model = HierarchicalAdaptiveRetrievalScenarioGenerator(
        base_model,
        top_k=payload.get("top_k", 16),
        base_temperature=payload.get("base_temperature", 0.2),
    )
    model.to(device).eval()
    return model, payload
