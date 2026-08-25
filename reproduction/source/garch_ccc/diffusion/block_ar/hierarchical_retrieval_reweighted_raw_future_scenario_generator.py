from __future__ import annotations

import torch

from diffusion.block_ar.hierarchical_retrieval_reweighted_scenario_generator import (
    HierarchicalReweightedRetrievalScenarioGenerator,
    HierarchicalReweightedRetrievalConfig,
)
from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    LearnedRetrievalRichTargetBackbone,
    LearnedRetrievalRichTargetConfig,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


class HierarchicalReweightedRawFutureScenarioGenerator(
    HierarchicalReweightedRetrievalScenarioGenerator
):
    """284b-v0: sample raw retrieved future paths directly instead of anchored deltas."""

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
        top_idx, top_scores, query_z = self.topk_candidates(history_norm)
        logits = self.candidate_logits(query_z, top_idx, top_scores)
        probs = torch.softmax(logits / self.cfg.sample_temperature, dim=-1)
        sampled_pos = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        batch = torch.arange(history_norm.shape[0], device=history_norm.device).unsqueeze(-1)
        chosen_idx = top_idx[batch, sampled_pos]
        return self.library_future_01[chosen_idx].contiguous()


def load_model(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_cfg = LearnedRetrievalRichTargetConfig(**payload["base_config"])
    base_model = LearnedRetrievalRichTargetBackbone(
        base_cfg,
        library_future_embeddings=payload["library_future_embeddings"].to(device),
        library_last_level_01=payload["library_last_level_01"].to(device),
        library_future_01=payload["library_future_01"].to(device),
    )
    base_model.load_state_dict(payload["base_model_state_dict"], strict=False)
    base_model.to(device).eval()
    cfg = HierarchicalReweightedRetrievalConfig(**payload["reweight_config"])
    model = HierarchicalReweightedRawFutureScenarioGenerator(base_model, cfg)
    model.scorer.load_state_dict(payload["scorer_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload
