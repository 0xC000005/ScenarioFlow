from __future__ import annotations

import torch

from diffusion.block_ar.hierarchical_retrieval_reweighted_history_affine_scenario_generator import (
    _build_library_history_stats,
)
from diffusion.block_ar.hierarchical_retrieval_reweighted_scenario_generator import (
    HierarchicalReweightedRetrievalConfig,
    HierarchicalReweightedRetrievalScenarioGenerator,
)
from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    LearnedRetrievalRichTargetBackbone,
    LearnedRetrievalRichTargetConfig,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


class HierarchicalReweightedHistoryMeanScenarioGenerator(
    HierarchicalReweightedRetrievalScenarioGenerator
):
    """286b-v0: translate retrieved futures by recent-history mean only."""

    def __init__(
        self,
        base_model: LearnedRetrievalRichTargetBackbone,
        cfg: HierarchicalReweightedRetrievalConfig,
        library_hist_mean_01: torch.Tensor,
    ):
        super().__init__(base_model=base_model, cfg=cfg)
        self.register_buffer("library_hist_mean_01", library_hist_mean_01.contiguous())

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
        query_mean = history_01.mean(dim=1).unsqueeze(1).unsqueeze(2)

        top_idx, top_scores, query_z = self.topk_candidates(history_norm)
        logits = self.candidate_logits(query_z, top_idx, top_scores)
        probs = torch.softmax(logits / self.cfg.sample_temperature, dim=-1)
        sampled_pos = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        batch = torch.arange(history_norm.shape[0], device=history_norm.device).unsqueeze(-1)
        chosen_idx = top_idx[batch, sampled_pos]

        raw_future = self.library_future_01[chosen_idx]
        library_mean = self.library_hist_mean_01[chosen_idx].unsqueeze(2)
        transported = torch.clamp(raw_future + (query_mean - library_mean), 0.0, 1.0)
        return transported.contiguous()


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
    library_hist_mean_01, _library_hist_std_01 = _build_library_history_stats(
        data_path="data/vol_surface_with_ret.npz",
        history_len=base_cfg.history_len,
        future_len=base_cfg.future_len,
        test_start=4511,
        val_size=441,
        device=device,
    )
    model = HierarchicalReweightedHistoryMeanScenarioGenerator(
        base_model=base_model,
        cfg=cfg,
        library_hist_mean_01=library_hist_mean_01.to(device),
    )
    model.scorer.load_state_dict(payload["scorer_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload
