from __future__ import annotations

import numpy as np
import torch

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
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


def _build_library_history_stats(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    train_hist, _train_future = build_multistep_windows(train_indices, surf_tensor, history_len, future_len)
    if train_hist.ndim == 3:
        train_hist = train_hist.view(train_hist.shape[0], train_hist.shape[1], 5, 5)
    hist_mean = train_hist.mean(dim=1)
    hist_std = train_hist.std(dim=1, unbiased=False).clamp_min(1e-4)
    return hist_mean.contiguous(), hist_std.contiguous()


class HierarchicalReweightedHistoryAffineScenarioGenerator(
    HierarchicalReweightedRetrievalScenarioGenerator
):
    """286a-v0: transport retrieved futures in a local affine history coordinate."""

    def __init__(
        self,
        base_model: LearnedRetrievalRichTargetBackbone,
        cfg: HierarchicalReweightedRetrievalConfig,
        library_hist_mean_01: torch.Tensor,
        library_hist_std_01: torch.Tensor,
    ):
        super().__init__(base_model=base_model, cfg=cfg)
        self.register_buffer("library_hist_mean_01", library_hist_mean_01.contiguous())
        self.register_buffer("library_hist_std_01", library_hist_std_01.contiguous())

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
        query_std = history_01.std(dim=1, unbiased=False).clamp_min(1e-4).unsqueeze(1).unsqueeze(2)

        top_idx, top_scores, query_z = self.topk_candidates(history_norm)
        logits = self.candidate_logits(query_z, top_idx, top_scores)
        probs = torch.softmax(logits / self.cfg.sample_temperature, dim=-1)
        sampled_pos = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        batch = torch.arange(history_norm.shape[0], device=history_norm.device).unsqueeze(-1)
        chosen_idx = top_idx[batch, sampled_pos]

        raw_future = self.library_future_01[chosen_idx]
        library_mean = self.library_hist_mean_01[chosen_idx].unsqueeze(2)
        library_std = self.library_hist_std_01[chosen_idx].unsqueeze(2)
        standardized = (raw_future - library_mean) / library_std
        transported = torch.clamp(query_mean + query_std * standardized, 0.0, 1.0)
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
    library_hist_mean_01, library_hist_std_01 = _build_library_history_stats(
        data_path="data/vol_surface_with_ret.npz",
        history_len=base_cfg.history_len,
        future_len=base_cfg.future_len,
        test_start=4511,
        val_size=441,
        device=device,
    )
    model = HierarchicalReweightedHistoryAffineScenarioGenerator(
        base_model=base_model,
        cfg=cfg,
        library_hist_mean_01=library_hist_mean_01.to(device),
        library_hist_std_01=library_hist_std_01.to(device),
    )
    model.scorer.load_state_dict(payload["scorer_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload
