from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    LearnedRetrievalRichTargetBackbone,
    LearnedRetrievalRichTargetConfig,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class HierarchicalScaledResidualRetrievalConfig:
    top_k: int = 16
    sample_temperature: float = 0.2
    residual_center_alpha: float = 0.5
    scale_hidden_dim: int = 128
    min_scale: float = 0.25


class HierarchicalScaledResidualRetrievalScenarioGenerator(torch.nn.Module):
    """280a-v0: fixed Stage A center path plus query-conditioned residual scaling."""

    def __init__(
        self,
        base_model: LearnedRetrievalRichTargetBackbone,
        top_k: int,
        sample_temperature: float,
        residual_center_alpha: float,
        scale_hidden_dim: int = 128,
        min_scale: float = 0.25,
    ):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        embed_dim = int(self.base_model.library_future_embeddings.shape[1])
        self.scale_head = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, scale_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(scale_hidden_dim, 1),
        )
        self.cfg = HierarchicalScaledResidualRetrievalConfig(
            top_k=top_k,
            sample_temperature=sample_temperature,
            residual_center_alpha=residual_center_alpha,
            scale_hidden_dim=scale_hidden_dim,
            min_scale=min_scale,
        )

    @property
    def library_future_embeddings(self) -> torch.Tensor:
        return self.base_model.library_future_embeddings

    @property
    def library_last_level_01(self) -> torch.Tensor:
        return self.base_model.library_last_level_01

    @property
    def library_future_01(self) -> torch.Tensor:
        return self.base_model.library_future_01

    def encode_query(self, history_norm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.base_model.encode_history(history_norm)

    def predict_scale_from_query(self, query_z: torch.Tensor) -> torch.Tensor:
        raw = self.scale_head(query_z).squeeze(-1)
        return F.softplus(raw) + self.cfg.min_scale

    def build_residual_bank(
        self,
        history_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if history_norm.ndim == 4:
            history_norm = history_norm.view(history_norm.shape[0], history_norm.shape[1], -1)
        query_z = self.encode_query(history_norm)
        scores = query_z @ self.library_future_embeddings.transpose(0, 1)
        k = min(self.cfg.top_k, self.library_future_embeddings.shape[0])
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)

        history_01 = denormalize_iv(history_norm).view(history_norm.shape[0], history_norm.shape[1], 5, 5)
        query_last = history_01[:, -1]

        center_idx = self.base_model.nearest_indices(history_norm)
        center_future = self.library_future_01[center_idx]
        center_prev = torch.cat(
            [self.library_last_level_01[center_idx].unsqueeze(1), center_future[:, :-1]], dim=1
        )
        center_deltas = center_future - center_prev
        anchored_center = torch.clamp(query_last.unsqueeze(1) + torch.cumsum(center_deltas, dim=1), 0.0, 1.0)

        retrieved_future = self.library_future_01[top_idx]
        retrieved_prev = torch.cat(
            [self.library_last_level_01[top_idx].unsqueeze(2), retrieved_future[:, :, :-1]], dim=2
        )
        deltas = retrieved_future - retrieved_prev
        anchored_candidates = torch.clamp(
            query_last.unsqueeze(1).unsqueeze(2) + torch.cumsum(deltas, dim=2), 0.0, 1.0
        )

        residuals = anchored_candidates - anchored_center.unsqueeze(1)
        residual_mean = residuals.mean(dim=1, keepdim=True)
        adjusted_residuals = residuals - self.cfg.residual_center_alpha * residual_mean
        return anchored_center, adjusted_residuals, top_scores, query_z

    def training_targets(
        self,
        history_norm: torch.Tensor,
        future_01: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchored_center, adjusted_residuals, _top_scores, query_z = self.build_residual_bank(history_norm)
        actual_residual = future_01 - anchored_center
        bank_rms = adjusted_residuals.pow(2).mean(dim=(1, 2, 3, 4)).sqrt().clamp_min(1e-4)
        target_rms = actual_residual.pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-4)
        target_scale = (target_rms / bank_rms).clamp(0.25, 3.0)
        return query_z, target_scale

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
        anchored_center, adjusted_residuals, top_scores, query_z = self.build_residual_bank(history_norm)
        probs = torch.softmax(top_scores / self.cfg.sample_temperature, dim=-1)
        sampled_pos = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        batch = torch.arange(history_norm.shape[0], device=history_norm.device).unsqueeze(-1)
        sampled_residuals = adjusted_residuals[batch, sampled_pos]
        scale = self.predict_scale_from_query(query_z).view(-1, 1, 1, 1, 1)
        scenarios = torch.clamp(anchored_center.unsqueeze(1) + scale * sampled_residuals, 0.0, 1.0)
        return scenarios.contiguous()


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
    model = HierarchicalScaledResidualRetrievalScenarioGenerator(
        base_model=base_model,
        top_k=payload.get("top_k", 16),
        sample_temperature=payload.get("sample_temperature", 0.2),
        residual_center_alpha=payload.get("residual_center_alpha", 0.5),
        scale_hidden_dim=payload.get("scale_hidden_dim", 128),
        min_scale=payload.get("min_scale", 0.25),
    )
    model.scale_head.load_state_dict(payload["scale_head_state_dict"])
    model.to(device).eval()
    return model, payload
