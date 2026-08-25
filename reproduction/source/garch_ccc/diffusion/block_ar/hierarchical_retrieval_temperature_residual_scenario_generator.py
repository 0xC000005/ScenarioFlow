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
class HierarchicalTemperatureResidualRetrievalConfig:
    top_k: int = 16
    tau_min: float = 0.05
    tau_max: float = 1.5
    hidden_dim: int = 128


class HierarchicalTemperatureResidualRetrievalScenarioGenerator(torch.nn.Module):
    """281a-v0: fixed center path plus learned residual temperature over retrieval scores."""

    def __init__(
        self,
        base_model: LearnedRetrievalRichTargetBackbone,
        top_k: int,
        tau_min: float = 0.05,
        tau_max: float = 1.5,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        embed_dim = int(self.base_model.library_future_embeddings.shape[1])
        self.temperature_head = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.cfg = HierarchicalTemperatureResidualRetrievalConfig(
            top_k=top_k,
            tau_min=tau_min,
            tau_max=tau_max,
            hidden_dim=hidden_dim,
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

    def predict_temperature(self, query_z: torch.Tensor) -> torch.Tensor:
        raw = self.temperature_head(query_z).squeeze(-1)
        sig = torch.sigmoid(raw)
        return self.cfg.tau_min + (self.cfg.tau_max - self.cfg.tau_min) * sig

    def build_weighted_residual_bank(
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

        tau = self.predict_temperature(query_z)
        weights = torch.softmax(top_scores / tau.unsqueeze(-1), dim=-1)
        weighted_mean = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * residuals).sum(dim=1, keepdim=True)
        centered_residuals = residuals - weighted_mean
        return anchored_center, centered_residuals, weights, query_z

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_01: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        anchored_center, centered_residuals, weights, query_z = self.build_weighted_residual_bank(history_norm)
        actual_residual = future_01 - anchored_center
        dists = (centered_residuals - actual_residual.unsqueeze(1)).pow(2).mean(dim=(2, 3, 4))
        loss = (weights * dists).sum(dim=1).mean()
        tau = self.predict_temperature(query_z)
        metrics = {
            "tau_mean": tau.mean(),
            "dist_mean": dists.mean(),
        }
        return loss, metrics

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
        anchored_center, centered_residuals, weights, _query_z = self.build_weighted_residual_bank(history_norm)
        sampled_pos = torch.multinomial(weights, num_samples=n_samples, replacement=True)
        batch = torch.arange(history_norm.shape[0], device=history_norm.device).unsqueeze(-1)
        sampled_residuals = centered_residuals[batch, sampled_pos]
        scenarios = torch.clamp(anchored_center.unsqueeze(1) + sampled_residuals, 0.0, 1.0)
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
    model = HierarchicalTemperatureResidualRetrievalScenarioGenerator(
        base_model=base_model,
        top_k=payload.get("top_k", 16),
        tau_min=payload.get("tau_min", 0.05),
        tau_max=payload.get("tau_max", 1.5),
        hidden_dim=payload.get("hidden_dim", 128),
    )
    model.temperature_head.load_state_dict(payload["temperature_head_state_dict"])
    model.to(device).eval()
    return model, payload
