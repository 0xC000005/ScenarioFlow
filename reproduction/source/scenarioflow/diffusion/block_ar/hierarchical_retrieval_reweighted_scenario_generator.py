from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
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
class HierarchicalReweightedRetrievalConfig:
    top_k: int = 16
    hidden_dim: int = 128
    label_temperature: float = 0.25
    sample_temperature: float = 1.0


class CandidateScorer(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        feat_dim = 2 * embed_dim + embed_dim + 1
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_z: torch.Tensor,
        candidate_z: torch.Tensor,
        sim: torch.Tensor,
    ) -> torch.Tensor:
        query_expand = query_z.unsqueeze(1).expand_as(candidate_z)
        feat = torch.cat(
            [
                query_expand,
                candidate_z,
                query_expand * candidate_z,
                sim.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(feat).squeeze(-1)


class HierarchicalReweightedRetrievalScenarioGenerator(nn.Module):
    """278c-v0: learned query-conditioned reweighting over retrieved future candidates."""

    def __init__(
        self,
        base_model: LearnedRetrievalRichTargetBackbone,
        cfg: HierarchicalReweightedRetrievalConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg
        embed_dim = base_model.cfg.embed_dim
        self.scorer = CandidateScorer(embed_dim=embed_dim, hidden_dim=cfg.hidden_dim)

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
    def topk_candidates(
        self,
        history_norm: torch.Tensor,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_z = self.base_model.encode_history(history_norm)
        scores = query_z @ self.library_future_embeddings.transpose(0, 1)
        k = min(top_k or self.cfg.top_k, self.library_future_embeddings.shape[0])
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        return top_idx, top_scores, query_z

    def candidate_logits(
        self,
        query_z: torch.Tensor,
        top_idx: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        candidate_z = self.library_future_embeddings[top_idx]
        return self.scorer(query_z, candidate_z, top_scores)

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
        top_idx, top_scores, query_z = self.topk_candidates(history_norm)
        logits = self.candidate_logits(query_z, top_idx, top_scores)
        probs = torch.softmax(logits / self.cfg.sample_temperature, dim=-1)
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
    base_model.load_state_dict(payload["base_model_state_dict"], strict=False)
    base_model.to(device).eval()
    cfg = HierarchicalReweightedRetrievalConfig(**payload["reweight_config"])
    model = HierarchicalReweightedRetrievalScenarioGenerator(base_model, cfg)
    model.scorer.load_state_dict(payload["scorer_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: HierarchicalReweightedRetrievalScenarioGenerator,
    extra: dict | None = None,
) -> None:
    payload = {
        "base_config": asdict(model.base_model.cfg),
        "base_model_state_dict": model.base_model.state_dict(),
        "library_future_embeddings": model.library_future_embeddings.cpu(),
        "library_last_level_01": model.library_last_level_01.cpu(),
        "library_future_01": model.library_future_01.cpu(),
        "reweight_config": asdict(model.cfg),
        "scorer_state_dict": model.scorer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
