from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    LearnedRetrievalRichTargetBackbone,
    load_model as load_277d_model,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class StructuralEmbeddingCenterConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    embed_dim: int = 96
    hidden_dim: int = 256
    dropout: float = 0.1
    scale_floor: float = 0.03
    scale_ceiling: float = 0.80
    sample_scale_mult: float = 1.0


class StructuralEmbeddingDensity(nn.Module):
    def __init__(self, cfg: StructuralEmbeddingCenterConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.mu_head = nn.Linear(cfg.hidden_dim, cfg.embed_dim)
        self.scale_head = nn.Linear(cfg.hidden_dim, cfg.embed_dim)

    def forward(self, history_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(history_embedding)
        mu = self.mu_head(h)
        raw_scale = self.scale_head(h)
        scale = F.softplus(raw_scale) + self.cfg.scale_floor
        scale = torch.clamp(scale, max=self.cfg.scale_ceiling)
        return mu, scale

    def training_loss(
        self,
        history_embedding: torch.Tensor,
        future_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mu, scale = self(history_embedding)
        target = future_embedding
        z = (target - mu) / scale
        nll = 0.5 * (z.pow(2) + 2.0 * torch.log(scale))
        loss = nll.mean()
        cos = F.cosine_similarity(F.normalize(mu, dim=-1), target, dim=-1).mean()
        metrics = {
            "total": loss.detach(),
            "scale_mean": scale.mean().detach(),
            "z_abs_mean": z.abs().mean().detach(),
            "mu_target_cos": cos.detach(),
        }
        return loss, metrics


class StructuralEmbeddingCenterGenerator(nn.Module):
    def __init__(
        self,
        density: StructuralEmbeddingDensity,
        backbone: LearnedRetrievalRichTargetBackbone,
        backbone_checkpoint_path: str,
        library_future_embeddings: torch.Tensor,
        library_last_level_01: torch.Tensor,
        library_future_01: torch.Tensor,
    ):
        super().__init__()
        self.density = density
        self.backbone = backbone.eval()
        self.backbone_checkpoint_path = backbone_checkpoint_path
        self.register_buffer(
            "library_future_embeddings",
            F.normalize(library_future_embeddings, dim=-1).contiguous(),
        )
        self.register_buffer("library_last_level_01", library_last_level_01.contiguous())
        self.register_buffer("library_future_01", library_future_01.contiguous())
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        sample_scale_mult: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.density.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.density.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        if history_norm.ndim == 4:
            history_norm = history_norm.view(history_norm.shape[0], history_norm.shape[1], -1)
        bsz = history_norm.shape[0]
        history_01 = denormalize_iv(history_norm).view(bsz, self.density.cfg.history_len, 5, 5)
        hist_z = self.backbone.encode_history(history_norm)
        mu, scale = self.density(hist_z)
        mult = self.density.cfg.sample_scale_mult if sample_scale_mult is None else float(sample_scale_mult)

        samples = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            eps = torch.randn(bsz, k, self.density.cfg.embed_dim, device=history_norm.device)
            z = F.normalize(mu.unsqueeze(1) + mult * scale.unsqueeze(1) * eps, dim=-1)
            scores = torch.einsum("bke,ne->bkn", z, self.library_future_embeddings)
            idx = scores.argmax(dim=-1)
            retrieved_future = self.library_future_01[idx]
            retrieved_prev = torch.cat(
                [
                    self.library_last_level_01[idx].unsqueeze(2),
                    retrieved_future[:, :, :-1],
                ],
                dim=2,
            )
            deltas = retrieved_future - retrieved_prev
            query_last = history_01[:, -1].unsqueeze(1).unsqueeze(2)
            anchored = torch.clamp(query_last + torch.cumsum(deltas, dim=2), 0.0, 1.0)
            samples.append(anchored)
        return torch.cat(samples, dim=1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[StructuralEmbeddingCenterGenerator, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = StructuralEmbeddingCenterConfig(**payload["config"])
    density = StructuralEmbeddingDensity(cfg)
    density.load_state_dict(payload["density_state_dict"], strict=True)
    backbone_checkpoint_path = payload["backbone_checkpoint_path"]
    backbone, _ = load_277d_model(backbone_checkpoint_path, device)
    model = StructuralEmbeddingCenterGenerator(
        density.to(device),
        backbone,
        backbone_checkpoint_path,
        payload["library_future_embeddings"].to(device),
        payload["library_last_level_01"].to(device),
        payload["library_future_01"].to(device),
    )
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    density: StructuralEmbeddingDensity,
    cfg: StructuralEmbeddingCenterConfig,
    epoch: int,
    best_val: float,
    backbone_checkpoint_path: str,
    library_future_embeddings: torch.Tensor,
    library_last_level_01: torch.Tensor,
    library_future_01: torch.Tensor,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "density_state_dict": density.state_dict(),
            "backbone_checkpoint_path": backbone_checkpoint_path,
            "library_future_embeddings": library_future_embeddings.cpu(),
            "library_last_level_01": library_last_level_01.cpu(),
            "library_future_01": library_future_01.cpu(),
        },
        path,
    )
