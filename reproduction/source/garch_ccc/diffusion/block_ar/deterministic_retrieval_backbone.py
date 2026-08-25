from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


@dataclass
class DeterministicRetrievalConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    distance: str = "l2"


class DeterministicRetrievalBackbone(nn.Module):
    """277a-v0: nearest-neighbor deterministic retrieval of an observed future path."""

    def __init__(
        self,
        cfg: DeterministicRetrievalConfig,
        library_history_norm: torch.Tensor,
        library_future_01: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        if library_history_norm.ndim != 3:
            raise ValueError("Expected history [N,T,D]")
        if library_future_01.ndim == 3:
            library_future_01 = library_future_01.view(
                library_future_01.shape[0], library_future_01.shape[1], 5, 5
            )
        elif library_future_01.ndim != 4:
            raise ValueError("Expected future [N,T,D] or [N,T,5,5]")
        self.register_buffer("library_history_norm", library_history_norm.contiguous())
        self.register_buffer("library_future_01", library_future_01.contiguous())

    def nearest_indices(self, history_norm: torch.Tensor) -> torch.Tensor:
        query = history_norm.view(history_norm.shape[0], -1)
        library = self.library_history_norm.view(self.library_history_norm.shape[0], -1)
        dists = torch.cdist(query, library)
        return dists.argmin(dim=-1)

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
        idx = self.nearest_indices(history_norm)
        future = self.library_future_01[idx]
        return future.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[DeterministicRetrievalBackbone, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = DeterministicRetrievalConfig(**payload["config"])
    model = DeterministicRetrievalBackbone(
        cfg,
        payload["library_history_norm"].to(device),
        payload["library_future_01"].to(device),
    )
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    cfg: DeterministicRetrievalConfig,
    library_history_norm: torch.Tensor,
    library_future_01: torch.Tensor,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "library_history_norm": library_history_norm.cpu(),
            "library_future_01": library_future_01.cpu(),
        },
        path,
    )
