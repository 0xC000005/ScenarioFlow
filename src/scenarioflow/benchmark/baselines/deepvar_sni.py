"""DeepVAR under the ScenarioFlow history-local normalization interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...state import normalize_increment_windows, unnormalize_increment_windows
from .deep import build_model


class DeepVARSNIInterfaceBaseline:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        scale_half_life: float | None,
        scale_floor: float,
        center_mode: str,
        device: str | torch.device = "cpu",
        future_len: int = 30,
        chunk_size: int = 512,
    ) -> None:
        self.model = model.to(device).eval()
        self.scale_half_life = scale_half_life
        self.scale_floor = float(scale_floor)
        self.center_mode = str(center_mode)
        self.device = torch.device(device)
        self.future_len = int(future_len)
        self.chunk_size = int(chunk_size)

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 64, **_: Any
    ) -> np.ndarray:
        history = np.asarray(history_changes, dtype=np.float32)
        batch, length, dimension = history.shape
        normalized, _unused, center, scale = normalize_increment_windows(
            history,
            history,
            half_life=self.scale_half_life,
            scale_floor=self.scale_floor,
            center_mode=self.center_mode,
        )
        repeated = np.repeat(normalized[:, None], int(n_samples), axis=1).reshape(
            -1, length, dimension
        )
        outputs = []
        with torch.no_grad():
            for start in range(0, repeated.shape[0], self.chunk_size):
                chunk = torch.from_numpy(repeated[start : start + self.chunk_size]).to(
                    self.device
                )
                outputs.append(
                    self.model.sample_trajectory(chunk, n_future=self.future_len)
                    .cpu()
                    .numpy()
                )
        normalized_future = np.concatenate(outputs).reshape(
            batch, int(n_samples), self.future_len, dimension
        )
        return unnormalize_increment_windows(normalized_future, center, scale).astype(
            np.float32
        )

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        chunk_size: int = 512,
    ) -> "DeepVARSNIInterfaceBaseline":
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = dict(checkpoint["config"])
        normalization = dict(checkpoint["normalization"])
        model = build_model(
            "deepvar",
            input_dim=int(config["input_dim"]),
            future_len=int(config.get("future_len", 30)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_layers=int(config.get("num_layers", 2)),
            rank=int(config.get("rank", 5)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(
            model,
            scale_half_life=normalization.get("scale_half_life"),
            scale_floor=float(normalization.get("scale_floor", 1e-4)),
            center_mode=str(normalization.get("center_mode", "zero")),
            device=device,
            future_len=int(config.get("future_len", 30)),
            chunk_size=chunk_size,
        )
