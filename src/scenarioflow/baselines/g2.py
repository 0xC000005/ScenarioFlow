"""G2: DeepVAR wrapped in the SNI observation interface.

This is the apples-to-apples width control demanded by the ICAIF review (R3).
The ONLY difference from the current-panel DeepVAR baseline is the normalizer:

  - current baseline: global per-dim z-score of encoded increments (constants
    pooled over the whole training set), reconstructed with the same constants;
  - this baseline: the SNI per-window EWMA-RMS local-scale normalization
    (center=0, ewma_rms scale, floor), reconstructed IDENTICALLY to the SNI model
    via ``unnormalize_increment_windows`` + ``reconstruct_state_from_increments``.

The DeepVAR model class, the state map ``g_d`` (bounded-logit / log / diff, applied
by ``build_blocks``), and the reconstruction path are untouched and were already
shared with the SNI generator. Only the observation-space normalizer is swapped.

The wrapper exposes the same ``sample_joint(history_increment, n_samples) -> raw
encoded increments`` interface as every other current-panel baseline, so the
existing ``evaluate_one`` metric battery runs against it byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .deep import build_model
from ..state import (
    normalize_increment_windows,
    unnormalize_increment_windows,
)


class DeepVARSNIInterfaceBaseline:
    """DeepVAR trained on normalized innovations; reconstructs like the SNI model."""

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
        seed: int = 0,
    ) -> None:
        self.model = model.to(device).eval()
        self.scale_half_life = scale_half_life
        self.scale_floor = float(scale_floor)
        self.center_mode = str(center_mode)
        self.device = torch.device(device)
        self.future_len = int(future_len)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        self._rng_state = generator.get_state()

    def eval(self) -> "DeepVARSNIInterfaceBaseline":
        self.model.eval()
        return self

    def sample_joint(
        self,
        history_changes: np.ndarray,
        n_samples: int = 64,
        **_: Any,
    ) -> np.ndarray:
        """Return raw encoded increments [B, n_samples, future_len, D].

        The per-window center/scale are computed history-only, exactly as the SNI
        generator does; the DeepVAR rollout produces normalized innovations that
        are un-normalized back to raw encoded increments before returning, so the
        downstream ``reconstruct_state_from_increments`` call is identical to the
        SNI reconstruction path.
        """
        history = np.asarray(history_changes, dtype=np.float32)
        bsz, hist_len, dim = history.shape
        # History-only local statistics (pass history as both args; future is unused here).
        history_norm, _future_norm, center, scale = normalize_increment_windows(
            history,
            history,
            half_life=self.scale_half_life,
            scale_floor=self.scale_floor,
            center_mode=self.center_mode,
        )
        repeated = np.repeat(
            history_norm[:, None, :, :], int(n_samples), axis=1
        ).reshape(
            bsz * int(n_samples),
            hist_len,
            dim,
        )
        outputs: list[np.ndarray] = []
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self._rng_state, self.device)
            else:
                torch.set_rng_state(self._rng_state)
            # Preserve one continuous per-trajectory stream across evaluator
            # calls, independent of the external window batch partition.
            for start in range(repeated.shape[0]):
                chunk = torch.from_numpy(repeated[start : start + 1]).to(self.device)
                norm_future = self.model.sample_trajectory(
                    chunk, n_future=self.future_len
                )
                outputs.append(norm_future.detach().cpu().numpy())
            self._rng_state = (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else torch.get_rng_state()
            )
        norm_future = np.concatenate(outputs, axis=0).reshape(
            bsz, int(n_samples), self.future_len, dim
        )
        raw_future = unnormalize_increment_windows(norm_future, center, scale)
        return raw_future.astype(np.float32)

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        chunk_size: int = 512,
        seed: int = 0,
    ) -> "DeepVARSNIInterfaceBaseline":
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = dict(ckpt["config"])
        norm = dict(ckpt["normalization"])
        model = build_model(
            "deepvar",
            input_dim=int(config["input_dim"]),
            future_len=int(config.get("future_len", 30)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_layers=int(config.get("num_layers", 2)),
            rank=int(config.get("rank", 5)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(
            model,
            scale_half_life=norm.get("scale_half_life"),
            scale_floor=float(norm.get("scale_floor", 1e-4)),
            center_mode=str(norm.get("center_mode", "zero")),
            device=device,
            future_len=int(config.get("future_len", 30)),
            chunk_size=int(chunk_size),
            seed=int(seed),
        )
