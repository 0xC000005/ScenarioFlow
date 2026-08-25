from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods.extend([nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)])
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


@dataclass
class ARDeterministicBackboneConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    hidden_dim: int = 192
    history_layers: int = 2
    dropout: float = 0.1

    change_hidden: int = 128
    change_layers: int = 2
    change_dropout: float = 0.1

    change_coord: str = "asinh_local_scale"
    change_scale_eps: float = 1e-3

    level_weight: float = 1.0
    change_weight: float = 1.0


class ARDeterministicChangeBackbone(nn.Module):
    """275a-v0: deterministic autoregressive next-change backbone for the 8/11 stage."""

    def __init__(self, cfg: ARDeterministicBackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.n_cells, cfg.hidden_dim)
        self.history_rnn = nn.GRU(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.history_layers,
            dropout=cfg.dropout if cfg.history_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder_cell = nn.GRUCell(cfg.n_cells, cfg.hidden_dim)
        self.change_head = _mlp(
            in_dim=cfg.hidden_dim,
            out_dim=cfg.n_cells,
            hidden=cfg.change_hidden,
            layers=cfg.change_layers,
            dropout=cfg.change_dropout,
        )

    @staticmethod
    def _flatten_history(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def compute_change_scale(self, history_norm: torch.Tensor) -> torch.Tensor:
        hist = self._flatten_history(history_norm)
        hist_change = hist[:, 1:] - hist[:, :-1]
        scale = hist_change.pow(2).mean(dim=1, keepdim=True).sqrt()
        return scale.clamp_min(self.cfg.change_scale_eps)

    def transform_change(self, raw_change: torch.Tensor, history_norm: torch.Tensor) -> torch.Tensor:
        if self.cfg.change_coord == "raw":
            return raw_change
        if self.cfg.change_coord == "asinh_local_scale":
            scale = self.compute_change_scale(history_norm)
            if raw_change.ndim == 2:
                scale = scale.squeeze(1)
            return torch.asinh(raw_change / scale)
        raise ValueError(f"Unknown change_coord={self.cfg.change_coord}")

    def inverse_transform_change(self, model_change: torch.Tensor, history_norm: torch.Tensor) -> torch.Tensor:
        if self.cfg.change_coord == "raw":
            return model_change
        if self.cfg.change_coord == "asinh_local_scale":
            scale = self.compute_change_scale(history_norm)
            if model_change.ndim == 2:
                scale = scale.squeeze(1)
            return torch.sinh(model_change) * scale
        raise ValueError(f"Unknown change_coord={self.cfg.change_coord}")

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        hist = self._flatten_history(history_norm)
        x = self.input_proj(hist)
        _, h_n = self.history_rnn(x)
        return h_n[-1]

    def rollout(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hist = self._flatten_history(history_norm)
        hidden = self.encode_history(hist)
        hist_window = hist.clone()
        curr = hist_window[:, -1]
        pred_levels = []
        pred_coords = []
        for _ in range(self.cfg.future_len):
            hidden = self.decoder_cell(curr, hidden)
            pred_coord = self.change_head(hidden)
            raw_change = self.inverse_transform_change(pred_coord, hist_window)
            next_level = torch.clamp(curr + raw_change, -1.0, 1.0)
            pred_levels.append(next_level)
            pred_coords.append(pred_coord)
            hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
            curr = next_level
        return torch.stack(pred_levels, dim=1), torch.stack(pred_coords, dim=1)

    def target_coords(self, history_norm: torch.Tensor, future_norm: torch.Tensor) -> torch.Tensor:
        hist_window = self._flatten_history(history_norm).clone()
        future = self._flatten_history(future_norm)
        curr = hist_window[:, -1]
        coords = []
        for step in range(self.cfg.future_len):
            next_level = future[:, step]
            raw_change = next_level - curr
            coord = self.transform_change(raw_change, hist_window)
            coords.append(coord)
            hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
            curr = next_level
        return torch.stack(coords, dim=1)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future = self._flatten_history(future_norm)
        pred_levels, pred_coords = self.rollout(history_norm)
        target_coords = self.target_coords(history_norm, future)
        level_loss = F.smooth_l1_loss(pred_levels, future)
        change_loss = F.smooth_l1_loss(pred_coords, target_coords)
        total = self.cfg.level_weight * level_loss + self.cfg.change_weight * change_loss
        metrics = {
            "total": total.detach(),
            "level_loss": level_loss.detach(),
            "change_loss": change_loss.detach(),
            "pred_level_std": pred_levels.std(unbiased=False).detach(),
        }
        return total, metrics

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
        pred_levels, _ = self.rollout(history_norm)
        future_01 = denormalize_iv(pred_levels)
        if self.cfg.n_cells == 25:
            future_01 = future_01.view(future_01.shape[0], self.cfg.future_len, 5, 5)
        else:
            future_01 = future_01.view(future_01.shape[0], self.cfg.future_len, self.cfg.n_cells)
        return future_01.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ARDeterministicChangeBackbone, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ARDeterministicBackboneConfig(**payload["config"])
    model = ARDeterministicChangeBackbone(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ARDeterministicChangeBackbone,
    cfg: ARDeterministicBackboneConfig,
    epoch: int,
    best_val: float,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
