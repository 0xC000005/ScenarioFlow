from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class DeterministicLatentWorldModelConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    latent_dim: int = 128
    history_hidden: int = 128
    history_layers: int = 2
    dropout: float = 0.1
    decoder_hidden: int = 192

    level_weight: float = 1.0
    change_weight: float = 1.0


class DeterministicLatentWorldModel(nn.Module):
    """289b-v0: deterministic latent world model with learned panel bottleneck."""

    def __init__(self, cfg: DeterministicLatentWorldModelConfig):
        super().__init__()
        self.cfg = cfg

        self.hist_in = nn.Linear(2 * cfg.n_cells, cfg.history_hidden)
        self.history_rnn = nn.GRU(
            input_size=cfg.history_hidden,
            hidden_size=cfg.latent_dim,
            num_layers=cfg.history_layers,
            dropout=cfg.dropout if cfg.history_layers > 1 else 0.0,
            batch_first=True,
        )
        self.transition_in = nn.Sequential(
            nn.Linear(2 * cfg.n_cells, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.latent_dim),
        )
        self.transition = nn.GRUCell(cfg.latent_dim, cfg.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.n_cells, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.n_cells),
        )

    @staticmethod
    def _flatten_levels(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def _history_tokens(self, levels: torch.Tensor) -> torch.Tensor:
        prev = torch.cat([levels[:, :1], levels[:, :-1]], dim=1)
        changes = levels - prev
        return torch.cat([levels, changes], dim=-1)

    def encode_history(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist = self._flatten_levels(history_norm)
        tokens = self._history_tokens(hist)
        x = torch.tanh(self.hist_in(tokens))
        _, h_last = self.history_rnn(x)
        curr_level = hist[:, -1]
        prev_change = hist[:, -1] - hist[:, -2]
        return h_last[-1], curr_level, prev_change

    def _transition_step(
        self,
        latent: torch.Tensor,
        curr_level: torch.Tensor,
        prev_change: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trans_in = self.transition_in(torch.cat([curr_level, prev_change], dim=-1))
        latent = self.transition(trans_in, latent)
        next_change = self.decoder(torch.cat([latent, curr_level], dim=-1))
        return latent, next_change

    def teacher_forced_changes(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> torch.Tensor:
        latent, curr_level, prev_change = self.encode_history(history_norm)
        future = self._flatten_levels(future_norm)
        preds: list[torch.Tensor] = []
        for step in range(self.cfg.future_len):
            latent, next_change = self._transition_step(latent, curr_level, prev_change)
            preds.append(next_change)
            next_level = future[:, step]
            prev_change = next_level - curr_level
            curr_level = next_level
        return torch.stack(preds, dim=1)

    def rollout(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, curr_level, prev_change = self.encode_history(history_norm)
        pred_levels: list[torch.Tensor] = []
        pred_changes: list[torch.Tensor] = []
        for _ in range(self.cfg.future_len):
            latent, next_change = self._transition_step(latent, curr_level, prev_change)
            next_level = torch.clamp(curr_level + next_change, -1.0, 1.0)
            pred_levels.append(next_level)
            pred_changes.append(next_change)
            prev_change = next_change
            curr_level = next_level
        return torch.stack(pred_levels, dim=1), torch.stack(pred_changes, dim=1)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hist = self._flatten_levels(history_norm)
        future = self._flatten_levels(future_norm)
        teacher_prev = torch.cat([hist[:, -1:], future[:, :-1]], dim=1)
        target_changes = future - teacher_prev
        teacher_pred_changes = self.teacher_forced_changes(hist, future)
        rollout_levels, rollout_changes = self.rollout(hist)
        level_loss = F.smooth_l1_loss(rollout_levels, future)
        change_loss = F.smooth_l1_loss(teacher_pred_changes, target_changes)
        rollout_change_loss = F.smooth_l1_loss(rollout_changes, target_changes)
        total = (
            self.cfg.level_weight * level_loss
            + self.cfg.change_weight * change_loss
            + 0.5 * self.cfg.change_weight * rollout_change_loss
        )
        metrics = {
            "total": total.detach(),
            "level_loss": level_loss.detach(),
            "change_loss": change_loss.detach(),
            "rollout_change_loss": rollout_change_loss.detach(),
            "pred_level_std": rollout_levels.std(unbiased=False).detach(),
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
) -> tuple[DeterministicLatentWorldModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = DeterministicLatentWorldModelConfig(**payload["config"])
    model = DeterministicLatentWorldModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: DeterministicLatentWorldModel,
    cfg: DeterministicLatentWorldModelConfig,
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
