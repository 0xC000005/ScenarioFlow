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
class DeterministicCausalTransformerWorldModelConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    d_model: int = 192
    nhead: int = 6
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    level_weight: float = 1.0
    change_weight: float = 1.0
    max_seq_len: int = 96


class DeterministicCausalTransformerWorldModel(nn.Module):
    """289a-v0: deterministic causal world model in normalized-change space."""

    def __init__(self, cfg: DeterministicCausalTransformerWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_proj = nn.Linear(2 * cfg.n_cells, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.out_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.n_cells),
        )

    @staticmethod
    def _flatten_levels(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def _build_tokens(self, levels: torch.Tensor) -> torch.Tensor:
        prev = torch.cat([levels[:, :1], levels[:, :-1]], dim=1)
        changes = levels - prev
        tokens = torch.cat([levels, changes], dim=-1)
        return tokens

    def _encode(self, levels: torch.Tensor) -> torch.Tensor:
        tokens = self._build_tokens(levels)
        seq_len = tokens.shape[1]
        if seq_len > self.cfg.max_seq_len:
            tokens = tokens[:, -self.cfg.max_seq_len :]
            seq_len = tokens.shape[1]
        pos = torch.arange(seq_len, device=levels.device).unsqueeze(0)
        x = self.token_proj(tokens) + self.pos_emb(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=levels.device)
        return self.backbone(x, mask=causal_mask)

    def predict_next_change(self, levels: torch.Tensor) -> torch.Tensor:
        h = self._encode(levels)
        return self.out_head(h[:, -1])

    def teacher_forced_changes(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> torch.Tensor:
        hist = self._flatten_levels(history_norm)
        future = self._flatten_levels(future_norm)
        seq = hist
        preds: list[torch.Tensor] = []
        for step in range(self.cfg.future_len):
            preds.append(self.predict_next_change(seq))
            seq = torch.cat([seq, future[:, step : step + 1]], dim=1)
        return torch.stack(preds, dim=1)

    def rollout(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self._flatten_levels(history_norm)
        preds: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        for _ in range(self.cfg.future_len):
            next_change = self.predict_next_change(seq)
            next_level = torch.clamp(seq[:, -1] + next_change, -1.0, 1.0)
            preds.append(next_level)
            deltas.append(next_change)
            seq = torch.cat([seq, next_level.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1), torch.stack(deltas, dim=1)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hist = self._flatten_levels(history_norm)
        future = self._flatten_levels(future_norm)
        teacher_pred_changes = self.teacher_forced_changes(hist, future)
        teacher_prev = torch.cat([hist[:, -1:], future[:, :-1]], dim=1)
        target_changes = future - teacher_prev
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
) -> tuple[DeterministicCausalTransformerWorldModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = DeterministicCausalTransformerWorldModelConfig(**payload["config"])
    model = DeterministicCausalTransformerWorldModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: DeterministicCausalTransformerWorldModel,
    cfg: DeterministicCausalTransformerWorldModelConfig,
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
