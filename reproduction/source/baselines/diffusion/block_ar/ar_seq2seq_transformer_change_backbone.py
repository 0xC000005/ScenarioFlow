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
class ARSeq2SeqTransformerConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    d_model: int = 192
    nhead: int = 6
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1

    change_coord: str = "asinh_local_scale"
    change_scale_eps: float = 1e-3

    level_weight: float = 1.0
    change_weight: float = 1.0


class ARSeq2SeqTransformerChangeBackbone(nn.Module):
    """275c-v0: deterministic observation-space seq2seq backbone with full history attention."""

    def __init__(self, cfg: ARSeq2SeqTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.history_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.coord_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.history_pos = nn.Embedding(cfg.history_len, cfg.d_model)
        self.future_pos = nn.Embedding(cfg.future_len + 1, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_decoder_layers)
        self.out_head = nn.Linear(cfg.d_model, cfg.n_cells)

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
        pos = torch.arange(self.cfg.history_len, device=hist.device).unsqueeze(0)
        x = self.history_proj(hist) + self.history_pos(pos)
        return self.encoder(x)

    def _decode_all(self, memory: torch.Tensor, prefix_coords: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(prefix_coords.shape[1], device=prefix_coords.device).unsqueeze(0)
        x = self.coord_proj(prefix_coords) + self.future_pos(pos)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            prefix_coords.shape[1], device=prefix_coords.device
        )
        return self.decoder(x, memory, tgt_mask=tgt_mask)

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

    def decode_teacher_forced(self, history_norm: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        memory = self.encode_history(history_norm)
        bos = torch.zeros(
            target_coords.shape[0], 1, self.cfg.n_cells, device=target_coords.device, dtype=target_coords.dtype
        )
        prefix = torch.cat([bos, target_coords[:, :-1]], dim=1)
        dec = self._decode_all(memory, prefix)
        return self.out_head(dec)

    def rollout(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hist_window = self._flatten_history(history_norm).clone()
        memory = self.encode_history(hist_window)
        curr = hist_window[:, -1]
        pred_levels = []
        pred_coords: list[torch.Tensor] = []
        bos = torch.zeros(
            hist_window.shape[0], 1, self.cfg.n_cells, device=hist_window.device, dtype=hist_window.dtype
        )
        for _ in range(self.cfg.future_len):
            if pred_coords:
                prefix = torch.cat([bos, torch.stack(pred_coords, dim=1)], dim=1)
            else:
                prefix = bos
            dec = self._decode_all(memory, prefix)
            pred_coord = self.out_head(dec[:, -1])
            raw_change = self.inverse_transform_change(pred_coord, hist_window)
            next_level = torch.clamp(curr + raw_change, -1.0, 1.0)
            pred_levels.append(next_level)
            pred_coords.append(pred_coord)
            hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
            curr = next_level
        return torch.stack(pred_levels, dim=1), torch.stack(pred_coords, dim=1)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future = self._flatten_history(future_norm)
        target_coords = self.target_coords(history_norm, future)
        teacher_coords = self.decode_teacher_forced(history_norm, target_coords)
        rollout_levels, _ = self.rollout(history_norm)
        level_loss = F.smooth_l1_loss(rollout_levels, future)
        change_loss = F.smooth_l1_loss(teacher_coords, target_coords)
        total = self.cfg.level_weight * level_loss + self.cfg.change_weight * change_loss
        metrics = {
            "total": total.detach(),
            "level_loss": level_loss.detach(),
            "change_loss": change_loss.detach(),
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
) -> tuple[ARSeq2SeqTransformerChangeBackbone, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ARSeq2SeqTransformerConfig(**payload["config"])
    model = ARSeq2SeqTransformerChangeBackbone(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ARSeq2SeqTransformerChangeBackbone,
    cfg: ARSeq2SeqTransformerConfig,
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
