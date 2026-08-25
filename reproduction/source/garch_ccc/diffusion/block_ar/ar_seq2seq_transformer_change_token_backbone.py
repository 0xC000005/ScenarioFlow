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
class ARSeq2SeqTransformerTokenConfig:
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

    n_bins: int = 255
    coord_limit: float = 4.0


class ARSeq2SeqTransformerChangeTokenBackbone(nn.Module):
    """276a-v0: deterministic tokenized next-change seq2seq model."""

    def __init__(self, cfg: ARSeq2SeqTransformerTokenConfig):
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
        self.out_head = nn.Linear(cfg.d_model, cfg.n_cells * cfg.n_bins)

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

    def quantize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        clipped = coords.clamp(-self.cfg.coord_limit, self.cfg.coord_limit)
        scaled = (clipped + self.cfg.coord_limit) / (2.0 * self.cfg.coord_limit)
        idx = torch.round(scaled * (self.cfg.n_bins - 1)).long()
        return idx.clamp(0, self.cfg.n_bins - 1)

    def dequantize_ids(self, ids: torch.Tensor) -> torch.Tensor:
        scaled = ids.float() / float(self.cfg.n_bins - 1)
        return scaled * (2.0 * self.cfg.coord_limit) - self.cfg.coord_limit

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

    def decode_teacher_forced_logits(self, history_norm: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        memory = self.encode_history(history_norm)
        bos = torch.zeros(
            target_ids.shape[0], 1, self.cfg.n_cells, device=target_ids.device, dtype=torch.float32
        )
        prefix_coords = torch.cat([bos, self.dequantize_ids(target_ids[:, :-1])], dim=1)
        dec = self._decode_all(memory, prefix_coords)
        logits = self.out_head(dec).view(
            target_ids.shape[0], target_ids.shape[1], self.cfg.n_cells, self.cfg.n_bins
        )
        return logits

    def rollout(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_window = self._flatten_history(history_norm).clone()
        memory = self.encode_history(hist_window)
        curr = hist_window[:, -1]
        pred_levels = []
        pred_coords = []
        pred_ids = []
        bos = torch.zeros(
            hist_window.shape[0], 1, self.cfg.n_cells, device=hist_window.device, dtype=hist_window.dtype
        )
        for _ in range(self.cfg.future_len):
            if pred_coords:
                prefix_coords = torch.cat([bos, torch.stack(pred_coords, dim=1)], dim=1)
            else:
                prefix_coords = bos
            dec = self._decode_all(memory, prefix_coords)
            logits = self.out_head(dec[:, -1]).view(hist_window.shape[0], self.cfg.n_cells, self.cfg.n_bins)
            next_ids = logits.argmax(dim=-1)
            next_coord = self.dequantize_ids(next_ids)
            raw_change = self.inverse_transform_change(next_coord, hist_window)
            next_level = torch.clamp(curr + raw_change, -1.0, 1.0)
            pred_levels.append(next_level)
            pred_coords.append(next_coord)
            pred_ids.append(next_ids)
            hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
            curr = next_level
        return (
            torch.stack(pred_levels, dim=1),
            torch.stack(pred_coords, dim=1),
            torch.stack(pred_ids, dim=1),
        )

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future = self._flatten_history(future_norm)
        target_coords = self.target_coords(history_norm, future)
        target_ids = self.quantize_coords(target_coords)
        logits = self.decode_teacher_forced_logits(history_norm, target_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, self.cfg.n_bins),
            target_ids.reshape(-1),
        )
        pred_ids = logits.argmax(dim=-1)
        acc = (pred_ids == target_ids).float().mean()
        metrics = {
            "total": loss.detach(),
            "token_ce": loss.detach(),
            "token_acc": acc.detach(),
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
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        pred_levels, _, _ = self.rollout(history_norm)
        future_01 = denormalize_iv(pred_levels)
        if self.cfg.n_cells == 25:
            future_01 = future_01.view(future_01.shape[0], self.cfg.future_len, 5, 5)
        else:
            future_01 = future_01.view(future_01.shape[0], self.cfg.future_len, self.cfg.n_cells)
        return future_01.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ARSeq2SeqTransformerChangeTokenBackbone, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ARSeq2SeqTransformerTokenConfig(**payload["config"])
    model = ARSeq2SeqTransformerChangeTokenBackbone(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ARSeq2SeqTransformerChangeTokenBackbone,
    cfg: ARSeq2SeqTransformerTokenConfig,
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
