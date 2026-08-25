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
class ProbabilisticJointTokenFutureControlModelConfig:
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

    codebook_size: int = 256
    n_knots: int = 6
    control_loss_weight: float = 1.0
    control_radius: int = 1
    label_smoothing: float = 0.0
    sample_temperature: float = 1.0


class ProbabilisticJointTokenFutureControlModel(nn.Module):
    """295a-v0: direct daily token law conditioned on a future control-state sequence."""

    def __init__(
        self,
        cfg: ProbabilisticJointTokenFutureControlModelConfig,
        residual_codebook: torch.Tensor,
        knot_matrix: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        if residual_codebook.shape != (cfg.codebook_size, cfg.n_cells):
            raise ValueError(
                f"Expected residual codebook {(cfg.codebook_size, cfg.n_cells)}, got {tuple(residual_codebook.shape)}"
            )
        if knot_matrix.shape != (cfg.future_len, cfg.n_knots):
            raise ValueError(
                f"Expected knot matrix {(cfg.future_len, cfg.n_knots)}, got {tuple(knot_matrix.shape)}"
            )

        self.history_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.token_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.control_proj = nn.Linear(cfg.n_cells, cfg.d_model)
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
        self.out_head = nn.Linear(cfg.d_model, cfg.codebook_size)
        self.control_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.n_knots * cfg.n_cells),
        )

        self.register_buffer("residual_codebook", residual_codebook.clone())
        self.register_buffer("knot_matrix", knot_matrix.clone())

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

    def _knot_indices(self, device: torch.device) -> torch.Tensor:
        return self.knot_matrix.argmax(dim=0).to(device=device, dtype=torch.long)

    def teacher_control_states(self, target_coords: torch.Tensor) -> torch.Tensor:
        knot_idx = self._knot_indices(target_coords.device)
        radius = max(0, int(self.cfg.control_radius))
        states = []
        for idx in knot_idx.tolist():
            left = max(0, idx - radius)
            right = min(self.cfg.future_len, idx + radius + 1)
            states.append(target_coords[:, left:right].mean(dim=1))
        return torch.stack(states, dim=1)

    def interpolate_control_states(self, knot_controls: torch.Tensor) -> torch.Tensor:
        return torch.einsum("tk,bkc->btc", self.knot_matrix.to(knot_controls.device), knot_controls)

    def assign_tokens(self, target_coords: torch.Tensor) -> torch.Tensor:
        flat = target_coords.reshape(-1, self.cfg.n_cells)
        dists = torch.cdist(flat, self.residual_codebook)
        ids = dists.argmin(dim=-1)
        return ids.view(target_coords.shape[0], target_coords.shape[1])

    def token_vectors(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.residual_codebook[token_ids]

    def predict_control_states(self, memory: torch.Tensor) -> torch.Tensor:
        controls = self.control_head(memory.mean(dim=1))
        return controls.view(memory.shape[0], self.cfg.n_knots, self.cfg.n_cells)

    def _decode_all(
        self,
        memory: torch.Tensor,
        prefix_vecs: torch.Tensor,
        control_levels: torch.Tensor,
    ) -> torch.Tensor:
        prefix_len = prefix_vecs.shape[1]
        pos = torch.arange(prefix_len, device=prefix_vecs.device).unsqueeze(0)
        x = self.token_proj(prefix_vecs) + self.future_pos(pos)
        x = x + self.control_proj(control_levels[:, :prefix_len])
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(prefix_len, device=prefix_vecs.device)
        return self.decoder(x, memory, tgt_mask=tgt_mask)

    def decode_teacher_forced_logits(
        self,
        history_norm: torch.Tensor,
        target_ids: torch.Tensor,
        control_levels: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode_history(history_norm)
        bos = torch.zeros(
            target_ids.shape[0], 1, self.cfg.n_cells, device=target_ids.device, dtype=torch.float32
        )
        prefix = torch.cat([bos, self.token_vectors(target_ids[:, :-1])], dim=1)
        dec = self._decode_all(memory, prefix, control_levels)
        return self.out_head(dec)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory = self.encode_history(history_norm)
        pred_knot_controls = self.predict_control_states(memory)
        target_coords = self.target_coords(history_norm, future_norm)
        teacher_knot_controls = self.teacher_control_states(target_coords)
        control_loss = F.mse_loss(pred_knot_controls, teacher_knot_controls)

        target_ids = self.assign_tokens(target_coords)
        control_levels = self.interpolate_control_states(pred_knot_controls)
        logits = self.decode_teacher_forced_logits(history_norm, target_ids, control_levels)
        token_nll = F.cross_entropy(
            logits.reshape(-1, self.cfg.codebook_size),
            target_ids.reshape(-1),
            label_smoothing=self.cfg.label_smoothing,
        )
        loss = token_nll + self.cfg.control_loss_weight * control_loss

        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)
        token_acc = (pred_ids == target_ids).float().mean()
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        control_mae = (pred_knot_controls - teacher_knot_controls).abs().mean()
        metrics = {
            "total": loss.detach(),
            "token_nll": token_nll.detach(),
            "control_loss": control_loss.detach(),
            "control_mae": control_mae.detach(),
            "token_acc": token_acc.detach(),
            "token_entropy": entropy.detach(),
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
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten_history(history_norm)
        bsz = history_norm.shape[0]
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        temp = max(temp, 1e-4)

        outputs = []
        remaining = int(n_samples)
        while remaining > 0:
            sample_chunk = min(chunk_size, remaining)
            hist_window = (
                history_norm.unsqueeze(1)
                .expand(-1, sample_chunk, -1, -1)
                .reshape(bsz * sample_chunk, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            memory = self.encode_history(hist_window)
            pred_knot_controls = self.predict_control_states(memory)
            control_levels = self.interpolate_control_states(pred_knot_controls)

            curr = hist_window[:, -1]
            pred_vecs: list[torch.Tensor] = []
            pred_levels: list[torch.Tensor] = []
            bos = torch.zeros(
                hist_window.shape[0], 1, self.cfg.n_cells, device=hist_window.device, dtype=hist_window.dtype
            )
            for step in range(self.cfg.future_len):
                if pred_vecs:
                    prefix = torch.cat([bos, torch.stack(pred_vecs, dim=1)], dim=1)
                else:
                    prefix = bos
                dec = self._decode_all(memory, prefix, control_levels)
                logits = self.out_head(dec[:, -1]) / temp
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                coord_vec = self.token_vectors(next_ids)
                raw_change = self.inverse_transform_change(coord_vec, hist_window)
                next_level = torch.clamp(curr + raw_change, -1.0, 1.0)
                pred_levels.append(next_level)
                pred_vecs.append(coord_vec)
                hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
                curr = next_level
            future_norm = torch.stack(pred_levels, dim=1)
            future_01 = denormalize_iv(future_norm)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, sample_chunk, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, sample_chunk, self.cfg.future_len, self.cfg.n_cells)
            outputs.append(future_01)
            remaining -= sample_chunk
        return torch.cat(outputs, dim=1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ProbabilisticJointTokenFutureControlModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ProbabilisticJointTokenFutureControlModelConfig(**payload["config"])
    model = ProbabilisticJointTokenFutureControlModel(
        cfg,
        payload["residual_codebook"].to(device),
        payload["knot_matrix"].to(device),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ProbabilisticJointTokenFutureControlModel,
    cfg: ProbabilisticJointTokenFutureControlModelConfig,
    epoch: int,
    best_val: float,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "residual_codebook": model.residual_codebook.detach().cpu(),
            "knot_matrix": model.knot_matrix.detach().cpu(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
