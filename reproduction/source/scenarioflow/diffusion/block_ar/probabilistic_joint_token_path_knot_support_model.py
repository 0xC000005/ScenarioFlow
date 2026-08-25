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
class ProbabilisticJointTokenPathKnotSupportModelConfig:
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
    coarse_codebook_size: int = 128
    n_knots: int = 5
    coarse_loss_weight: float = 1.0
    label_smoothing: float = 0.0
    sample_temperature: float = 1.0


class ProbabilisticJointTokenPathKnotSupportModel(nn.Module):
    """293e-v0: fixed-horizon joint-token law with compositional knot-token support."""

    def __init__(
        self,
        cfg: ProbabilisticJointTokenPathKnotSupportModelConfig,
        token_codebook: torch.Tensor,
        coarse_codebook: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        if token_codebook.shape != (cfg.codebook_size, cfg.n_cells):
            raise ValueError(
                f"Expected token codebook {(cfg.codebook_size, cfg.n_cells)}, got {tuple(token_codebook.shape)}"
            )
        if coarse_codebook.shape != (cfg.coarse_codebook_size, cfg.n_cells):
            raise ValueError(
                f"Expected coarse codebook {(cfg.coarse_codebook_size, cfg.n_cells)}, got {tuple(coarse_codebook.shape)}"
            )

        self.history_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.token_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.coarse_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.history_pos = nn.Embedding(cfg.history_len, cfg.d_model)
        self.future_pos = nn.Embedding(cfg.future_len + 1, cfg.d_model)
        self.coarse_pos = nn.Embedding(cfg.n_knots + 1, cfg.d_model)

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
        self.coarse_decoder = nn.TransformerDecoder(dec_layer, num_layers=1)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_decoder_layers)
        self.out_head = nn.Linear(cfg.d_model, cfg.codebook_size)
        self.coarse_out_head = nn.Linear(cfg.d_model, cfg.coarse_codebook_size)

        knot_steps, step_weights = self._build_knot_interpolation(cfg.future_len, cfg.n_knots)
        self.register_buffer("knot_steps", knot_steps)
        self.register_buffer("step_interp_weights", step_weights)
        self.register_buffer("token_codebook", token_codebook.clone())
        self.register_buffer("coarse_codebook", coarse_codebook.clone())

    @staticmethod
    def _build_knot_interpolation(future_len: int, n_knots: int) -> tuple[torch.Tensor, torch.Tensor]:
        knot_steps = torch.round(torch.linspace(0, future_len - 1, steps=n_knots)).long()
        weights = torch.zeros(future_len, n_knots, dtype=torch.float32)
        for step in range(future_len):
            if step <= int(knot_steps[0].item()):
                weights[step, 0] = 1.0
                continue
            if step >= int(knot_steps[-1].item()):
                weights[step, -1] = 1.0
                continue
            for knot_idx in range(n_knots - 1):
                left = int(knot_steps[knot_idx].item())
                right = int(knot_steps[knot_idx + 1].item())
                if left <= step <= right:
                    alpha = float(step - left) / max(right - left, 1)
                    weights[step, knot_idx] = 1.0 - alpha
                    weights[step, knot_idx + 1] = alpha
                    break
        return knot_steps, weights

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

    def assign_tokens(self, coords: torch.Tensor) -> torch.Tensor:
        flat = coords.reshape(-1, self.cfg.n_cells)
        dists = torch.cdist(flat, self.token_codebook)
        ids = dists.argmin(dim=-1)
        return ids.view(coords.shape[0], coords.shape[1])

    def token_vectors(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_codebook[token_ids]

    def coarse_targets(self, future_norm: torch.Tensor) -> torch.Tensor:
        future = self._flatten_history(future_norm)
        return future[:, self.knot_steps.to(future.device)]

    def assign_coarse_tokens(self, coarse_targets: torch.Tensor) -> torch.Tensor:
        flat = coarse_targets.reshape(-1, self.cfg.n_cells)
        dists = torch.cdist(flat, self.coarse_codebook)
        ids = dists.argmin(dim=-1)
        return ids.view(coarse_targets.shape[0], coarse_targets.shape[1])

    def coarse_vectors(self, coarse_ids: torch.Tensor) -> torch.Tensor:
        return self.coarse_codebook[coarse_ids]

    def expand_coarse_scaffold(self, coarse_vecs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("tk,bkc->btc", self.step_interp_weights.to(coarse_vecs.device), coarse_vecs)

    def decode_coarse_teacher_forced_logits(self, memory: torch.Tensor, coarse_ids: torch.Tensor) -> torch.Tensor:
        bos = torch.zeros(
            coarse_ids.shape[0], 1, self.cfg.n_cells, device=coarse_ids.device, dtype=torch.float32
        )
        prefix = torch.cat([bos, self.coarse_vectors(coarse_ids[:, :-1])], dim=1)
        pos = torch.arange(prefix.shape[1], device=prefix.device).unsqueeze(0)
        x = self.coarse_proj(prefix) + self.coarse_pos(pos)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(prefix.shape[1], device=prefix.device)
        dec = self.coarse_decoder(x, memory, tgt_mask=tgt_mask)
        return self.coarse_out_head(dec)

    def _decode_all(
        self,
        memory: torch.Tensor,
        prefix_vecs: torch.Tensor,
        coarse_scaffold: torch.Tensor,
    ) -> torch.Tensor:
        prefix_len = prefix_vecs.shape[1]
        pos = torch.arange(prefix_len, device=prefix_vecs.device).unsqueeze(0)
        x = self.token_proj(prefix_vecs) + self.future_pos(pos)
        x = x + self.coarse_proj(coarse_scaffold[:, :prefix_len])
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(prefix_len, device=prefix_vecs.device)
        return self.decoder(x, memory, tgt_mask=tgt_mask)

    def decode_teacher_forced_logits(
        self,
        history_norm: torch.Tensor,
        target_ids: torch.Tensor,
        coarse_scaffold: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode_history(history_norm)
        bos = torch.zeros(
            target_ids.shape[0], 1, self.cfg.n_cells, device=target_ids.device, dtype=torch.float32
        )
        prefix = torch.cat([bos, self.token_vectors(target_ids[:, :-1])], dim=1)
        dec = self._decode_all(memory, prefix, coarse_scaffold)
        return self.out_head(dec)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future = self._flatten_history(future_norm)
        target_coords = self.target_coords(history_norm, future)
        target_ids = self.assign_tokens(target_coords)

        coarse_targets = self.coarse_targets(future_norm)
        coarse_ids = self.assign_coarse_tokens(coarse_targets)
        coarse_vecs = self.coarse_vectors(coarse_ids)
        coarse_scaffold = self.expand_coarse_scaffold(coarse_vecs)

        memory = self.encode_history(history_norm)
        coarse_logits = self.decode_coarse_teacher_forced_logits(memory, coarse_ids)
        coarse_ce = F.cross_entropy(
            coarse_logits.reshape(-1, self.cfg.coarse_codebook_size),
            coarse_ids.reshape(-1),
            label_smoothing=self.cfg.label_smoothing,
        )
        logits = self.decode_teacher_forced_logits(history_norm, target_ids, coarse_scaffold)
        token_nll = F.cross_entropy(
            logits.reshape(-1, self.cfg.codebook_size),
            target_ids.reshape(-1),
            label_smoothing=self.cfg.label_smoothing,
        )
        loss = token_nll + self.cfg.coarse_loss_weight * coarse_ce

        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)
        token_acc = (pred_ids == target_ids).float().mean()
        coarse_pred = coarse_logits.argmax(dim=-1)
        coarse_acc = (coarse_pred == coarse_ids).float().mean()
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        metrics = {
            "total": loss.detach(),
            "token_nll": token_nll.detach(),
            "coarse_ce": coarse_ce.detach(),
            "token_acc": token_acc.detach(),
            "coarse_acc": coarse_acc.detach(),
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

            coarse_bos = torch.zeros(
                hist_window.shape[0], 1, self.cfg.n_cells, device=hist_window.device, dtype=hist_window.dtype
            )
            coarse_vecs_list: list[torch.Tensor] = []
            for _k in range(self.cfg.n_knots):
                if coarse_vecs_list:
                    prefix = torch.cat([coarse_bos, torch.stack(coarse_vecs_list, dim=1)], dim=1)
                else:
                    prefix = coarse_bos
                pos = torch.arange(prefix.shape[1], device=prefix.device).unsqueeze(0)
                x = self.coarse_proj(prefix) + self.coarse_pos(pos)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(prefix.shape[1], device=prefix.device)
                dec = self.coarse_decoder(x, memory, tgt_mask=tgt_mask)
                logits = self.coarse_out_head(dec[:, -1]) / temp
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                coarse_vecs_list.append(self.coarse_vectors(next_ids))
            coarse_vecs = torch.stack(coarse_vecs_list, dim=1)
            coarse_scaffold = self.expand_coarse_scaffold(coarse_vecs)

            curr = hist_window[:, -1]
            pred_vecs: list[torch.Tensor] = []
            pred_levels: list[torch.Tensor] = []
            bos = torch.zeros(
                hist_window.shape[0], 1, self.cfg.n_cells, device=hist_window.device, dtype=hist_window.dtype
            )
            for _step in range(self.cfg.future_len):
                if pred_vecs:
                    prefix = torch.cat([bos, torch.stack(pred_vecs, dim=1)], dim=1)
                else:
                    prefix = bos
                dec = self._decode_all(memory, prefix, coarse_scaffold)
                logits = self.out_head(dec[:, -1]) / temp
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                next_vec = self.token_vectors(next_ids)
                raw_change = self.inverse_transform_change(next_vec, hist_window)
                next_level = torch.clamp(curr + raw_change, -1.0, 1.0)
                pred_levels.append(next_level)
                pred_vecs.append(next_vec)
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
) -> tuple[ProbabilisticJointTokenPathKnotSupportModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ProbabilisticJointTokenPathKnotSupportModelConfig(**payload["config"])
    model = ProbabilisticJointTokenPathKnotSupportModel(
        cfg,
        payload["token_codebook"].to(device),
        payload["coarse_codebook"].to(device),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ProbabilisticJointTokenPathKnotSupportModel,
    cfg: ProbabilisticJointTokenPathKnotSupportModelConfig,
    epoch: int,
    best_val: float,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "token_codebook": model.token_codebook.detach().cpu(),
            "coarse_codebook": model.coarse_codebook.detach().cpu(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
