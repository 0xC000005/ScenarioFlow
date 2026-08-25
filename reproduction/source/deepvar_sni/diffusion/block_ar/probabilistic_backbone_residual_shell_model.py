from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    load_model as load_277d_model,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


def flatten_panels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.view(x.shape[0], x.shape[1], -1)
    return x


def compute_change_scale(history_norm: torch.Tensor, eps: float) -> torch.Tensor:
    hist = flatten_panels(history_norm)
    hist_change = hist[:, 1:] - hist[:, :-1]
    scale = hist_change.pow(2).mean(dim=1, keepdim=True).sqrt()
    return scale.clamp_min(eps)


def compute_backbone_stats(
    history_norm: torch.Tensor,
    center_future_norm: torch.Tensor,
    eps: float,
    change_coord: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hist_window = flatten_panels(history_norm).clone()
    center_future = flatten_panels(center_future_norm)
    curr_center = hist_window[:, -1]
    center_raw: list[torch.Tensor] = []
    center_coords: list[torch.Tensor] = []
    center_scales: list[torch.Tensor] = []
    for step in range(center_future.shape[1]):
        next_center = center_future[:, step]
        raw_change = next_center - curr_center
        scale = compute_change_scale(hist_window, eps).squeeze(1)
        if change_coord == "raw":
            coord = raw_change
        elif change_coord == "asinh_local_scale":
            coord = torch.asinh(raw_change / scale)
        else:
            raise ValueError(f"Unknown change_coord={change_coord}")
        center_raw.append(raw_change)
        center_coords.append(coord)
        center_scales.append(scale)
        hist_window = torch.cat([hist_window[:, 1:], next_center.unsqueeze(1)], dim=1)
        curr_center = next_center
    return (
        torch.stack(center_raw, dim=1),
        torch.stack(center_coords, dim=1),
        torch.stack(center_scales, dim=1),
    )


def compute_residual_coords(
    history_norm: torch.Tensor,
    center_future_norm: torch.Tensor,
    future_norm: torch.Tensor,
    eps: float,
    change_coord: str,
) -> torch.Tensor:
    hist_actual = flatten_panels(history_norm).clone()
    hist_center = flatten_panels(history_norm).clone()
    center_future = flatten_panels(center_future_norm)
    future = flatten_panels(future_norm)
    curr_actual = hist_actual[:, -1]
    curr_center = hist_center[:, -1]
    residuals: list[torch.Tensor] = []
    for step in range(future.shape[1]):
        next_center = center_future[:, step]
        center_raw = next_center - curr_center
        scale = compute_change_scale(hist_center, eps).squeeze(1)
        next_actual = future[:, step]
        actual_raw = next_actual - curr_actual
        residual_raw = actual_raw - center_raw
        if change_coord == "raw":
            residual_coord = residual_raw
        elif change_coord == "asinh_local_scale":
            residual_coord = torch.asinh(residual_raw / scale)
        else:
            raise ValueError(f"Unknown change_coord={change_coord}")
        residuals.append(residual_coord)
        hist_actual = torch.cat([hist_actual[:, 1:], next_actual.unsqueeze(1)], dim=1)
        hist_center = torch.cat([hist_center[:, 1:], next_center.unsqueeze(1)], dim=1)
        curr_actual = next_actual
        curr_center = next_center
    return torch.stack(residuals, dim=1)


@dataclass
class ProbabilisticBackboneResidualShellConfig:
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
    label_smoothing: float = 0.0
    sample_temperature: float = 1.0


class ResidualShellCore(nn.Module):
    """296a-v0 shell: residual token law around a frozen 277d center path."""

    def __init__(self, cfg: ProbabilisticBackboneResidualShellConfig, codebook: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        if codebook.shape != (cfg.codebook_size, cfg.n_cells):
            raise ValueError(
                f"Expected codebook {(cfg.codebook_size, cfg.n_cells)}, got {tuple(codebook.shape)}"
            )

        self.history_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.token_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.cond_proj = nn.Linear(2 * cfg.n_cells, cfg.d_model)
        self.history_pos = nn.Embedding(cfg.history_len, cfg.d_model)
        self.future_pos = nn.Embedding(cfg.future_len + 1, cfg.d_model)
        self.segment_embed = nn.Embedding(2, cfg.d_model)

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

        self.register_buffer("codebook", codebook.clone())

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        hist = flatten_panels(history_norm)
        pos = torch.arange(self.cfg.history_len, device=hist.device).unsqueeze(0)
        seg = torch.zeros_like(pos)
        x = self.history_proj(hist) + self.history_pos(pos) + self.segment_embed(seg)
        return self.encoder(x)

    def build_center_context(
        self,
        center_future_norm: torch.Tensor,
        center_coords: torch.Tensor,
    ) -> torch.Tensor:
        center_future = flatten_panels(center_future_norm)
        feats = torch.cat([center_future, center_coords], dim=-1)
        pos = torch.arange(self.cfg.future_len, device=feats.device).unsqueeze(0) + 1
        seg = torch.ones_like(pos)
        return self.cond_proj(feats) + self.future_pos(pos) + self.segment_embed(seg)

    def assign_tokens(self, residual_coords: torch.Tensor) -> torch.Tensor:
        flat = residual_coords.reshape(-1, self.cfg.n_cells)
        dists = torch.cdist(flat, self.codebook)
        ids = dists.argmin(dim=-1)
        return ids.view(residual_coords.shape[0], residual_coords.shape[1])

    def token_vectors(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.codebook[token_ids]

    def inverse_residual_change(self, residual_coord: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.cfg.change_coord == "raw":
            return residual_coord
        if self.cfg.change_coord == "asinh_local_scale":
            return torch.sinh(residual_coord) * scale
        raise ValueError(f"Unknown change_coord={self.cfg.change_coord}")

    def _decode_all(
        self,
        memory: torch.Tensor,
        prefix_tokens: torch.Tensor,
        center_ctx: torch.Tensor,
    ) -> torch.Tensor:
        prefix_len = prefix_tokens.shape[1]
        pos = torch.arange(prefix_len, device=prefix_tokens.device).unsqueeze(0) + 1
        tgt = self.token_proj(prefix_tokens) + self.future_pos(pos) + center_ctx[:, :prefix_len]
        return self.decoder(tgt, memory)

    def decode_teacher_forced_logits(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory = self.encode_history(history_norm)
        center_raw, center_coords, center_scales = compute_backbone_stats(
            history_norm,
            center_future_norm,
            eps=self.cfg.change_scale_eps,
            change_coord=self.cfg.change_coord,
        )
        center_ctx = self.build_center_context(center_future_norm, center_coords)
        bos = torch.zeros(
            target_ids.shape[0], 1, self.cfg.n_cells, device=target_ids.device, dtype=self.codebook.dtype
        )
        prefix = torch.cat([bos, self.token_vectors(target_ids[:, :-1])], dim=1)
        dec = self._decode_all(memory, prefix, center_ctx)
        return self.out_head(dec), center_raw, center_scales

    def training_loss(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual_coords = compute_residual_coords(
            history_norm,
            center_future_norm,
            future_norm,
            eps=self.cfg.change_scale_eps,
            change_coord=self.cfg.change_coord,
        )
        target_ids = self.assign_tokens(residual_coords)
        logits, center_raw, center_scales = self.decode_teacher_forced_logits(
            history_norm,
            center_future_norm,
            target_ids,
        )
        loss = F.cross_entropy(
            logits.reshape(-1, self.cfg.codebook_size),
            target_ids.reshape(-1),
            label_smoothing=self.cfg.label_smoothing,
        )
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)
        pred_coords = self.token_vectors(pred_ids)
        pred_residual_raw = self.inverse_residual_change(pred_coords, center_scales)
        target_residual_raw = self.inverse_residual_change(residual_coords, center_scales)
        raw_mae = (pred_residual_raw - target_residual_raw).abs().mean()
        token_acc = (pred_ids == target_ids).float().mean()
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        metrics = {
            "total": loss.detach(),
            "token_nll": loss.detach(),
            "token_acc": token_acc.detach(),
            "token_entropy": entropy.detach(),
            "residual_raw_mae": raw_mae.detach(),
            "center_raw_abs_mean": center_raw.abs().mean().detach(),
            "residual_coord_abs_mean": residual_coords.abs().mean().detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample_with_center(
        self,
        history_norm: torch.Tensor,
        center_future_norm: torch.Tensor,
        n_samples: int,
        chunk_size: int,
        temperature: float | None = None,
    ) -> torch.Tensor:
        history_norm = flatten_panels(history_norm)
        center_future_norm = flatten_panels(center_future_norm)
        bsz = history_norm.shape[0]
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        temp = max(temp, 1e-4)

        center_raw, center_coords, center_scales = compute_backbone_stats(
            history_norm,
            center_future_norm,
            eps=self.cfg.change_scale_eps,
            change_coord=self.cfg.change_coord,
        )
        base_memory = self.encode_history(history_norm)
        base_center_ctx = self.build_center_context(center_future_norm, center_coords)

        outputs: list[torch.Tensor] = []
        remaining = int(n_samples)
        while remaining > 0:
            sample_chunk = min(chunk_size, remaining)
            memory = (
                base_memory.unsqueeze(1)
                .expand(-1, sample_chunk, -1, -1)
                .reshape(bsz * sample_chunk, self.cfg.history_len, self.cfg.d_model)
            )
            center_ctx = (
                base_center_ctx.unsqueeze(1)
                .expand(-1, sample_chunk, -1, -1)
                .reshape(bsz * sample_chunk, self.cfg.future_len, self.cfg.d_model)
            )
            center_raw_chunk = (
                center_raw.unsqueeze(1)
                .expand(-1, sample_chunk, -1, -1)
                .reshape(bsz * sample_chunk, self.cfg.future_len, self.cfg.n_cells)
            )
            center_scales_chunk = (
                center_scales.unsqueeze(1)
                .expand(-1, sample_chunk, -1, -1)
                .reshape(bsz * sample_chunk, self.cfg.future_len, self.cfg.n_cells)
            )

            curr = (
                history_norm[:, -1]
                .unsqueeze(1)
                .expand(-1, sample_chunk, -1)
                .reshape(bsz * sample_chunk, self.cfg.n_cells)
                .clone()
            )
            pred_vecs: list[torch.Tensor] = []
            pred_levels: list[torch.Tensor] = []
            bos = torch.zeros(
                curr.shape[0], 1, self.cfg.n_cells, device=curr.device, dtype=curr.dtype
            )
            for step in range(self.cfg.future_len):
                if pred_vecs:
                    prefix = torch.cat([bos, torch.stack(pred_vecs, dim=1)], dim=1)
                else:
                    prefix = bos
                dec = self._decode_all(memory, prefix, center_ctx)
                logits = self.out_head(dec[:, -1]) / temp
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                residual_coord = self.token_vectors(next_ids)
                residual_raw = self.inverse_residual_change(
                    residual_coord, center_scales_chunk[:, step]
                )
                total_raw = center_raw_chunk[:, step] + residual_raw
                next_level = torch.clamp(curr + total_raw, -1.0, 1.0)
                pred_vecs.append(residual_coord)
                pred_levels.append(next_level)
                curr = next_level
            future_norm = torch.stack(pred_levels, dim=1)
            future_01 = denormalize_iv(future_norm)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, sample_chunk, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(
                    bsz, sample_chunk, self.cfg.future_len, self.cfg.n_cells
                )
            outputs.append(future_01)
            remaining -= sample_chunk
        return torch.cat(outputs, dim=1).contiguous()


class BackboneResidualShellGenerator(nn.Module):
    def __init__(
        self,
        shell: ResidualShellCore,
        backbone: nn.Module,
        backbone_checkpoint_path: str,
    ):
        super().__init__()
        self.shell = shell
        self.backbone = backbone.eval()
        self.backbone_checkpoint_path = backbone_checkpoint_path
        for param in self.backbone.parameters():
            param.requires_grad_(False)

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
        if n_steps != self.shell.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.shell.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = flatten_panels(history_norm)
        center_future_01 = self.backbone.sample_batched(
            history_norm,
            n_samples=1,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
        ).squeeze(1)
        center_future_norm = normalize_iv(center_future_01)
        return self.shell.sample_with_center(
            history_norm,
            center_future_norm,
            n_samples=n_samples,
            chunk_size=chunk_size,
            temperature=temperature,
        )


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[BackboneResidualShellGenerator, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ProbabilisticBackboneResidualShellConfig(**payload["config"])
    shell = ResidualShellCore(cfg, payload["codebook"].to(device))
    shell.load_state_dict(payload["shell_state_dict"], strict=True)
    backbone_checkpoint_path = payload["backbone_checkpoint_path"]
    backbone, _ = load_277d_model(backbone_checkpoint_path, device)
    model = BackboneResidualShellGenerator(shell.to(device), backbone, backbone_checkpoint_path)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    shell: ResidualShellCore,
    cfg: ProbabilisticBackboneResidualShellConfig,
    epoch: int,
    best_val: float,
    backbone_checkpoint_path: str,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "codebook": shell.codebook.detach().cpu(),
            "shell_state_dict": shell.state_dict(),
            "backbone_checkpoint_path": backbone_checkpoint_path,
        },
        path,
    )
