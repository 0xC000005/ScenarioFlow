from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class EmpiricalNormalScorePathCouplingConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 256
    history_hidden: int = 160
    encoder_dropout: float = 0.1

    coupling_layers: int = 8
    coupling_hidden: int = 768
    coupling_dropout: float = 0.1
    scale_clip: float = 2.0

    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    sample_temperature: float = 1.0
    max_sample_chunk: int = 8


class CouplingLayer(nn.Module):
    def __init__(
        self,
        path_dim: int,
        context_dim: int,
        hidden_dim: int,
        dropout: float,
        scale_clip: float,
        mask: torch.Tensor,
    ):
        super().__init__()
        self.path_dim = int(path_dim)
        self.scale_clip = float(scale_clip)
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(
            nn.Linear(path_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, path_dim * 2),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(last.bias)

    def _shift_log_scale(
        self,
        x_masked: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat([x_masked, context], dim=-1))
        shift, log_scale = raw.chunk(2, dim=-1)
        inv_mask = 1.0 - self.mask
        shift = shift * inv_mask
        log_scale = torch.tanh(log_scale) * self.scale_clip * inv_mask
        return shift, log_scale

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        shift, log_scale = self._shift_log_scale(x_masked, context)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(log_scale) + shift)
        log_det = log_scale.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        y_masked = y * self.mask
        shift, log_scale = self._shift_log_scale(y_masked, context)
        return y_masked + (1.0 - self.mask) * ((y - shift) * torch.exp(-log_scale))


class EmpiricalNormalScorePathCouplingDensity(nn.Module):
    """346a: full future-path exact likelihood with vector coupling dependence."""

    def __init__(self, cfg: EmpiricalNormalScorePathCouplingConfig):
        super().__init__()
        self.cfg = cfg
        self.path_dim = cfg.future_len * cfg.n_cells
        hist_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            gru_hidden_dim=cfg.history_hidden,
            bottleneck_dim=cfg.context_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=0.0,
        )
        self.history_encoder = GRUEncoder(hist_cfg)
        layers: list[CouplingLayer] = []
        base = torch.arange(self.path_dim)
        for idx in range(cfg.coupling_layers):
            mask = ((base + idx) % 2 == 0).float()
            layers.append(
                CouplingLayer(
                    path_dim=self.path_dim,
                    context_dim=cfg.context_dim,
                    hidden_dim=cfg.coupling_hidden,
                    dropout=cfg.coupling_dropout,
                    scale_clip=cfg.scale_clip,
                    mask=mask,
                )
            )
        self.layers = nn.ModuleList(layers)
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("history_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("future_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def set_empirical_quantiles(
        self,
        history_quantiles: torch.Tensor,
        future_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if history_quantiles.shape != expected or future_quantiles.shape != expected:
            raise ValueError(f"Expected quantile tensors with shape {expected}")
        self.history_quantiles.copy_(history_quantiles.to(self.history_quantiles))
        self.future_quantiles.copy_(future_quantiles.to(self.future_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("Expected quantile_levels with shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("Empirical quantiles must be set before training or sampling")

    def _values_to_scores(self, values_01: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        values_01 = self._flatten(values_01)
        levels = self.quantile_levels.to(device=values_01.device, dtype=values_01.dtype)
        table = table.to(device=values_01.device, dtype=values_01.dtype)
        cols: list[torch.Tensor] = []
        for cell in range(self.cfg.n_cells):
            q = table[cell]
            flat = values_01[..., cell].reshape(-1)
            idx = torch.searchsorted(q.contiguous(), flat.contiguous(), right=False)
            idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
            idx_lo = idx_hi - 1
            q_lo = q[idx_lo]
            q_hi = q[idx_hi]
            u_lo = levels[idx_lo]
            u_hi = levels[idx_hi]
            alpha = (flat - q_lo) / (q_hi - q_lo).clamp_min(1e-12)
            u = u_lo + alpha.clamp(0.0, 1.0) * (u_hi - u_lo)
            u = torch.where(flat <= q[0], levels[0], u)
            u = torch.where(flat >= q[-1], levels[-1], u)
            eps = float(self.cfg.cdf_eps)
            z = torch.special.ndtri(u.clamp(eps, 1.0 - eps))
            cols.append(z.view(values_01.shape[:-1]))
        return torch.stack(cols, dim=-1)

    def _scores_to_values(self, scores: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = table.to(device=scores.device, dtype=scores.dtype)
        eps = float(self.cfg.cdf_eps)
        u_all = (0.5 * (1.0 + torch.erf(scores / math.sqrt(2.0)))).clamp(eps, 1.0 - eps)
        cols: list[torch.Tensor] = []
        for cell in range(self.cfg.n_cells):
            q = table[cell]
            flat = u_all[..., cell].reshape(-1)
            idx = torch.searchsorted(levels.contiguous(), flat.contiguous(), right=False)
            idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
            idx_lo = idx_hi - 1
            u_lo = levels[idx_lo]
            u_hi = levels[idx_hi]
            q_lo = q[idx_lo]
            q_hi = q[idx_hi]
            alpha = (flat - u_lo) / (u_hi - u_lo).clamp_min(1e-12)
            x = q_lo + alpha.clamp(0.0, 1.0) * (q_hi - q_lo)
            x = torch.where(flat <= levels[0], q[0], x)
            x = torch.where(flat >= levels[-1], q[-1], x)
            cols.append(x.view(scores.shape[:-1]))
        return torch.stack(cols, dim=-1).clamp(0.0, 1.0)

    def history_scores(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_01 = denormalize_iv(self._flatten(history_norm))
        return self._values_to_scores(history_01, self.history_quantiles)

    def target_future_scores(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_01 = denormalize_iv(self._flatten(future_norm))
        return self._values_to_scores(future_01, self.future_quantiles)

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self.history_scores(history_norm))

    def encode_history_scores(self, history_scores: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history_scores)

    def forward_to_base(
        self,
        future_scores: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = future_scores.reshape(future_scores.shape[0], self.path_dim)
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            z, ld = layer(z, context)
            log_det = log_det + ld
        return z, log_det

    def inverse_from_base(self, base: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = base
        for layer in reversed(self.layers):
            x = layer.inverse(x, context)
        return x.view(base.shape[0], self.cfg.future_len, self.cfg.n_cells)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_scores = self.history_scores(history_norm)
        future_scores = self.target_future_scores(future_norm)
        context = self.encode_history_scores(history_scores)
        z, log_det = self.forward_to_base(future_scores, context)
        base_nll = 0.5 * z.square().sum(dim=-1) + 0.5 * self.path_dim * math.log(2.0 * math.pi)
        nll = (base_nll - log_det) / float(self.path_dim)
        loss = nll.mean()
        transitions = future_scores - torch.cat(
            [history_scores[:, -1:], future_scores[:, :-1]], dim=1
        )
        metrics = {
            "total": loss.detach(),
            "nll": loss.detach(),
            "base_abs": z.abs().mean().detach(),
            "base_std": z.std(unbiased=False).detach(),
            "log_det_per_dim": (log_det / float(self.path_dim)).mean().detach(),
            "target_std": future_scores.std(unbiased=False).detach(),
            "transition_std": transitions.std(unbiased=False).detach(),
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
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(f"Expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        context = self.encode_history(history_norm)
        bsz = history_norm.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            ctx = context.repeat_interleave(k, dim=0)
            base = temp * torch.randn(
                bsz * k,
                self.path_dim,
                device=context.device,
                dtype=context.dtype,
            )
            future_scores = self.inverse_from_base(base, ctx)[:, :n_steps]
            future_01 = self._scores_to_values(future_scores, self.future_quantiles)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, n_steps, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, n_steps, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.sample_batched(
            normalize_iv(history_01),
            n_samples=n_samples,
            n_steps=1,
            history_is_normalized=True,
            **kwargs,
        )[:, :, 0]


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[EmpiricalNormalScorePathCouplingDensity, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScorePathCouplingConfig(**payload["config"])
    model = EmpiricalNormalScorePathCouplingDensity(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScorePathCouplingDensity,
    cfg: EmpiricalNormalScorePathCouplingConfig,
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
