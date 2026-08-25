from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class LatentPathManifoldFlowConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    latent_dim: int = 96
    ae_hidden: int = 512
    ae_layers: int = 3
    history_hidden: int = 160
    history_layers: int = 2
    flow_hidden: int = 384
    flow_layers: int = 4
    time_dim: int = 32
    dropout: float = 0.05

    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    prefix_feature_mode: str = "scale"
    flow_steps: int = 32
    sample_temperature: float = 1.0
    max_sample_chunk: int = 8
    recon_change_weight: float = 0.5


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(1, n_layers - 1)):
        layers.extend(
            [
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        return t[:, None]
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            half,
            device=t.device,
            dtype=t.dtype,
        )
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class LatentPathManifoldFlow(nn.Module):
    """476a: deterministic future-path manifold plus conditional latent flow.

    The model learns a compact coordinate for the full empirical-normal-score
    future path, then learns p(latent | history) with vanilla rectified flow.
    """

    def __init__(self, cfg: LatentPathManifoldFlowConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.prefix_feature_mode not in {"basic", "scale"}:
            raise ValueError("prefix_feature_mode must be 'basic' or 'scale'")
        self.path_dim = cfg.future_len * cfg.n_cells

        self.future_encoder = nn.Sequential(
            nn.LayerNorm(self.path_dim),
            _make_mlp(
                self.path_dim,
                cfg.ae_hidden,
                cfg.latent_dim,
                cfg.ae_layers,
                cfg.dropout,
            ),
        )
        self.future_decoder = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim),
            _make_mlp(
                cfg.latent_dim,
                cfg.ae_hidden,
                self.path_dim,
                cfg.ae_layers,
                cfg.dropout,
            ),
        )

        feature_mult = 4 if cfg.prefix_feature_mode == "scale" else 2
        self.history_gru = nn.GRU(
            input_size=feature_mult * cfg.n_cells,
            hidden_size=cfg.history_hidden,
            num_layers=cfg.history_layers,
            dropout=cfg.dropout if cfg.history_layers > 1 else 0.0,
            batch_first=True,
        )
        self.history_norm = nn.LayerNorm(cfg.history_hidden)
        self.velocity = _make_mlp(
            cfg.latent_dim + cfg.history_hidden + cfg.time_dim,
            cfg.flow_hidden,
            cfg.latent_dim,
            cfg.flow_layers,
            cfg.dropout,
        )

        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("level_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def set_empirical_quantiles(
        self,
        level_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if level_quantiles.shape != expected:
            raise ValueError(f"Expected level quantiles with shape {expected}")
        self.level_quantiles.copy_(level_quantiles.to(self.level_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("Expected quantile_levels with shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("Empirical quantiles must be set before use")

    def _values_to_scores(self, values_01: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        values_01 = self._flatten(values_01)
        levels = self.quantile_levels.to(device=values_01.device, dtype=values_01.dtype)
        table = self.level_quantiles.to(device=values_01.device, dtype=values_01.dtype)
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
            cols.append(torch.special.ndtri(u.clamp(eps, 1.0 - eps)).view(values_01.shape[:-1]))
        return torch.stack(cols, dim=-1)

    def _scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = self.level_quantiles.to(device=scores.device, dtype=scores.dtype)
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
        return self._values_to_scores(history_01)

    def target_future_scores(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_01 = denormalize_iv(self._flatten(future_norm))
        return self._values_to_scores(future_01)

    def _score_features_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(scores)
        deltas[:, 1:] = scores[:, 1:] - scores[:, :-1]
        if self.cfg.prefix_feature_mode == "scale":
            return torch.cat([scores, deltas, deltas.abs(), deltas.square()], dim=-1)
        return torch.cat([scores, deltas], dim=-1)

    def encode_history_scores(self, history_scores: torch.Tensor) -> torch.Tensor:
        features = self._score_features_from_scores(history_scores)
        out, _hidden = self.history_gru(features)
        return self.history_norm(out[:, -1])

    def encode_future_scores(self, future_scores: torch.Tensor) -> torch.Tensor:
        flat = future_scores.reshape(future_scores.shape[0], self.path_dim)
        return self.future_encoder(flat)

    def decode_future_scores(self, latent: torch.Tensor) -> torch.Tensor:
        flat = self.future_decoder(latent)
        return flat.view(latent.shape[0], self.cfg.future_len, self.cfg.n_cells)

    def autoencoder_loss(
        self,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future_scores = self.target_future_scores(future_norm)
        latent = self.encode_future_scores(future_scores)
        recon_scores = self.decode_future_scores(latent)
        level_loss = F.mse_loss(recon_scores, future_scores)
        change_loss = F.mse_loss(
            recon_scores[:, 1:] - recon_scores[:, :-1],
            future_scores[:, 1:] - future_scores[:, :-1],
        )
        loss = level_loss + float(self.cfg.recon_change_weight) * change_loss
        metrics = {
            "total": loss.detach(),
            "ae_loss": loss.detach(),
            "recon_level_mse": level_loss.detach(),
            "recon_change_mse": change_loss.detach(),
            "latent_std": latent.std(unbiased=False).detach(),
            "target_score_std": future_scores.std(unbiased=False).detach(),
            "recon_score_std": recon_scores.std(unbiased=False).detach(),
        }
        return loss, metrics

    def flow_matching_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_scores = self.history_scores(history_norm)
        future_scores = self.target_future_scores(future_norm)
        context = self.encode_history_scores(history_scores)
        target_latent = self.encode_future_scores(future_scores).detach()
        x0 = torch.randn_like(target_latent)
        t = torch.rand(target_latent.shape[0], device=target_latent.device, dtype=target_latent.dtype)
        x_t = (1.0 - t[:, None]) * x0 + t[:, None] * target_latent
        target_velocity = target_latent - x0
        pred_velocity = self.velocity(
            torch.cat(
                [
                    x_t,
                    context,
                    _sinusoidal_time_embedding(t, self.cfg.time_dim),
                ],
                dim=-1,
            )
        )
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "latent_std": target_latent.std(unbiased=False).detach(),
            "target_velocity_std": target_velocity.std(unbiased=False).detach(),
            "context_abs": context.abs().mean().detach(),
        }
        return fm_loss, metrics

    def freeze_autoencoder(self) -> None:
        for module in (self.future_encoder, self.future_decoder):
            for param in module.parameters():
                param.requires_grad_(False)

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
        history_scores = self.history_scores(history_norm)
        context = self.encode_history_scores(history_scores)
        bsz = history_scores.shape[0]
        chunk_size = max(
            1,
            min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)),
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)

        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            ctx = context.repeat_interleave(k, dim=0)
            latent = temp * torch.randn(
                bsz * k,
                self.cfg.latent_dim,
                device=history_scores.device,
                dtype=history_scores.dtype,
            )
            for flow_step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (flow_step + 0.5) * dt,
                    device=history_scores.device,
                    dtype=history_scores.dtype,
                )
                velocity = self.velocity(
                    torch.cat(
                        [
                            latent,
                            ctx,
                            _sinusoidal_time_embedding(t, self.cfg.time_dim),
                        ],
                        dim=-1,
                    )
                )
                latent = latent + dt * velocity
            future_scores = self.decode_future_scores(latent)[:, :n_steps]
            future_01 = self._scores_to_values(future_scores.reshape(-1, self.cfg.n_cells))
            future_01 = future_01.view(bsz, k, n_steps, 5, 5)
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
) -> tuple[LatentPathManifoldFlow, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LatentPathManifoldFlowConfig(**payload["config"])
    model = LatentPathManifoldFlow(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: LatentPathManifoldFlow,
    cfg: LatentPathManifoldFlowConfig,
    epoch: int,
    best_val: float,
    stage: str,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "stage": stage,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
