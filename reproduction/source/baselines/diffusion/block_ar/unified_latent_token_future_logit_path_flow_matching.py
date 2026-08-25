from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.joint_token_logit_transition_flow_matching import (
    _flow_time_features,
)
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class UnifiedLatentTokenFutureLogitPathFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 160
    history_hidden: int = 96
    encoder_dropout: float = 0.1

    latent_tokens: int = 6
    latent_dim: int = 64
    token_dim: int = 128
    token_layers: int = 3
    token_heads: int = 4
    token_ff: int = 256
    model_dropout: float = 0.1
    flow_time_dim: int = 32

    logit_eps: float = 1e-4
    flow_steps: int = 24
    sample_temperature: float = 1.0
    max_sample_chunk: int = 2


def _encoder_layer(cfg: UnifiedLatentTokenFutureLogitPathFMConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=cfg.token_dim,
        nhead=cfg.token_heads,
        dim_feedforward=cfg.token_ff,
        dropout=cfg.model_dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class FutureLogitPathLatentEncoder(nn.Module):
    def __init__(self, cfg: UnifiedLatentTokenFutureLogitPathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(1, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.mixer = nn.TransformerEncoder(_encoder_layer(cfg), num_layers=cfg.token_layers)
        self.latent_proj = nn.Linear(cfg.token_dim, cfg.latent_dim)
        self.latent_norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, future_logits: torch.Tensor) -> torch.Tensor:
        bsz, horizon, n_cells = future_logits.shape
        h_idx = torch.arange(horizon, device=future_logits.device)
        c_idx = torch.arange(n_cells, device=future_logits.device)
        pos = (
            self.horizon_embed(h_idx)[:, None, :]
            + self.cell_embed(c_idx)[None, :, :]
        ).reshape(horizon * n_cells, self.cfg.token_dim)
        token = self.value_proj(future_logits.reshape(bsz, horizon * n_cells, 1))
        hidden = self.mixer(token + pos[None, :, :])
        pooled = F.adaptive_avg_pool1d(
            hidden.transpose(1, 2), self.cfg.latent_tokens
        ).transpose(1, 2)
        return self.latent_norm(self.latent_proj(pooled))


class LatentTokenVelocity(nn.Module):
    def __init__(self, cfg: UnifiedLatentTokenFutureLogitPathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.latent_pos = nn.Embedding(cfg.latent_tokens, cfg.token_dim)
        self.prefix_embed = nn.Parameter(torch.zeros(2, cfg.token_dim))
        self.mixer = nn.TransformerEncoder(_encoder_layer(cfg), num_layers=cfg.token_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, cfg.latent_dim),
        )

    def forward(
        self,
        latent_t: torch.Tensor,
        context: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_tokens, _ = latent_t.shape
        pos_idx = torch.arange(n_tokens, device=latent_t.device)
        latent_token = (
            self.latent_proj(latent_t)
            + self.latent_pos(pos_idx)[None, :, :]
        )
        ctx_token = self.context_proj(context) + self.prefix_embed[0][None, :]
        time_token = self.flow_time_proj(
            _flow_time_features(flow_t, self.cfg.flow_time_dim)
        ) + self.prefix_embed[1][None, :]
        hidden = torch.cat([torch.stack([ctx_token, time_token], dim=1), latent_token], dim=1)
        hidden = self.mixer(hidden)
        return self.out(hidden[:, 2:, :])


class FutureLogitPathDecoder(nn.Module):
    def __init__(self, cfg: UnifiedLatentTokenFutureLogitPathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.token_type = nn.Parameter(torch.zeros(3, cfg.token_dim))
        self.latent_pos = nn.Embedding(cfg.latent_tokens, cfg.token_dim)
        self.mixer = nn.TransformerEncoder(_encoder_layer(cfg), num_layers=cfg.token_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def forward(
        self,
        context: torch.Tensor,
        latent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        bsz, horizon, n_cells = context.shape[0], self.cfg.future_len, self.cfg.n_cells
        h_idx = torch.arange(horizon, device=context.device)
        c_idx = torch.arange(n_cells, device=context.device)
        future_pos = (
            self.horizon_embed(h_idx)[:, None, :]
            + self.cell_embed(c_idx)[None, :, :]
        ).reshape(horizon * n_cells, self.cfg.token_dim)
        query = future_pos[None, :, :] + self.token_type[2][None, None, :]

        latent_idx = torch.arange(latent_tokens.shape[1], device=latent_tokens.device)
        latent_tok = (
            self.latent_proj(latent_tokens)
            + self.latent_pos(latent_idx)[None, :, :]
            + self.token_type[1][None, None, :]
        )
        ctx_token = (
            self.context_proj(context) + self.token_type[0][None, :]
        )[:, None, :]
        hidden = torch.cat([ctx_token, latent_tok, query.expand(bsz, -1, -1)], dim=1)
        hidden = self.mixer(hidden)
        logits = self.out(hidden[:, 1 + latent_tokens.shape[1] :, :]).squeeze(-1)
        return logits.view(bsz, horizon, n_cells)


class UnifiedLatentTokenFutureLogitPathFlowMatching(nn.Module):
    """313a-v0: unified future-logit path law with a narrow shared latent-token bottleneck."""

    def __init__(self, cfg: UnifiedLatentTokenFutureLogitPathFMConfig):
        super().__init__()
        self.cfg = cfg
        hist_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            gru_hidden_dim=cfg.history_hidden,
            bottleneck_dim=cfg.context_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=0.0,
        )
        self.history_encoder = GRUEncoder(hist_cfg)
        self.future_encoder = FutureLogitPathLatentEncoder(cfg)
        self.velocity = LatentTokenVelocity(cfg)
        self.decoder = FutureLogitPathDecoder(cfg)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten(history_norm))

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        future_01 = denormalize_iv(future_norm)
        return iv_to_logit(future_01, self.cfg.logit_eps)

    def encode_future_latents(self, future_logits: torch.Tensor) -> torch.Tensor:
        return self.future_encoder(future_logits)

    def decode_future_logits(
        self,
        context: torch.Tensor,
        latent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(context, latent_tokens)

    def predict_velocity(
        self,
        latent_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity(latent_t, context, t)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
        recon_weight: float = 1.0,
        fm_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        future_logits = self.target_future_logits(future_norm)
        latent_target = self.encode_future_latents(future_logits)
        recon_logits = self.decode_future_logits(context, latent_target)
        recon_loss = F.smooth_l1_loss(recon_logits, future_logits)

        bsz = history_norm.shape[0]
        latent_base = torch.randn_like(latent_target)
        t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
        latent_t = (1.0 - t)[:, None, None] * latent_base + t[:, None, None] * latent_target
        target_velocity = latent_target - latent_base
        pred_velocity = self.predict_velocity(latent_t, context, t)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)

        total = recon_weight * recon_loss + fm_weight * fm_loss
        metrics = {
            "total": total.detach(),
            "recon_loss": recon_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "latent_std": latent_target.std(unbiased=False).detach(),
            "future_logit_std": future_logits.std(unbiased=False).detach(),
            "future_logit_abs": future_logits.abs().mean().detach(),
            "recon_logit_abs": (recon_logits - future_logits).abs().mean().detach(),
        }
        return total, metrics

    @torch.no_grad()
    def sample_latents(self, context: torch.Tensor, n_samples: int) -> torch.Tensor:
        bsz = context.shape[0]
        latent = torch.randn(
            bsz * n_samples,
            self.cfg.latent_tokens,
            self.cfg.latent_dim,
            device=context.device,
            dtype=context.dtype,
        )
        ctx = context.repeat_interleave(n_samples, dim=0)
        dt = 1.0 / float(self.cfg.flow_steps)
        for step in range(self.cfg.flow_steps):
            t = torch.full(
                (bsz * n_samples,),
                (step + 0.5) * dt,
                device=context.device,
                dtype=context.dtype,
            )
            latent = latent + dt * self.predict_velocity(latent, ctx, t)
        return latent.view(bsz, n_samples, self.cfg.latent_tokens, self.cfg.latent_dim)

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
        history_norm = self._flatten(history_norm)
        context = self.encode_history(history_norm)
        bsz = context.shape[0]
        outs: list[torch.Tensor] = []
        chunk_size = max(1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            latent = self.sample_latents(context, n_samples=k)
            flat_context = context.repeat_interleave(k, dim=0)
            flat_latent = latent.reshape(
                bsz * k, self.cfg.latent_tokens, self.cfg.latent_dim
            )
            if temp != 1.0:
                flat_latent = flat_latent * temp
            future_logits = self.decode_future_logits(flat_context, flat_latent)
            future_01 = logit_to_iv(future_logits)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(
                    bsz, k, self.cfg.future_len, self.cfg.n_cells
                )
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[UnifiedLatentTokenFutureLogitPathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = UnifiedLatentTokenFutureLogitPathFMConfig(**payload["config"])
    model = UnifiedLatentTokenFutureLogitPathFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: UnifiedLatentTokenFutureLogitPathFlowMatching,
    cfg: UnifiedLatentTokenFutureLogitPathFMConfig,
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
