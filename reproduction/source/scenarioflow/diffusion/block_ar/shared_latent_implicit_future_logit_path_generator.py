from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.axial_mixer_future_logit_path_flow_matching import AxialMixerBlock
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.joint_token_logit_transition_flow_matching import (
    JointTokenLogitTransitionFMConfig,
)
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class SharedLatentImplicitPathGeneratorConfig(JointTokenLogitTransitionFMConfig):
    latent_tokens: int = 8
    latent_dim: int = 64
    use_local_noise: bool = False
    demean_local_noise: bool = False
    standardize_logits: bool = True
    logit_std_floor: float = 1e-3
    train_sample_count: int = 4
    score_eps: float = 1e-6


class SharedLatentPathDecoder(nn.Module):
    def __init__(self, cfg: SharedLatentImplicitPathGeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.token_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.token_dim)
        self.local_noise_proj = nn.Linear(1, cfg.token_dim)
        self.horizon_embed = nn.Embedding(cfg.future_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.latent_pos = nn.Embedding(cfg.latent_tokens, cfg.token_dim)
        self.cross_attn = nn.MultiheadAttention(
            cfg.token_dim,
            cfg.token_heads,
            dropout=cfg.model_dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(cfg.token_dim)
        self.latent_norm = nn.LayerNorm(cfg.token_dim)
        self.blocks = nn.ModuleList([AxialMixerBlock(cfg) for _ in range(cfg.token_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        local_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n_latent, _ = latent.shape
        h_idx = torch.arange(self.cfg.future_len, device=latent.device)
        c_idx = torch.arange(self.cfg.n_cells, device=latent.device)
        l_idx = torch.arange(n_latent, device=latent.device)

        ctx = self.context_proj(context)
        latent_tokens = self.latent_proj(latent)
        latent_tokens = latent_tokens + self.latent_pos(l_idx)[None, :, :]
        latent_tokens = latent_tokens + ctx[:, None, :]
        latent_tokens = self.latent_norm(latent_tokens)

        query = self.horizon_embed(h_idx)[:, None, :] + self.cell_embed(c_idx)[None, :, :]
        query = query.reshape(self.cfg.future_len * self.cfg.n_cells, self.cfg.token_dim)
        query = query[None, :, :] + ctx[:, None, :]
        if local_noise is not None:
            query = query + self.local_noise_proj(
                local_noise.reshape(bsz, self.cfg.future_len * self.cfg.n_cells, 1)
            )
        attn, _ = self.cross_attn(
            self.query_norm(query),
            latent_tokens,
            latent_tokens,
            need_weights=False,
        )
        x = (query + attn).view(
            bsz,
            self.cfg.future_len,
            self.cfg.n_cells,
            self.cfg.token_dim,
        )
        for block in self.blocks:
            x = block(x)
        return self.out(x).squeeze(-1)


class SharedLatentImplicitFutureLogitPathGenerator(nn.Module):
    """326a-v0: full-path generator with stochasticity only through shared latent tokens."""

    def __init__(self, cfg: SharedLatentImplicitPathGeneratorConfig):
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
        self.decoder = SharedLatentPathDecoder(cfg)
        self.register_buffer("cell_logit_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_logit_std", torch.ones(cfg.n_cells))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def set_logit_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.cfg.n_cells,) or std.shape != (self.cfg.n_cells,):
            raise ValueError("Expected per-cell logit stats with shape (n_cells,)")
        self.cell_logit_mean.copy_(mean.to(self.cell_logit_mean))
        self.cell_logit_std.copy_(
            std.clamp_min(self.cfg.logit_std_floor).to(self.cell_logit_std)
        )

    def _to_model_coord(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return logits
        return (logits - self.cell_logit_mean) / self.cell_logit_std

    def _from_model_coord(self, coord: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return coord
        return coord * self.cell_logit_std + self.cell_logit_mean

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten(history_norm))

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        future_01 = denormalize_iv(future_norm)
        return self._to_model_coord(iv_to_logit(future_01, self.cfg.logit_eps))

    def generate_model_coord(self, history_norm: torch.Tensor, n_samples: int) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        context = self.encode_history(history_norm)
        bsz = history_norm.shape[0]
        k = int(n_samples)
        latent = torch.randn(
            bsz * k,
            self.cfg.latent_tokens,
            self.cfg.latent_dim,
            device=history_norm.device,
            dtype=history_norm.dtype,
        )
        local_noise = None
        if self.cfg.use_local_noise:
            local_noise = torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=history_norm.device,
                dtype=history_norm.dtype,
            )
            if self.cfg.demean_local_noise:
                local_noise = local_noise - local_noise.mean(dim=2, keepdim=True)
        ctx = context.repeat_interleave(k, dim=0)
        out = self.decoder(latent, ctx, local_noise=local_noise)
        return out.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)

    def energy_score(
        self,
        samples: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, k, horizon, n_cells = samples.shape
        scale = float(horizon * n_cells) ** 0.5
        sample_flat = samples.reshape(bsz, k, horizon * n_cells)
        target_flat = target.reshape(bsz, horizon * n_cells)
        target_dist = torch.sqrt(
            (sample_flat - target_flat[:, None, :]).pow(2).sum(dim=-1)
            + self.cfg.score_eps
        ).mean(dim=1) / scale
        pair_dist = torch.cdist(sample_flat, sample_flat, p=2).mean(dim=(1, 2)) / scale
        score = target_dist - 0.5 * pair_dist
        return score.mean(), target_dist.mean(), pair_dist.mean()

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        target = self.target_future_logits(future_norm)
        samples = self.generate_model_coord(history_norm, self.cfg.train_sample_count)
        total, target_dist, pair_dist = self.energy_score(samples, target)
        metrics = {
            "total": total.detach(),
            "target_dist": target_dist.detach(),
            "pair_dist": pair_dist.detach(),
            "target_std": target.std(unbiased=False).detach(),
            "sample_std": samples.std(unbiased=False).detach(),
            "sample_abs": samples.abs().mean().detach(),
        }
        return total, metrics

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 16,
        history_is_normalized: bool = True,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        chunk_size = max(1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)))
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            coord = self.generate_model_coord(history_norm, k)
            future_01 = logit_to_iv(self._from_model_coord(coord))
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(history_norm.shape[0], k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(history_norm.shape[0], k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[SharedLatentImplicitFutureLogitPathGenerator, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = SharedLatentImplicitPathGeneratorConfig(**payload["config"])
    model = SharedLatentImplicitFutureLogitPathGenerator(cfg)
    result = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = {"cell_logit_mean", "cell_logit_std"}
    if not cfg.use_local_noise:
        allowed_missing.update(
            {"decoder.local_noise_proj.weight", "decoder.local_noise_proj.bias"}
        )
    missing = set(result.missing_keys) - allowed_missing
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    if cfg.standardize_logits and allowed_missing.intersection(result.missing_keys):
        raise RuntimeError("Standardized-logit checkpoint is missing saved logit stats")
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: SharedLatentImplicitFutureLogitPathGenerator,
    cfg: SharedLatentImplicitPathGeneratorConfig,
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
