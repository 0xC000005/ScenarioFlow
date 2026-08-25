from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.dual_timescale_low_rank_temporal import TemporalConvTower, _mlp
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class JointProbabilisticLatentFactorFMECConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    latent_dim: int = 8

    encoder_hidden: int = 64
    bottleneck_dim: int = 128
    encoder_dropout: float = 0.1
    cond_aug_sigma: float = 0.0

    model_hidden: int = 128
    cond_layers: int = 3
    flow_layers: int = 3
    kernel_size: int = 5
    dilation: int = 2
    model_dropout: float = 0.1
    future_pos_embed: int = 16
    flow_time_embed: int = 16

    head_hidden: int = 256
    head_layers: int = 2
    head_dropout: float = 0.1

    max_idio_ratio: float = 0.20
    sigma_floor: float = 0.05
    ode_steps: int = 16
    ortho_reg_weight: float = 0.0
    pinv_ridge: float = 1e-4
    change_coord: str = "asinh_local_scale"
    change_scale_eps: float = 1e-3

    ec_gain_max: float = 0.20
    short_ec_boost_max: float = 1.0
    short_ec_horizons: int = 3


class JointProbabilisticLatentFactorFMEC(nn.Module):
    """262b-v0: 262a joint latent FM with a bounded history-mean EC baseline."""

    def __init__(self, cfg: JointProbabilisticLatentFactorFMECConfig):
        super().__init__()
        self.cfg = cfg

        enc_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            extra_features=0,
            gru_hidden_dim=cfg.encoder_hidden,
            bottleneck_dim=cfg.bottleneck_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=cfg.cond_aug_sigma,
        )
        self.encoder = GRUEncoder(enc_cfg)

        self.future_pos = nn.Embedding(cfg.future_len, cfg.future_pos_embed)
        self.cond_token_proj = nn.Sequential(
            nn.Linear(cfg.bottleneck_dim + cfg.future_pos_embed, cfg.model_hidden),
            nn.GELU(),
            nn.Linear(cfg.model_hidden, cfg.model_hidden),
        )
        self.cond_backbone = TemporalConvTower(
            channels=cfg.model_hidden,
            layers=cfg.cond_layers,
            kernel_size=cfg.kernel_size,
            dilation=cfg.dilation,
            dropout=cfg.model_dropout,
        )
        self.loadings_head = _mlp(
            cfg.bottleneck_dim,
            cfg.n_cells * cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.mu_head = _mlp(
            cfg.model_hidden,
            cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.sigma_head = _mlp(
            cfg.model_hidden,
            cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.idio_head = _mlp(
            cfg.model_hidden + cfg.n_cells,
            cfg.n_cells,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.idio_budget_head = _mlp(
            cfg.model_hidden + cfg.n_cells,
            1,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.ec_gain_head = _mlp(
            cfg.bottleneck_dim,
            cfg.n_cells,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.short_ec_head = _mlp(
            cfg.bottleneck_dim,
            1,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )

        self.flow_time = nn.Linear(1, cfg.flow_time_embed)
        self.eps_input_proj = nn.Linear(cfg.latent_dim, cfg.model_hidden)
        self.mu_proj = nn.Linear(cfg.latent_dim, cfg.model_hidden)
        self.sigma_proj = nn.Linear(cfg.latent_dim, cfg.model_hidden)
        self.ctx_proj = nn.Linear(cfg.model_hidden, cfg.model_hidden)
        self.flow_token_proj = nn.Sequential(
            nn.Linear(
                4 * cfg.model_hidden + cfg.future_pos_embed + cfg.flow_time_embed,
                cfg.model_hidden,
            ),
            nn.GELU(),
            nn.Linear(cfg.model_hidden, cfg.model_hidden),
        )
        self.flow_backbone = TemporalConvTower(
            channels=cfg.model_hidden,
            layers=cfg.flow_layers,
            kernel_size=cfg.kernel_size,
            dilation=cfg.dilation,
            dropout=cfg.model_dropout,
        )
        self.flow_head = _mlp(
            cfg.model_hidden,
            cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self._init_heads()

    def _init_heads(self) -> None:
        for net in [
            self.loadings_head,
            self.mu_head,
            self.sigma_head,
            self.idio_head,
            self.idio_budget_head,
            self.ec_gain_head,
            self.short_ec_head,
            self.flow_head,
        ]:
            last = net[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.bias.zero_()

    @staticmethod
    def _flatten_history(history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            return history.view(history.shape[0], history.shape[1], -1)
        return history

    def compute_change_scale(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten_history(history_norm)
        hist_change = history_norm[:, 1:] - history_norm[:, :-1]
        scale = hist_change.pow(2).mean(dim=1, keepdim=True).sqrt()
        return scale.clamp_min(self.cfg.change_scale_eps)

    def transform_change(self, raw_change: torch.Tensor, history_norm: torch.Tensor) -> torch.Tensor:
        if self.cfg.change_coord == "raw":
            return raw_change
        if self.cfg.change_coord == "asinh_local_scale":
            scale = self.compute_change_scale(history_norm)
            return torch.asinh(raw_change / scale)
        raise ValueError(f"Unknown change_coord={self.cfg.change_coord}")

    def inverse_transform_change(self, model_change: torch.Tensor, history_norm: torch.Tensor) -> torch.Tensor:
        if self.cfg.change_coord == "raw":
            return model_change
        if self.cfg.change_coord == "asinh_local_scale":
            scale = self.compute_change_scale(history_norm)
            if model_change.ndim == 4:
                scale = scale.unsqueeze(1)
            return torch.sinh(model_change) * scale
        raise ValueError(f"Unknown change_coord={self.cfg.change_coord}")

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten_history(history_norm)
        return self.encoder(history_norm)

    def compute_factor_pinv(self, loadings: torch.Tensor) -> torch.Tensor:
        lt = loadings.transpose(1, 2)
        gram = torch.bmm(lt, loadings)
        eye = torch.eye(self.cfg.latent_dim, device=loadings.device, dtype=loadings.dtype).unsqueeze(0)
        gram = gram + self.cfg.pinv_ridge * eye
        return torch.linalg.solve(gram, lt)

    def decode_common(self, loadings: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        if factors.ndim == 3:
            return torch.einsum("bdl,btl->btd", loadings, factors)
        return torch.einsum("bdl,bktl->bktd", loadings, factors)

    def compute_anchor(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten_history(history_norm)
        return history_norm.mean(dim=1)

    def error_correction_baseline(
        self,
        prev_level: torch.Tensor,
        cond: dict[str, torch.Tensor],
        *,
        step_idx: int = 0,
    ) -> torch.Tensor:
        anchor = cond["anchor"]
        ec_gain = cond["ec_gain"]
        if prev_level.ndim == 3:
            anchor = anchor.unsqueeze(1)
            ec_gain = ec_gain.unsqueeze(1)
        boost = 1.0
        if self.cfg.short_ec_boost_max > 0.0 and self.cfg.short_ec_horizons > 0:
            decay = max(0.0, 1.0 - float(step_idx) / float(self.cfg.short_ec_horizons))
            short_boost = cond["short_ec_boost"]
            if prev_level.ndim == 3:
                short_boost = short_boost.unsqueeze(1)
            boost = 1.0 + decay * short_boost
        return boost * ec_gain * (anchor - prev_level)

    def residualize_raw_change(
        self,
        raw_change: torch.Tensor,
        history_norm: torch.Tensor,
        future_levels_norm: torch.Tensor,
        cond: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        history_norm = self._flatten_history(history_norm)
        future_levels_norm = self._flatten_history(future_levels_norm)
        cond = self.condition(history_norm) if cond is None else cond
        prev = history_norm[:, -1, :]
        residuals = []
        for t in range(raw_change.shape[1]):
            baseline = self.error_correction_baseline(prev, cond, step_idx=t)
            residuals.append(raw_change[:, t, :] - baseline)
            prev = future_levels_norm[:, t, :]
        return torch.stack(residuals, dim=1)

    def compose_raw_change(
        self,
        residual_change: torch.Tensor,
        history_norm: torch.Tensor,
        cond: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        history_norm = self._flatten_history(history_norm)
        cond = self.condition(history_norm) if cond is None else cond
        prev = history_norm[:, -1, :]
        if residual_change.ndim == 4:
            prev = prev.unsqueeze(1).expand(-1, residual_change.shape[1], -1)
        pieces = []
        for t in range(residual_change.shape[-2]):
            baseline = self.error_correction_baseline(prev, cond, step_idx=t)
            total = residual_change[..., t, :] + baseline
            pieces.append(total)
            prev = prev + total
        return torch.stack(pieces, dim=-2)

    def condition(self, history_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        history_norm = self._flatten_history(history_norm)
        bsz = history_norm.shape[0]
        h = self.encode_history(history_norm)
        pos_idx = torch.arange(self.cfg.future_len, device=history_norm.device)
        pos = self.future_pos(pos_idx)[None, :, :].expand(bsz, -1, -1)
        h_rep = h[:, None, :].expand(-1, self.cfg.future_len, -1)
        cond_tokens = self.cond_token_proj(torch.cat([h_rep, pos], dim=-1))
        ctx = self.cond_backbone(cond_tokens)

        loadings = self.loadings_head(h).view(bsz, self.cfg.n_cells, self.cfg.latent_dim)
        mu = self.mu_head(ctx)
        sigma = self.cfg.sigma_floor + F.softplus(self.sigma_head(ctx))
        mean_common = self.decode_common(loadings, mu)
        common_rms = mean_common.pow(2).mean(dim=-1, keepdim=True).sqrt()
        idio_ctx = torch.cat([ctx, mean_common], dim=-1)
        idio_budget = self.cfg.max_idio_ratio * torch.sigmoid(self.idio_budget_head(idio_ctx)) * common_rms
        mean_idio = torch.tanh(self.idio_head(idio_ctx)) * idio_budget

        anchor = self.compute_anchor(history_norm)
        ec_gain = self.cfg.ec_gain_max * torch.sigmoid(self.ec_gain_head(h))
        short_ec_boost = self.cfg.short_ec_boost_max * torch.sigmoid(self.short_ec_head(h))
        pinv = self.compute_factor_pinv(loadings)
        return {
            "h": h,
            "ctx": ctx,
            "loadings": loadings,
            "pinv": pinv,
            "mu": mu,
            "sigma": sigma,
            "mean_common": mean_common,
            "mean_idio": mean_idio,
            "anchor": anchor,
            "ec_gain": ec_gain,
            "short_ec_boost": short_ec_boost,
        }

    def target_factors(
        self,
        target_coord: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pinv = cond["pinv"].detach()
        mu = cond["mu"].detach()
        sigma = cond["sigma"].detach().clamp_min(self.cfg.sigma_floor)
        mean_idio = cond["mean_idio"].detach()
        common_target = target_coord - mean_idio
        z_target = torch.einsum("bld,btd->btl", pinv, common_target)
        eps_target = (z_target - mu) / sigma
        return z_target, eps_target

    def velocity(
        self,
        x_t: torch.Tensor,
        flow_t: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if flow_t.ndim == 1:
            flow_t = flow_t[:, None]
        bsz, steps, _ = x_t.shape
        pos_idx = torch.arange(steps, device=x_t.device)
        pos = self.future_pos(pos_idx)[None, :, :].expand(bsz, -1, -1)
        flow_emb = self.flow_time(flow_t[:, None]).expand(-1, steps, -1)
        eps_feat = self.eps_input_proj(x_t)
        mu_feat = self.mu_proj(cond["mu"])
        sigma_feat = self.sigma_proj(torch.log(cond["sigma"]))
        ctx_feat = self.ctx_proj(cond["ctx"])
        tokens = self.flow_token_proj(torch.cat([eps_feat, mu_feat, sigma_feat, ctx_feat, pos, flow_emb], dim=-1))
        hidden = self.flow_backbone(tokens)
        return self.flow_head(hidden)

    def expand_condition(self, cond: dict[str, torch.Tensor], repeat: int) -> dict[str, torch.Tensor]:
        return {k: v.repeat_interleave(repeat, dim=0) for k, v in cond.items()}

    def sample_eps_paths(
        self,
        history_norm: torch.Tensor,
        n_samples: int,
        *,
        cond: dict[str, torch.Tensor] | None = None,
        ode_steps: int | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        history_norm = self._flatten_history(history_norm)
        bsz = history_norm.shape[0]
        steps = self.cfg.future_len
        dims = self.cfg.latent_dim
        n_ode = int(ode_steps or self.cfg.ode_steps)
        cond = self.condition(history_norm) if cond is None else cond
        cond_k = self.expand_condition(cond, n_samples)
        if x_init is None:
            x = torch.randn(bsz * n_samples, steps, dims, device=history_norm.device, dtype=history_norm.dtype)
        else:
            if x_init.ndim == 3:
                x = x_init.repeat_interleave(n_samples, dim=0)
            elif x_init.ndim == 4:
                x = x_init.reshape(bsz * n_samples, steps, dims)
            else:
                raise ValueError(f"Unexpected x_init shape: {tuple(x_init.shape)}")
        dt = 1.0 / float(n_ode)
        for i in range(n_ode):
            t = torch.full((bsz * n_samples,), float(i) / float(n_ode), device=x.device, dtype=x.dtype)
            v = self.velocity(x, t, cond_k)
            x = x + dt * v
        return x.view(bsz, n_samples, steps, dims)

    def decode_residual_change_coord(
        self,
        eps: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if eps.ndim == 3:
            factors = cond["mu"] + cond["sigma"] * eps
            return self.decode_common(cond["loadings"], factors) + cond["mean_idio"]
        factors = cond["mu"].unsqueeze(1) + cond["sigma"].unsqueeze(1) * eps
        return self.decode_common(cond["loadings"], factors) + cond["mean_idio"].unsqueeze(1)

    def deterministic_center_path(
        self,
        history_norm: torch.Tensor,
        cond: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_norm = self._flatten_history(history_norm)
        cond = self.condition(history_norm) if cond is None else cond
        zero_eps = torch.zeros(
            history_norm.shape[0],
            self.cfg.future_len,
            self.cfg.latent_dim,
            device=history_norm.device,
            dtype=history_norm.dtype,
        )
        residual_coord = self.decode_residual_change_coord(zero_eps, cond)
        residual_raw = self.inverse_transform_change(residual_coord, history_norm)
        raw_change = self.compose_raw_change(residual_raw, history_norm, cond=cond)
        last_level = history_norm[:, -1:, :]
        levels_norm = (last_level + torch.cumsum(raw_change, dim=1)).clamp(-1.0, 1.0)
        return raw_change, levels_norm

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
        history_norm = self._flatten_history(history_norm)
        bsz = history_norm.shape[0]
        cond = self.condition(history_norm)
        last_level = history_norm[:, -1:, :].unsqueeze(1)
        outs: list[torch.Tensor] = []
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            eps = self.sample_eps_paths(history_norm, n_samples=k, cond=cond)
            residual_coord = self.decode_residual_change_coord(eps, cond)
            residual_raw = self.inverse_transform_change(residual_coord, history_norm)
            raw_change = self.compose_raw_change(residual_raw, history_norm, cond=cond)
            levels_norm = (last_level + torch.cumsum(raw_change, dim=2)).clamp(-1.0, 1.0)
            levels_01 = denormalize_iv(levels_norm)
            if self.cfg.n_cells == 25:
                levels_01 = levels_01.view(bsz, k, self.cfg.future_len, 5, 5)
            outs.append(levels_01)
        return torch.cat(outs, dim=1)

    def ortho_penalty(self, loadings: torch.Tensor) -> torch.Tensor:
        gram = torch.einsum("bdl,bdm->blm", loadings, loadings) / float(self.cfg.n_cells)
        eye = torch.eye(self.cfg.latent_dim, device=loadings.device, dtype=loadings.dtype)
        return (gram - eye).pow(2).mean()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[JointProbabilisticLatentFactorFMEC, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = JointProbabilisticLatentFactorFMECConfig(**payload["config"])
    model = JointProbabilisticLatentFactorFMEC(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: JointProbabilisticLatentFactorFMEC,
    cfg: JointProbabilisticLatentFactorFMECConfig,
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
