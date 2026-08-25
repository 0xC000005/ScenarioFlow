from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.dual_timescale_low_rank_temporal import TemporalConvTower, _mlp
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class MinimalFactorFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    latent_dim: int = 8

    encoder_hidden: int = 64
    bottleneck_dim: int = 128
    encoder_dropout: float = 0.1
    cond_aug_sigma: float = 0.0

    model_hidden: int = 128
    model_layers: int = 4
    kernel_size: int = 5
    dilation: int = 2
    model_dropout: float = 0.1
    flow_time_embed: int = 16
    future_pos_embed: int = 16

    head_hidden: int = 256
    head_layers: int = 2
    head_dropout: float = 0.1

    max_idio_ratio: float = 0.25
    support_lo: float = 0.01
    support_hi: float = 1.0
    ode_steps: int = 16
    ortho_reg_weight: float = 0.0
    change_coord: str = "raw"  # "raw" | "asinh_local_scale"
    change_scale_eps: float = 1e-3
    ec_anchor_mode: str = "none"  # "none" | "history_mean" | "learned_history_residual"
    ec_gain_max: float = 0.0
    anchor_delta_mult: float = 0.0
    short_ec_boost_max: float = 0.0
    short_ec_horizons: int = 0


class MinimalFactorFM(nn.Module):
    """260a-v0: minimal conditional flow-matching baseline in change space.

    The generative core is standard conditional flow matching over future change paths.
    Structural bias lives only in:
    - GRU history encoder
    - explicit low-rank readout Lambda(h)
    - bounded idiosyncratic residual path
    """

    def __init__(self, cfg: MinimalFactorFMConfig):
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
        self.flow_time = nn.Linear(1, cfg.flow_time_embed)
        self.input_proj = nn.Linear(cfg.n_cells, cfg.model_hidden)
        self.context_proj = nn.Linear(cfg.bottleneck_dim, cfg.model_hidden)
        self.token_proj = nn.Sequential(
            nn.Linear(cfg.model_hidden + cfg.model_hidden + cfg.future_pos_embed + cfg.flow_time_embed, cfg.model_hidden),
            nn.GELU(),
            nn.Linear(cfg.model_hidden, cfg.model_hidden),
        )
        self.backbone = TemporalConvTower(
            channels=cfg.model_hidden,
            layers=cfg.model_layers,
            kernel_size=cfg.kernel_size,
            dilation=cfg.dilation,
            dropout=cfg.model_dropout,
        )
        self.factor_head = _mlp(
            cfg.model_hidden,
            cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.loading_head = _mlp(
            cfg.bottleneck_dim,
            cfg.n_cells * cfg.latent_dim,
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
        self.anchor_delta_head = _mlp(
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
        self._init_heads()

    def _init_heads(self) -> None:
        for net in [self.factor_head, self.loading_head, self.idio_head, self.idio_budget_head]:
            last = net[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.bias.zero_()
        last = self.ec_gain_head[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                last.bias.zero_()
        last = self.anchor_delta_head[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                last.bias.zero_()
        last = self.short_ec_head[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                last.bias.zero_()

    @staticmethod
    def _flatten_history(history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            return history.view(history.shape[0], history.shape[1], -1)
        return history

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten_history(history_norm)
        return self.encoder(history_norm)

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

    def compute_anchor(self, history_norm: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten_history(history_norm)
        if self.cfg.ec_anchor_mode == "none":
            return torch.zeros(
                history_norm.shape[0],
                self.cfg.n_cells,
                device=history_norm.device,
                dtype=history_norm.dtype,
            )
        if self.cfg.ec_anchor_mode == "history_mean":
            return history_norm.mean(dim=1)
        if self.cfg.ec_anchor_mode == "learned_history_residual":
            mean_anchor = history_norm.mean(dim=1)
            scale = self.compute_change_scale(history_norm).squeeze(1)
            delta = torch.tanh(self.anchor_delta_head(h)) * (self.cfg.anchor_delta_mult * scale)
            return (mean_anchor + delta).clamp(-1.0, 1.0)
        raise ValueError(f"Unknown ec_anchor_mode={self.cfg.ec_anchor_mode}")

    @staticmethod
    def _expand_path_tensor(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(1) if target.ndim == 3 else tensor

    def error_correction_baseline(
        self,
        prev_level: torch.Tensor,
        cond: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> torch.Tensor:
        if self.cfg.ec_anchor_mode == "none" or self.cfg.ec_gain_max <= 0.0:
            return torch.zeros_like(prev_level)
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
    ) -> torch.Tensor:
        history_norm = self._flatten_history(history_norm)
        future_levels_norm = self._flatten_history(future_levels_norm)
        cond = self.condition(history_norm)
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
        h = self.encode_history(history_norm)
        loadings = self.loading_head(h).view(history_norm.shape[0], self.cfg.n_cells, self.cfg.latent_dim)
        ctx = self.context_proj(h)
        anchor = self.compute_anchor(history_norm, h)
        ec_gain = self.cfg.ec_gain_max * torch.sigmoid(self.ec_gain_head(h))
        short_ec_boost = self.cfg.short_ec_boost_max * torch.sigmoid(self.short_ec_head(h))
        return {
            "h": h,
            "loadings": loadings,
            "ctx": ctx,
            "anchor": anchor,
            "ec_gain": ec_gain,
            "short_ec_boost": short_ec_boost,
        }

    def expand_condition(self, cond: dict[str, torch.Tensor], repeat: int) -> dict[str, torch.Tensor]:
        expanded: dict[str, torch.Tensor] = {}
        for k, v in cond.items():
            expanded[k] = v.repeat_interleave(repeat, dim=0)
        return expanded

    def velocity(
        self,
        x_t: torch.Tensor,
        flow_t: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """x_t: (B, T, D), flow_t: (B,) or (B,1), cond from condition()."""
        if flow_t.ndim == 1:
            flow_t = flow_t[:, None]
        B, T, D = x_t.shape
        pos_idx = torch.arange(T, device=x_t.device)
        pos = self.future_pos(pos_idx)[None, :, :].expand(B, -1, -1)
        flow_emb = self.flow_time(flow_t[:, None]).expand(-1, T, -1)
        x_feat = self.input_proj(x_t)
        h_feat = cond["ctx"][:, None, :].expand(-1, T, -1)
        tokens = self.token_proj(torch.cat([x_feat, h_feat, pos, flow_emb], dim=-1))
        hidden = self.backbone(tokens)

        factor_state = self.factor_head(hidden)
        common = torch.einsum("bdl,btl->btd", cond["loadings"], factor_state)
        common_rms = common.pow(2).mean(dim=-1, keepdim=True).sqrt()

        idio_ctx = torch.cat([hidden, x_t], dim=-1)
        idio_budget = self.cfg.max_idio_ratio * torch.sigmoid(self.idio_budget_head(idio_ctx)) * common_rms
        idio = torch.tanh(self.idio_head(idio_ctx)) * idio_budget
        velocity = common + idio

        aux = {
            "common": common,
            "idio": idio,
            "idios_budget": idio_budget,
            "factor_state": factor_state,
            "loadings": cond["loadings"],
        }
        return velocity, aux

    def sample_change_paths(
        self,
        history_norm: torch.Tensor,
        n_samples: int,
        ode_steps: int | None = None,
        x_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten_history(history_norm)
        B = history_norm.shape[0]
        T = self.cfg.future_len
        D = self.cfg.n_cells
        steps = int(ode_steps or self.cfg.ode_steps)
        cond = self.condition(history_norm)
        cond_k = self.expand_condition(cond, n_samples)
        if x_init is None:
            x = torch.randn(B * n_samples, T, D, device=history_norm.device, dtype=history_norm.dtype)
        else:
            if x_init.ndim == 3:
                x = x_init.repeat_interleave(n_samples, dim=0)
            elif x_init.ndim == 4:
                x = x_init.reshape(B * n_samples, T, D)
            else:
                raise ValueError(f"Unexpected x_init shape: {tuple(x_init.shape)}")
        dt = 1.0 / float(steps)
        last_aux: dict[str, torch.Tensor] | None = None
        for i in range(steps):
            t = torch.full((B * n_samples,), float(i) / float(steps), device=x.device, dtype=x.dtype)
            v, last_aux = self.velocity(x, t, cond_k)
            x = x + dt * v
        assert last_aux is not None
        return x.view(B, n_samples, T, D), last_aux

    def deterministic_center_path(
        self,
        history_norm: torch.Tensor,
        ode_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_norm = self._flatten_history(history_norm)
        zeros = torch.zeros(
            history_norm.shape[0],
            self.cfg.future_len,
            self.cfg.n_cells,
            device=history_norm.device,
            dtype=history_norm.dtype,
        )
        residual_change, _ = self.sample_change_paths(
            history_norm,
            n_samples=1,
            ode_steps=ode_steps,
            x_init=zeros,
        )
        residual_change = residual_change[:, 0]
        raw_change = self.inverse_transform_change(residual_change, history_norm)
        raw_change = self.compose_raw_change(raw_change, history_norm)
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
        B = history_norm.shape[0]
        outs: list[torch.Tensor] = []
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            change_coord, _ = self.sample_change_paths(history_norm, n_samples=k)
            change_norm = self.inverse_transform_change(change_coord, history_norm)
            change_norm = self.compose_raw_change(change_norm, history_norm)
            last_level = history_norm[:, -1:, :].unsqueeze(1)
            levels_norm = last_level + torch.cumsum(change_norm, dim=2)
            levels_norm = levels_norm.clamp(-1.0, 1.0)
            levels_01 = denormalize_iv(levels_norm)
            if self.cfg.n_cells == 25:
                levels_01 = levels_01.view(B, k, self.cfg.future_len, 5, 5)
            outs.append(levels_01)
        return torch.cat(outs, dim=1)

    def forward(
        self,
        history_norm: torch.Tensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten_history(history_norm)
        change_coord, aux = self.sample_change_paths(history_norm, n_samples=n_samples)
        change_norm = self.inverse_transform_change(change_coord, history_norm)
        change_norm = self.compose_raw_change(change_norm, history_norm)
        last_level = history_norm[:, -1:, :].unsqueeze(1)
        levels_norm = last_level + torch.cumsum(change_norm, dim=2)
        levels_norm = levels_norm.clamp(-1.0, 1.0)
        levels_01 = denormalize_iv(levels_norm)
        return levels_01, aux

    def ortho_penalty(self, loadings: torch.Tensor) -> torch.Tensor:
        # loadings: (B, D, L)
        gram = torch.einsum("bdl,bdm->blm", loadings, loadings) / float(self.cfg.n_cells)
        eye = torch.eye(self.cfg.latent_dim, device=loadings.device, dtype=loadings.dtype)
        return (gram - eye).pow(2).mean()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[MinimalFactorFM, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = MinimalFactorFMConfig(**payload["config"])
    model = MinimalFactorFM(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: MinimalFactorFM,
    cfg: MinimalFactorFMConfig,
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
