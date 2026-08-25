from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.dual_timescale_low_rank_temporal import TemporalConvTower, _mlp
from diffusion.block_ar.minimal_factor_fm import MinimalFactorFM, load_model as load_260_model
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class ScaleCalibratedLatentFactorResidualFMConfig:
    base_checkpoint: str
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    latent_dim: int = 8

    model_hidden: int = 96
    model_layers: int = 2
    kernel_size: int = 5
    dilation: int = 2
    model_dropout: float = 0.1
    flow_time_embed: int = 16
    future_pos_embed: int = 16

    head_hidden: int = 128
    head_layers: int = 2
    head_dropout: float = 0.1
    ode_steps: int = 16
    pinv_ridge: float = 1e-4
    scale_floor: float = 1e-3


class ScaleCalibratedLatentFactorResidualFM(nn.Module):
    """261d-v0: latent-factor residual FM with supervised conditional scale profile."""

    def __init__(
        self,
        cfg: ScaleCalibratedLatentFactorResidualFMConfig,
        *,
        base_model: MinimalFactorFM,
    ):
        super().__init__()
        self.cfg = cfg
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad_(False)
        object.__setattr__(self, "base_model", base_model)

        self.future_pos = nn.Embedding(cfg.future_len, cfg.future_pos_embed)
        self.flow_time = nn.Linear(1, cfg.flow_time_embed)
        self.input_proj = nn.Linear(cfg.latent_dim, cfg.model_hidden)
        self.center_proj = nn.Linear(cfg.n_cells, cfg.model_hidden)
        self.ctx_proj = nn.Linear(base_model.cfg.model_hidden, cfg.model_hidden)
        self.token_proj = nn.Sequential(
            nn.Linear(
                cfg.model_hidden + cfg.model_hidden + cfg.model_hidden + cfg.future_pos_embed + cfg.flow_time_embed,
                cfg.model_hidden,
            ),
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
        self.scale_head = _mlp(
            base_model.cfg.model_hidden,
            cfg.future_len,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self._init_heads()

    def _init_heads(self) -> None:
        for net in (self.factor_head, self.scale_head):
            last = net[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.bias.zero_()

    @staticmethod
    def _flatten_history(history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            return history.view(history.shape[0], history.shape[1], -1)
        return history

    def compute_factor_pinv(self, loadings: torch.Tensor) -> torch.Tensor:
        lt = loadings.transpose(1, 2)
        gram = torch.bmm(lt, loadings)
        eye = torch.eye(self.cfg.latent_dim, device=loadings.device, dtype=loadings.dtype).unsqueeze(0)
        gram = gram + self.cfg.pinv_ridge * eye
        return torch.linalg.solve(gram, lt)

    def compute_scale_profile(self, ctx: torch.Tensor) -> torch.Tensor:
        raw = self.scale_head(ctx)
        return torch.nn.functional.softplus(raw).unsqueeze(-1) + self.cfg.scale_floor

    def condition(self, history_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        history_norm = self._flatten_history(history_norm)
        with torch.no_grad():
            base_cond = self.base_model.condition(history_norm)
            center_raw_change, center_levels_norm = self.base_model.deterministic_center_path(history_norm)
            center_coord = self.base_model.transform_change(center_raw_change, history_norm)
            pinv = self.compute_factor_pinv(base_cond["loadings"])
        scale_profile = self.compute_scale_profile(base_cond["ctx"])
        return {
            "ctx": base_cond["ctx"],
            "loadings": base_cond["loadings"],
            "pinv": pinv,
            "center_raw_change": center_raw_change,
            "center_levels_norm": center_levels_norm,
            "center_coord": center_coord,
            "scale_profile": scale_profile,
        }

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
        x_feat = self.input_proj(x_t)
        center_feat = self.center_proj(cond["center_coord"])
        ctx_feat = self.ctx_proj(cond["ctx"])[:, None, :].expand(-1, steps, -1)
        tokens = self.token_proj(torch.cat([x_feat, center_feat, ctx_feat, pos, flow_emb], dim=-1))
        hidden = self.backbone(tokens)
        return self.factor_head(hidden)

    def decode_panel_residual(self, factor_resid: torch.Tensor, loadings: torch.Tensor) -> torch.Tensor:
        if factor_resid.ndim == 3:
            return torch.einsum("bdl,btl->btd", loadings, factor_resid)
        return torch.einsum("bdl,bktl->bktd", loadings, factor_resid)

    def sample_standardized_factor_residuals(
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
        cond_k = {
            "ctx": cond["ctx"].repeat_interleave(n_samples, dim=0),
            "center_coord": cond["center_coord"].repeat_interleave(n_samples, dim=0),
        }
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
        center_coord = cond["center_coord"]
        loadings = cond["loadings"]
        scale_profile = cond["scale_profile"]
        last_level = history_norm[:, -1:, :].unsqueeze(1)

        outs: list[torch.Tensor] = []
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            factor_resid = self.sample_standardized_factor_residuals(history_norm, n_samples=k, cond=cond)
            factor_resid = factor_resid - factor_resid.mean(dim=1, keepdim=True)
            factor_resid = factor_resid * scale_profile.unsqueeze(1)
            panel_resid_coord = self.decode_panel_residual(factor_resid, loadings)
            total_coord = center_coord.unsqueeze(1) + panel_resid_coord
            raw_change = self.base_model.inverse_transform_change(total_coord, history_norm)
            levels_norm = (last_level + torch.cumsum(raw_change, dim=2)).clamp(-1.0, 1.0)
            levels_01 = denormalize_iv(levels_norm)
            if self.cfg.n_cells == 25:
                levels_01 = levels_01.view(bsz, k, self.cfg.future_len, 5, 5)
            outs.append(levels_01)
        return torch.cat(outs, dim=1)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[ScaleCalibratedLatentFactorResidualFM, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ScaleCalibratedLatentFactorResidualFMConfig(**payload["config"])
    base_model, _ = load_260_model(cfg.base_checkpoint, device)
    model = ScaleCalibratedLatentFactorResidualFM(cfg, base_model=base_model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ScaleCalibratedLatentFactorResidualFM,
    cfg: ScaleCalibratedLatentFactorResidualFMConfig,
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
