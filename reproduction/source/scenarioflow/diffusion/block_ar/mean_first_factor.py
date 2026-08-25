"""
Mean-first explicit factor model (252a).

Purpose:
    Separate deterministic center-path modeling from stochastic uncertainty modeling.
    This is the architecture-level follow-up to the 250/251 audit:
      - first close the deterministic 8/11 ceiling as far as possible
      - then add uncertainty around a frozen center path

Model family:
    Same broad non-AR low-rank family as 250/251, but no stochastic latent path.
    The model predicts a single center path in (B, T, D):

        history -> encoder -> h
        h -> Lambda(h)         in R^{T,D,L}
        h -> factor_path(h)    in R^{T,L}
        h -> mean_idio(h)      in R^{T,D}
        mean_change = einsum("btdl,btl->btd", Lambda, factor_path) + mean_idio
        mean_level  = last_obs + cumsum(mean_change, dim=1)

This preserves explicit low-rank common structure while allowing small cell-specific
mean corrections through mean_idio.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder


@dataclass
class MeanFirstFactorConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    latent_dim: int = 8

    encoder_hidden: int = 64
    bottleneck_dim: int = 128
    encoder_dropout: float = 0.1
    cond_aug_sigma: float = 0.0

    head_hidden: int = 256
    head_layers: int = 2
    head_dropout: float = 0.1

    support_lo: float = 0.01
    support_hi: float = 1.0


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)]
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


class MeanLoadingHead(nn.Module):
    def __init__(self, cfg: MeanFirstFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            in_dim=cfg.bottleneck_dim,
            out_dim=cfg.future_len * cfg.n_cells * cfg.latent_dim,
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.1)
                last.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        return self.net(h).view(B, self.cfg.future_len, self.cfg.n_cells, self.cfg.latent_dim)


class MeanFactorPathHead(nn.Module):
    def __init__(self, cfg: MeanFirstFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            in_dim=cfg.bottleneck_dim,
            out_dim=cfg.future_len * cfg.latent_dim,
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.1)
                last.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        return self.net(h).view(B, self.cfg.future_len, self.cfg.latent_dim)


class MeanIdiosyncraticHead(nn.Module):
    def __init__(self, cfg: MeanFirstFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            in_dim=cfg.bottleneck_dim,
            out_dim=cfg.future_len * cfg.n_cells,
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )
        # Keep cell-specific mean corrections small at initialization so the
        # common factor path learns first.
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.05)
                last.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        return self.net(h).view(B, self.cfg.future_len, self.cfg.n_cells)


class MeanFirstFactorModel(nn.Module):
    def __init__(self, cfg: MeanFirstFactorConfig):
        super().__init__()
        self.cfg = cfg
        encoder_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            extra_features=0,
            gru_hidden_dim=cfg.encoder_hidden,
            bottleneck_dim=cfg.bottleneck_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=cfg.cond_aug_sigma,
        )
        self.encoder = GRUEncoder(encoder_cfg)
        self.loading_head = MeanLoadingHead(cfg)
        self.factor_path_head = MeanFactorPathHead(cfg)
        self.mean_idio_head = MeanIdiosyncraticHead(cfg)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history)

    @staticmethod
    def _flatten_history(history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            return history.view(history.shape[0], history.shape[1], -1)
        return history

    def predict_mean(
        self, history: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history = self._flatten_history(history)
        B, _, D = history.shape
        assert D == self.cfg.n_cells, (
            f"history last dim {D} != cfg.n_cells {self.cfg.n_cells}"
        )

        h = self.encode_history(history)
        Lambda = self.loading_head(h)          # (B, T, D, L)
        factor_path = self.factor_path_head(h) # (B, T, L)
        mean_idio = self.mean_idio_head(h)     # (B, T, D)

        mean_factor = torch.einsum("btdl,btl->btd", Lambda, factor_path)
        mean_change = mean_factor + mean_idio

        last_iv = 0.5 * (history[:, -1:, :] + 1.0)
        last_iv = last_iv.clamp(self.cfg.support_lo, self.cfg.support_hi)
        mean_level = last_iv + torch.cumsum(mean_change, dim=1)
        mean_level = mean_level.clamp(self.cfg.support_lo, self.cfg.support_hi)

        aux = {
            "h": h,
            "Lambda": Lambda,
            "factor_path": factor_path,
            "mean_factor": mean_factor,
            "mean_idio": mean_idio,
            "mean_change": mean_change,
        }
        return mean_level, aux

    def forward(
        self,
        history: torch.Tensor,
        **_ignored_kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.predict_mean(history)

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        **_ignored_kwargs,
    ) -> torch.Tensor:
        history_ndim = history.ndim
        orig_h = history.shape[-2] if history_ndim == 4 else None
        orig_w = history.shape[-1] if history_ndim == 4 else None
        mean_level, _ = self.predict_mean(history)
        samples = mean_level.unsqueeze(1).expand(-1, n_samples, -1, -1)
        if history_ndim == 4 and orig_h is not None and orig_w is not None:
            return samples.view(samples.shape[0], samples.shape[1], samples.shape[2], orig_h, orig_w)
        return samples

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        **_ignored_kwargs,
    ) -> torch.Tensor:
        return self.sample(history, n_samples=n_samples)

    def orthogonality_penalty(self, Lambda: torch.Tensor) -> torch.Tensor:
        B, T, D, L = Lambda.shape
        Lm = Lambda.reshape(B * T, D, L)
        gram = torch.einsum("bdl, bdm -> blm", Lm, Lm)
        eye = torch.eye(L, device=Lambda.device, dtype=Lambda.dtype).expand(B * T, L, L)
        off = gram - gram * eye
        return off.pow(2).mean()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[MeanFirstFactorModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = MeanFirstFactorConfig(**payload["config"])
    model = MeanFirstFactorModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def config_to_dict(cfg: MeanFirstFactorConfig) -> dict:
    return asdict(cfg)
