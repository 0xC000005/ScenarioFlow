"""
254a-v0: deterministic dual-timescale temporal backbone with low-rank readout.

Purpose:
    First prototype after the 253 family was judged structurally capped.
    Separate slow mean-reverting structure and fast event-like dynamics in the
    temporal backbone itself, while preserving low-rank common structure and
    bounded idiosyncratic corrections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder


@dataclass
class DualTimescaleTemporalConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    latent_dim: int = 8

    encoder_hidden: int = 64
    bottleneck_dim: int = 128
    encoder_dropout: float = 0.1
    cond_aug_sigma: float = 0.0

    token_hidden: int = 96
    slow_layers: int = 3
    fast_layers: int = 2
    slow_kernel: int = 7
    fast_kernel: int = 3
    fast_dilation: int = 2
    branch_dropout: float = 0.1

    head_hidden: int = 256
    head_layers: int = 2
    head_dropout: float = 0.1

    obs_hidden: int = 64
    time_embed_dim: int = 16

    max_idio_ratio: float = 0.60
    ec_max_strength: float = 0.25
    support_lo: float = 0.01
    support_hi: float = 1.0


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)]
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


class TemporalResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.dropout(y)
        return x + y


class TemporalConvTower(nn.Module):
    def __init__(self, channels: int, layers: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        for i in range(layers):
            d = dilation**i if dilation > 1 else 1
            blocks.append(TemporalResBlock(channels, kernel_size, d, dropout))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        y = x.transpose(1, 2)
        for block in self.blocks:
            y = block(y)
        return y.transpose(1, 2)


class DualTimescaleLowRankTemporal(nn.Module):
    def __init__(self, cfg: DualTimescaleTemporalConfig):
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
        self.time_embed = nn.Embedding(cfg.future_len, cfg.time_embed_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(cfg.bottleneck_dim + cfg.time_embed_dim, cfg.token_hidden),
            nn.GELU(),
            nn.Linear(cfg.token_hidden, cfg.token_hidden),
        )

        self.slow_backbone = TemporalConvTower(
            channels=cfg.token_hidden,
            layers=cfg.slow_layers,
            kernel_size=cfg.slow_kernel,
            dilation=1,
            dropout=cfg.branch_dropout,
        )
        self.fast_backbone = TemporalConvTower(
            channels=cfg.token_hidden,
            layers=cfg.fast_layers,
            kernel_size=cfg.fast_kernel,
            dilation=cfg.fast_dilation,
            dropout=cfg.branch_dropout,
        )

        self.loading_head = _mlp(
            cfg.bottleneck_dim, cfg.n_cells * cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.slow_latent_head = _mlp(
            cfg.token_hidden, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.fast_latent_head = _mlp(
            cfg.token_hidden, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.fast_gate_head = _mlp(
            cfg.bottleneck_dim + 2 * cfg.token_hidden + cfg.time_embed_dim,
            1,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.latent_gate_head = _mlp(
            cfg.bottleneck_dim + 2 * cfg.token_hidden + cfg.time_embed_dim,
            cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )

        self.obs_proj = nn.Sequential(
            nn.Linear(cfg.n_cells, cfg.obs_hidden),
            nn.GELU(),
            nn.Linear(cfg.obs_hidden, cfg.obs_hidden),
        )
        idio_ctx_dim = cfg.bottleneck_dim + 2 * cfg.token_hidden + cfg.obs_hidden + cfg.time_embed_dim
        self.idio_head = _mlp(idio_ctx_dim, cfg.n_cells, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)
        self.idio_budget_head = _mlp(idio_ctx_dim, 1, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)

        ec_ctx_dim = cfg.bottleneck_dim + cfg.obs_hidden + cfg.time_embed_dim
        self.anchor_head = _mlp(
            cfg.bottleneck_dim + cfg.time_embed_dim,
            cfg.n_cells,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.ec_gate = _mlp(ec_ctx_dim, 1, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)

        self._init_heads()

    def _init_heads(self) -> None:
        for head in [
            self.loading_head,
            self.slow_latent_head,
            self.fast_latent_head,
            self.fast_gate_head,
            self.latent_gate_head,
            self.idio_head,
            self.idio_budget_head,
            self.anchor_head,
            self.ec_gate,
        ]:
            last = head[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    scale = 0.1 if head in [self.loading_head, self.slow_latent_head, self.fast_latent_head] else 0.05
                    last.weight.mul_(scale)
                    last.bias.zero_()
        ec_last = self.ec_gate[-1]
        if isinstance(ec_last, nn.Linear):
            with torch.no_grad():
                ec_last.weight.mul_(0.01)
                ec_last.bias.fill_(-3.0)

    @staticmethod
    def _flatten_history(history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            return history.view(history.shape[0], history.shape[1], -1)
        return history

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history)

    def predict_mean(self, history: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history = self._flatten_history(history)
        B, _, D = history.shape
        assert D == self.cfg.n_cells, f"history last dim {D} != cfg.n_cells {self.cfg.n_cells}"

        h = self.encode_history(history)
        base_loadings = self.loading_head(h).view(B, self.cfg.n_cells, self.cfg.latent_dim)

        t_idx = torch.arange(self.cfg.future_len, device=history.device)
        t_emb = self.time_embed(t_idx).unsqueeze(0).expand(B, -1, -1)
        h_rep = h.unsqueeze(1).expand(-1, self.cfg.future_len, -1)
        tokens = self.token_proj(torch.cat([h_rep, t_emb], dim=-1))
        slow_feat = self.slow_backbone(tokens)
        fast_feat = self.fast_backbone(tokens)

        fast_ctx = torch.cat([h_rep, slow_feat, fast_feat, t_emb], dim=-1)
        fast_gate = torch.sigmoid(self.fast_gate_head(fast_ctx))
        latent_gate = 1.0 + 0.25 * torch.tanh(self.latent_gate_head(fast_ctx))
        slow_latent = self.slow_latent_head(slow_feat)
        fast_latent = self.fast_latent_head(fast_feat)
        common_latent = latent_gate * (slow_latent + fast_gate * fast_latent)
        mean_common = torch.einsum("bdl,btl->btd", base_loadings, common_latent)

        current = 0.5 * (history[:, -1, :] + 1.0)
        current = current.clamp(self.cfg.support_lo, self.cfg.support_hi)

        level_steps: list[torch.Tensor] = []
        change_steps: list[torch.Tensor] = []
        idio_steps: list[torch.Tensor] = []
        ec_steps: list[torch.Tensor] = []
        budget_steps: list[torch.Tensor] = []
        anchor_steps: list[torch.Tensor] = []

        for t in range(self.cfg.future_len):
            obs = self.obs_proj(current)
            idio_ctx = torch.cat([h, slow_feat[:, t], fast_feat[:, t], obs, t_emb[:, t]], dim=-1)
            common_t = mean_common[:, t]
            common_rms = common_t.pow(2).mean(dim=1, keepdim=True).sqrt()

            budget = (
                self.cfg.max_idio_ratio
                * torch.sigmoid(self.idio_budget_head(idio_ctx))
                * common_rms
            )
            idio = torch.tanh(self.idio_head(idio_ctx)) * budget

            ec_ctx = torch.cat([h, obs, t_emb[:, t]], dim=-1)
            anchor = self.anchor_head(torch.cat([h, t_emb[:, t]], dim=-1))
            anchor = anchor.clamp(self.cfg.support_lo, self.cfg.support_hi)
            ec_strength = self.cfg.ec_max_strength * torch.sigmoid(self.ec_gate(ec_ctx))
            ec_delta = ec_strength * (anchor - current)

            delta = common_t + idio + ec_delta
            current = (current + delta).clamp(self.cfg.support_lo, self.cfg.support_hi)

            level_steps.append(current)
            change_steps.append(delta)
            idio_steps.append(idio)
            ec_steps.append(ec_delta)
            budget_steps.append(budget)
            anchor_steps.append(anchor)

        mean_level = torch.stack(level_steps, dim=1)
        mean_change = torch.stack(change_steps, dim=1)
        mean_idio = torch.stack(idio_steps, dim=1)
        ec_term = torch.stack(ec_steps, dim=1)
        idio_budget = torch.stack(budget_steps, dim=1)
        anchors = torch.stack(anchor_steps, dim=1)

        aux = {
            "h": h,
            "base_loadings": base_loadings,
            "slow_feat": slow_feat,
            "fast_feat": fast_feat,
            "slow_latent": slow_latent,
            "fast_latent": fast_latent,
            "fast_gate": fast_gate,
            "common_latent": common_latent,
            "mean_common": mean_common,
            "mean_idio": mean_idio,
            "mean_change": mean_change,
            "ec_term": ec_term,
            "idio_budget": idio_budget,
            "anchors": anchors,
        }
        return mean_level, aux

    def forward(self, history: torch.Tensor, **_ignored_kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.predict_mean(history)

    @torch.no_grad()
    def sample(self, history: torch.Tensor, n_samples: int = 48, **_ignored_kwargs) -> torch.Tensor:
        history_ndim = history.ndim
        orig_h = history.shape[-2] if history_ndim == 4 else None
        orig_w = history.shape[-1] if history_ndim == 4 else None
        mean_level, _ = self.predict_mean(history)
        samples = mean_level.unsqueeze(1).expand(-1, n_samples, -1, -1)
        if history_ndim == 4 and orig_h is not None and orig_w is not None:
            return samples.view(samples.shape[0], samples.shape[1], samples.shape[2], orig_h, orig_w)
        return samples

    @torch.no_grad()
    def sample_batched(self, history: torch.Tensor, n_samples: int = 48, **_ignored_kwargs) -> torch.Tensor:
        return self.sample(history, n_samples=n_samples)

    def orthogonality_penalty(self, loadings: torch.Tensor) -> torch.Tensor:
        gram = torch.einsum("bdl,bdm->blm", loadings, loadings)
        eye = torch.eye(self.cfg.latent_dim, device=loadings.device, dtype=loadings.dtype)
        off = gram - gram * eye.unsqueeze(0)
        return off.pow(2).mean()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DualTimescaleLowRankTemporal, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = DualTimescaleTemporalConfig(**payload["config"])
    model = DualTimescaleLowRankTemporal(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def config_to_dict(cfg: DualTimescaleTemporalConfig) -> dict:
    return asdict(cfg)
