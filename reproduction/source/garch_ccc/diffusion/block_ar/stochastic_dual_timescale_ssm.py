"""
258a-v0: stochastic dual-timescale latent state-space generator.

Purpose:
    Replace the static latent-token decoder family with dynamic stochastic latent
    states over the forecast horizon while preserving a low-rank readout and bounded
    idiosyncratic path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.dual_timescale_low_rank_temporal import TemporalConvTower, _mlp
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder


@dataclass
class StochasticDualTimescaleSSMConfig:
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

    max_idio_ratio: float = 0.30
    max_slow_noise: float = 0.08
    max_fast_noise: float = 0.16
    support_lo: float = 0.01
    support_hi: float = 1.0


class StochasticDualTimescaleSSM(nn.Module):
    def __init__(self, cfg: StochasticDualTimescaleSSMConfig):
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

        self.loadings_head = _mlp(
            cfg.bottleneck_dim, cfg.n_cells * cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.slow_init_head = _mlp(
            cfg.bottleneck_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.fast_init_head = _mlp(
            cfg.bottleneck_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.slow_scale_head = _mlp(
            cfg.bottleneck_dim + cfg.token_hidden + cfg.time_embed_dim,
            cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.fast_scale_head = _mlp(
            cfg.bottleneck_dim + 2 * cfg.token_hidden + cfg.time_embed_dim + cfg.latent_dim,
            cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.slow_cell = nn.GRUCell(cfg.token_hidden, cfg.latent_dim)
        self.fast_input_proj = nn.Linear(cfg.token_hidden + cfg.latent_dim, cfg.token_hidden)
        self.fast_cell = nn.GRUCell(cfg.token_hidden, cfg.latent_dim)
        self.fast_gate_head = _mlp(
            cfg.bottleneck_dim + 2 * cfg.token_hidden + cfg.time_embed_dim + 2 * cfg.latent_dim,
            1,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )

        self.obs_proj = nn.Sequential(
            nn.Linear(cfg.n_cells, cfg.obs_hidden),
            nn.GELU(),
            nn.Linear(cfg.obs_hidden, cfg.obs_hidden),
        )
        self.idio_head = _mlp(
            cfg.bottleneck_dim + 2 * cfg.token_hidden + cfg.obs_hidden + cfg.time_embed_dim + 2 * cfg.latent_dim,
            cfg.n_cells,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.idio_budget_head = _mlp(
            cfg.bottleneck_dim + 2 * cfg.token_hidden + cfg.obs_hidden + cfg.time_embed_dim + 2 * cfg.latent_dim,
            1,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )

        self._init_heads()

    def _init_heads(self) -> None:
        nets = [
            self.loadings_head,
            self.slow_init_head,
            self.fast_init_head,
            self.slow_scale_head,
            self.fast_scale_head,
            self.fast_gate_head,
            self.idio_head,
            self.idio_budget_head,
        ]
        for net in nets:
            last = net[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.bias.zero_()

    @staticmethod
    def _flatten_history(history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            return history.view(history.shape[0], history.shape[1], -1)
        return history

    def _build_features(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        history = self._flatten_history(history)
        B = history.shape[0]
        h = self.encoder(history)
        t_idx = torch.arange(self.cfg.future_len, device=history.device)
        t_emb = self.time_embed(t_idx).unsqueeze(0).expand(B, -1, -1)
        h_rep = h.unsqueeze(1).expand(-1, self.cfg.future_len, -1)
        tokens = self.token_proj(torch.cat([h_rep, t_emb], dim=-1))
        slow_feat = self.slow_backbone(tokens)
        fast_feat = self.fast_backbone(tokens)
        loadings = self.loadings_head(h).view(B, self.cfg.n_cells, self.cfg.latent_dim)
        return {
            "history": history,
            "h": h,
            "h_rep": h_rep,
            "t_emb": t_emb,
            "slow_feat": slow_feat,
            "fast_feat": fast_feat,
            "loadings": loadings,
        }

    def _rollout_from_noise(
        self,
        history: torch.Tensor,
        features: dict[str, torch.Tensor],
        eps_slow: torch.Tensor,
        eps_fast: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history = self._flatten_history(history)
        B = history.shape[0]
        h = features["h"]
        h_rep = features["h_rep"]
        t_emb = features["t_emb"]
        slow_feat = features["slow_feat"]
        fast_feat = features["fast_feat"]
        loadings = features["loadings"]

        z_slow = self.slow_init_head(h)
        z_fast = self.fast_init_head(h)
        current = 0.5 * (history[:, -1, :] + 1.0)
        current = current.clamp(self.cfg.support_lo, self.cfg.support_hi)

        level_steps = []
        change_steps = []
        common_steps = []
        idio_steps = []
        fast_gate_steps = []
        slow_state_steps = []
        fast_state_steps = []
        slow_scale_steps = []
        fast_scale_steps = []

        for t in range(self.cfg.future_len):
            slow_ctx = torch.cat([h_rep[:, t], slow_feat[:, t], t_emb[:, t]], dim=-1)
            slow_scale = self.cfg.max_slow_noise * torch.sigmoid(self.slow_scale_head(slow_ctx))
            z_slow = self.slow_cell(slow_feat[:, t], z_slow) + slow_scale * eps_slow[:, t]

            fast_in = self.fast_input_proj(torch.cat([fast_feat[:, t], z_slow], dim=-1))
            fast_ctx = torch.cat([h_rep[:, t], slow_feat[:, t], fast_feat[:, t], t_emb[:, t], z_slow], dim=-1)
            fast_scale = self.cfg.max_fast_noise * torch.sigmoid(self.fast_scale_head(fast_ctx))
            z_fast = self.fast_cell(fast_in, z_fast) + fast_scale * eps_fast[:, t]
            fast_gate = torch.sigmoid(self.fast_gate_head(torch.cat([fast_ctx, z_fast], dim=-1)))

            common_latent = z_slow + fast_gate * z_fast
            common = torch.einsum("bdl,bl->bd", loadings, common_latent)

            obs = self.obs_proj(current)
            idio_ctx = torch.cat([h_rep[:, t], slow_feat[:, t], fast_feat[:, t], obs, t_emb[:, t], z_slow, z_fast], dim=-1)
            common_rms = common.pow(2).mean(dim=1, keepdim=True).sqrt()
            idio_budget = self.cfg.max_idio_ratio * torch.sigmoid(self.idio_budget_head(idio_ctx)) * common_rms
            idio = torch.tanh(self.idio_head(idio_ctx)) * idio_budget

            delta = common + idio
            current = (current + delta).clamp(self.cfg.support_lo, self.cfg.support_hi)

            level_steps.append(current)
            change_steps.append(delta)
            common_steps.append(common)
            idio_steps.append(idio)
            fast_gate_steps.append(fast_gate)
            slow_state_steps.append(z_slow)
            fast_state_steps.append(z_fast)
            slow_scale_steps.append(slow_scale)
            fast_scale_steps.append(fast_scale)

        levels = torch.stack(level_steps, dim=1)
        aux = {
            "loadings": loadings,
            "mean_change": torch.stack(change_steps, dim=1),
            "mean_common": torch.stack(common_steps, dim=1),
            "mean_idio": torch.stack(idio_steps, dim=1),
            "fast_gate": torch.stack(fast_gate_steps, dim=1),
            "slow_state": torch.stack(slow_state_steps, dim=1),
            "fast_state": torch.stack(fast_state_steps, dim=1),
            "slow_scale": torch.stack(slow_scale_steps, dim=1),
            "fast_scale": torch.stack(fast_scale_steps, dim=1),
        }
        return levels, aux

    def draw_samples(self, history: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self._build_features(history)
        B = features["history"].shape[0]
        rep_features = {}
        for k, v in features.items():
            if k == "history":
                rep_features[k] = v.unsqueeze(1).expand(B, n_samples, *v.shape[1:]).reshape(B * n_samples, *v.shape[1:])
            else:
                rep_features[k] = v.unsqueeze(1).expand(B, n_samples, *v.shape[1:]).reshape(B * n_samples, *v.shape[1:])
        history_rep = rep_features["history"]
        eps_slow = torch.randn(
            B * n_samples, self.cfg.future_len, self.cfg.latent_dim, device=history_rep.device, dtype=history_rep.dtype
        )
        eps_fast = torch.randn_like(eps_slow)
        levels, aux = self._rollout_from_noise(history_rep, rep_features, eps_slow, eps_fast)
        levels = levels.view(B, n_samples, self.cfg.future_len, self.cfg.n_cells)
        # summarize batch-independent features on the base batch
        base_aux = {
            "loadings": features["loadings"],
        }
        for k in ["mean_change", "mean_common", "mean_idio", "fast_gate", "slow_state", "fast_state", "slow_scale", "fast_scale"]:
            v = aux[k].view(B, n_samples, *aux[k].shape[1:])
            base_aux[k] = v.mean(dim=1)
        return levels, base_aux

    def forward(self, history: torch.Tensor, n_samples: int = 1, **_ignored_kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        samples, aux = self.draw_samples(history, n_samples=max(1, n_samples))
        return samples.mean(dim=1), aux

    @torch.no_grad()
    def sample_batched(self, history: torch.Tensor, n_samples: int = 48, **_ignored_kwargs) -> torch.Tensor:
        history_ndim = history.ndim
        orig_h = history.shape[-2] if history_ndim == 4 else None
        orig_w = history.shape[-1] if history_ndim == 4 else None
        history = self._flatten_history(history)
        samples, _ = self.draw_samples(history, n_samples=n_samples)
        if history_ndim == 4 and orig_h is not None and orig_w is not None:
            return samples.view(samples.shape[0], samples.shape[1], self.cfg.future_len, orig_h, orig_w)
        return samples

    def orthogonality_penalty(self, loadings: torch.Tensor) -> torch.Tensor:
        gram = torch.matmul(loadings.transpose(1, 2), loadings)
        eye = torch.eye(self.cfg.latent_dim, device=loadings.device, dtype=loadings.dtype).unsqueeze(0)
        return (gram - eye).pow(2).mean()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[StochasticDualTimescaleSSM, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = StochasticDualTimescaleSSMConfig(**payload["config"])
    model = StochasticDualTimescaleSSM(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def config_to_dict(cfg: StochasticDualTimescaleSSMConfig) -> dict:
    return asdict(cfg)
