"""
255a-v0: deterministic sparse future-motif model in change space.

Paradigm shift after the continuous common-path families were judged capped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)]
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


@dataclass
class SparseFutureMotifConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    encoder_hidden: int = 64
    bottleneck_dim: int = 128
    encoder_dropout: float = 0.1
    cond_aug_sigma: float = 0.0

    n_motifs: int = 16
    route_hidden: int = 256
    route_layers: int = 2
    route_dropout: float = 0.1
    route_temperature: float = 1.0
    route_topk: int = 4
    route_entropy_floor: float = 0.50

    adapter_hidden: int = 128
    adapter_layers: int = 2
    adapter_dropout: float = 0.1
    obs_hidden: int = 64
    time_embed_dim: int = 16

    max_resid_ratio: float = 0.25
    support_lo: float = 0.01
    support_hi: float = 1.0


class SparseFutureMotifModel(nn.Module):
    def __init__(self, cfg: SparseFutureMotifConfig):
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
        self.route_head = _mlp(
            cfg.bottleneck_dim,
            cfg.n_motifs,
            cfg.route_hidden,
            cfg.route_layers,
            cfg.route_dropout,
        )
        self.motif_bank = nn.Parameter(torch.randn(cfg.n_motifs, cfg.future_len, cfg.n_cells) * 0.01)

        self.time_embed = nn.Embedding(cfg.future_len, cfg.time_embed_dim)
        self.obs_proj = nn.Sequential(
            nn.Linear(cfg.n_cells, cfg.obs_hidden),
            nn.GELU(),
            nn.Linear(cfg.obs_hidden, cfg.obs_hidden),
        )
        ctx_dim = cfg.bottleneck_dim + cfg.obs_hidden + cfg.time_embed_dim
        self.resid_head = _mlp(ctx_dim, cfg.n_cells, cfg.adapter_hidden, cfg.adapter_layers, cfg.adapter_dropout)
        self.resid_budget_head = _mlp(ctx_dim, 1, cfg.adapter_hidden, cfg.adapter_layers, cfg.adapter_dropout)

        self._init_heads()

    def _init_heads(self) -> None:
        for head in [self.route_head, self.resid_head, self.resid_budget_head]:
            last = head[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.mul_(0.05)
                    last.bias.zero_()

    @staticmethod
    def _flatten_history(history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            return history.view(history.shape[0], history.shape[1], -1)
        return history

    def _sparse_topk_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        k = min(self.cfg.route_topk, logits.shape[-1])
        vals, idx = torch.topk(logits, k=k, dim=-1)
        sparse = torch.full_like(logits, float("-inf"))
        sparse.scatter_(dim=-1, index=idx, src=vals / self.cfg.route_temperature)
        return torch.softmax(sparse, dim=-1)

    def predict_mean(self, history: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history = self._flatten_history(history)
        B, _, D = history.shape
        assert D == self.cfg.n_cells

        h = self.encoder(history)
        route_logits = self.route_head(h)
        motif_weights = self._sparse_topk_softmax(route_logits)
        motif_change = torch.einsum("bm,mtd->btd", motif_weights, self.motif_bank)

        current = 0.5 * (history[:, -1, :] + 1.0)
        current = current.clamp(self.cfg.support_lo, self.cfg.support_hi)

        t_idx = torch.arange(self.cfg.future_len, device=history.device)
        t_emb = self.time_embed(t_idx).unsqueeze(0).expand(B, -1, -1)

        level_steps: list[torch.Tensor] = []
        change_steps: list[torch.Tensor] = []
        resid_steps: list[torch.Tensor] = []
        budget_steps: list[torch.Tensor] = []

        for t in range(self.cfg.future_len):
            base_t = motif_change[:, t]
            obs = self.obs_proj(current)
            ctx = torch.cat([h, obs, t_emb[:, t]], dim=-1)
            base_rms = base_t.pow(2).mean(dim=1, keepdim=True).sqrt()
            budget = self.cfg.max_resid_ratio * torch.sigmoid(self.resid_budget_head(ctx)) * base_rms
            resid = torch.tanh(self.resid_head(ctx)) * budget
            delta = base_t + resid
            current = (current + delta).clamp(self.cfg.support_lo, self.cfg.support_hi)

            level_steps.append(current)
            change_steps.append(delta)
            resid_steps.append(resid)
            budget_steps.append(budget)

        mean_level = torch.stack(level_steps, dim=1)
        mean_change = torch.stack(change_steps, dim=1)
        mean_resid = torch.stack(resid_steps, dim=1)
        resid_budget = torch.stack(budget_steps, dim=1)

        aux = {
            "h": h,
            "route_logits": route_logits,
            "motif_weights": motif_weights,
            "motif_change": motif_change,
            "mean_resid": mean_resid,
            "resid_budget": resid_budget,
            "mean_change": mean_change,
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


def load_model(checkpoint_path: str, device: torch.device) -> tuple[SparseFutureMotifModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = SparseFutureMotifConfig(**payload["config"])
    model = SparseFutureMotifModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def config_to_dict(cfg: SparseFutureMotifConfig) -> dict:
    return asdict(cfg)
