"""
256a-v0: deterministic hierarchical future-token model.

Future representation is history-conditioned and tokenized rather than forced into
one continuous common path or one fixed motif library.
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
class HierarchicalFutureTokenConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    latent_dim: int = 8
    token_dim: int = 64
    n_fast_tokens: int = 4

    encoder_hidden: int = 64
    bottleneck_dim: int = 128
    encoder_dropout: float = 0.1
    cond_aug_sigma: float = 0.0

    head_hidden: int = 256
    head_layers: int = 2
    head_dropout: float = 0.1

    obs_hidden: int = 64
    time_embed_dim: int = 16

    max_resid_ratio: float = 0.30
    support_lo: float = 0.01
    support_hi: float = 1.0


class HierarchicalFutureTokenModel(nn.Module):
    def __init__(self, cfg: HierarchicalFutureTokenConfig):
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

        self.slow_token_head = _mlp(
            cfg.bottleneck_dim, cfg.token_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.fast_token_head = _mlp(
            cfg.bottleneck_dim,
            cfg.n_fast_tokens * cfg.token_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.token_type = nn.Parameter(torch.randn(1 + cfg.n_fast_tokens, cfg.token_dim) * 0.02)

        self.time_embed = nn.Embedding(cfg.future_len, cfg.time_embed_dim)
        self.obs_proj = nn.Sequential(
            nn.Linear(cfg.n_cells, cfg.obs_hidden),
            nn.GELU(),
            nn.Linear(cfg.obs_hidden, cfg.obs_hidden),
        )

        query_dim = cfg.bottleneck_dim + cfg.obs_hidden + cfg.time_embed_dim + cfg.latent_dim
        self.query_head = _mlp(query_dim, cfg.token_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)
        self.key_proj = nn.Linear(cfg.token_dim, cfg.token_dim)
        self.value_proj = nn.Linear(cfg.token_dim, cfg.token_dim)

        self.state_init = _mlp(
            cfg.bottleneck_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.state_cell = nn.GRUCell(cfg.token_dim, cfg.latent_dim)

        self.loading_head = _mlp(
            cfg.bottleneck_dim,
            cfg.n_cells * cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        factor_ctx_dim = cfg.bottleneck_dim + cfg.token_dim + cfg.latent_dim + cfg.time_embed_dim
        self.factor_head = _mlp(
            factor_ctx_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )

        resid_ctx_dim = cfg.bottleneck_dim + cfg.token_dim + cfg.obs_hidden + cfg.time_embed_dim + cfg.latent_dim
        self.resid_head = _mlp(
            resid_ctx_dim, cfg.n_cells, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.resid_budget_head = _mlp(
            resid_ctx_dim, 1, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )

        self._init_heads()

    def _init_heads(self) -> None:
        strong_heads = [
            self.slow_token_head,
            self.fast_token_head,
            self.query_head,
            self.state_init,
            self.loading_head,
            self.factor_head,
        ]
        weak_heads = [self.resid_head, self.resid_budget_head]
        for head in strong_heads:
            last = head[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.mul_(0.1)
                    last.bias.zero_()
        for head in weak_heads:
            last = head[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.mul_(0.05)
                    last.bias.zero_()
        with torch.no_grad():
            self.key_proj.weight.mul_(0.2)
            self.value_proj.weight.mul_(0.2)
            if self.key_proj.bias is not None:
                self.key_proj.bias.zero_()
            if self.value_proj.bias is not None:
                self.value_proj.bias.zero_()

    @staticmethod
    def _flatten_history(history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            return history.view(history.shape[0], history.shape[1], -1)
        return history

    def _build_tokens(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        slow = self.slow_token_head(h).unsqueeze(1)
        fast = self.fast_token_head(h).view(B, self.cfg.n_fast_tokens, self.cfg.token_dim)
        tokens = torch.cat([slow, fast], dim=1)
        tokens = tokens + self.token_type.unsqueeze(0)
        return tokens

    def predict_mean(self, history: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history = self._flatten_history(history)
        B, _, D = history.shape
        assert D == self.cfg.n_cells

        h = self.encoder(history)
        tokens = self._build_tokens(h)
        keys = self.key_proj(tokens)
        values = self.value_proj(tokens)
        loadings = self.loading_head(h).view(B, self.cfg.n_cells, self.cfg.latent_dim)

        z = torch.tanh(self.state_init(h))
        current = 0.5 * (history[:, -1, :] + 1.0)
        current = current.clamp(self.cfg.support_lo, self.cfg.support_hi)

        level_steps: list[torch.Tensor] = []
        factor_steps: list[torch.Tensor] = []
        resid_steps: list[torch.Tensor] = []
        change_steps: list[torch.Tensor] = []
        attn_steps: list[torch.Tensor] = []
        token_ctx_steps: list[torch.Tensor] = []
        budget_steps: list[torch.Tensor] = []
        state_steps: list[torch.Tensor] = []

        scale = float(self.cfg.token_dim) ** -0.5
        for t in range(self.cfg.future_len):
            t_idx = torch.full((B,), t, device=history.device, dtype=torch.long)
            t_emb = self.time_embed(t_idx)
            obs = self.obs_proj(current)

            query_ctx = torch.cat([h, obs, t_emb, z], dim=-1)
            q = self.query_head(query_ctx)
            attn_logits = torch.einsum("bd,bnd->bn", q, keys) * scale
            attn = torch.softmax(attn_logits, dim=-1)
            token_ctx = torch.einsum("bn,bnd->bd", attn, values)

            z = self.state_cell(token_ctx, z)
            factor_ctx = torch.cat([h, token_ctx, z, t_emb], dim=-1)
            factor_scores = self.factor_head(factor_ctx)
            factor = torch.einsum("bdl,bl->bd", loadings, factor_scores)

            resid_ctx = torch.cat([h, token_ctx, obs, t_emb, z], dim=-1)
            factor_rms = factor.pow(2).mean(dim=1, keepdim=True).sqrt()
            budget = self.cfg.max_resid_ratio * torch.sigmoid(self.resid_budget_head(resid_ctx)) * factor_rms
            resid = torch.tanh(self.resid_head(resid_ctx)) * budget

            delta = factor + resid
            current = (current + delta).clamp(self.cfg.support_lo, self.cfg.support_hi)

            level_steps.append(current)
            factor_steps.append(factor)
            resid_steps.append(resid)
            change_steps.append(delta)
            attn_steps.append(attn)
            token_ctx_steps.append(token_ctx)
            budget_steps.append(budget)
            state_steps.append(z)

        mean_level = torch.stack(level_steps, dim=1)
        mean_factor = torch.stack(factor_steps, dim=1)
        mean_resid = torch.stack(resid_steps, dim=1)
        mean_change = torch.stack(change_steps, dim=1)
        attn_weights = torch.stack(attn_steps, dim=1)
        token_ctx_path = torch.stack(token_ctx_steps, dim=1)
        resid_budget = torch.stack(budget_steps, dim=1)
        state_path = torch.stack(state_steps, dim=1)

        aux = {
            "h": h,
            "tokens": tokens,
            "loadings": loadings,
            "state_path": state_path,
            "token_ctx_path": token_ctx_path,
            "attn_weights": attn_weights,
            "mean_factor": mean_factor,
            "mean_resid": mean_resid,
            "mean_change": mean_change,
            "resid_budget": resid_budget,
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


def load_model(checkpoint_path: str, device: torch.device) -> tuple[HierarchicalFutureTokenModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = HierarchicalFutureTokenConfig(**payload["config"])
    model = HierarchicalFutureTokenModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def config_to_dict(cfg: HierarchicalFutureTokenConfig) -> dict:
    return asdict(cfg)
