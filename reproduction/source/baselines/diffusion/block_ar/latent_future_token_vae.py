"""
257a-v0: conditional latent future-token VAE.

History conditions a prior over latent future tokens. During training, a posterior
encoder uses the realized future to infer token latents; a decoder then maps sampled
tokens to a future path.
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
class LatentFutureTokenVAEConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    latent_dim: int = 8
    token_dim: int = 64
    n_tokens: int = 4

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
    query_use_history: bool = True
    factor_use_history: bool = True
    resid_use_history: bool = True
    token_dependent_loadings: bool = False
    loading_delta_scale: float = 0.5


class LatentFutureTokenVAE(nn.Module):
    def __init__(self, cfg: LatentFutureTokenVAEConfig):
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
        self.history_encoder = GRUEncoder(enc_cfg)
        self.future_encoder = GRUEncoder(enc_cfg)

        stat_dim = cfg.n_tokens * cfg.token_dim * 2
        self.prior_head = _mlp(
            cfg.bottleneck_dim, stat_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.posterior_head = _mlp(
            2 * cfg.bottleneck_dim, stat_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.token_slot = nn.Parameter(torch.randn(cfg.n_tokens, cfg.token_dim) * 0.05)

        self.time_embed = nn.Embedding(cfg.future_len, cfg.time_embed_dim)
        self.obs_proj = nn.Sequential(
            nn.Linear(cfg.n_cells, cfg.obs_hidden),
            nn.GELU(),
            nn.Linear(cfg.obs_hidden, cfg.obs_hidden),
        )
        query_dim = cfg.obs_hidden + cfg.time_embed_dim + cfg.latent_dim
        if cfg.query_use_history:
            query_dim += cfg.bottleneck_dim
        self.query_head = _mlp(query_dim, cfg.token_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)
        self.key_proj = nn.Linear(cfg.token_dim, cfg.token_dim)
        self.value_proj = nn.Linear(cfg.token_dim, cfg.token_dim)

        self.state_init = _mlp(
            cfg.bottleneck_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.state_cell = nn.GRUCell(cfg.token_dim, cfg.latent_dim)
        self.loading_head = _mlp(
            cfg.bottleneck_dim, cfg.n_cells * cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        if cfg.token_dependent_loadings:
            self.loading_delta_head = _mlp(
                cfg.bottleneck_dim + cfg.token_dim,
                cfg.n_cells * cfg.latent_dim,
                cfg.head_hidden,
                cfg.head_layers,
                cfg.head_dropout,
            )
        else:
            self.loading_delta_head = None
        factor_ctx_dim = cfg.token_dim + cfg.latent_dim + cfg.time_embed_dim
        if cfg.factor_use_history:
            factor_ctx_dim += cfg.bottleneck_dim
        self.factor_head = _mlp(
            factor_ctx_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        resid_ctx_dim = cfg.token_dim + cfg.obs_hidden + cfg.time_embed_dim + cfg.latent_dim
        if cfg.resid_use_history:
            resid_ctx_dim += cfg.bottleneck_dim
        self.resid_head = _mlp(
            resid_ctx_dim, cfg.n_cells, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.resid_budget_head = _mlp(
            resid_ctx_dim, 1, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )

        self._init_heads()

    def _init_heads(self) -> None:
        nets = [
            self.prior_head,
            self.posterior_head,
            self.query_head,
            self.state_init,
            self.loading_head,
            self.factor_head,
            self.resid_head,
            self.resid_budget_head,
        ]
        if self.loading_delta_head is not None:
            nets.append(self.loading_delta_head)
        for net in nets:
            last = net[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.bias.zero_()
        with torch.no_grad():
            self.key_proj.bias.zero_()
            self.value_proj.bias.zero_()

    @staticmethod
    def _flatten_seq(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def _encode_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten_seq(history))

    def _future_changes(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        future = self._flatten_seq(future)
        prev = torch.cat([history[:, -1:, :], future[:, :-1, :]], dim=1)
        return future - prev

    def _stats_to_dist(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = stats.shape[0]
        stats = stats.view(B, self.cfg.n_tokens, 2 * self.cfg.token_dim)
        mu, logvar = stats[..., : self.cfg.token_dim], stats[..., self.cfg.token_dim :]
        logvar = logvar.clamp(-6.0, 4.0)
        return mu, logvar

    def _sample_latents(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def _kl_div(self, post_mu: torch.Tensor, post_logvar: torch.Tensor, prior_mu: torch.Tensor, prior_logvar: torch.Tensor) -> torch.Tensor:
        post_var = torch.exp(post_logvar)
        prior_var = torch.exp(prior_logvar)
        kl = 0.5 * (
            prior_logvar - post_logvar
            + (post_var + (post_mu - prior_mu).pow(2)) / prior_var
            - 1.0
        )
        return kl.mean()

    def _decode(self, history: torch.Tensor, h: torch.Tensor, token_latents: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history = self._flatten_seq(history)
        B, _, D = history.shape
        tokens = token_latents + self.token_slot.unsqueeze(0)
        keys = self.key_proj(tokens)
        values = self.value_proj(tokens)
        loadings = self.loading_head(h).view(B, self.cfg.n_cells, self.cfg.latent_dim)
        if self.loading_delta_head is not None:
            token_summary = token_latents.mean(dim=1)
            delta = self.loading_delta_head(torch.cat([h, token_summary], dim=-1))
            delta = delta.view(B, self.cfg.n_cells, self.cfg.latent_dim)
            loadings = loadings + self.cfg.loading_delta_scale * delta
        z = torch.tanh(self.state_init(h))

        current = 0.5 * (history[:, -1, :] + 1.0)
        current = current.clamp(self.cfg.support_lo, self.cfg.support_hi)
        scale = float(self.cfg.token_dim) ** -0.5

        level_steps: list[torch.Tensor] = []
        factor_steps: list[torch.Tensor] = []
        resid_steps: list[torch.Tensor] = []
        change_steps: list[torch.Tensor] = []
        attn_steps: list[torch.Tensor] = []
        budget_steps: list[torch.Tensor] = []
        state_steps: list[torch.Tensor] = []

        for t in range(self.cfg.future_len):
            t_idx = torch.full((B,), t, device=history.device, dtype=torch.long)
            t_emb = self.time_embed(t_idx)
            obs = self.obs_proj(current)
            query_parts = [obs, t_emb, z]
            if self.cfg.query_use_history:
                query_parts.insert(0, h)
            query_ctx = torch.cat(query_parts, dim=-1)
            q = self.query_head(query_ctx)
            attn_logits = torch.einsum("bd,bnd->bn", q, keys) * scale
            attn = torch.softmax(attn_logits, dim=-1)
            token_ctx = torch.einsum("bn,bnd->bd", attn, values)

            z = self.state_cell(token_ctx, z)
            factor_parts = [token_ctx, z, t_emb]
            if self.cfg.factor_use_history:
                factor_parts.insert(0, h)
            factor_ctx = torch.cat(factor_parts, dim=-1)
            factor_scores = self.factor_head(factor_ctx)
            factor = torch.einsum("bdl,bl->bd", loadings, factor_scores)

            resid_parts = [token_ctx, obs, t_emb, z]
            if self.cfg.resid_use_history:
                resid_parts.insert(0, h)
            resid_ctx = torch.cat(resid_parts, dim=-1)
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
            budget_steps.append(budget)
            state_steps.append(z)

        mean_level = torch.stack(level_steps, dim=1)
        mean_factor = torch.stack(factor_steps, dim=1)
        mean_resid = torch.stack(resid_steps, dim=1)
        mean_change = torch.stack(change_steps, dim=1)
        attn_weights = torch.stack(attn_steps, dim=1)
        resid_budget = torch.stack(budget_steps, dim=1)
        state_path = torch.stack(state_steps, dim=1)
        aux = {
            "tokens": tokens,
            "loadings": loadings,
            "state_path": state_path,
            "attn_weights": attn_weights,
            "mean_factor": mean_factor,
            "mean_resid": mean_resid,
            "mean_change": mean_change,
            "resid_budget": resid_budget,
        }
        return mean_level, aux

    def forward(self, history: torch.Tensor, future: torch.Tensor | None = None, **_ignored_kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history = self._flatten_seq(history)
        h = self._encode_history(history)
        prior_mu, prior_logvar = self._stats_to_dist(self.prior_head(h))

        if future is None:
            token_latents = prior_mu
            kl = torch.zeros((), device=history.device, dtype=history.dtype)
            post_mu, post_logvar = prior_mu, prior_logvar
        else:
            future = self._flatten_seq(future)
            future_change = self._future_changes(history, future)
            g = self.future_encoder(future_change)
            post_mu, post_logvar = self._stats_to_dist(self.posterior_head(torch.cat([h, g], dim=-1)))
            token_latents = self._sample_latents(post_mu, post_logvar)
            kl = self._kl_div(post_mu, post_logvar, prior_mu, prior_logvar)

        mean_level, aux = self._decode(history, h, token_latents)
        aux.update(
            {
                "h": h,
                "prior_mu": prior_mu,
                "prior_logvar": prior_logvar,
                "post_mu": post_mu,
                "post_logvar": post_logvar,
                "kl": kl,
            }
        )
        return mean_level, aux

    @torch.no_grad()
    def sample_batched(self, history: torch.Tensor, n_samples: int = 48, **_ignored_kwargs) -> torch.Tensor:
        history_ndim = history.ndim
        orig_h = history.shape[-2] if history_ndim == 4 else None
        orig_w = history.shape[-1] if history_ndim == 4 else None
        history = self._flatten_seq(history)
        B, T_hist, D = history.shape
        h = self._encode_history(history)
        prior_mu, prior_logvar = self._stats_to_dist(self.prior_head(h))
        history_rep = history.unsqueeze(1).expand(B, n_samples, T_hist, D).reshape(B * n_samples, T_hist, D)
        h_rep = h.unsqueeze(1).expand(B, n_samples, h.shape[-1]).reshape(B * n_samples, h.shape[-1])
        mu_rep = prior_mu.unsqueeze(1).expand(B, n_samples, self.cfg.n_tokens, self.cfg.token_dim)
        logvar_rep = prior_logvar.unsqueeze(1).expand(B, n_samples, self.cfg.n_tokens, self.cfg.token_dim)
        token_latents = self._sample_latents(mu_rep.reshape(B * n_samples, self.cfg.n_tokens, self.cfg.token_dim), logvar_rep.reshape(B * n_samples, self.cfg.n_tokens, self.cfg.token_dim))
        levels, _ = self._decode(history_rep, h_rep, token_latents)
        levels = levels.view(B, n_samples, self.cfg.future_len, self.cfg.n_cells)
        if history_ndim == 4 and orig_h is not None and orig_w is not None:
            return levels.view(B, n_samples, self.cfg.future_len, orig_h, orig_w)
        return levels


def load_model(checkpoint_path: str, device: torch.device) -> tuple[LatentFutureTokenVAE, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LatentFutureTokenVAEConfig(**payload["config"])
    model = LatentFutureTokenVAE(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def config_to_dict(cfg: LatentFutureTokenVAEConfig) -> dict:
    return asdict(cfg)
