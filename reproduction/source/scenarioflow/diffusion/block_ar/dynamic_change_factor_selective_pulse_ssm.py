"""
253c: dynamic change-space factor SSM with selective supervised pulse channel.

Purpose:
    Last justified 253-family attempt after 253b. Keep the 253a-ec backbone,
    but replace the dormant shock branch with an explicitly supervised,
    selective common pulse path that cannot silently collapse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder


@dataclass
class DynamicFactorSelectivePulseSSMConfig:
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

    obs_hidden: int = 64
    time_embed_dim: int = 16

    max_idio_ratio: float = 0.60
    use_error_correction: bool = True
    ec_max_strength: float = 0.25

    max_pulse_ratio: float = 1.0

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


class DynamicChangeFactorSelectivePulseSSM(nn.Module):
    def __init__(self, cfg: DynamicFactorSelectivePulseSSMConfig):
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
        self.obs_proj = nn.Sequential(
            nn.Linear(cfg.n_cells, cfg.obs_hidden),
            nn.GELU(),
            nn.Linear(cfg.obs_hidden, cfg.obs_hidden),
        )

        self.state_init = _mlp(
            cfg.bottleneck_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.loading_head = _mlp(
            cfg.bottleneck_dim, cfg.n_cells * cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )

        ctx_dim = cfg.bottleneck_dim + cfg.latent_dim + cfg.obs_hidden + cfg.time_embed_dim
        self.state_gate = _mlp(ctx_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)
        self.state_prop = _mlp(ctx_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)
        self.factor_gate = _mlp(ctx_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)

        self.idio_head = _mlp(ctx_dim, cfg.n_cells, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)
        self.idio_budget_head = _mlp(ctx_dim, 1, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)

        self.pulse_latent_head = _mlp(
            ctx_dim, cfg.latent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.pulse_gate_head = _mlp(
            ctx_dim, 1, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )
        self.pulse_budget_head = _mlp(
            ctx_dim, 1, cfg.head_hidden, cfg.head_layers, cfg.head_dropout
        )

        ec_ctx_dim = cfg.bottleneck_dim + cfg.obs_hidden + cfg.time_embed_dim
        if cfg.use_error_correction:
            self.anchor_head = _mlp(
                cfg.bottleneck_dim + cfg.time_embed_dim,
                cfg.n_cells,
                cfg.head_hidden,
                cfg.head_layers,
                cfg.head_dropout,
            )
            self.ec_gate = _mlp(ec_ctx_dim, 1, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)
        else:
            self.anchor_head = None
            self.ec_gate = None

        self._init_heads()

    def _init_heads(self) -> None:
        for head in [
            self.state_init,
            self.loading_head,
            self.state_gate,
            self.state_prop,
            self.factor_gate,
        ]:
            last = head[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.mul_(0.1)
                    last.bias.zero_()

        for head in [
            self.idio_head,
            self.idio_budget_head,
            self.pulse_latent_head,
            self.pulse_gate_head,
            self.pulse_budget_head,
        ]:
            last = head[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.mul_(0.05)
                    last.bias.zero_()
        pulse_gate_last = self.pulse_gate_head[-1]
        if isinstance(pulse_gate_last, nn.Linear):
            with torch.no_grad():
                pulse_gate_last.weight.mul_(0.01)
                pulse_gate_last.bias.fill_(-2.0)

        if self.anchor_head is not None:
            last = self.anchor_head[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.mul_(0.05)
                    last.bias.zero_()
        if self.ec_gate is not None:
            last = self.ec_gate[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.mul_(0.01)
                    last.bias.fill_(-3.0)

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
        z = torch.tanh(self.state_init(h))

        current = 0.5 * (history[:, -1, :] + 1.0)
        current = current.clamp(self.cfg.support_lo, self.cfg.support_hi)

        level_steps: list[torch.Tensor] = []
        change_steps: list[torch.Tensor] = []
        factor_steps: list[torch.Tensor] = []
        pulse_steps: list[torch.Tensor] = []
        idio_steps: list[torch.Tensor] = []
        ec_steps: list[torch.Tensor] = []
        budget_steps: list[torch.Tensor] = []
        pulse_budget_steps: list[torch.Tensor] = []
        pulse_gate_steps: list[torch.Tensor] = []
        state_steps: list[torch.Tensor] = []
        anchor_steps: list[torch.Tensor] = []

        for t in range(self.cfg.future_len):
            t_idx = torch.full((B,), t, device=history.device, dtype=torch.long)
            t_emb = self.time_embed(t_idx)
            obs = self.obs_proj(current)
            ctx = torch.cat([h, z, obs, t_emb], dim=-1)

            retain = torch.sigmoid(self.state_gate(ctx))
            proposal = torch.tanh(self.state_prop(ctx))
            z = retain * z + (1.0 - retain) * proposal

            fac_scale = 1.0 + 0.25 * torch.tanh(self.factor_gate(ctx))
            factor = torch.einsum("bdl,bl->bd", base_loadings, fac_scale * z)

            factor_rms = factor.pow(2).mean(dim=1, keepdim=True).sqrt()

            pulse_latent = torch.tanh(self.pulse_latent_head(ctx))
            pulse_raw = torch.einsum("bdl,bl->bd", base_loadings, pulse_latent)
            pulse_gate = torch.sigmoid(self.pulse_gate_head(ctx))
            pulse_budget = (
                self.cfg.max_pulse_ratio
                * torch.sigmoid(self.pulse_budget_head(ctx))
                * factor_rms
            )
            pulse = pulse_gate * torch.tanh(pulse_raw) * pulse_budget

            budget = (
                self.cfg.max_idio_ratio
                * torch.sigmoid(self.idio_budget_head(ctx))
                * factor_rms
            )
            idio = torch.tanh(self.idio_head(ctx)) * budget

            ec_delta = torch.zeros_like(factor)
            anchor = torch.zeros_like(factor)
            if self.cfg.use_error_correction and self.anchor_head is not None and self.ec_gate is not None:
                ec_ctx = torch.cat([h, obs, t_emb], dim=-1)
                anchor = self.anchor_head(torch.cat([h, t_emb], dim=-1))
                anchor = anchor.clamp(self.cfg.support_lo, self.cfg.support_hi)
                ec_strength = self.cfg.ec_max_strength * torch.sigmoid(self.ec_gate(ec_ctx))
                ec_delta = ec_strength * (anchor - current)

            delta = factor + pulse + idio + ec_delta
            current = (current + delta).clamp(self.cfg.support_lo, self.cfg.support_hi)

            level_steps.append(current)
            change_steps.append(delta)
            factor_steps.append(factor)
            pulse_steps.append(pulse)
            idio_steps.append(idio)
            ec_steps.append(ec_delta)
            budget_steps.append(budget)
            pulse_budget_steps.append(pulse_budget)
            pulse_gate_steps.append(pulse_gate)
            state_steps.append(z)
            anchor_steps.append(anchor)

        mean_level = torch.stack(level_steps, dim=1)
        mean_change = torch.stack(change_steps, dim=1)
        mean_factor = torch.stack(factor_steps, dim=1)
        mean_pulse = torch.stack(pulse_steps, dim=1)
        mean_common = mean_factor + mean_pulse
        mean_idio = torch.stack(idio_steps, dim=1)
        ec_term = torch.stack(ec_steps, dim=1)
        idio_budget = torch.stack(budget_steps, dim=1)
        pulse_budget = torch.stack(pulse_budget_steps, dim=1)
        pulse_gate = torch.stack(pulse_gate_steps, dim=1)
        state_path = torch.stack(state_steps, dim=1)
        anchors = torch.stack(anchor_steps, dim=1)

        aux = {
            "h": h,
            "base_loadings": base_loadings,
            "state_path": state_path,
            "mean_factor": mean_factor,
            "mean_pulse": mean_pulse,
            "mean_common": mean_common,
            "mean_idio": mean_idio,
            "mean_change": mean_change,
            "ec_term": ec_term,
            "idio_budget": idio_budget,
            "pulse_budget": pulse_budget,
            "pulse_gate": pulse_gate,
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


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DynamicChangeFactorSelectivePulseSSM, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = DynamicFactorSelectivePulseSSMConfig(**payload["config"])
    model = DynamicChangeFactorSelectivePulseSSM(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def config_to_dict(cfg: DynamicFactorSelectivePulseSSMConfig) -> dict:
    return asdict(cfg)
