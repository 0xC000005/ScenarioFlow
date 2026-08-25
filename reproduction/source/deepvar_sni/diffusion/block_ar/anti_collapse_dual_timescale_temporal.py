"""
254b: anti-collapse dual-timescale temporal backbone with dynamic loading modulation.

Purpose:
    Follow-up to 254a-v0 after the first dual-timescale prototype collapsed into a
    near-rank-1 common path. Keep the same broad deterministic family, but let the
    cross-sectional loading map move modestly over time and add explicit structural
    penalties against PC1 domination / rank collapse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.dual_timescale_low_rank_temporal import (
    DualTimescaleLowRankTemporal,
    DualTimescaleTemporalConfig,
    _mlp,
)


@dataclass
class AntiCollapseDualTimescaleConfig(DualTimescaleTemporalConfig):
    slow_loading_scale: float = 0.15
    fast_loading_scale: float = 0.20
    top1_share_target: float = 0.55
    rank_floor: float = 3.0


class AntiCollapseDualTimescaleTemporal(DualTimescaleLowRankTemporal):
    def __init__(self, cfg: AntiCollapseDualTimescaleConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.slow_loading_delta_head = _mlp(
            cfg.token_hidden,
            cfg.n_cells * cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self.fast_loading_delta_head = _mlp(
            cfg.token_hidden,
            cfg.n_cells * cfg.latent_dim,
            cfg.head_hidden,
            cfg.head_layers,
            cfg.head_dropout,
        )
        self._init_anti_collapse_heads()

    def _init_anti_collapse_heads(self) -> None:
        for head in [self.slow_loading_delta_head, self.fast_loading_delta_head]:
            last = head[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.mul_(0.01)
                    last.bias.zero_()

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

        slow_delta = self.cfg.slow_loading_scale * torch.tanh(
            self.slow_loading_delta_head(slow_feat).view(B, self.cfg.future_len, self.cfg.n_cells, self.cfg.latent_dim)
        )
        fast_delta = self.cfg.fast_loading_scale * fast_gate.unsqueeze(-1) * torch.tanh(
            self.fast_loading_delta_head(fast_feat).view(B, self.cfg.future_len, self.cfg.n_cells, self.cfg.latent_dim)
        )
        dynamic_loadings = base_loadings.unsqueeze(1) + slow_delta + fast_delta

        mean_common = torch.einsum("btdl,btl->btd", dynamic_loadings, common_latent)

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
            "dynamic_loadings": dynamic_loadings,
            "slow_loading_delta": slow_delta,
            "fast_loading_delta": fast_delta,
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

    def spectral_collapse_stats(self, dynamic_loadings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_load = dynamic_loadings.mean(dim=(0, 1))  # [D, L]
        s = torch.linalg.svdvals(mean_load)
        power = s.square()
        p = power / power.sum().clamp_min(1e-8)
        top1_share = p[0]
        eff_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-8))).sum())
        return top1_share, eff_rank

    def spectral_collapse_penalty(self, dynamic_loadings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        top1_share, eff_rank = self.spectral_collapse_stats(dynamic_loadings)
        p_top1 = torch.relu(top1_share - self.cfg.top1_share_target).square()
        p_rank = torch.relu(self.cfg.rank_floor - eff_rank).square()
        return p_top1 + p_rank, top1_share, eff_rank


def load_model(checkpoint_path: str, device: torch.device) -> tuple[AntiCollapseDualTimescaleTemporal, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = AntiCollapseDualTimescaleConfig(**payload["config"])
    model = AntiCollapseDualTimescaleTemporal(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def config_to_dict(cfg: AntiCollapseDualTimescaleConfig) -> dict:
    return asdict(cfg)
