from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GenericRealNVPTransitionConfig:
    """Exact-likelihood AR transition flow for generic financial state panels."""

    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    prefix_feature_mode: str = "basic"
    memory_dim: int = 128
    memory_layers: int = 3
    memory_heads: int = 4
    memory_ff: int = 256
    coupling_layers: int = 6
    coupling_hidden: int = 256
    coupling_depth: int = 2
    model_dropout: float = 0.05
    scale_clip: float = 1.5
    sample_temperature: float = 1.0


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, cfg: GenericRealNVPTransitionConfig, mask: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mask", mask.view(1, cfg.n_cells))
        in_dim = cfg.n_cells + cfg.n_cells + cfg.memory_dim
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        hidden = int(cfg.coupling_hidden)
        last = in_dim
        for _ in range(int(cfg.coupling_depth)):
            layers += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(float(cfg.model_dropout))]
            last = hidden
        layers.append(nn.Linear(last, 2 * cfg.n_cells))
        self.net = nn.Sequential(*layers)
        self._init_last()

    def _init_last(self) -> None:
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _shift_log_scale(
        self,
        masked_value: torch.Tensor,
        current_score: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat([masked_value, current_score, memory_state], dim=-1))
        shift, raw_log_scale = raw.chunk(2, dim=-1)
        inv_mask = 1.0 - self.mask
        log_scale = torch.tanh(raw_log_scale) * float(self.cfg.scale_clip) * inv_mask
        shift = shift * inv_mask
        return shift, log_scale

    def forward(
        self,
        z: torch.Tensor,
        current_score: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = z * self.mask
        shift, log_scale = self._shift_log_scale(masked, current_score, memory_state)
        inv_mask = 1.0 - self.mask
        y = masked + inv_mask * (z * torch.exp(log_scale) + shift)
        log_det = log_scale.sum(dim=-1)
        return y, log_det

    def inverse(
        self,
        y: torch.Tensor,
        current_score: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = y * self.mask
        shift, log_scale = self._shift_log_scale(masked, current_score, memory_state)
        inv_mask = 1.0 - self.mask
        z = masked + inv_mask * ((y - shift) * torch.exp(-log_scale))
        log_det_inv = -log_scale.sum(dim=-1)
        return z, log_det_inv


class GenericRealNVPTransitionLaw(nn.Module):
    """Autoregressive conditional normalizing flow over score increments."""

    def __init__(self, cfg: GenericRealNVPTransitionConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.prefix_feature_mode not in {"basic", "scale"}:
            raise ValueError("prefix_feature_mode must be 'basic' or 'scale'")
        if int(cfg.coupling_layers) < 1:
            raise ValueError("coupling_layers must be >= 1")
        feature_mult = 4 if cfg.prefix_feature_mode == "scale" else 2
        self.feature_proj = nn.Linear(feature_mult * cfg.n_cells, cfg.memory_dim)
        self.pos_embed = nn.Embedding(cfg.history_len + cfg.future_len, cfg.memory_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.memory_dim,
            nhead=cfg.memory_heads,
            dim_feedforward=cfg.memory_ff,
            dropout=cfg.model_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory = nn.TransformerEncoder(layer, num_layers=cfg.memory_layers)
        self.memory_norm = nn.LayerNorm(cfg.memory_dim)
        self.couplings = nn.ModuleList(self._build_couplings())
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(cfg.n_quantiles)
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("value_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))

    def _build_couplings(self) -> list[ConditionalAffineCoupling]:
        masks: list[ConditionalAffineCoupling] = []
        base = (torch.arange(self.cfg.n_cells) % 2).float()
        for idx in range(int(self.cfg.coupling_layers)):
            mask = base if idx % 2 == 0 else 1.0 - base
            masks.append(ConditionalAffineCoupling(self.cfg, mask))
        return masks

    def set_empirical_quantiles(
        self,
        value_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if value_quantiles.shape != expected:
            raise ValueError(f"expected value_quantiles shape {expected}, got {tuple(value_quantiles.shape)}")
        self.value_quantiles.copy_(value_quantiles.to(self.value_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("quantile_levels must have shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("empirical quantiles must be set before use")

    def values_to_scores(self, values: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        if values.ndim != 3 or values.shape[-1] != self.cfg.n_cells:
            raise ValueError(
                f"values must have shape (batch,time,{self.cfg.n_cells}), got {tuple(values.shape)}"
            )
        levels = self.quantile_levels.to(device=values.device, dtype=values.dtype)
        table = self.value_quantiles.to(device=values.device, dtype=values.dtype)
        cols: list[torch.Tensor] = []
        for var in range(self.cfg.n_cells):
            q = table[var]
            flat = values[..., var].reshape(-1)
            idx = torch.searchsorted(q.contiguous(), flat.contiguous(), right=False)
            idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
            idx_lo = idx_hi - 1
            q_lo = q[idx_lo]
            q_hi = q[idx_hi]
            u_lo = levels[idx_lo]
            u_hi = levels[idx_hi]
            alpha = (flat - q_lo) / (q_hi - q_lo).clamp_min(1e-12)
            u = u_lo + alpha.clamp(0.0, 1.0) * (u_hi - u_lo)
            u = torch.where(flat <= q[0], levels[0], u)
            u = torch.where(flat >= q[-1], levels[-1], u)
            eps = float(self.cfg.cdf_eps)
            cols.append(torch.special.ndtri(u.clamp(eps, 1.0 - eps)).view(values.shape[:-1]))
        return torch.stack(cols, dim=-1)

    def scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        if scores.ndim not in {2, 3}:
            raise ValueError("scores must have shape (batch,time,vars) or (batch,vars)")
        if scores.shape[-1] != self.cfg.n_cells:
            raise ValueError(f"last dimension must be {self.cfg.n_cells}")
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = self.value_quantiles.to(device=scores.device, dtype=scores.dtype)
        eps = float(self.cfg.cdf_eps)
        u_all = (0.5 * (1.0 + torch.erf(scores / math.sqrt(2.0)))).clamp(eps, 1.0 - eps)
        cols: list[torch.Tensor] = []
        for var in range(self.cfg.n_cells):
            q = table[var]
            flat = u_all[..., var].reshape(-1)
            idx = torch.searchsorted(levels.contiguous(), flat.contiguous(), right=False)
            idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
            idx_lo = idx_hi - 1
            u_lo = levels[idx_lo]
            u_hi = levels[idx_hi]
            q_lo = q[idx_lo]
            q_hi = q[idx_hi]
            alpha = (flat - u_lo) / (u_hi - u_lo).clamp_min(1e-12)
            value = q_lo + alpha.clamp(0.0, 1.0) * (q_hi - q_lo)
            value = torch.where(flat <= levels[0], q[0], value)
            value = torch.where(flat >= levels[-1], q[-1], value)
            cols.append(value.view(scores.shape[:-1]))
        return torch.stack(cols, dim=-1)

    def _score_features(self, scores: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(scores)
        deltas[:, 1:] = scores[:, 1:] - scores[:, :-1]
        if self.cfg.prefix_feature_mode == "scale":
            return torch.cat([scores, deltas, deltas.abs(), deltas.square()], dim=-1)
        return torch.cat([scores, deltas], dim=-1)

    def _encode_prefix_scores(self, prefix_scores: torch.Tensor) -> torch.Tensor:
        seq_len = prefix_scores.shape[1]
        if seq_len > self.cfg.history_len + self.cfg.future_len:
            raise ValueError(f"prefix length {seq_len} exceeds {self.cfg.history_len + self.cfg.future_len}")
        pos = torch.arange(seq_len, device=prefix_scores.device)
        x = self.feature_proj(self._score_features(prefix_scores))
        x = x + self.pos_embed(pos)[None, :, :]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=prefix_scores.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.memory_norm(self.memory(x, mask=mask))

    def _flow_forward(
        self,
        z: torch.Tensor,
        current_score: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        y = z
        for layer in self.couplings:
            y, log_det = layer(y, current_score, memory_state)
            total_log_det = total_log_det + log_det
        return y, total_log_det

    def _flow_inverse(
        self,
        delta: torch.Tensor,
        current_score: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_log_det = torch.zeros(delta.shape[0], device=delta.device, dtype=delta.dtype)
        z = delta
        for layer in reversed(self.couplings):
            z, log_det = layer.inverse(z, current_score, memory_state)
            total_log_det = total_log_det + log_det
        return z, total_log_det

    def training_loss(
        self,
        history_values: torch.Tensor,
        future_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_scores = self.values_to_scores(history_values)
        future_scores = self.values_to_scores(future_values)
        prefix_scores = torch.cat([history_scores, future_scores[:, :-1]], dim=1)
        hidden = self._encode_prefix_scores(prefix_scores)
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        current_scores = prefix_scores[:, start : start + self.cfg.future_len]
        target = future_scores - current_scores
        bsz, horizon, n_vars = target.shape
        flat_target = target.reshape(bsz * horizon, n_vars)
        flat_current = current_scores.reshape(bsz * horizon, n_vars)
        flat_memory = memory_states.reshape(bsz * horizon, self.cfg.memory_dim)
        z, log_det_inv = self._flow_inverse(flat_target, flat_current, flat_memory)
        base_log_prob = -0.5 * (z.square().sum(dim=-1) + n_vars * math.log(2.0 * math.pi))
        log_prob = base_log_prob + log_det_inv
        nll = -log_prob.mean()
        metrics = {
            "total": nll.detach(),
            "nll": nll.detach(),
            "transition_std": target.std(unbiased=False).detach(),
            "transition_abs": target.abs().mean().detach(),
            "z_std": z.std(unbiased=False).detach(),
            "z_abs": z.abs().mean().detach(),
            "log_det_inv_mean": log_det_inv.mean().detach(),
            "memory_abs": memory_states.abs().mean().detach(),
        }
        return nll, metrics

    @torch.no_grad()
    def sample_batched(
        self,
        history_values: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(f"expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        history_scores = self.values_to_scores(history_values)
        bsz = history_scores.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        outs: list[torch.Tensor] = []
        for start in range(0, int(n_samples), chunk_size):
            k = min(chunk_size, int(n_samples) - start)
            prefix = (
                history_scores.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            frames: list[torch.Tensor] = []
            for _step in range(n_steps):
                memory_state = self._encode_prefix_scores(prefix)[:, -1]
                current_score = prefix[:, -1]
                z = temp * torch.randn(
                    current_score.shape[0],
                    self.cfg.n_cells,
                    device=current_score.device,
                    dtype=current_score.dtype,
                )
                transition, _log_det = self._flow_forward(z, current_score, memory_state)
                next_score = current_score + transition
                next_values = self.scores_to_values(next_score)
                frames.append(next_values.view(bsz, k, self.cfg.n_cells))
                prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GenericRealNVPTransitionLaw, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GenericRealNVPTransitionConfig(**payload["config"])
    model = GenericRealNVPTransitionLaw(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: GenericRealNVPTransitionLaw,
    cfg: GenericRealNVPTransitionConfig,
    epoch: int,
    best_val: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": asdict(cfg),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)

