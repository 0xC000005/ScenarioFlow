from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class EmpiricalNormalScoreScalarARMixtureConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 256
    history_hidden: int = 160
    encoder_dropout: float = 0.1

    ar_hidden: int = 384
    ar_layers: int = 2
    ar_dropout: float = 0.1
    n_mixtures: int = 9

    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    scale_floor: float = 1e-3
    sample_temperature: float = 1.0
    max_sample_chunk: int = 8
    use_same_cell_feedback: bool = True


class EmpiricalNormalScoreScalarARMixtureDensity(nn.Module):
    """343a: scalar chain-rule density in empirical normal-score coordinates."""

    def __init__(self, cfg: EmpiricalNormalScoreScalarARMixtureConfig):
        super().__init__()
        self.cfg = cfg
        hist_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            gru_hidden_dim=cfg.history_hidden,
            bottleneck_dim=cfg.context_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=0.0,
        )
        self.history_encoder = GRUEncoder(hist_cfg)
        self.init_hidden = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.ar_layers * cfg.ar_hidden),
            nn.Tanh(),
        )
        scalar_input_dim = 3 if cfg.use_same_cell_feedback else 2
        self.scalar_in = nn.Linear(scalar_input_dim, cfg.ar_hidden)
        self.day_embed = nn.Embedding(cfg.future_len, cfg.ar_hidden)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.ar_hidden)
        self.ar = nn.GRU(
            input_size=cfg.ar_hidden,
            hidden_size=cfg.ar_hidden,
            num_layers=cfg.ar_layers,
            dropout=cfg.ar_dropout if cfg.ar_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.ar_hidden, cfg.ar_hidden),
            nn.GELU(),
            nn.Dropout(cfg.ar_dropout),
            nn.Linear(cfg.ar_hidden, cfg.n_mixtures * 3),
        )
        token_idx = torch.arange(cfg.future_len * cfg.n_cells)
        self.register_buffer("token_day", token_idx // cfg.n_cells)
        self.register_buffer("token_cell", token_idx % cfg.n_cells)
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("level_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def set_empirical_quantiles(
        self,
        level_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if level_quantiles.shape != expected:
            raise ValueError(f"Expected level quantiles with shape {expected}")
        self.level_quantiles.copy_(level_quantiles.to(self.level_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("Expected quantile_levels with shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("Empirical quantiles must be set before training or sampling")

    def _values_to_scores(self, values_01: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        values_01 = self._flatten(values_01)
        levels = self.quantile_levels.to(device=values_01.device, dtype=values_01.dtype)
        table = self.level_quantiles.to(device=values_01.device, dtype=values_01.dtype)
        cols: list[torch.Tensor] = []
        for cell in range(self.cfg.n_cells):
            q = table[cell]
            flat = values_01[..., cell].reshape(-1)
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
            cols.append(torch.special.ndtri(u.clamp(eps, 1.0 - eps)).view(values_01.shape[:-1]))
        return torch.stack(cols, dim=-1)

    def _scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = self.level_quantiles.to(device=scores.device, dtype=scores.dtype)
        eps = float(self.cfg.cdf_eps)
        u_all = (0.5 * (1.0 + torch.erf(scores / math.sqrt(2.0)))).clamp(eps, 1.0 - eps)
        cols: list[torch.Tensor] = []
        for cell in range(self.cfg.n_cells):
            q = table[cell]
            flat = u_all[..., cell].reshape(-1)
            idx = torch.searchsorted(levels.contiguous(), flat.contiguous(), right=False)
            idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
            idx_lo = idx_hi - 1
            u_lo = levels[idx_lo]
            u_hi = levels[idx_hi]
            q_lo = q[idx_lo]
            q_hi = q[idx_hi]
            alpha = (flat - u_lo) / (u_hi - u_lo).clamp_min(1e-12)
            x = q_lo + alpha.clamp(0.0, 1.0) * (q_hi - q_lo)
            x = torch.where(flat <= levels[0], q[0], x)
            x = torch.where(flat >= levels[-1], q[-1], x)
            cols.append(x.view(scores.shape[:-1]))
        return torch.stack(cols, dim=-1).clamp(0.0, 1.0)

    def history_scores(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_01 = denormalize_iv(self._flatten(history_norm))
        return self._values_to_scores(history_01)

    def target_future_scores(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_01 = denormalize_iv(self._flatten(future_norm))
        return self._values_to_scores(future_01)

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self.history_scores(history_norm))

    def _initial_hidden(self, context: torch.Tensor) -> torch.Tensor:
        hidden = self.init_hidden(context)
        hidden = hidden.view(context.shape[0], self.cfg.ar_layers, self.cfg.ar_hidden)
        return hidden.transpose(0, 1).contiguous()

    def _token_inputs(
        self,
        prev_scalar: torch.Tensor,
        anchor_scalar: torch.Tensor,
        same_cell_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cfg.use_same_cell_feedback:
            if same_cell_scalar is None:
                raise ValueError("same_cell_scalar is required when use_same_cell_feedback=True")
            scalar_feat = torch.stack([prev_scalar, anchor_scalar, same_cell_scalar], dim=-1)
        else:
            scalar_feat = torch.stack([prev_scalar, anchor_scalar], dim=-1)
        x = self.scalar_in(scalar_feat)
        x = x + self.day_embed(self.token_day)[None]
        x = x + self.cell_embed(self.token_cell)[None]
        return x

    def _split_params(
        self,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bshape = raw.shape[:-1]
        params = raw.view(*bshape, self.cfg.n_mixtures, 3)
        logits = params[..., 0]
        means = params[..., 1]
        scales = F.softplus(params[..., 2]) + self.cfg.scale_floor
        return logits, means, scales

    @staticmethod
    def mixture_nll(
        target: torch.Tensor,
        logits: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        z = (target.unsqueeze(-1) - means) / scales
        log_comp = -0.5 * z.pow(2) - torch.log(scales) - 0.5 * math.log(2.0 * math.pi)
        log_mix = F.log_softmax(logits, dim=-1) + log_comp
        return -torch.logsumexp(log_mix, dim=-1)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_scores = self.history_scores(history_norm)
        future_scores = self.target_future_scores(future_norm)
        context = self.history_encoder(history_scores)
        target = future_scores.reshape(future_scores.shape[0], -1)

        prev_scalar = torch.empty_like(target)
        prev_scalar[:, 0] = history_scores[:, -1, self.token_cell[0]]
        prev_scalar[:, 1:] = target[:, :-1]
        anchor_scalar = history_scores[:, -1, self.token_cell]
        same_cell_scalar = None
        if self.cfg.use_same_cell_feedback:
            same_cell = torch.empty_like(future_scores)
            same_cell[:, 0] = history_scores[:, -1]
            same_cell[:, 1:] = future_scores[:, :-1]
            same_cell_scalar = same_cell.reshape(future_scores.shape[0], -1)

        inputs = self._token_inputs(prev_scalar, anchor_scalar, same_cell_scalar)
        states, _ = self.ar(inputs, self._initial_hidden(context))
        logits, means, scales = self._split_params(self.head(states))
        nll_grid = self.mixture_nll(target, logits, means, scales)
        nll = nll_grid.mean()
        weights = F.softmax(logits, dim=-1)
        expected = (weights * means).sum(dim=-1)
        metrics = {
            "total": nll.detach(),
            "nll": nll.detach(),
            "target_std": target.std(unbiased=False).detach(),
            "target_abs": target.abs().mean().detach(),
            "scale_mean": (weights * scales).sum(dim=-1).mean().detach(),
            "mean_abs_err": (expected - target).abs().mean().detach(),
        }
        return nll, metrics

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        history_scores = self.history_scores(history_norm)
        context = self.history_encoder(history_scores)
        bsz = history_norm.shape[0]
        n_tokens = self.cfg.future_len * self.cfg.n_cells
        chunk_size = max(1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        outs: list[torch.Tensor] = []

        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            ctx = context.repeat_interleave(k, dim=0)
            hist_scores = history_scores.repeat_interleave(k, dim=0)
            hidden = self._initial_hidden(ctx)
            prev_scalar = hist_scores[:, -1, self.token_cell[0]]
            last_cell_values = hist_scores[:, -1].clone()
            draws: list[torch.Tensor] = []
            for token in range(n_tokens):
                cell = int(self.token_cell[token].item())
                day = int(self.token_day[token].item())
                anchor = hist_scores[:, -1, cell]
                if self.cfg.use_same_cell_feedback:
                    scalar_feat = torch.stack(
                        [prev_scalar, anchor, last_cell_values[:, cell]],
                        dim=-1,
                    )
                else:
                    scalar_feat = torch.stack([prev_scalar, anchor], dim=-1)
                inp = self.scalar_in(scalar_feat)
                inp = inp + self.day_embed.weight[day][None]
                inp = inp + self.cell_embed.weight[cell][None]
                state, hidden = self.ar(inp.unsqueeze(1), hidden)
                logits, means, scales = self._split_params(self.head(state[:, 0]))
                mix_idx = torch.distributions.Categorical(logits=logits).sample()
                chosen_mean = means.gather(-1, mix_idx.unsqueeze(-1)).squeeze(-1)
                chosen_scale = scales.gather(-1, mix_idx.unsqueeze(-1)).squeeze(-1)
                draw = chosen_mean + temp * chosen_scale * torch.randn_like(chosen_mean)
                draws.append(draw)
                last_cell_values[:, cell] = draw
                prev_scalar = draw
            future_scores = torch.stack(draws, dim=1).view(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
            )
            future_01 = self._scores_to_values(future_scores)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[EmpiricalNormalScoreScalarARMixtureDensity, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScoreScalarARMixtureConfig(**payload["config"])
    model = EmpiricalNormalScoreScalarARMixtureDensity(cfg)
    result = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = {"level_quantiles", "quantile_levels", "_quantiles_ready"}
    missing = set(result.missing_keys) - allowed_missing
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    if allowed_missing.intersection(result.missing_keys):
        raise RuntimeError("Checkpoint is missing saved empirical quantiles")
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScoreScalarARMixtureDensity,
    cfg: EmpiricalNormalScoreScalarARMixtureConfig,
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
