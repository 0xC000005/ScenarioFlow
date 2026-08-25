from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class FutureScalarARMixtureConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 192
    history_hidden: int = 128
    encoder_dropout: float = 0.1

    ar_hidden: int = 256
    ar_layers: int = 2
    ar_dropout: float = 0.1
    n_mixtures: int = 5

    logit_eps: float = 1e-4
    scale_floor: float = 1e-3
    sample_temperature: float = 1.0
    max_sample_chunk: int = 8
    use_same_cell_feedback: bool = False
    use_history_delta_features: bool = False
    standardize_logits: bool = False
    logit_std_floor: float = 1e-3
    target_mode: str = "level"
    standardize_level_features: bool = False
    aux_transition_weight: float = 0.0
    standardize_aux_transition: bool = False


class FutureScalarARMixtureDensityModel(nn.Module):
    """320a-v0: scalar chain-rule density over the flattened future logit path."""

    def __init__(self, cfg: FutureScalarARMixtureConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.target_mode not in {"level", "transition"}:
            raise ValueError("target_mode must be 'level' or 'transition'")
        hist_cfg = EncoderConfig(
            input_dim=cfg.n_cells * (2 if cfg.use_history_delta_features else 1),
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
        self.transition_head: nn.Module | None = None
        if cfg.aux_transition_weight > 0.0:
            self.transition_head = nn.Sequential(
                nn.Linear(cfg.ar_hidden, cfg.ar_hidden),
                nn.GELU(),
                nn.Dropout(cfg.ar_dropout),
                nn.Linear(cfg.ar_hidden, cfg.n_mixtures * 3),
            )
        token_idx = torch.arange(cfg.future_len * cfg.n_cells)
        self.register_buffer("token_day", token_idx // cfg.n_cells)
        self.register_buffer("token_cell", token_idx % cfg.n_cells)
        self.register_buffer("cell_logit_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_logit_std", torch.ones(cfg.n_cells))
        self.register_buffer("cell_level_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_level_std", torch.ones(cfg.n_cells))
        self.register_buffer("cell_transition_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_transition_std", torch.ones(cfg.n_cells))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_flat = self._flatten(history_norm)
        if not self.cfg.use_history_delta_features:
            return self.history_encoder(history_flat)
        history_logits = self.to_logits(history_flat)
        deltas = torch.zeros_like(history_logits)
        deltas[:, 1:] = history_logits[:, 1:] - history_logits[:, :-1]
        history_features = torch.cat([history_flat, deltas], dim=-1)
        return self.history_encoder(history_features)

    def to_logits(self, levels_norm: torch.Tensor) -> torch.Tensor:
        levels_norm = self._flatten(levels_norm)
        levels_01 = denormalize_iv(levels_norm)
        return iv_to_logit(levels_01, self.cfg.logit_eps)

    def set_logit_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.cfg.n_cells,) or std.shape != (self.cfg.n_cells,):
            raise ValueError("Expected per-cell logit stats with shape (n_cells,)")
        self.cell_logit_mean.copy_(mean.to(self.cell_logit_mean))
        self.cell_logit_std.copy_(std.clamp_min(self.cfg.logit_std_floor).to(self.cell_logit_std))

    def set_level_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.cfg.n_cells,) or std.shape != (self.cfg.n_cells,):
            raise ValueError("Expected per-cell level stats with shape (n_cells,)")
        self.cell_level_mean.copy_(mean.to(self.cell_level_mean))
        self.cell_level_std.copy_(std.clamp_min(self.cfg.logit_std_floor).to(self.cell_level_std))

    def set_transition_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.cfg.n_cells,) or std.shape != (self.cfg.n_cells,):
            raise ValueError("Expected per-cell transition stats with shape (n_cells,)")
        self.cell_transition_mean.copy_(mean.to(self.cell_transition_mean))
        self.cell_transition_std.copy_(
            std.clamp_min(self.cfg.logit_std_floor).to(self.cell_transition_std)
        )

    def _to_model_coord(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return logits
        return (logits - self.cell_logit_mean) / self.cell_logit_std

    def _from_model_coord(self, coord: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return coord
        return coord * self.cell_logit_std + self.cell_logit_mean

    def _scalar_from_model_coord(self, coord: torch.Tensor, cell: int) -> torch.Tensor:
        if not self.cfg.standardize_logits:
            return coord
        return coord * self.cell_logit_std[cell] + self.cell_logit_mean[cell]

    def _level_feature(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_level_features:
            return logits
        return (logits - self.cell_level_mean) / self.cell_level_std

    def _scalar_level_feature(self, logit: torch.Tensor, cell: int) -> torch.Tensor:
        if not self.cfg.standardize_level_features:
            return logit
        return (logit - self.cell_level_mean[cell]) / self.cell_level_std[cell]

    def _to_transition_coord(self, transition: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_aux_transition:
            return transition
        return (transition - self.cell_transition_mean) / self.cell_transition_std

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
        inp = self.scalar_in(scalar_feat)
        inp = inp + self.day_embed(self.token_day)[None]
        inp = inp + self.cell_embed(self.token_cell)[None]
        return inp

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
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        context = self.encode_history(history_norm)
        history_logits = self.to_logits(history_norm)
        future_logits = self.to_logits(future_norm)
        future_trans_raw = torch.cat(
            [
                future_logits[:, :1] - history_logits[:, -1:, :],
                future_logits[:, 1:] - future_logits[:, :-1],
            ],
            dim=1,
        )
        if self.cfg.target_mode == "level":
            history_coord = self._to_model_coord(history_logits)
            future_coord = self._to_model_coord(future_logits)
            target_seq = future_coord
            first_prev = history_coord[:, -1, self.token_cell[0]]
            anchor_scalar = history_coord[:, -1, self.token_cell]
            same_cell_source = future_coord
            same_cell_initial = history_coord[:, -1]
        else:
            history_trans = torch.zeros_like(history_logits)
            history_trans[:, 1:] = history_logits[:, 1:] - history_logits[:, :-1]
            target_seq = self._to_model_coord(future_trans_raw)
            history_trans_coord = self._to_model_coord(history_trans)
            first_prev = history_trans_coord[:, -1, self.token_cell[0]]
            current_level = torch.empty_like(future_logits)
            current_level[:, 0] = history_logits[:, -1]
            current_level[:, 1:] = future_logits[:, :-1]
            anchor_scalar = self._level_feature(current_level).reshape(
                future_logits.shape[0],
                -1,
            )
            same_cell_source = target_seq
            same_cell_initial = history_trans_coord[:, -1]
        target = target_seq.reshape(target_seq.shape[0], -1)

        prev_scalar = torch.empty_like(target)
        prev_scalar[:, 0] = first_prev
        prev_scalar[:, 1:] = target[:, :-1]
        same_cell_scalar = None
        if self.cfg.use_same_cell_feedback:
            same_cell = torch.empty_like(same_cell_source)
            same_cell[:, 0] = same_cell_initial
            same_cell[:, 1:] = same_cell_source[:, :-1]
            same_cell_scalar = same_cell.reshape(same_cell_source.shape[0], -1)

        inputs = self._token_inputs(prev_scalar, anchor_scalar, same_cell_scalar)
        states, _ = self.ar(inputs, self._initial_hidden(context))
        logits, means, scales = self._split_params(self.head(states))
        nll_grid = self.mixture_nll(target, logits, means, scales)
        nll = nll_grid.mean()
        total = nll
        aux_nll = None
        if self.transition_head is not None and self.cfg.aux_transition_weight > 0.0:
            aux_target = self._to_transition_coord(future_trans_raw).reshape(
                future_trans_raw.shape[0],
                -1,
            )
            aux_logits, aux_means, aux_scales = self._split_params(
                self.transition_head(states)
            )
            aux_nll = self.mixture_nll(aux_target, aux_logits, aux_means, aux_scales).mean()
            total = total + float(self.cfg.aux_transition_weight) * aux_nll

        weights = F.softmax(logits, dim=-1)
        expected = (weights * means).sum(dim=-1)
        metrics = {
            "total": total.detach(),
            "nll": nll.detach(),
            "target_std": target.std(unbiased=False).detach(),
            "target_abs": target.abs().mean().detach(),
            "scale_mean": (weights * scales).sum(dim=-1).mean().detach(),
            "mean_abs_err": (expected - target).abs().mean().detach(),
        }
        if aux_nll is not None:
            metrics["aux_transition_nll"] = aux_nll.detach()
        return total, metrics

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
        history_logits = self.to_logits(history_norm)
        context = self.encode_history(history_norm)
        if self.cfg.target_mode == "level":
            history_coord = self._to_model_coord(history_logits)
        else:
            history_trans = torch.zeros_like(history_logits)
            history_trans[:, 1:] = history_logits[:, 1:] - history_logits[:, :-1]
            history_coord = self._to_model_coord(history_trans)
        bsz = history_norm.shape[0]
        n_tokens = self.cfg.future_len * self.cfg.n_cells
        chunk_size = max(1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        outs: list[torch.Tensor] = []

        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            ctx = context.repeat_interleave(k, dim=0)
            hist_coord = history_coord.repeat_interleave(k, dim=0)
            hist_logits = history_logits.repeat_interleave(k, dim=0)
            hidden = self._initial_hidden(ctx)
            prev_scalar = hist_coord[:, -1, self.token_cell[0]]
            last_cell_values = hist_coord[:, -1].clone()
            current_logits = hist_logits[:, -1].clone()
            draws: list[torch.Tensor] = []
            level_frames: list[torch.Tensor] = []
            for token in range(n_tokens):
                cell = int(self.token_cell[token].item())
                day = int(self.token_day[token].item())
                if self.cfg.target_mode == "level":
                    anchor = hist_coord[:, -1, cell]
                else:
                    anchor = self._scalar_level_feature(current_logits[:, cell], cell)
                if self.cfg.use_same_cell_feedback:
                    scalar_feat = torch.stack([prev_scalar, anchor, last_cell_values[:, cell]], dim=-1)
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
                if self.cfg.target_mode == "transition":
                    current_logits[:, cell] = current_logits[:, cell] + self._scalar_from_model_coord(
                        draw,
                        cell,
                    )
                    if cell == self.cfg.n_cells - 1:
                        level_frames.append(current_logits.clone())
            if self.cfg.target_mode == "level":
                future_coord = torch.stack(draws, dim=1).view(
                    bsz * k,
                    self.cfg.future_len,
                    self.cfg.n_cells,
                )
                future_logits = self._from_model_coord(future_coord)
            else:
                future_logits = torch.stack(level_frames, dim=1)
            future_01 = logit_to_iv(future_logits)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[FutureScalarARMixtureDensityModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = FutureScalarARMixtureConfig(**payload["config"])
    model = FutureScalarARMixtureDensityModel(cfg)
    state_dict = payload["model_state_dict"]
    result = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "cell_logit_mean",
        "cell_logit_std",
        "cell_level_mean",
        "cell_level_std",
        "cell_transition_mean",
        "cell_transition_std",
    }
    if cfg.aux_transition_weight <= 0.0:
        allowed_missing.update(
            key for key in result.missing_keys if key.startswith("transition_head.")
        )
    missing = set(result.missing_keys) - allowed_missing
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    if cfg.standardize_logits and allowed_missing.intersection(result.missing_keys):
        missing_stats = {"cell_logit_mean", "cell_logit_std"}.intersection(result.missing_keys)
        if missing_stats:
            raise RuntimeError("Standardized-logit checkpoint is missing saved logit stats")
    if cfg.standardize_level_features and {"cell_level_mean", "cell_level_std"}.intersection(
        result.missing_keys
    ):
        raise RuntimeError("Checkpoint is missing saved level feature stats")
    if cfg.standardize_aux_transition and {
        "cell_transition_mean",
        "cell_transition_std",
    }.intersection(result.missing_keys):
        raise RuntimeError("Checkpoint is missing saved transition stats")
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: FutureScalarARMixtureDensityModel,
    cfg: FutureScalarARMixtureConfig,
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
