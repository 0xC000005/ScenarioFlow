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
class DailyJointCholeskyTransitionConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 160
    history_hidden: int = 96
    encoder_dropout: float = 0.1

    decoder_hidden: int = 160
    decoder_layers: int = 2
    decoder_dropout: float = 0.1

    logit_eps: float = 1e-4
    diag_floor: float = 1e-3
    sample_temperature: float = 1.0
    max_sample_chunk: int = 8
    use_level_feedback: bool = False
    target_mode: str = "transition"
    level_mean_residual: bool = False
    standardize_target: bool = False
    target_std_floor: float = 1e-3
    distribution: str = "gaussian"
    nu_floor: float = 4.0


def _mlp(
    in_dim: int,
    out_dim: int,
    hidden: int,
    layers: int,
    dropout: float,
) -> nn.Sequential:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(max(0, layers - 1)):
        mods.extend([nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)])
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


class DailyJointCholeskyTransitionModel(nn.Module):
    """318a-v0: chain-rule daily joint law with full 25-cell covariance."""

    def __init__(self, cfg: DailyJointCholeskyTransitionConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.target_mode not in {"transition", "level"}:
            raise ValueError("target_mode must be 'transition' or 'level'")
        if cfg.distribution not in {"gaussian", "student_t"}:
            raise ValueError("distribution must be 'gaussian' or 'student_t'")
        hist_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            gru_hidden_dim=cfg.history_hidden,
            bottleneck_dim=cfg.context_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=0.0,
        )
        self.history_encoder = GRUEncoder(hist_cfg)
        self.init_state = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.decoder_hidden),
            nn.Tanh(),
        )
        decoder_input_dim = cfg.n_cells * 2 if cfg.use_level_feedback else cfg.n_cells
        self.transition_in = nn.Linear(decoder_input_dim, cfg.decoder_hidden)
        self.step_embed = nn.Embedding(cfg.future_len, cfg.decoder_hidden)
        self.decoder = nn.GRUCell(cfg.decoder_hidden, cfg.decoder_hidden)
        n_tril = cfg.n_cells * (cfg.n_cells + 1) // 2
        dist_extra = 1 if cfg.distribution == "student_t" else 0
        self.head = _mlp(
            cfg.decoder_hidden,
            cfg.n_cells + n_tril + dist_extra,
            cfg.decoder_hidden,
            cfg.decoder_layers,
            cfg.decoder_dropout,
        )
        tril = torch.tril_indices(cfg.n_cells, cfg.n_cells)
        self.register_buffer("tril_row", tril[0])
        self.register_buffer("tril_col", tril[1])
        self.register_buffer("cell_target_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_target_std", torch.ones(cfg.n_cells))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._flatten(history_norm))

    def to_logits(self, levels_norm: torch.Tensor) -> torch.Tensor:
        levels_norm = self._flatten(levels_norm)
        levels_01 = denormalize_iv(levels_norm)
        return iv_to_logit(levels_01, self.cfg.logit_eps)

    def set_target_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.cfg.n_cells,) or std.shape != (self.cfg.n_cells,):
            raise ValueError("Expected per-cell target stats with shape (n_cells,)")
        self.cell_target_mean.copy_(mean.to(self.cell_target_mean))
        self.cell_target_std.copy_(
            std.clamp_min(self.cfg.target_std_floor).to(self.cell_target_std)
        )

    def _to_target_coord(self, target: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_target:
            return target
        return (target - self.cell_target_mean) / self.cell_target_std

    def _from_target_coord(self, coord: torch.Tensor) -> torch.Tensor:
        if not self.cfg.standardize_target:
            return coord
        return coord * self.cell_target_std + self.cell_target_mean

    def future_transitions(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_logits = self.to_logits(history_norm)
        future_logits = self.to_logits(future_norm)
        last_logit = history_logits[:, -1]
        prev_logit = history_logits[:, -2]
        last_transition = last_logit - prev_logit
        future_trans = torch.cat(
            [
                future_logits[:, :1] - last_logit[:, None],
                future_logits[:, 1:] - future_logits[:, :-1],
            ],
            dim=1,
        )
        return last_transition, future_trans

    def _distribution_params(
        self,
        state: torch.Tensor,
        current_logit: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.head(state)
        mean = raw[:, : self.cfg.n_cells]
        if (
            self.cfg.target_mode == "level"
            and self.cfg.level_mean_residual
            and current_logit is not None
        ):
            mean = current_logit + mean
        packed_end = self.cfg.n_cells + self.cfg.n_cells * (self.cfg.n_cells + 1) // 2
        packed = raw[:, self.cfg.n_cells : packed_end]
        batch = raw.shape[0]
        chol = raw.new_zeros(batch, self.cfg.n_cells, self.cfg.n_cells)
        chol[:, self.tril_row, self.tril_col] = packed
        diag = F.softplus(torch.diagonal(chol, dim1=-2, dim2=-1)) + self.cfg.diag_floor
        chol = chol - torch.diag_embed(torch.diagonal(chol, dim1=-2, dim2=-1))
        chol = chol + torch.diag_embed(diag)
        nu = None
        if self.cfg.distribution == "student_t":
            nu = F.softplus(raw[:, packed_end]) + self.cfg.nu_floor
        return mean, chol, nu

    def _decoder_input(
        self,
        prev_transition: torch.Tensor,
        current_logit: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.use_level_feedback:
            return torch.cat([prev_transition, current_logit], dim=-1)
        return prev_transition

    @staticmethod
    def gaussian_nll(
        target: torch.Tensor,
        mean: torch.Tensor,
        chol: torch.Tensor,
    ) -> torch.Tensor:
        diff = (target - mean).unsqueeze(-1)
        white = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)
        quad = white.pow(2).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        return 0.5 * (quad + logdet + target.shape[-1] * math.log(2.0 * math.pi))

    def student_t_nll(
        self,
        target: torch.Tensor,
        mean: torch.Tensor,
        chol: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        d = target.shape[-1]
        diff = (target - mean).unsqueeze(-1)
        white = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)
        quad = white.pow(2).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        pi = target.new_tensor(math.pi)
        log_norm = (
            torch.lgamma((nu + d) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * (d * torch.log(nu * pi) + logdet)
        )
        log_kernel = -0.5 * (nu + d) * torch.log1p(quad / nu)
        return -(log_norm + log_kernel)

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
        current_logit = history_logits[:, -1]
        prev_transition, future_trans = self.future_transitions(history_norm, future_norm)
        target_raw_seq = future_trans if self.cfg.target_mode == "transition" else future_logits
        target_seq = self._to_target_coord(target_raw_seq)
        state = self.init_state(context)
        losses = []
        diag_means = []
        offdiag_abs = []
        mean_abs_err = []
        for step in range(self.cfg.future_len):
            step_idx = torch.full(
                (history_norm.shape[0],),
                step,
                dtype=torch.long,
                device=history_norm.device,
            )
            dec_in = self.transition_in(
                self._decoder_input(prev_transition, current_logit)
            ) + self.step_embed(step_idx)
            state = self.decoder(dec_in, state)
            mean, chol, nu = self._distribution_params(state, current_logit)
            target = target_seq[:, step]
            if self.cfg.distribution == "student_t":
                if nu is None:
                    raise RuntimeError("Student-t distribution did not produce nu")
                losses.append(self.student_t_nll(target, mean, chol, nu))
            else:
                losses.append(self.gaussian_nll(target, mean, chol))
            diag = torch.diagonal(chol, dim1=-2, dim2=-1)
            diag_means.append(diag.mean())
            offdiag_abs.append((chol - torch.diag_embed(diag)).abs().mean())
            mean_abs_err.append((mean - target).abs().mean())
            if self.cfg.target_mode == "transition":
                target_raw = target_raw_seq[:, step]
                current_logit = current_logit + target_raw
                prev_transition = target_raw
            else:
                target_raw = target_raw_seq[:, step]
                prev_transition = target_raw - current_logit
                current_logit = target_raw
        nll = torch.stack(losses, dim=1).mean()
        metrics = {
            "total": nll.detach(),
            "nll": nll.detach(),
            "transition_std": future_trans.std(unbiased=False).detach(),
            "transition_abs": future_trans.abs().mean().detach(),
            "diag_mean": torch.stack(diag_means).mean().detach(),
            "offdiag_abs": torch.stack(offdiag_abs).mean().detach(),
            "mean_abs_err": torch.stack(mean_abs_err).mean().detach(),
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
        history_logits = self.to_logits(history_norm)
        context = self.encode_history(history_norm)
        bsz = history_norm.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            ctx = context.repeat_interleave(k, dim=0)
            current_logit = history_logits[:, -1].repeat_interleave(k, dim=0)
            prev_logit = history_logits[:, -2].repeat_interleave(k, dim=0)
            prev_transition = current_logit - prev_logit
            state = self.init_state(ctx)
            frames = []
            for step in range(self.cfg.future_len):
                step_idx = torch.full(
                    (bsz * k,),
                    step,
                    dtype=torch.long,
                    device=history_norm.device,
                )
                dec_in = self.transition_in(
                    self._decoder_input(prev_transition, current_logit)
                ) + self.step_embed(step_idx)
                state = self.decoder(dec_in, state)
                mean, chol, nu = self._distribution_params(state, current_logit)
                eps = temp * torch.randn(
                    bsz * k,
                    self.cfg.n_cells,
                    device=history_norm.device,
                    dtype=history_norm.dtype,
                )
                draw = torch.bmm(chol, eps.unsqueeze(-1)).squeeze(-1)
                if self.cfg.distribution == "student_t":
                    if nu is None:
                        raise RuntimeError("Student-t distribution did not produce nu")
                    gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
                    mix = gamma.sample().to(draw.dtype).clamp_min(1e-6)
                    draw = draw * torch.rsqrt(mix).unsqueeze(-1)
                draw = mean + draw
                draw_raw = self._from_target_coord(draw)
                if self.cfg.target_mode == "transition":
                    transition = draw_raw
                    current_logit = current_logit + transition
                else:
                    next_logit = draw_raw
                    transition = next_logit - current_logit
                    current_logit = next_logit
                frames.append(logit_to_iv(current_logit))
                prev_transition = transition
            future_01 = torch.stack(frames, dim=1)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, self.cfg.future_len, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[DailyJointCholeskyTransitionModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = DailyJointCholeskyTransitionConfig(**payload["config"])
    model = DailyJointCholeskyTransitionModel(cfg)
    result = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = {"cell_target_mean", "cell_target_std"}
    missing = set(result.missing_keys) - allowed_missing
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    if cfg.standardize_target and allowed_missing.intersection(result.missing_keys):
        raise RuntimeError("Standardized-target checkpoint is missing saved target stats")
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: DailyJointCholeskyTransitionModel,
    cfg: DailyJointCholeskyTransitionConfig,
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
