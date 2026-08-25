from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.conditional_masked_path_flow_matching import MaskedPathAxialBlock
from diffusion.block_ar.empirical_normal_score_path_flow_matching import (
    EmpiricalNormalScorePathFMConfig,
    EmpiricalNormalScorePathFlowMatching,
)
from diffusion.block_ar.joint_token_logit_transition_flow_matching import (
    _flow_time_features,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    normalize_iv,
)


@dataclass
class EmpiricalNormalScoreMaskedFuturePathFMConfig(EmpiricalNormalScorePathFMConfig):
    pass


class MaskedFuturePathVelocity(nn.Module):
    def __init__(self, cfg: EmpiricalNormalScoreMaskedFuturePathFMConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.history_len + cfg.future_len
        input_dim = 3 if cfg.transition_features else 2
        self.value_proj = nn.Linear(input_dim, cfg.token_dim)
        self.flow_time_proj = nn.Linear(cfg.flow_time_dim, cfg.token_dim)
        self.time_embed = nn.Embedding(self.seq_len, cfg.token_dim)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.type_embed = nn.Embedding(2, cfg.token_dim)
        self.blocks = nn.ModuleList(
            [
                MaskedPathAxialBlock(
                    seq_len=self.seq_len,
                    n_cells=cfg.n_cells,
                    token_dim=cfg.token_dim,
                    token_ff=cfg.token_ff,
                    dropout=cfg.model_dropout,
                    global_mixer=cfg.global_mixer,
                )
                for _ in range(cfg.token_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def forward(
        self,
        path_t: torch.Tensor,
        future_observed: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, n_cells = path_t.shape
        if seq_len != self.seq_len or n_cells != self.cfg.n_cells:
            raise ValueError(
                f"Expected path shape (*,{self.seq_len},{self.cfg.n_cells}), "
                f"got {tuple(path_t.shape)}"
            )
        history_observed = torch.ones(
            bsz,
            self.cfg.history_len,
            n_cells,
            device=path_t.device,
            dtype=path_t.dtype,
        )
        full_observed = torch.cat(
            [history_observed, future_observed.to(dtype=path_t.dtype)], dim=1
        )
        if self.cfg.transition_features:
            prev = torch.cat([path_t[:, :1], path_t[:, :-1]], dim=1)
            values = torch.stack([path_t, path_t - prev, full_observed], dim=-1)
        else:
            values = torch.stack([path_t, full_observed], dim=-1)

        time_ids = torch.arange(seq_len, device=path_t.device)
        cell_ids = torch.arange(n_cells, device=path_t.device)
        type_ids = torch.cat(
            [
                torch.zeros(self.cfg.history_len, device=path_t.device, dtype=torch.long),
                torch.ones(self.cfg.future_len, device=path_t.device, dtype=torch.long),
            ],
            dim=0,
        )
        x = self.value_proj(values)
        x = x + self.time_embed(time_ids)[None, :, None, :]
        x = x + self.cell_embed(cell_ids)[None, None, :, :]
        x = x + self.type_embed(type_ids)[None, :, None, :]
        x = x + self.flow_time_proj(_flow_time_features(flow_t, self.cfg.flow_time_dim))[
            :, None, None, :
        ]
        for block in self.blocks:
            x = block(x)
        future = x[:, self.cfg.history_len :, :, :]
        return self.out(future).squeeze(-1)


class EmpiricalNormalScoreMaskedFuturePathFlowMatching(
    EmpiricalNormalScorePathFlowMatching
):
    """345a: full future-path FM trained with random future inpainting masks."""

    def __init__(self, cfg: EmpiricalNormalScoreMaskedFuturePathFMConfig):
        if cfg.mixer_type != "axial":
            raise ValueError("345a masked future-path flow currently supports axial mixer only")
        super().__init__(cfg)
        self.cfg = cfg
        self.velocity = MaskedFuturePathVelocity(cfg)

    @staticmethod
    def _sample_future_observed_mask(x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        reveal_prob = torch.rand(
            bsz, 1, 1, device=x.device, dtype=x.dtype
        ).square()
        return torch.rand_like(x) < reveal_prob

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_z = self.history_scores(history_norm)
        x1 = self.target_future_scores(future_norm)
        x0 = torch.randn_like(x1)
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        observed = self._sample_future_observed_mask(x1)
        future_input = torch.where(observed, x1, x_t)
        path_t = torch.cat([history_z, future_input], dim=1)
        target_velocity = x1 - x0
        pred_velocity = self.velocity(path_t, observed, t)
        hidden = (~observed).to(dtype=x1.dtype)
        sq_err = (pred_velocity - target_velocity).square()
        loss = (sq_err * hidden).sum() / hidden.sum().clamp_min(1.0)
        transitions = x1 - torch.cat([history_z[:, -1:], x1[:, :-1]], dim=1)
        metrics = {
            "total": loss.detach(),
            "fm_loss": loss.detach(),
            "future_score_std": x1.std(unbiased=False).detach(),
            "future_score_abs": x1.abs().mean().detach(),
            "implied_transition_std": transitions.std(unbiased=False).detach(),
            "implied_transition_abs": transitions.abs().mean().detach(),
            "observed_fraction": observed.to(dtype=x1.dtype).mean().detach(),
        }
        return loss, metrics

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
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(f"Expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_z = self.history_scores(history_norm)
        bsz = history_z.shape[0]
        chunk_size = max(
            1, min(int(chunk_size), int(n_samples), int(self.cfg.max_sample_chunk))
        )
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            x = temp * torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=history_z.device,
                dtype=history_z.dtype,
            )
            hist = history_z.repeat_interleave(k, dim=0)
            observed = torch.zeros_like(x, dtype=torch.bool)
            for step in range(self.cfg.flow_steps):
                t = torch.full(
                    (bsz * k,),
                    (step + 0.5) * dt,
                    device=history_z.device,
                    dtype=history_z.dtype,
                )
                path_t = torch.cat([hist, x], dim=1)
                x = x + dt * self.velocity(path_t, observed, t)
            future_01 = self._scores_to_values(x[:, :n_steps], self.future_quantiles)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, k, n_steps, 5, 5)
            else:
                future_01 = future_01.view(bsz, k, n_steps, self.cfg.n_cells)
            outs.append(future_01)
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.sample_batched(
            normalize_iv(history_01),
            n_samples=n_samples,
            n_steps=1,
            history_is_normalized=True,
            **kwargs,
        )[:, :, 0]


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[EmpiricalNormalScoreMaskedFuturePathFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScoreMaskedFuturePathFMConfig(**payload["config"])
    model = EmpiricalNormalScoreMaskedFuturePathFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScoreMaskedFuturePathFlowMatching,
    cfg: EmpiricalNormalScoreMaskedFuturePathFMConfig,
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
