from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from diffusion.block_ar.empirical_normal_score_path_flow_matching import (
    EmpiricalNormalScorePathFMConfig,
    EmpiricalNormalScorePathFlowMatching,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


@dataclass
class EmpiricalNormalScorePathDiffusionConfig(EmpiricalNormalScorePathFMConfig):
    """354a uses the 339 path representation with a VP/DDIM denoising core."""


class EmpiricalNormalScorePathDiffusion(EmpiricalNormalScorePathFlowMatching):
    """Full future-path denoising diffusion in empirical normal-score coordinates."""

    cfg: EmpiricalNormalScorePathDiffusionConfig

    @staticmethod
    def _vp_alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = 0.5 * math.pi * t
        return torch.cos(theta), torch.sin(theta)

    def _predict_noise(
        self,
        noisy_future: torch.Tensor,
        history_z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.mixer_type == "transformer":
            context = self.encode_history_scores(history_z)
            return self.velocity(noisy_future, context, t, history_z[:, -1])
        return self.velocity(torch.cat([history_z, noisy_future], dim=1), t)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_z = self.history_scores(history_norm)
        clean_future = self.target_future_scores(future_norm)
        noise = torch.randn_like(clean_future)
        bsz = clean_future.shape[0]
        t = torch.rand(bsz, device=clean_future.device, dtype=clean_future.dtype)
        alpha, sigma = self._vp_alpha_sigma(t)
        noisy_future = alpha[:, None, None] * clean_future + sigma[:, None, None] * noise
        pred_noise = self._predict_noise(noisy_future, history_z, t)
        denoise_loss = F.mse_loss(pred_noise, noise)
        transitions = clean_future - torch.cat([history_z[:, -1:], clean_future[:, :-1]], dim=1)
        metrics = {
            "total": denoise_loss.detach(),
            "denoise_loss": denoise_loss.detach(),
            "future_score_std": clean_future.std(unbiased=False).detach(),
            "future_score_abs": clean_future.abs().mean().detach(),
            "implied_transition_std": transitions.std(unbiased=False).detach(),
            "implied_transition_abs": transitions.abs().mean().detach(),
            "noise_std": noise.std(unbiased=False).detach(),
            "noisy_future_std": noisy_future.std(unbiased=False).detach(),
        }
        return denoise_loss, metrics

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
        n_ddim = max(1, int(self.cfg.flow_steps))
        dt = 1.0 / float(n_ddim)
        context = None
        if self.cfg.mixer_type == "transformer":
            context = self.encode_history_scores(history_z)

        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            hist = history_z.repeat_interleave(k, dim=0)
            ctx = context.repeat_interleave(k, dim=0) if context is not None else None
            x = temp * torch.randn(
                bsz * k,
                self.cfg.future_len,
                self.cfg.n_cells,
                device=history_z.device,
                dtype=history_z.dtype,
            )
            for ddim_step in range(n_ddim - 1, -1, -1):
                t_value = (ddim_step + 0.5) * dt
                t_next_value = max((ddim_step - 0.5) * dt, 0.0)
                t = torch.full(
                    (bsz * k,),
                    t_value,
                    device=history_z.device,
                    dtype=history_z.dtype,
                )
                if self.cfg.mixer_type == "transformer":
                    if ctx is None:
                        raise RuntimeError("Missing transformer context")
                    pred_noise = self.velocity(x, ctx, t, hist[:, -1])
                else:
                    pred_noise = self.velocity(torch.cat([hist, x], dim=1), t)
                alpha_t, sigma_t = self._vp_alpha_sigma(t)
                x0_hat = (x - sigma_t[:, None, None] * pred_noise) / alpha_t[
                    :, None, None
                ].clamp_min(1e-4)
                t_next = torch.full_like(t, t_next_value)
                alpha_next, sigma_next = self._vp_alpha_sigma(t_next)
                x = alpha_next[:, None, None] * x0_hat + sigma_next[:, None, None] * pred_noise
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
) -> tuple[EmpiricalNormalScorePathDiffusion, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScorePathDiffusionConfig(**payload["config"])
    model = EmpiricalNormalScorePathDiffusion(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScorePathDiffusion,
    cfg: EmpiricalNormalScorePathDiffusionConfig,
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
