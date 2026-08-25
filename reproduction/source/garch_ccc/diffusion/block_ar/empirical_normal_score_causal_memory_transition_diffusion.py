from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
    EmpiricalNormalScoreCausalMemoryTransitionFMConfig,
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


@dataclass
class EmpiricalNormalScoreCausalMemoryTransitionDiffusionConfig(
    EmpiricalNormalScoreCausalMemoryTransitionFMConfig
):
    """352a uses the 340c structure with a VP/DDIM denoising core."""


class EmpiricalNormalScoreCausalMemoryTransitionDiffusion(
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching
):
    """Causal-memory AR transition diffusion in shared empirical normal-score coordinates."""

    cfg: EmpiricalNormalScoreCausalMemoryTransitionDiffusionConfig

    @staticmethod
    def _vp_alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = 0.5 * math.pi * t
        return torch.cos(theta), torch.sin(theta)

    def predict_epsilon(
        self,
        x_t: torch.Tensor,
        current_score: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_velocity(x_t, current_score, memory_state, t)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory_states, current_scores, future_scores = self.teacher_forced_memory(
            history_norm,
            future_norm,
        )
        clean_transition = future_scores - current_scores
        noise = torch.randn_like(clean_transition)
        bsz, horizon, n_cells = clean_transition.shape
        t = torch.rand(
            bsz,
            horizon,
            device=clean_transition.device,
            dtype=clean_transition.dtype,
        )
        alpha, sigma = self._vp_alpha_sigma(t)
        noisy_transition = alpha[..., None] * clean_transition + sigma[..., None] * noise
        pred_noise = self.predict_epsilon(
            noisy_transition.reshape(bsz * horizon, n_cells),
            current_scores.reshape(bsz * horizon, n_cells),
            memory_states.reshape(bsz * horizon, self.cfg.memory_dim),
            t.reshape(bsz * horizon),
        ).view_as(clean_transition)
        denoise_loss = F.mse_loss(pred_noise, noise)
        metrics = {
            "total": denoise_loss.detach(),
            "denoise_loss": denoise_loss.detach(),
            "transition_std": clean_transition.std(unbiased=False).detach(),
            "transition_abs": clean_transition.abs().mean().detach(),
            "noise_std": noise.std(unbiased=False).detach(),
            "noisy_transition_std": noisy_transition.std(unbiased=False).detach(),
            "memory_abs": memory_states.abs().mean().detach(),
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
        history_scores = self.history_scores(history_norm)
        bsz = history_scores.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        n_ddim = max(1, int(self.cfg.flow_steps))
        dt = 1.0 / float(n_ddim)

        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
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
                x = temp * torch.randn_like(current_score)
                for ddim_step in range(n_ddim - 1, -1, -1):
                    t_value = (ddim_step + 0.5) * dt
                    t_next_value = max((ddim_step - 0.5) * dt, 0.0)
                    t = torch.full(
                        (bsz * k,),
                        t_value,
                        device=history_scores.device,
                        dtype=history_scores.dtype,
                    )
                    pred_noise = self.predict_epsilon(x, current_score, memory_state, t)
                    alpha_t, sigma_t = self._vp_alpha_sigma(t)
                    x0_hat = (x - sigma_t[:, None] * pred_noise) / alpha_t[:, None].clamp_min(
                        1e-4
                    )
                    t_next = torch.full_like(t, t_next_value)
                    alpha_next, sigma_next = self._vp_alpha_sigma(t_next)
                    x = alpha_next[:, None] * x0_hat + sigma_next[:, None] * pred_noise
                next_score = current_score + x
                next_iv = self._scores_to_values(next_score)
                frames.append(next_iv.view(bsz, k, 5, 5))
                prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
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
) -> tuple[EmpiricalNormalScoreCausalMemoryTransitionDiffusion, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EmpiricalNormalScoreCausalMemoryTransitionDiffusionConfig(**payload["config"])
    model = EmpiricalNormalScoreCausalMemoryTransitionDiffusion(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: EmpiricalNormalScoreCausalMemoryTransitionDiffusion,
    cfg: EmpiricalNormalScoreCausalMemoryTransitionDiffusionConfig,
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
