from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from diffusion.block_ar.causal_future_memory_transition_flow_matching import (
    CausalFutureMemoryTransitionFMConfig,
    CausalFutureMemoryTransitionFlowMatching,
)
from diffusion.block_ar.logit_level_flow_matching import logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


@dataclass
class CausalMemoryLevelFMConfig(CausalFutureMemoryTransitionFMConfig):
    standardize_logits: bool = True


class CausalMemoryLevelFlowMatching(CausalFutureMemoryTransitionFlowMatching):
    """336a: causal-memory AR flow over next logit levels instead of transitions."""

    def __init__(self, cfg: CausalMemoryLevelFMConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory_states, current_logits, future_logits = self.teacher_forced_memory(
            history_norm,
            future_norm,
        )
        x1 = future_logits
        x0 = torch.randn_like(x1)
        bsz, horizon, n_cells = x1.shape
        t = torch.rand(bsz, horizon, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t[..., None]) * x0 + t[..., None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.predict_velocity(
            x_t.reshape(bsz * horizon, n_cells),
            current_logits.reshape(bsz * horizon, n_cells),
            memory_states.reshape(bsz * horizon, self.cfg.memory_dim),
            t.reshape(bsz * horizon),
        ).view_as(x1)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        transitions = future_logits - current_logits
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "level_std": x1.std(unbiased=False).detach(),
            "level_abs": x1.abs().mean().detach(),
            "transition_std": transitions.std(unbiased=False).detach(),
            "transition_abs": transitions.abs().mean().detach(),
            "target_velocity_std": target_velocity.std(unbiased=False).detach(),
            "memory_abs": memory_states.abs().mean().detach(),
        }
        return fm_loss, metrics

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
            raise ValueError(
                f"Expected n_steps in [1,{self.cfg.future_len}], got {n_steps}"
            )
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_logits = self.history_logits(history_norm)
        bsz = history_logits.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(
            self.cfg.sample_temperature if temperature is None else temperature
        )
        dt = 1.0 / float(self.cfg.flow_steps)

        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            prefix = (
                history_logits.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            frames: list[torch.Tensor] = []
            for _step in range(n_steps):
                memory_state = self._encode_prefix_logits(prefix)[:, -1]
                current_logit = prefix[:, -1]
                x = temp * torch.randn_like(current_logit)
                for flow_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (flow_step + 0.5) * dt,
                        device=history_logits.device,
                        dtype=history_logits.dtype,
                    )
                    x = x + dt * self.predict_velocity(
                        x,
                        current_logit,
                        memory_state,
                        t,
                    )
                next_logit = x
                next_iv = logit_to_iv(self._from_model_coord(next_logit))
                frames.append(next_iv.view(bsz, k, 5, 5))
                prefix = torch.cat([prefix, next_logit[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[CausalMemoryLevelFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = CausalMemoryLevelFMConfig(**payload["config"])
    model = CausalMemoryLevelFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: CausalMemoryLevelFlowMatching,
    cfg: CausalMemoryLevelFMConfig,
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
