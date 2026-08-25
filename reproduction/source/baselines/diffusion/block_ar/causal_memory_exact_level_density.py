from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, pi

import torch

from diffusion.block_ar.causal_future_memory_transition_flow_matching import (
    CausalFutureMemoryTransitionFMConfig,
    CausalFutureMemoryTransitionFlowMatching,
)
from diffusion.block_ar.causal_memory_exact_transition_density import (
    ConditionalCouplingFlow,
)
from diffusion.block_ar.logit_level_flow_matching import logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


@dataclass
class CausalMemoryExactLevelDensityConfig(CausalFutureMemoryTransitionFMConfig):
    flow_layers: int = 8
    flow_hidden: int = 256
    scale_clip: float = 2.0
    standardize_logits: bool = True


class CausalMemoryExactLevelDensity(CausalFutureMemoryTransitionFlowMatching):
    """337a: exact chain-rule density for next standardized logit levels."""

    def __init__(self, cfg: CausalMemoryExactLevelDensityConfig):
        super().__init__(cfg)
        self.cfg = cfg
        del self.velocity
        self.flow = ConditionalCouplingFlow(cfg)

    def _condition(
        self,
        memory_states: torch.Tensor,
        current_logits: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([memory_states, current_logits], dim=-1)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory_states, current_logits, future_logits = self.teacher_forced_memory(
            history_norm,
            future_norm,
        )
        x = future_logits
        transitions = future_logits - current_logits
        bsz, horizon, n_cells = x.shape
        cond = self._condition(memory_states, current_logits)
        z, logdet = self.flow.data_to_base(
            x.reshape(bsz * horizon, n_cells),
            cond.reshape(bsz * horizon, self.cfg.memory_dim + self.cfg.n_cells),
        )
        base_nll = 0.5 * (z.square() + log(2.0 * pi)).sum(dim=-1)
        nll = (base_nll - logdet).mean()
        metrics = {
            "total": nll.detach(),
            "nll": nll.detach(),
            "level_std": x.std(unbiased=False).detach(),
            "level_abs": x.abs().mean().detach(),
            "transition_std": transitions.std(unbiased=False).detach(),
            "transition_abs": transitions.abs().mean().detach(),
            "base_std": z.std(unbiased=False).detach(),
            "logdet": logdet.mean().detach(),
            "memory_abs": memory_states.abs().mean().detach(),
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
                cond = self._condition(memory_state, current_logit)
                z = temp * torch.randn_like(current_logit)
                next_logit, _logdet = self.flow.base_to_data(z, cond)
                next_iv = logit_to_iv(self._from_model_coord(next_logit))
                frames.append(next_iv.view(bsz, k, 5, 5))
                prefix = torch.cat([prefix, next_logit[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[CausalMemoryExactLevelDensity, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = CausalMemoryExactLevelDensityConfig(**payload["config"])
    model = CausalMemoryExactLevelDensity(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: CausalMemoryExactLevelDensity,
    cfg: CausalMemoryExactLevelDensityConfig,
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
