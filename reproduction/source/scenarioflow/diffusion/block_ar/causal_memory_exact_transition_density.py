from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from diffusion.block_ar.causal_future_memory_transition_flow_matching import (
    CausalFutureMemoryTransitionFMConfig,
    CausalFutureMemoryTransitionFlowMatching,
)
from diffusion.block_ar.logit_level_flow_matching import logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


@dataclass
class CausalMemoryExactTransitionDensityConfig(CausalFutureMemoryTransitionFMConfig):
    flow_layers: int = 8
    flow_hidden: int = 256
    scale_clip: float = 2.0
    standardize_logits: bool = True


class ConditionalAffineCoupling(nn.Module):
    def __init__(
        self,
        n_cells: int,
        cond_dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        scale_clip: float,
    ):
        super().__init__()
        self.register_buffer("mask", mask.view(1, n_cells))
        self.scale_clip = float(scale_clip)
        self.net = nn.Sequential(
            nn.Linear(n_cells + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_cells),
        )

    def _shift_scale(self, x_masked: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.net(torch.cat([x_masked, cond], dim=-1)).chunk(2, dim=-1)
        inv_mask = 1.0 - self.mask
        log_scale = self.scale_clip * torch.tanh(log_scale / self.scale_clip) * inv_mask
        shift = shift * inv_mask
        return shift, log_scale

    def data_to_base(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        shift, log_scale = self._shift_scale(x_masked, cond)
        inv_mask = 1.0 - self.mask
        z = x_masked + inv_mask * ((x - shift) * torch.exp(-log_scale))
        logdet = -log_scale.sum(dim=-1)
        return z, logdet

    def base_to_data(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_masked = z * self.mask
        shift, log_scale = self._shift_scale(z_masked, cond)
        inv_mask = 1.0 - self.mask
        x = z_masked + inv_mask * (z * torch.exp(log_scale) + shift)
        logdet = log_scale.sum(dim=-1)
        return x, logdet


class ConditionalCouplingFlow(nn.Module):
    def __init__(self, cfg: CausalMemoryExactTransitionDensityConfig):
        super().__init__()
        cond_dim = cfg.memory_dim + cfg.n_cells
        layers = []
        base = (torch.arange(cfg.n_cells) % 2).float()
        for layer_idx in range(cfg.flow_layers):
            mask = base if layer_idx % 2 == 0 else 1.0 - base
            layers.append(
                ConditionalAffineCoupling(
                    n_cells=cfg.n_cells,
                    cond_dim=cond_dim,
                    hidden_dim=cfg.flow_hidden,
                    mask=mask,
                    scale_clip=cfg.scale_clip,
                )
            )
        self.layers = nn.ModuleList(layers)

    def data_to_base(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, ld = layer.data_to_base(z, cond)
            logdet = logdet + ld
        return z, logdet

    def base_to_data(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            x, ld = layer.base_to_data(x, cond)
            logdet = logdet + ld
        return x, logdet


class CausalMemoryExactTransitionDensity(CausalFutureMemoryTransitionFlowMatching):
    """331a: causal future-memory AR model with exact daily transition likelihood."""

    def __init__(self, cfg: CausalMemoryExactTransitionDensityConfig):
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
        x = future_logits - current_logits
        bsz, horizon, n_cells = x.shape
        cond = self._condition(memory_states, current_logits)
        z, logdet = self.flow.data_to_base(
            x.reshape(bsz * horizon, n_cells),
            cond.reshape(bsz * horizon, self.cfg.memory_dim + self.cfg.n_cells),
        )
        base_nll = 0.5 * (z.square() + torch.log(torch.tensor(2.0 * torch.pi, device=z.device))).sum(dim=-1)
        nll = (base_nll - logdet).mean()
        metrics = {
            "total": nll.detach(),
            "nll": nll.detach(),
            "transition_std": x.std(unbiased=False).detach(),
            "transition_abs": x.abs().mean().detach(),
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
            raise ValueError(f"Expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_logits = self.history_logits(history_norm)
        bsz = history_logits.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)

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
                x, _logdet = self.flow.base_to_data(z, cond)
                next_logit = current_logit + x
                next_iv = logit_to_iv(self._from_model_coord(next_logit))
                frames.append(next_iv.view(bsz, k, 5, 5))
                prefix = torch.cat([prefix, next_logit[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[CausalMemoryExactTransitionDensity, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = CausalMemoryExactTransitionDensityConfig(**payload["config"])
    model = CausalMemoryExactTransitionDensity(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: CausalMemoryExactTransitionDensity,
    cfg: CausalMemoryExactTransitionDensityConfig,
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
