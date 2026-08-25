from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


def _time_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        torch.linspace(
            0.0,
            -torch.log(torch.tensor(10000.0, device=device)),
            half,
            device=device,
        )
    )
    angles = t[:, None] * freqs[None, :] * 2.0 * torch.pi
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


@dataclass
class RecurrentLogitTransitionFMConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    hidden_dim: int = 128
    gru_layers: int = 2
    gru_dropout: float = 0.1

    model_hidden: int = 256
    model_layers: int = 4
    model_dropout: float = 0.1
    time_dim: int = 32

    logit_eps: float = 1e-4
    flow_steps: int = 32
    sample_temperature: float = 1.0


class TransitionVelocityMLP(nn.Module):
    def __init__(self, cfg: RecurrentLogitTransitionFMConfig):
        super().__init__()
        in_dim = cfg.n_cells + cfg.n_cells + cfg.hidden_dim + cfg.time_dim
        layers: list[nn.Module] = [
            nn.Linear(in_dim, cfg.model_hidden),
            nn.GELU(),
        ]
        for _ in range(max(0, cfg.model_layers - 1)):
            layers.extend(
                [
                    nn.Dropout(cfg.model_dropout),
                    nn.Linear(cfg.model_hidden, cfg.model_hidden),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(cfg.model_hidden, cfg.n_cells))
        self.net = nn.Sequential(*layers)
        self.cfg = cfg

    def forward(
        self,
        x_t: torch.Tensor,
        current_logit: torch.Tensor,
        state_top: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        time = _time_features(t, self.cfg.time_dim)
        return self.net(torch.cat([x_t, current_logit, state_top, time], dim=-1))


class RecurrentLogitTransitionFlowMatching(nn.Module):
    """303a-v0: recurrent vanilla flow over support-valid logit transitions."""

    def __init__(self, cfg: RecurrentLogitTransitionFMConfig):
        super().__init__()
        self.cfg = cfg
        self.input_dim = 2 * cfg.n_cells
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.gru_layers,
            dropout=cfg.gru_dropout if cfg.gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.recurrent_cells = nn.ModuleList(
            [
                nn.GRUCell(self.input_dim if i == 0 else cfg.hidden_dim, cfg.hidden_dim)
                for i in range(cfg.gru_layers)
            ]
        )
        self.velocity = TransitionVelocityMLP(cfg)
        self.init_recurrent_cells_from_gru()

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def init_recurrent_cells_from_gru(self) -> None:
        with torch.no_grad():
            for layer_idx, cell in enumerate(self.recurrent_cells):
                cell.weight_ih.copy_(getattr(self.gru, f"weight_ih_l{layer_idx}"))
                cell.weight_hh.copy_(getattr(self.gru, f"weight_hh_l{layer_idx}"))
                cell.bias_ih.copy_(getattr(self.gru, f"bias_ih_l{layer_idx}"))
                cell.bias_hh.copy_(getattr(self.gru, f"bias_hh_l{layer_idx}"))

    def _logit_features_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(logits)
        deltas[:, 1:] = logits[:, 1:] - logits[:, :-1]
        return torch.cat([logits, deltas], dim=-1)

    def _step_feature_from_logits(
        self,
        prev_logit: torch.Tensor,
        next_logit: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([next_logit, next_logit - prev_logit], dim=-1)

    def encode_history(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history_norm = self._flatten(history_norm)
        history_01 = denormalize_iv(history_norm)
        logits = iv_to_logit(history_01, self.cfg.logit_eps)
        features = self._logit_features_from_logits(logits)
        _out, state_stack = self.gru(features)
        return state_stack, logits[:, -1]

    def recurrent_step(
        self,
        state_stack: torch.Tensor,
        prev_logit: torch.Tensor,
        next_logit: torch.Tensor,
    ) -> torch.Tensor:
        layer_input = self._step_feature_from_logits(prev_logit, next_logit)
        next_states = []
        for layer_idx, cell in enumerate(self.recurrent_cells):
            h_next = cell(layer_input, state_stack[layer_idx])
            next_states.append(h_next)
            layer_input = h_next
        return torch.stack(next_states, dim=0)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        current_logit: torch.Tensor,
        state_stack: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity(x_t, current_logit, state_stack[-1], t)

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        return iv_to_logit(denormalize_iv(future_norm), self.cfg.logit_eps)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_logits = self.target_future_logits(future_norm)
        state_stack, current_logit = self.encode_history(history_norm)

        losses: list[torch.Tensor] = []
        transition_abs = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)
        transition_std = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)
        velocity_std = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)
        state_abs = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)
        bsz = history_norm.shape[0]

        for step in range(future_logits.shape[1]):
            next_logit = future_logits[:, step]
            x1 = next_logit - current_logit
            x0 = torch.randn_like(x1)
            t = torch.rand(bsz, device=history_norm.device, dtype=history_norm.dtype)
            x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
            target_velocity = x1 - x0
            pred_velocity = self.predict_velocity(x_t, current_logit, state_stack, t)
            losses.append(F.mse_loss(pred_velocity, target_velocity))
            transition_abs = transition_abs + x1.abs().mean()
            transition_std = transition_std + x1.std(unbiased=False)
            velocity_std = velocity_std + target_velocity.std(unbiased=False)
            state_abs = state_abs + current_logit.abs().mean()

            state_stack = self.recurrent_step(state_stack, current_logit, next_logit)
            current_logit = next_logit

        horizon = float(future_logits.shape[1])
        fm_loss = torch.stack(losses).mean()
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "transition_std": (transition_std / horizon).detach(),
            "transition_abs": (transition_abs / horizon).detach(),
            "target_velocity_std": (velocity_std / horizon).detach(),
            "state_logit_abs": (state_abs / horizon).detach(),
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
        if n_steps < 1:
            raise ValueError(f"Expected positive n_steps, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        base_state, base_logit = self.encode_history(history_norm)
        bsz = history_norm.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)

        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            state_stack = (
                base_state.unsqueeze(2)
                .expand(self.cfg.gru_layers, bsz, k, self.cfg.hidden_dim)
                .reshape(self.cfg.gru_layers, bsz * k, self.cfg.hidden_dim)
                .clone()
            )
            current_logit = (
                base_logit.unsqueeze(1)
                .expand(bsz, k, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.n_cells)
                .clone()
            )

            frames: list[torch.Tensor] = []
            for _step in range(n_steps):
                x = temp * torch.randn(
                    bsz * k,
                    self.cfg.n_cells,
                    device=base_logit.device,
                    dtype=base_logit.dtype,
                )
                for flow_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (flow_step + 0.5) * dt,
                        device=base_logit.device,
                        dtype=base_logit.dtype,
                    )
                    x = x + dt * self.predict_velocity(x, current_logit, state_stack, t)
                next_logit = current_logit + x
                next_iv = logit_to_iv(next_logit)
                frames.append(next_iv.view(bsz, k, 5, 5))
                state_stack = self.recurrent_step(state_stack, current_logit, next_logit)
                current_logit = next_logit
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        **kwargs: object,
    ) -> torch.Tensor:
        history_norm = normalize_iv(history_01)
        return self.sample_batched(
            history_norm,
            n_samples=n_samples,
            n_steps=1,
            history_is_normalized=True,
            **kwargs,
        )[:, :, 0]


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[RecurrentLogitTransitionFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = RecurrentLogitTransitionFMConfig(**payload["config"])
    model = RecurrentLogitTransitionFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: RecurrentLogitTransitionFlowMatching,
    cfg: RecurrentLogitTransitionFMConfig,
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
