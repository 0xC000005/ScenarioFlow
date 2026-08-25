from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from diffusion.block_ar.generic_state_conditioned_level_score_flow_matching import (
    GenericStateConditionedLevelScoreFMConfig,
    GenericStateConditionedLevelScoreFlowMatching,
)


@dataclass
class GenericStateConditionedMixedCoordinateFMConfig(
    GenericStateConditionedLevelScoreFMConfig
):
    """State-conditioned flow with a fixed generated-coordinate policy per channel."""

    level_score_channels: list[int] = field(default_factory=list)


class GenericStateConditionedMixedCoordinateFlowMatching(
    GenericStateConditionedLevelScoreFlowMatching
):
    """One shared transition model with channel-specific generated coordinates.

    Channels listed in ``level_score_channels`` generate next empirical
    level-score changes. Other channels generate encoded increments. The model
    still uses one memory, one velocity network, one optimizer, and one rollout.
    """

    cfg: GenericStateConditionedMixedCoordinateFMConfig

    def __init__(self, cfg: GenericStateConditionedMixedCoordinateFMConfig):
        super().__init__(cfg)
        mask = torch.zeros(cfg.n_cells, dtype=torch.bool)
        if cfg.level_score_channels:
            idx = torch.tensor(cfg.level_score_channels, dtype=torch.long)
            if idx.min().item() < 0 or idx.max().item() >= cfg.n_cells:
                raise ValueError("level_score_channels contains out-of-range indices")
            mask[idx] = True
        self.register_buffer("level_score_mask", mask)

    def training_loss(
        self,
        history_level_values: torch.Tensor,
        history_increment_values: torch.Tensor,
        future_level_values: torch.Tensor,
        future_increment_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_level_scores = self.level_values_to_scores(history_level_values)
        future_level_scores = self.level_values_to_scores(future_level_values)
        history_increment_scores = self.increment_values_to_scores(
            history_increment_values
        )
        future_increment_scores = self.increment_values_to_scores(
            future_increment_values
        )
        prefix_level_scores = torch.cat(
            [history_level_scores, future_level_scores[:, :-1]], dim=1
        )
        prefix_increment_scores = torch.cat(
            [history_increment_scores, future_increment_scores[:, :-1]], dim=1
        )
        hidden = self._encode_prefix(prefix_level_scores, prefix_increment_scores)
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        current_level_scores = prefix_level_scores[
            :, start : start + self.cfg.future_len
        ]

        level_delta_scores = future_level_scores - current_level_scores
        mask = self.level_score_mask.to(device=future_level_scores.device)[
            None, None, :
        ]
        x1 = torch.where(mask, level_delta_scores, future_increment_scores)
        x0 = torch.randn_like(x1)
        source_scale = self._conditional_source_scale(memory_states)
        if source_scale is not None:
            x0 = x0 * source_scale
        bsz, horizon, n_vars = x1.shape
        t = torch.rand(bsz, horizon, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t[..., None]) * x0 + t[..., None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.velocity(
            x_t.reshape(bsz * horizon, n_vars),
            current_level_scores.reshape(bsz * horizon, n_vars),
            memory_states.reshape(bsz * horizon, self.cfg.memory_dim),
            t.reshape(bsz * horizon),
        ).view_as(x1)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "target_mixed_coord_std": x1.std(unbiased=False).detach(),
            "target_mixed_coord_abs": x1.abs().mean().detach(),
            "target_level_delta_score_std": level_delta_scores.std(
                unbiased=False
            ).detach(),
            "target_increment_score_std": future_increment_scores.std(
                unbiased=False
            ).detach(),
            "level_score_channel_frac": self.level_score_mask.float().mean().detach(),
            "memory_abs": memory_states.abs().mean().detach(),
        }
        if source_scale is not None:
            metrics.update(
                {
                    "source_scale_mean": source_scale.mean().detach(),
                    "source_scale_std": source_scale.std(unbiased=False).detach(),
                    "source_scale_min": source_scale.min().detach(),
                    "source_scale_max": source_scale.max().detach(),
                }
            )
        return fm_loss, metrics

    @torch.no_grad()
    def sample_batched(
        self,
        history_level_values: torch.Tensor,
        history_increment_values: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(
                f"expected n_steps in [1,{self.cfg.future_len}], got {n_steps}"
            )
        level_scores = self.level_values_to_scores(history_level_values)
        increment_scores = self.increment_values_to_scores(history_increment_values)
        bsz = int(level_scores.shape[0])
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(
            self.cfg.sample_temperature if temperature is None else temperature
        )
        dt = 1.0 / float(self.cfg.flow_steps)
        mask = self.level_score_mask.to(device=level_scores.device)[None, :]
        outs: list[torch.Tensor] = []
        for start in range(0, int(n_samples), chunk_size):
            k = min(chunk_size, int(n_samples) - start)
            prefix_level_scores = (
                level_scores.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            prefix_increment_scores = (
                increment_scores.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            prefix_level_values = (
                history_level_values.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            frames: list[torch.Tensor] = []
            for _step in range(n_steps):
                memory_state = self._encode_prefix(
                    prefix_level_scores, prefix_increment_scores
                )[:, -1]
                current_level_score = prefix_level_scores[:, -1]
                x = temp * torch.randn(
                    bsz * k,
                    self.cfg.n_cells,
                    device=level_scores.device,
                    dtype=level_scores.dtype,
                )
                source_scale = self._conditional_source_scale(memory_state)
                if source_scale is not None:
                    x = x * source_scale
                for flow_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (flow_step + 0.5) * dt,
                        device=level_scores.device,
                        dtype=level_scores.dtype,
                    )
                    x = x + dt * self.velocity(x, current_level_score, memory_state, t)

                level_next_score = current_level_score + x
                level_next_value = self._level_scores_to_values(level_next_score)
                increment_score_next = x
                increment_next_value = self.increment_scores_to_values(
                    increment_score_next
                )
                increment_next_level = prefix_level_values[:, -1] + increment_next_value
                increment_next_level_score = self.level_values_to_scores(
                    increment_next_level
                )

                next_level_value = torch.where(
                    mask, level_next_value, increment_next_level
                )
                next_level_score = torch.where(
                    mask, level_next_score, increment_next_level_score
                )
                next_increment_value = next_level_value - prefix_level_values[:, -1]
                next_increment_score = self.increment_values_to_scores(
                    next_increment_value
                )

                frames.append(next_increment_value.view(bsz, k, self.cfg.n_cells))
                prefix_level_values = torch.cat(
                    [prefix_level_values, next_level_value[:, None, :]], dim=1
                )
                prefix_level_scores = torch.cat(
                    [prefix_level_scores, next_level_score[:, None, :]], dim=1
                )
                prefix_increment_scores = torch.cat(
                    [prefix_increment_scores, next_increment_score[:, None, :]], dim=1
                )
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GenericStateConditionedMixedCoordinateFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GenericStateConditionedMixedCoordinateFMConfig(**payload["config"])
    model = GenericStateConditionedMixedCoordinateFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: GenericStateConditionedMixedCoordinateFlowMatching,
    cfg: GenericStateConditionedMixedCoordinateFMConfig,
    epoch: int,
    best_val: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": asdict(cfg),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
