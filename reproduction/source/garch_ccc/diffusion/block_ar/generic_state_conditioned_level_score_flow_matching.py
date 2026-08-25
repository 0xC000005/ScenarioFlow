from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from diffusion.block_ar.generic_state_conditioned_increment_flow_matching import (
    GenericStateConditionedIncrementFMConfig,
    GenericStateConditionedIncrementFlowMatching,
)


@dataclass
class GenericStateConditionedLevelScoreFMConfig(
    GenericStateConditionedIncrementFMConfig
):
    """Flow for p(delta level_score_t | level_score history, increment_score history)."""


class GenericStateConditionedLevelScoreFlowMatching(
    GenericStateConditionedIncrementFlowMatching
):
    """State-conditioned transition that samples next level-score changes.

    The model conditions on the same level and increment histories as 629a/631a,
    but the stochastic variable is the next empirical level-score change. During
    rollout, sampled next levels are decoded to encoded state values and encoded
    increments are derived by differencing consecutive encoded levels.
    """

    cfg: GenericStateConditionedLevelScoreFMConfig

    def _level_scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        return self._scores_to_values(scores, self.level_quantiles)

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

        x1 = future_level_scores - current_level_scores
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
            "target_level_delta_score_std": x1.std(unbiased=False).detach(),
            "target_level_delta_score_abs": x1.abs().mean().detach(),
            "target_increment_score_std": future_increment_scores.std(
                unbiased=False
            ).detach(),
            "target_increment_score_abs": future_increment_scores.abs().mean().detach(),
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
                next_level_score = current_level_score + x
                next_level_value = self._level_scores_to_values(next_level_score)
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
                    [prefix_increment_scores, next_increment_score[:, None, :]],
                    dim=1,
                )
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GenericStateConditionedLevelScoreFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GenericStateConditionedLevelScoreFMConfig(**payload["config"])
    model = GenericStateConditionedLevelScoreFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: GenericStateConditionedLevelScoreFlowMatching,
    cfg: GenericStateConditionedLevelScoreFMConfig,
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
