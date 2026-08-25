from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class DeterministicHistoryMemoryDiscreteWorldModelConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    latent_dim: int = 128
    history_hidden: int = 128
    history_layers: int = 2
    obs_hidden: int = 160
    context_dim: int = 128
    nhead: int = 4
    decoder_hidden: int = 192
    dropout: float = 0.1

    n_bins: int = 255
    change_limit: float = 0.5
    level_weight: float = 1.0
    ce_weight: float = 1.0
    inference_hard_decode: bool = True


class DeterministicHistoryMemoryDiscreteWorldModel(nn.Module):
    """290a-v0: history-memory world model with discretized next-change target."""

    def __init__(self, cfg: DeterministicHistoryMemoryDiscreteWorldModelConfig):
        super().__init__()
        self.cfg = cfg

        self.hist_in = nn.Linear(2 * cfg.n_cells, cfg.history_hidden)
        self.history_rnn = nn.GRU(
            input_size=cfg.history_hidden,
            hidden_size=cfg.latent_dim,
            num_layers=cfg.history_layers,
            dropout=cfg.dropout if cfg.history_layers > 1 else 0.0,
            batch_first=True,
        )
        self.history_proj = nn.Linear(cfg.latent_dim, cfg.context_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(2 * cfg.n_cells, cfg.obs_hidden),
            nn.GELU(),
            nn.Linear(cfg.obs_hidden, cfg.context_dim),
        )
        self.query_proj = nn.Linear(cfg.latent_dim + cfg.context_dim, cfg.context_dim)
        self.history_attn = nn.MultiheadAttention(
            embed_dim=cfg.context_dim,
            num_heads=cfg.nhead,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transition = nn.GRUCell(2 * cfg.context_dim, cfg.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim + 2 * cfg.context_dim, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.n_cells * cfg.n_bins),
        )

        centers = torch.linspace(-cfg.change_limit, cfg.change_limit, cfg.n_bins)
        self.register_buffer("bin_centers", centers)

    @staticmethod
    def _flatten_levels(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def _history_tokens(self, levels: torch.Tensor) -> torch.Tensor:
        prev = torch.cat([levels[:, :1], levels[:, :-1]], dim=1)
        changes = levels - prev
        return torch.cat([levels, changes], dim=-1)

    def encode_history(
        self, history_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hist = self._flatten_levels(history_norm)
        tokens = self._history_tokens(hist)
        x = torch.tanh(self.hist_in(tokens))
        memory_raw, h_last = self.history_rnn(x)
        memory = self.history_proj(memory_raw)
        latent = h_last[-1]
        curr_level = hist[:, -1]
        prev_change = hist[:, -1] - hist[:, -2]
        return latent, memory, curr_level, prev_change

    def _obs_embed(self, curr_level: torch.Tensor, prev_change: torch.Tensor) -> torch.Tensor:
        return self.obs_encoder(torch.cat([curr_level, prev_change], dim=-1))

    def _history_context(self, latent: torch.Tensor, obs_emb: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(torch.cat([latent, obs_emb], dim=-1)).unsqueeze(1)
        context, _ = self.history_attn(query, memory, memory, need_weights=False)
        return context.squeeze(1)

    def _decode_logits(self, latent: torch.Tensor, obs_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(torch.cat([latent, obs_emb, context], dim=-1))
        return logits.view(latent.shape[0], self.cfg.n_cells, self.cfg.n_bins)

    def _soft_change_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        return torch.sum(probs * self.bin_centers.view(1, 1, -1), dim=-1)

    def _hard_change_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        idx = logits.argmax(dim=-1)
        return self.bin_centers[idx]

    def _targets_to_bins(self, target_changes: torch.Tensor) -> torch.Tensor:
        clipped = target_changes.clamp(-self.cfg.change_limit, self.cfg.change_limit)
        edges = 0.5 * (self.bin_centers[:-1] + self.bin_centers[1:])
        bins = torch.bucketize(clipped, edges)
        return bins.long()

    def _transition_step(
        self,
        latent: torch.Tensor,
        memory: torch.Tensor,
        curr_level: torch.Tensor,
        prev_change: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_emb = self._obs_embed(curr_level, prev_change)
        context = self._history_context(latent, obs_emb, memory)
        latent = self.transition(torch.cat([obs_emb, context], dim=-1), latent)
        logits = self._decode_logits(latent, obs_emb, context)
        hard_change = self._hard_change_from_logits(logits)
        soft_change = self._soft_change_from_logits(logits)
        return latent, logits, hard_change, soft_change

    def teacher_forced_logits(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> torch.Tensor:
        latent, memory, curr_level, prev_change = self.encode_history(history_norm)
        future = self._flatten_levels(future_norm)
        logits_list: list[torch.Tensor] = []
        for step in range(self.cfg.future_len):
            latent, logits, _, _ = self._transition_step(latent, memory, curr_level, prev_change)
            logits_list.append(logits)
            next_level = future[:, step]
            prev_change = next_level - curr_level
            curr_level = next_level
        return torch.stack(logits_list, dim=1)

    def rollout(
        self,
        history_norm: torch.Tensor,
        use_hard_decode: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, memory, curr_level, prev_change = self.encode_history(history_norm)
        pred_levels: list[torch.Tensor] = []
        pred_changes: list[torch.Tensor] = []
        for _ in range(self.cfg.future_len):
            latent, logits, hard_change, soft_change = self._transition_step(latent, memory, curr_level, prev_change)
            next_change = hard_change if use_hard_decode else soft_change
            next_level = torch.clamp(curr_level + next_change, -1.0, 1.0)
            pred_levels.append(next_level)
            pred_changes.append(next_change)
            prev_change = next_change
            curr_level = next_level
        return torch.stack(pred_levels, dim=1), torch.stack(pred_changes, dim=1)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hist = self._flatten_levels(history_norm)
        future = self._flatten_levels(future_norm)
        teacher_prev = torch.cat([hist[:, -1:], future[:, :-1]], dim=1)
        target_changes = future - teacher_prev
        target_bins = self._targets_to_bins(target_changes)

        teacher_logits = self.teacher_forced_logits(hist, future)
        rollout_levels, rollout_changes = self.rollout(hist, use_hard_decode=False)

        ce_loss = F.cross_entropy(
            teacher_logits.view(-1, self.cfg.n_bins),
            target_bins.view(-1),
        )
        level_loss = F.smooth_l1_loss(rollout_levels, future)
        rollout_change_loss = F.smooth_l1_loss(rollout_changes, target_changes)
        total = (
            self.cfg.ce_weight * ce_loss
            + self.cfg.level_weight * level_loss
            + 0.5 * self.cfg.level_weight * rollout_change_loss
        )
        metrics = {
            "total": total.detach(),
            "ce_loss": ce_loss.detach(),
            "level_loss": level_loss.detach(),
            "rollout_change_loss": rollout_change_loss.detach(),
            "pred_level_std": rollout_levels.std(unbiased=False).detach(),
        }
        return total, metrics

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        pred_levels, _ = self.rollout(
            history_norm,
            use_hard_decode=self.cfg.inference_hard_decode,
        )
        future_01 = denormalize_iv(pred_levels)
        if self.cfg.n_cells == 25:
            future_01 = future_01.view(future_01.shape[0], self.cfg.future_len, 5, 5)
        else:
            future_01 = future_01.view(future_01.shape[0], self.cfg.future_len, self.cfg.n_cells)
        return future_01.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[DeterministicHistoryMemoryDiscreteWorldModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = DeterministicHistoryMemoryDiscreteWorldModelConfig(**payload["config"])
    model = DeterministicHistoryMemoryDiscreteWorldModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def load_model_soft_decode(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[DeterministicHistoryMemoryDiscreteWorldModel, dict]:
    model, payload = load_model(checkpoint_path, device)
    model.cfg.inference_hard_decode = False
    return model, payload


def save_checkpoint(
    path: str,
    model: DeterministicHistoryMemoryDiscreteWorldModel,
    cfg: DeterministicHistoryMemoryDiscreteWorldModelConfig,
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
