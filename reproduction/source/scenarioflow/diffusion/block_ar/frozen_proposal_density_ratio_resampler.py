from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
    load_model as load_empirical_score_ar_model,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


@dataclass
class FrozenProposalDensityRatioConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    hidden_dim: int = 192
    gru_layers: int = 2
    stats_hidden: int = 192
    head_hidden: int = 256
    dropout: float = 0.10
    candidate_multiplier: int = 4
    candidate_chunk_size: int = 8
    max_score_chunk: int = 64
    logit_strength: float = 1.0


class FrozenProposalDensityRatioScorer(nn.Module):
    """Conditional density-ratio scorer for frozen proposal paths.

    The scorer does not generate new support. It assigns a scalar log weight to a
    complete future path given history, so deployment can resample frozen base
    candidates without changing their path geometry.
    """

    def __init__(self, cfg: FrozenProposalDensityRatioConfig):
        super().__init__()
        self.cfg = cfg
        dropout = float(cfg.dropout) if cfg.gru_layers > 1 else 0.0
        self.history_gru = nn.GRU(
            cfg.n_cells,
            cfg.hidden_dim,
            num_layers=cfg.gru_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.future_gru = nn.GRU(
            cfg.n_cells,
            cfg.hidden_dim,
            num_layers=cfg.gru_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.stats_proj = nn.Sequential(
            nn.LayerNorm(8 * cfg.n_cells),
            nn.Linear(8 * cfg.n_cells, cfg.stats_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.stats_hidden, cfg.stats_hidden),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * cfg.hidden_dim + cfg.stats_hidden),
            nn.Linear(2 * cfg.hidden_dim + cfg.stats_hidden, cfg.head_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def _path_stats(
        self,
        history_scores: torch.Tensor,
        future_scores: torch.Tensor,
    ) -> torch.Tensor:
        last_hist = history_scores[:, -1, :]
        first = future_scores[:, 0, :]
        terminal = future_scores[:, -1, :]
        mean = future_scores.mean(dim=1)
        std = future_scores.std(dim=1, unbiased=False)
        deltas = future_scores[:, 1:, :] - future_scores[:, :-1, :]
        delta_mean = deltas.mean(dim=1)
        delta_std = deltas.std(dim=1, unbiased=False)
        delta_max_abs = deltas.abs().amax(dim=1)
        return torch.cat(
            [
                last_hist,
                first - last_hist,
                terminal - last_hist,
                mean - last_hist,
                std,
                delta_mean,
                delta_std,
                delta_max_abs,
            ],
            dim=-1,
        )

    def forward(
        self,
        history_scores: torch.Tensor,
        future_scores: torch.Tensor,
    ) -> torch.Tensor:
        if history_scores.ndim != 3 or future_scores.ndim != 3:
            raise ValueError("Expected history and future score tensors with shape [B,T,C]")
        hist_out, _ = self.history_gru(history_scores)
        fut_out, _ = self.future_gru(future_scores)
        hist_ctx = hist_out[:, -1, :]
        fut_ctx = fut_out[:, -1, :]
        stats_ctx = self.stats_proj(self._path_stats(history_scores, future_scores))
        features = torch.cat([hist_ctx, fut_ctx, stats_ctx], dim=-1)
        return self.head(features).squeeze(-1)

    def score_candidates(
        self,
        history_scores: torch.Tensor,
        future_scores: torch.Tensor,
        score_chunk: int | None = None,
    ) -> torch.Tensor:
        if future_scores.ndim != 4:
            raise ValueError("Expected future candidates with shape [B,K,T,C]")
        bsz, n_candidates, horizon, n_cells = future_scores.shape
        if horizon != self.cfg.future_len or n_cells != self.cfg.n_cells:
            raise ValueError(
                f"Expected candidates [B,K,{self.cfg.future_len},{self.cfg.n_cells}], "
                f"got {tuple(future_scores.shape)}"
            )
        chunk = int(score_chunk or self.cfg.max_score_chunk)
        chunk = max(1, min(chunk, n_candidates))
        logits: list[torch.Tensor] = []
        for start in range(0, n_candidates, chunk):
            k = min(chunk, n_candidates - start)
            hist = history_scores.repeat_interleave(k, dim=0)
            fut = future_scores[:, start : start + k].reshape(
                bsz * k,
                horizon,
                n_cells,
            )
            logits.append(self.forward(hist, fut).view(bsz, k))
        return torch.cat(logits, dim=1)


def density_ratio_nce_loss(
    model: FrozenProposalDensityRatioScorer,
    history_scores: torch.Tensor,
    positive_future_scores: torch.Tensor,
    negative_future_scores: torch.Tensor,
    logit_l2: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    futures = torch.cat(
        [positive_future_scores[:, None, :, :], negative_future_scores],
        dim=1,
    )
    logits = model.score_candidates(history_scores, futures)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    ce = F.cross_entropy(logits, labels)
    penalty = float(logit_l2) * logits.square().mean()
    loss = ce + penalty
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        metrics = {
            "total": loss.detach(),
            "ce": ce.detach(),
            "logit_l2": penalty.detach(),
            "acc": (pred == 0).float().mean().detach(),
            "pos_logit": logits[:, 0].mean().detach(),
            "neg_logit": logits[:, 1:].mean().detach(),
            "logit_std": logits.std(unbiased=False).detach(),
            "entropy": torch.distributions.Categorical(logits=logits).entropy().mean().detach(),
        }
    return loss, metrics


class FrozenProposalDensityRatioScenarioGenerator(nn.Module):
    def __init__(
        self,
        scorer: FrozenProposalDensityRatioScorer,
        base_model: nn.Module,
        base_checkpoint_path: str,
    ):
        super().__init__()
        self.scorer = scorer
        self.base_model = base_model.eval()
        self.base_checkpoint_path = base_checkpoint_path
        self.cfg = scorer.cfg
        for param in self.base_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def _candidate_scores(
        self,
        candidates_01: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_candidates, horizon = candidates_01.shape[:3]
        flat = candidates_01.reshape(bsz * n_candidates, horizon, 5, 5)
        flat_norm = normalize_iv(flat).view(bsz * n_candidates, horizon, -1)
        return self.base_model.target_future_scores(flat_norm).view(
            bsz,
            n_candidates,
            horizon,
            self.cfg.n_cells,
        )

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
        history_norm = history_norm.view(history_norm.shape[0], history_norm.shape[1], -1)
        history_scores = self.base_model.history_scores(history_norm)
        candidate_count = max(
            int(n_samples) + 1,
            int(n_samples) * max(1, int(self.cfg.candidate_multiplier)),
        )
        candidate_chunk = max(
            1,
            min(int(chunk_size), int(self.cfg.candidate_chunk_size), candidate_count),
        )
        candidates_01 = self.base_model.sample_batched(
            history_norm,
            n_samples=candidate_count,
            n_steps=n_steps,
            chunk_size=candidate_chunk,
            history_is_normalized=True,
        )
        future_scores = self._candidate_scores(candidates_01)
        logits = self.scorer.score_candidates(
            history_scores,
            future_scores,
            score_chunk=self.cfg.max_score_chunk,
        )
        logits = float(self.cfg.logit_strength) * logits
        probs = torch.softmax(logits.float(), dim=1)
        replacement = candidate_count < int(n_samples)
        index = torch.multinomial(probs, num_samples=int(n_samples), replacement=replacement)
        selected = torch.stack(
            [candidates_01[b].index_select(0, index[b]) for b in range(candidates_01.shape[0])],
            dim=0,
        )
        return selected


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[FrozenProposalDensityRatioScenarioGenerator, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = FrozenProposalDensityRatioConfig(**payload["config"])
    scorer = FrozenProposalDensityRatioScorer(cfg)
    scorer.load_state_dict(payload["model_state_dict"], strict=True)
    base_checkpoint_path = payload["base_checkpoint_path"]
    base_model, _base_payload = load_empirical_score_ar_model(base_checkpoint_path, device)
    model = FrozenProposalDensityRatioScenarioGenerator(
        scorer=scorer.to(device),
        base_model=base_model,
        base_checkpoint_path=base_checkpoint_path,
    )
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: FrozenProposalDensityRatioScorer,
    cfg: FrozenProposalDensityRatioConfig,
    epoch: int,
    best_val: float,
    base_checkpoint_path: str,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
            "base_checkpoint_path": base_checkpoint_path,
        },
        path,
    )
