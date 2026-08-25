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


@dataclass
class RecurrentPathLatentTokenLikelihoodConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    hidden_dim: int = 128
    gru_layers: int = 2
    gru_dropout: float = 0.1

    latent_dim: int = 8
    posterior_hidden: int = 128

    token_dim: int = 128
    token_layers: int = 3
    token_heads: int = 4
    token_ff: int = 256
    model_dropout: float = 0.1

    logit_eps: float = 1e-4
    scale_floor: float = 1e-3
    min_df: float = 2.5
    max_df: float = 20.0


class PathPosteriorEncoder(nn.Module):
    def __init__(self, cfg: RecurrentPathLatentTokenLikelihoodConfig):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=2 * cfg.n_cells,
            hidden_size=cfg.posterior_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.mean = nn.Linear(cfg.posterior_hidden + cfg.hidden_dim, cfg.latent_dim)
        self.logvar = nn.Linear(cfg.posterior_hidden + cfg.hidden_dim, cfg.latent_dim)

    def forward(
        self,
        future_logits: torch.Tensor,
        history_top: torch.Tensor,
        current_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev = torch.cat([current_logit[:, None, :], future_logits[:, :-1]], dim=1)
        future_feats = torch.cat([future_logits, future_logits - prev], dim=-1)
        _out, h = self.gru(future_feats)
        posterior_state = torch.cat([h[-1], history_top], dim=-1)
        return self.mean(posterior_state), self.logvar(posterior_state)


class LatentTokenTransitionHead(nn.Module):
    def __init__(self, cfg: RecurrentPathLatentTokenLikelihoodConfig):
        super().__init__()
        self.cfg = cfg
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.value_proj = nn.Linear(1, cfg.token_dim)
        self.state_proj = nn.Linear(cfg.hidden_dim, cfg.token_dim)
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.token_heads,
            dim_feedforward=cfg.token_ff,
            dropout=cfg.model_dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=cfg.token_layers)
        self.loc_out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )
        self.scale_out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )
        self.df_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.latent_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def forward(
        self,
        current_logit: torch.Tensor,
        state_top: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_cells = current_logit.shape
        cell_ids = torch.arange(n_cells, device=current_logit.device)
        tokens = self.value_proj(current_logit.unsqueeze(-1))
        tokens = tokens + self.cell_embed(cell_ids)[None, :, :]
        tokens = tokens + self.state_proj(state_top)[:, None, :]
        tokens = tokens + self.latent_proj(z)[:, None, :]
        hidden = self.mixer(tokens)
        loc = self.loc_out(hidden).squeeze(-1)
        scale = F.softplus(self.scale_out(hidden).squeeze(-1)) + self.cfg.scale_floor
        df_unit = torch.sigmoid(self.df_head(torch.cat([state_top, z], dim=-1))).squeeze(-1)
        df = self.cfg.min_df + (self.cfg.max_df - self.cfg.min_df) * df_unit
        return loc, scale, df


class RecurrentPathLatentTokenLikelihood(nn.Module):
    """307a-v0: recurrent AR decoder with one global path latent and token mixing."""

    def __init__(self, cfg: RecurrentPathLatentTokenLikelihoodConfig):
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
        self.posterior = PathPosteriorEncoder(cfg)
        self.decoder = LatentTokenTransitionHead(cfg)
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

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        return iv_to_logit(denormalize_iv(future_norm), self.cfg.logit_eps)

    @staticmethod
    def diagonal_student_t_nll(
        target: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        df: torch.Tensor,
    ) -> torch.Tensor:
        df = df[:, None]
        z2 = ((target - loc) / scale).pow(2)
        log_norm = (
            torch.lgamma((df + 1.0) * 0.5)
            - torch.lgamma(df * 0.5)
            - 0.5 * (torch.log(df) + torch.log(torch.tensor(torch.pi, device=target.device, dtype=target.dtype)))
            - torch.log(scale)
        )
        log_prob = log_norm - 0.5 * (df + 1.0) * torch.log1p(z2 / df)
        return -log_prob.mean(dim=-1)

    def sample_latent(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        kl = -0.5 * torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        return z, kl

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_logits = self.target_future_logits(future_norm)
        state_stack, current_logit = self.encode_history(history_norm)
        mean, logvar = self.posterior(future_logits, state_stack[-1], current_logit)
        z, kl = self.sample_latent(mean, logvar)

        losses: list[torch.Tensor] = []
        loc_abs = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)
        scale_mean = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)
        df_mean = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)
        transition_std = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)

        for step in range(future_logits.shape[1]):
            next_logit = future_logits[:, step]
            transition = next_logit - current_logit
            loc, scale, df = self.decoder(current_logit, state_stack[-1], z)
            losses.append(self.diagonal_student_t_nll(transition, loc, scale, df).mean())
            loc_abs = loc_abs + loc.abs().mean()
            scale_mean = scale_mean + scale.mean()
            df_mean = df_mean + df.mean()
            transition_std = transition_std + transition.std(unbiased=False)
            state_stack = self.recurrent_step(state_stack, current_logit, next_logit)
            current_logit = next_logit

        horizon = float(future_logits.shape[1])
        nll = torch.stack(losses).mean()
        # Normalize the path-level KL to the same per-step scale as the NLL term.
        total = nll + kl.mean() / horizon
        metrics = {
            "total": total.detach(),
            "nll": nll.detach(),
            "kl": kl.mean().detach(),
            "loc_abs": (loc_abs / horizon).detach(),
            "scale_mean": (scale_mean / horizon).detach(),
            "df_mean": (df_mean / horizon).detach(),
            "transition_std": (transition_std / horizon).detach(),
            "latent_abs": z.abs().mean().detach(),
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
        if n_steps < 1:
            raise ValueError(f"Expected positive n_steps, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        base_state, base_logit = self.encode_history(history_norm)
        bsz = history_norm.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))

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
            z = torch.randn(
                bsz * k,
                self.cfg.latent_dim,
                device=base_logit.device,
                dtype=base_logit.dtype,
            )

            frames: list[torch.Tensor] = []
            for _step in range(n_steps):
                loc, scale, df = self.decoder(current_logit, state_stack[-1], z)
                eps = torch.randn_like(loc)
                chi2 = torch.distributions.Chi2(df).sample().clamp_min(1e-6)
                noise = eps / torch.sqrt((chi2 / df).unsqueeze(-1))
                transition = loc + scale * noise
                next_logit = current_logit + transition
                frames.append(logit_to_iv(next_logit).view(bsz, k, 5, 5))
                state_stack = self.recurrent_step(state_stack, current_logit, next_logit)
                current_logit = next_logit
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[RecurrentPathLatentTokenLikelihood, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = RecurrentPathLatentTokenLikelihoodConfig(**payload["config"])
    model = RecurrentPathLatentTokenLikelihood(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: RecurrentPathLatentTokenLikelihood,
    cfg: RecurrentPathLatentTokenLikelihoodConfig,
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
