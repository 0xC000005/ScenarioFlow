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
class RecurrentStudentTTransitionConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    hidden_dim: int = 160
    gru_layers: int = 2
    gru_dropout: float = 0.1

    param_hidden: int = 320
    param_layers: int = 3
    param_dropout: float = 0.1

    logit_eps: float = 1e-4
    scale_floor: float = 1e-3
    min_df: float = 2.5
    max_df: float = 40.0


class TransitionStudentTHead(nn.Module):
    def __init__(self, cfg: RecurrentStudentTTransitionConfig):
        super().__init__()
        self.cfg = cfg
        self.tril_size = cfg.n_cells * (cfg.n_cells + 1) // 2
        in_dim = cfg.hidden_dim + cfg.n_cells
        layers: list[nn.Module] = [
            nn.Linear(in_dim, cfg.param_hidden),
            nn.GELU(),
        ]
        for _ in range(max(0, cfg.param_layers - 1)):
            layers.extend(
                [
                    nn.Dropout(cfg.param_dropout),
                    nn.Linear(cfg.param_hidden, cfg.param_hidden),
                    nn.GELU(),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.loc_head = nn.Linear(cfg.param_hidden, cfg.n_cells)
        self.tril_head = nn.Linear(cfg.param_hidden, self.tril_size)
        self.df_head = nn.Linear(cfg.param_hidden, 1)
        tril = torch.tril_indices(cfg.n_cells, cfg.n_cells)
        self.register_buffer("tril_row", tril[0])
        self.register_buffer("tril_col", tril[1])
        self.register_buffer("diag_mask", tril[0] == tril[1])

    def forward(
        self,
        state_top: torch.Tensor,
        current_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.net(torch.cat([state_top, current_logit], dim=-1))
        loc = self.loc_head(hidden)
        raw_tril = self.tril_head(hidden)
        scale_tril = raw_tril.new_zeros(raw_tril.shape[0], self.cfg.n_cells, self.cfg.n_cells)
        vals = raw_tril.clone()
        vals[:, self.diag_mask] = F.softplus(vals[:, self.diag_mask]) + self.cfg.scale_floor
        scale_tril[:, self.tril_row, self.tril_col] = vals
        df_unit = torch.sigmoid(self.df_head(hidden)).squeeze(-1)
        df = self.cfg.min_df + (self.cfg.max_df - self.cfg.min_df) * df_unit
        return loc, scale_tril, df


class RecurrentStudentTTransitionLikelihood(nn.Module):
    """306a-v0: support-valid AR logit-transition model with full-cov Student-t NLL."""

    def __init__(self, cfg: RecurrentStudentTTransitionConfig):
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
        self.head = TransitionStudentTHead(cfg)
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

    def transition_params(
        self,
        state_stack: torch.Tensor,
        current_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(state_stack[-1], current_logit)

    def student_t_nll(
        self,
        target: torch.Tensor,
        loc: torch.Tensor,
        scale_tril: torch.Tensor,
        df: torch.Tensor,
    ) -> torch.Tensor:
        diff = target - loc
        sol = torch.linalg.solve_triangular(
            scale_tril,
            diff.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        maha = sol.pow(2).sum(dim=-1)
        logdet = torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(dim=-1)
        dim = float(self.cfg.n_cells)
        log_norm = (
            torch.lgamma((df + dim) * 0.5)
            - torch.lgamma(df * 0.5)
            - 0.5 * dim * (torch.log(df) + torch.log(torch.tensor(torch.pi, device=df.device, dtype=df.dtype)))
            - logdet
        )
        log_kernel = -0.5 * (df + dim) * torch.log1p(maha / df)
        return -(log_norm + log_kernel)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_logits = self.target_future_logits(future_norm)
        state_stack, current_logit = self.encode_history(history_norm)

        losses: list[torch.Tensor] = []
        loc_abs = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)
        scale_mean = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)
        df_mean = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)
        transition_std = torch.tensor(0.0, device=history_norm.device, dtype=history_norm.dtype)

        for step in range(future_logits.shape[1]):
            next_logit = future_logits[:, step]
            transition = next_logit - current_logit
            loc, scale_tril, df = self.transition_params(state_stack, current_logit)
            losses.append(self.student_t_nll(transition, loc, scale_tril, df).mean())
            loc_abs = loc_abs + loc.abs().mean()
            diag = torch.diagonal(scale_tril, dim1=-2, dim2=-1)
            scale_mean = scale_mean + diag.mean()
            df_mean = df_mean + df.mean()
            transition_std = transition_std + transition.std(unbiased=False)
            state_stack = self.recurrent_step(state_stack, current_logit, next_logit)
            current_logit = next_logit

        horizon = float(future_logits.shape[1])
        nll = torch.stack(losses).mean()
        metrics = {
            "total": nll.detach(),
            "nll": nll.detach(),
            "loc_abs": (loc_abs / horizon).detach(),
            "scale_diag_mean": (scale_mean / horizon).detach(),
            "df_mean": (df_mean / horizon).detach(),
            "transition_std": (transition_std / horizon).detach(),
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

            frames: list[torch.Tensor] = []
            for _step in range(n_steps):
                loc, scale_tril, df = self.transition_params(state_stack, current_logit)
                eps = torch.randn_like(loc)
                chi2 = torch.distributions.Chi2(df).sample().clamp_min(1e-6)
                t_noise = eps / torch.sqrt((chi2 / df).unsqueeze(-1))
                transition = loc + torch.matmul(scale_tril, t_noise.unsqueeze(-1)).squeeze(-1)
                next_logit = current_logit + transition
                frames.append(logit_to_iv(next_logit).view(bsz, k, 5, 5))
                state_stack = self.recurrent_step(state_stack, current_logit, next_logit)
                current_logit = next_logit
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[RecurrentStudentTTransitionLikelihood, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = RecurrentStudentTTransitionConfig(**payload["config"])
    model = RecurrentStudentTTransitionLikelihood(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: RecurrentStudentTTransitionLikelihood,
    cfg: RecurrentStudentTTransitionConfig,
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
