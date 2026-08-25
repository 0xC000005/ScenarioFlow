from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.support_transforms import SupportTransform


def inverse_softplus(x: float) -> float:
    return float(np.log(np.expm1(x)))


def effective_rank(cov: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(eps)
    probs = eigvals / eigvals.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy.exp()


def sample_relaxed_onehot(logits: torch.Tensor, temperature: float, hard: bool) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    if temperature <= 0.0:
        idx = probs.argmax(dim=-1)
        onehot = F.one_hot(idx, num_classes=probs.shape[-1]).to(probs.dtype)
        return onehot, probs
    sample = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
    return sample, probs


def aggregate_slope_ratio(
    prev: torch.Tensor,
    gt_next: torch.Tensor,
    pred_next: torch.Tensor,
) -> float:
    prev_flat = prev.reshape(prev.shape[0], -1)
    gt_next_flat = gt_next.reshape(gt_next.shape[0], -1)
    pred_next_flat = pred_next.reshape(pred_next.shape[0], -1)
    x = prev_flat.reshape(-1).detach().cpu().numpy().astype(np.float64)
    gt_delta = (gt_next_flat - prev_flat).reshape(-1).detach().cpu().numpy().astype(np.float64)
    pred_delta = (pred_next_flat - prev_flat).reshape(-1).detach().cpu().numpy().astype(np.float64)
    x_centered = x - x.mean()
    denom = float(np.square(x_centered).sum())
    if denom <= 1e-12:
        return float("nan")
    gt_slope = float((x_centered * (gt_delta - gt_delta.mean())).sum() / denom)
    pred_slope = float((x_centered * (pred_delta - pred_delta.mean())).sum() / denom)
    if abs(gt_slope) <= 1e-12:
        return float("nan")
    return pred_slope / gt_slope


class StickyRegimePrior(nn.Module):
    def __init__(
        self,
        context_dim: int,
        n_regimes: int,
        n_blocks: int,
        hidden_dim: int = 128,
        sticky_bias: float = 2.0,
    ):
        super().__init__()
        self.n_regimes = n_regimes
        self.n_blocks = n_blocks
        self.init_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_regimes),
        )
        self.trans_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, (n_blocks - 1) * n_regimes * n_regimes),
        )
        self.sticky_bias = sticky_bias
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def transition_bank(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        init_logits = self.init_head(context)
        trans = self.trans_head(context).view(batch, self.n_blocks - 1, self.n_regimes, self.n_regimes)
        sticky = torch.eye(self.n_regimes, device=context.device, dtype=context.dtype)
        trans = trans + self.sticky_bias * sticky.view(1, 1, self.n_regimes, self.n_regimes)
        return init_logits, trans

    def sequential_logits(
        self,
        context: torch.Tensor,
        prev_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        init_logits, trans = self.transition_bank(context)
        batch = context.shape[0]
        logits = []
        running_probs = prev_probs[:, 0] if prev_probs is not None else F.softmax(init_logits, dim=-1)
        logits.append(init_logits)
        for block_idx in range(1, self.n_blocks):
            step_logits = torch.einsum("bk,bkj->bj", running_probs, trans[:, block_idx - 1])
            logits.append(step_logits)
            if prev_probs is not None:
                running_probs = prev_probs[:, block_idx]
            else:
                running_probs = F.softmax(step_logits, dim=-1)
        return torch.stack(logits, dim=1)


class RegimePosteriorEncoder(nn.Module):
    def __init__(
        self,
        n_cells: int,
        n_blocks: int,
        block_len: int,
        context_dim: int,
        n_regimes: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_len = block_len
        self.summary_proj = nn.Sequential(
            nn.Linear(n_cells * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.rnn = nn.GRUCell(hidden_dim + context_dim + n_regimes, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_regimes)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, context: torch.Tensor, future_u: torch.Tensor) -> torch.Tensor:
        batch, horizon, n_cells = future_u.shape
        if horizon != self.n_blocks * self.block_len:
            raise ValueError(
                f"Expected horizon={self.n_blocks * self.block_len}, got {horizon}"
            )
        blocks = future_u.view(batch, self.n_blocks, self.block_len, n_cells)
        block_mean = blocks.mean(dim=2)
        block_std = blocks.std(dim=2)
        block_last = blocks[:, :, -1]
        summary = torch.cat([block_mean, block_std, block_last], dim=-1)
        summary_h = self.summary_proj(summary)

        hidden = context.new_zeros(batch, summary_h.shape[-1])
        prev_probs = context.new_full((batch, self.out.out_features), 1.0 / self.out.out_features)
        logits = []
        for block_idx in range(self.n_blocks):
            rnn_in = torch.cat([summary_h[:, block_idx], context, prev_probs], dim=-1)
            hidden = self.rnn(rnn_in, hidden)
            step_logits = self.out(hidden)
            logits.append(step_logits)
            prev_probs = F.softmax(step_logits, dim=-1)
        return torch.stack(logits, dim=1)


class MeanRevertingStateTransition(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        regime_dim: int,
        time_emb_dim: int,
        hidden_dim: int = 128,
        kappa_max: float = 0.75,
        anchor_clip: float = 0.20,
        correction_clip: float = 0.10,
    ):
        super().__init__()
        self.kappa_max = kappa_max
        self.anchor_clip = anchor_clip
        self.correction_clip = correction_clip
        inp_dim = latent_dim * 2 + context_dim + regime_dim + time_emb_dim
        self.trunk = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.kappa_head = nn.Linear(hidden_dim, latent_dim)
        self.anchor_head = nn.Linear(hidden_dim, latent_dim)
        self.correction_head = nn.Linear(hidden_dim, latent_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.kappa_head.weight)
        nn.init.zeros_(self.anchor_head.weight)
        nn.init.zeros_(self.correction_head.weight)

    def forward(
        self,
        z_prev: torch.Tensor,
        anchor_state: torch.Tensor,
        context: torch.Tensor,
        regime_emb: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        hidden = self.trunk(torch.cat([z_prev, anchor_state, context, regime_emb, time_emb], dim=-1))
        kappa = torch.sigmoid(self.kappa_head(hidden)) * self.kappa_max
        equilibrium = anchor_state + torch.tanh(self.anchor_head(hidden)) * self.anchor_clip
        correction = torch.tanh(self.correction_head(hidden)) * self.correction_clip
        z_next = z_prev + kappa * (equilibrium - z_prev) + correction
        aux = {
            "kappa_mean": kappa.mean(dim=-1),
            "equilibrium_norm": equilibrium.norm(dim=-1),
            "correction_rms": correction.pow(2).mean(dim=-1).sqrt(),
        }
        return z_next, aux


class StructuredObservationHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        regime_dim: int,
        time_emb_dim: int,
        n_cells: int,
        cell_rank: int,
        resid_rank: int,
        hidden_dim: int = 128,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        local_scale_clip: float = 0.35,
        n_regimes: int = 3,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.cell_rank = cell_rank
        self.resid_rank = resid_rank
        self.diag_floor = diag_floor
        self.scale_floor = scale_floor
        self.local_scale_clip = local_scale_clip
        inp_dim = latent_dim + context_dim + regime_dim + time_emb_dim
        self.trunk = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, n_cells)
        self.factor_head = nn.Linear(hidden_dim, n_cells * cell_rank)
        self.diag_head = nn.Linear(hidden_dim, n_cells)
        self.scale_head = nn.Linear(hidden_dim, 1)
        self.local_scale_head = nn.Linear(hidden_dim, n_cells)

        self.regime_factor_bank = nn.Parameter(torch.zeros(n_regimes, n_cells, resid_rank))
        self.regime_diag_bank = nn.Parameter(torch.zeros(n_regimes, n_cells))
        self.regime_local_bank = nn.Parameter(torch.zeros(n_regimes, n_cells))
        nn.init.normal_(self.regime_factor_bank, mean=0.0, std=5e-3)
        nn.init.normal_(self.regime_diag_bank, mean=0.0, std=5e-3)
        nn.init.normal_(self.regime_local_bank, mean=0.0, std=5e-3)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.diag_head.weight)
        nn.init.constant_(self.diag_head.bias, inverse_softplus(max(0.05 - self.diag_floor, 1e-6)))
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, inverse_softplus(max(0.10 - self.scale_floor, 1e-6)))
        nn.init.zeros_(self.local_scale_head.weight)
        nn.init.zeros_(self.local_scale_head.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        context: torch.Tensor,
        regime_onehot: torch.Tensor,
        regime_emb: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        hidden = self.trunk(torch.cat([z_t, context, regime_emb, time_emb], dim=-1))
        mu = self.mu_head(hidden)
        shared_factor = self.factor_head(hidden).view(-1, self.n_cells, self.cell_rank)
        shared_diag = F.softplus(self.diag_head(hidden)) + self.diag_floor
        global_scale = F.softplus(self.scale_head(hidden).squeeze(-1)) + self.scale_floor
        local_delta = torch.tanh(self.local_scale_head(hidden)) * self.local_scale_clip

        regime_factor = torch.einsum("bk,kcr->bcr", regime_onehot, self.regime_factor_bank)
        regime_diag = torch.einsum("bk,kc->bc", regime_onehot, self.regime_diag_bank)
        regime_local = torch.einsum("bk,kc->bc", regime_onehot, self.regime_local_bank)

        factor = torch.cat([shared_factor, regime_factor], dim=-1)
        diag = F.softplus(shared_diag + regime_diag) + self.diag_floor
        local_scale = torch.exp(0.5 * torch.tanh(local_delta + regime_local) * self.local_scale_clip)

        aux = {
            "local_delta_rms": local_delta.pow(2).mean(dim=-1).sqrt(),
            "local_scale_min": local_scale.amin(dim=-1),
            "local_scale_max": local_scale.amax(dim=-1),
        }
        return mu, factor, diag, global_scale, local_scale, aux


class ResidualLaw(nn.Module):
    def log_prob(
        self,
        value: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        global_scale: torch.Tensor,
        local_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def sample(
        self,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        global_scale: torch.Tensor,
        local_scale: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        raise NotImplementedError


class StudentTResidualLaw(ResidualLaw):
    def __init__(self, base_nu: float = 8.0, jitter: float = 1e-5, scale_floor: float = 1e-4):
        super().__init__()
        self.base_nu = float(base_nu)
        self.jitter = jitter
        self.scale_floor = scale_floor

    def _base_logprob(self, z: torch.Tensor) -> torch.Tensor:
        nu = z.new_tensor(self.base_nu)
        log_norm = (
            torch.lgamma((nu + 1.0) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * torch.log(nu * z.new_tensor(math.pi))
        )
        log_kernel = -0.5 * (nu + 1.0) * torch.log1p(z.pow(2) / nu)
        return (log_norm + log_kernel).sum(dim=-1)

    def normalized_components(self, factor: torch.Tensor, diag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_diag = factor.pow(2).sum(dim=-1) + diag.pow(2) + self.jitter
        avg_var = raw_diag.mean(dim=-1).clamp_min(self.jitter)
        norm = avg_var.sqrt().unsqueeze(-1)
        factor_norm = factor / norm.unsqueeze(-1)
        diag_norm = diag / norm
        return factor_norm, diag_norm

    def covariance(self, factor: torch.Tensor, diag: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
        factor_norm, diag_norm = self.normalized_components(factor, diag)
        cov = factor_norm @ factor_norm.transpose(-1, -2)
        cov = cov + torch.diag_embed(diag_norm.pow(2) + self.jitter)
        scale = global_scale.clamp_min(self.scale_floor)
        return cov * scale.view(-1, 1, 1)

    def log_prob(
        self,
        value: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        global_scale: torch.Tensor,
        local_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        cov = self.covariance(factor, diag, global_scale)
        chol = torch.linalg.cholesky(cov)
        diff = (value - mu) / local_scale
        white = torch.linalg.solve_triangular(chol, diff.unsqueeze(-1), upper=False).squeeze(-1)
        base_logprob = self._base_logprob(white)
        logdet_cov = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_local = 2.0 * torch.log(local_scale).sum(dim=-1)
        logprob = base_logprob - 0.5 * (logdet_cov + logdet_local)
        aux = {
            "cov": cov,
            "white_std": white.std(dim=-1),
            "z_std": white.std(dim=-1),
        }
        return logprob, aux

    def sample(
        self,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        global_scale: torch.Tensor,
        local_scale: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        batch, cells = mu.shape
        cov = self.covariance(factor, diag, global_scale)
        chol = torch.linalg.cholesky(cov)
        base = torch.distributions.StudentT(df=self.base_nu)
        z = base.sample((batch, n_samples, cells)).to(device=mu.device, dtype=mu.dtype)
        noise = torch.einsum("bij,bsj->bsi", chol, z)
        return mu.unsqueeze(1) + noise * local_scale.unsqueeze(1)


class FlowResidualLaw(ResidualLaw):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("FlowResidualLaw is reserved for a later 178x extension.")


@dataclass
class RegimeModelOutputs:
    mu_u: torch.Tensor
    factor: torch.Tensor
    diag: torch.Tensor
    global_scale: torch.Tensor
    local_scale: torch.Tensor
    prior_logits: torch.Tensor
    posterior_logits: torch.Tensor | None
    regime_probs: torch.Tensor
    regime_onehot: torch.Tensor
    z_path: torch.Tensor
    aux: dict


class RegimeCoupledStateSpaceModel(nn.Module):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        support_transform: SupportTransform,
        future_len: int,
        n_cells: int = 25,
        n_blocks: int = 5,
        n_regimes: int = 3,
        latent_dim: int = 64,
        regime_dim: int = 16,
        time_emb_dim: int = 16,
        cell_rank: int = 5,
        cov_resid_rank: int = 3,
        hidden_dim: int = 128,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        local_scale_clip: float = 0.35,
        base_nu: float = 8.0,
        cov_jitter: float = 1e-5,
    ):
        super().__init__()
        if future_len % n_blocks != 0:
            raise ValueError(f"future_len={future_len} must be divisible by n_blocks={n_blocks}")
        self.encoder = GRUEncoder(encoder_config)
        self.support_transform = support_transform
        self.future_len = future_len
        self.n_cells = n_cells
        self.n_blocks = n_blocks
        self.block_len = future_len // n_blocks
        self.n_regimes = n_regimes
        self.latent_dim = latent_dim
        self.regime_dim = regime_dim
        self.time_emb_dim = time_emb_dim
        self.cell_rank = cell_rank
        self.cov_resid_rank = cov_resid_rank
        self.hidden_dim = hidden_dim
        self.diag_floor = diag_floor
        self.scale_floor = scale_floor
        self.local_scale_clip = local_scale_clip
        self.base_nu = base_nu
        self.cov_jitter = cov_jitter

        context_dim = encoder_config.bottleneck_dim
        self.prior = StickyRegimePrior(context_dim=context_dim, n_regimes=n_regimes, n_blocks=n_blocks, hidden_dim=hidden_dim)
        self.posterior = RegimePosteriorEncoder(
            n_cells=n_cells,
            n_blocks=n_blocks,
            block_len=self.block_len,
            context_dim=context_dim,
            n_regimes=n_regimes,
            hidden_dim=hidden_dim,
        )
        self.regime_embedding = nn.Embedding(n_regimes, regime_dim)
        self.time_embedding = nn.Embedding(future_len, time_emb_dim)
        self.init_state = nn.Sequential(
            nn.Linear(context_dim + 2 * n_cells, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.anchor_state = nn.Sequential(
            nn.Linear(context_dim + 2 * n_cells, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.transition = MeanRevertingStateTransition(
            latent_dim=latent_dim,
            context_dim=context_dim,
            regime_dim=regime_dim,
            time_emb_dim=time_emb_dim,
            hidden_dim=hidden_dim,
        )
        self.observation = StructuredObservationHead(
            latent_dim=latent_dim,
            context_dim=context_dim,
            regime_dim=regime_dim,
            time_emb_dim=time_emb_dim,
            n_cells=n_cells,
            cell_rank=cell_rank,
            resid_rank=cov_resid_rank,
            hidden_dim=hidden_dim,
            diag_floor=diag_floor,
            scale_floor=scale_floor,
            local_scale_clip=local_scale_clip,
            n_regimes=n_regimes,
        )
        self.residual_law = StudentTResidualLaw(base_nu=base_nu, jitter=cov_jitter, scale_floor=scale_floor)

    def encode_history(self, history_01: torch.Tensor) -> torch.Tensor:
        history_norm = history_01 * 2.0 - 1.0
        return self.encoder(history_norm)

    def _history_features(self, history_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_flat = history_01.reshape(history_01.shape[0], history_01.shape[1], -1)
        history_u, _ = self.support_transform.forward(history_flat)
        last_u = history_u[:, -1]
        mean_u = history_u.mean(dim=1)
        return history_u, last_u, mean_u

    def _prior_logits_from_probs(self, context: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return self.prior.sequential_logits(context, prev_probs=probs)

    def _sample_regimes(
        self,
        context: torch.Tensor,
        future_u: torch.Tensor | None,
        temperature: float,
        hard: bool,
        use_posterior: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if use_posterior:
            if future_u is None:
                raise ValueError("future_u required when use_posterior=True")
            posterior_logits = self.posterior(context, future_u)
            regime_onehot, regime_probs = sample_relaxed_onehot(
                posterior_logits.reshape(-1, self.n_regimes),
                temperature=temperature,
                hard=hard,
            )
            regime_onehot = regime_onehot.view(context.shape[0], self.n_blocks, self.n_regimes)
            regime_probs = regime_probs.view(context.shape[0], self.n_blocks, self.n_regimes)
            prior_logits = self._prior_logits_from_probs(context, regime_probs.detach())
            return prior_logits, regime_onehot, posterior_logits

        prior_logits = self.prior.sequential_logits(context)
        if temperature < 0.0:
            prior_probs = F.softmax(prior_logits, dim=-1)
            comp_idx = torch.multinomial(prior_probs.view(-1, self.n_regimes), num_samples=1).squeeze(-1)
            regime_onehot = F.one_hot(comp_idx, num_classes=self.n_regimes).to(prior_probs.dtype)
            regime_onehot = regime_onehot.view(context.shape[0], self.n_blocks, self.n_regimes)
        else:
            regime_onehot, _ = sample_relaxed_onehot(
                prior_logits.reshape(-1, self.n_regimes),
                temperature=temperature,
                hard=True,
            )
            regime_onehot = regime_onehot.view(context.shape[0], self.n_blocks, self.n_regimes)
        return prior_logits, regime_onehot, None

    def rollout(
        self,
        history_01: torch.Tensor,
        future_01: torch.Tensor | None = None,
        temperature: float = 0.5,
        hard_regimes: bool = True,
        use_posterior: bool = True,
    ) -> RegimeModelOutputs:
        context = self.encode_history(history_01)
        _, last_u, mean_u = self._history_features(history_01)
        init_in = torch.cat([context, last_u, mean_u], dim=-1)
        z_t = self.init_state(init_in)
        anchor_state = self.anchor_state(init_in)

        future_u = None
        if future_01 is not None:
            future_flat = future_01.reshape(future_01.shape[0], future_01.shape[1], -1)
            future_u = self.support_transform.forward(future_flat)[0]

        prior_logits, regime_onehot, posterior_logits = self._sample_regimes(
            context=context,
            future_u=future_u,
            temperature=temperature,
            hard=hard_regimes,
            use_posterior=use_posterior and future_u is not None,
        )
        regime_probs = F.softmax(prior_logits if posterior_logits is None else posterior_logits, dim=-1)

        mu_list = []
        factor_list = []
        diag_list = []
        scale_list = []
        local_scale_list = []
        z_list = []
        kappa_means = []
        local_scale_min = []
        local_scale_max = []
        corr_rms = []

        for t in range(self.future_len):
            block_idx = t // self.block_len
            onehot = regime_onehot[:, block_idx]
            regime_emb = onehot @ self.regime_embedding.weight
            time_emb = self.time_embedding.weight[t].unsqueeze(0).expand(history_01.shape[0], -1)
            z_t, tr_aux = self.transition(z_t, anchor_state, context, regime_emb, time_emb)
            mu_t, factor_t, diag_t, scale_t, local_scale_t, obs_aux = self.observation(
                z_t=z_t,
                context=context,
                regime_onehot=onehot,
                regime_emb=regime_emb,
                time_emb=time_emb,
            )
            mu_list.append(mu_t)
            factor_list.append(factor_t)
            diag_list.append(diag_t)
            scale_list.append(scale_t)
            local_scale_list.append(local_scale_t)
            z_list.append(z_t)
            kappa_means.append(tr_aux["kappa_mean"])
            local_scale_min.append(obs_aux["local_scale_min"])
            local_scale_max.append(obs_aux["local_scale_max"])
            corr_rms.append(tr_aux["correction_rms"])

        aux = {
            "kappa_mean": torch.stack(kappa_means, dim=1),
            "local_scale_min": torch.stack(local_scale_min, dim=1),
            "local_scale_max": torch.stack(local_scale_max, dim=1),
            "correction_rms": torch.stack(corr_rms, dim=1),
        }
        return RegimeModelOutputs(
            mu_u=torch.stack(mu_list, dim=1),
            factor=torch.stack(factor_list, dim=1),
            diag=torch.stack(diag_list, dim=1),
            global_scale=torch.stack(scale_list, dim=1),
            local_scale=torch.stack(local_scale_list, dim=1),
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
            regime_probs=regime_probs,
            regime_onehot=regime_onehot,
            z_path=torch.stack(z_list, dim=1),
            aux=aux,
        )

    def log_prob_future(
        self,
        history_01: torch.Tensor,
        future_01: torch.Tensor,
        temperature: float = 0.5,
    ) -> tuple[torch.Tensor, dict]:
        outputs = self.rollout(
            history_01=history_01,
            future_01=future_01,
            temperature=temperature,
            hard_regimes=True,
            use_posterior=True,
        )
        future_flat = future_01.reshape(future_01.shape[0], future_01.shape[1], -1)
        target_u, _ = self.support_transform.forward(future_flat)

        batch, horizon, cells = target_u.shape
        value = target_u.reshape(batch * horizon, cells)
        mu = outputs.mu_u.reshape(batch * horizon, cells)
        factor = outputs.factor.reshape(batch * horizon, cells, -1)
        diag = outputs.diag.reshape(batch * horizon, cells)
        global_scale = outputs.global_scale.reshape(batch * horizon)
        local_scale = outputs.local_scale.reshape(batch * horizon, cells)
        logprob_step, resid_aux = self.residual_law.log_prob(
            value=value,
            mu=mu,
            factor=factor,
            diag=diag,
            global_scale=global_scale,
            local_scale=local_scale,
        )
        logprob = logprob_step.view(batch, horizon).sum(dim=1)

        prior_probs = F.softmax(outputs.prior_logits, dim=-1)
        posterior_probs = (
            F.softmax(outputs.posterior_logits, dim=-1)
            if outputs.posterior_logits is not None
            else outputs.regime_probs
        )
        kl = (
            posterior_probs
            * (torch.log(posterior_probs.clamp_min(1e-8)) - torch.log(prior_probs.clamp_min(1e-8)))
        ).sum(dim=-1)
        kl = kl.sum(dim=-1)

        det_next, _ = self.support_transform.inverse(outputs.mu_u[:, 0])
        prev_native = history_01[:, -1].reshape(batch, -1)
        gt_next = future_flat[:, 0]
        pred_delta = det_next - prev_native
        gt_delta = gt_next - prev_native
        x = prev_native.reshape(-1)
        x_centered = x - x.mean()
        gt_delta_flat = gt_delta.reshape(-1).detach()
        pred_delta_flat = pred_delta.reshape(-1)
        denom = x_centered.pow(2).sum().clamp_min(1e-8)
        gt_slope = (x_centered * (gt_delta_flat - gt_delta_flat.mean())).sum() / denom
        pred_slope = (x_centered * (pred_delta_flat - pred_delta_flat.mean())).sum() / denom
        neg_mask = (gt_slope.detach() < 0).to(pred_slope.dtype)
        safe_gt = torch.where(gt_slope.detach() < -1e-8, gt_slope, pred_slope.new_tensor(-1.0))
        mr_penalty = neg_mask * (pred_slope / safe_gt - 1.0).pow(2)

        aux = {
            "cov": resid_aux["cov"].view(batch, horizon, cells, cells),
            "white_std": resid_aux["white_std"].view(batch, horizon),
            "z_std": resid_aux["z_std"].view(batch, horizon),
            "kl_regime": kl,
            "prior_entropy": (-(prior_probs * torch.log(prior_probs.clamp_min(1e-8))).sum(dim=-1)).mean(dim=-1),
            "posterior_entropy": (-(posterior_probs * torch.log(posterior_probs.clamp_min(1e-8))).sum(dim=-1)).mean(dim=-1),
            "regime_usage": posterior_probs.mean(dim=(0, 1)),
            "mr_penalty": mr_penalty,
            "det_next_native": det_next,
            "gt_next_native": gt_next,
            "prev_native": prev_native,
        }
        return logprob, outputs, aux

    @torch.no_grad()
    def sample_future_u(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        repeated = history_01.repeat_interleave(n_samples, dim=0)
        outputs = self.rollout(
            history_01=repeated,
            future_01=None,
            temperature=-1.0 if temperature <= 0.0 else temperature,
            hard_regimes=True,
            use_posterior=False,
        )
        batch_rep, horizon, cells = outputs.mu_u.shape
        mu = outputs.mu_u.reshape(batch_rep * horizon, cells)
        factor = outputs.factor.reshape(batch_rep * horizon, cells, -1)
        diag = outputs.diag.reshape(batch_rep * horizon, cells)
        global_scale = outputs.global_scale.reshape(batch_rep * horizon)
        local_scale = outputs.local_scale.reshape(batch_rep * horizon, cells)
        samples = self.residual_law.sample(
            mu=mu,
            factor=factor,
            diag=diag,
            global_scale=global_scale,
            local_scale=local_scale,
            n_samples=1,
        ).squeeze(1)
        samples = samples.view(history_01.shape[0], n_samples, horizon, cells)
        regimes = outputs.regime_onehot.view(history_01.shape[0], n_samples, self.n_blocks, self.n_regimes)
        return samples, regimes

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        **kwargs,
    ) -> torch.Tensor:
        history_01 = (history + 1.0) / 2.0
        batch_size = history_01.shape[0]
        chunks = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            samples_u, _ = self.sample_future_u(history_01, n_samples=k)
            samples_native = self.support_transform.inverse(samples_u.reshape(batch_size * k, self.future_len, self.n_cells))[0]
            chunks.append(samples_native.view(batch_size, k, self.future_len, 5, 5))
        return torch.cat(chunks, dim=1)
