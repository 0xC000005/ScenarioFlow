"""
Neural Factor Model (250 series) — explicit low-rank non-AR joint generator.

Stack-agnostic by construction: model internals are strictly (B, K, T, D). Callers
do any factor-family-specific reshape (5x5 for IV, curve-length for rates, etc.)
outside the model.

Stage A (NeuralFactorModel):
    history -> encoder -> h
    h -> (mu_z, sigma_z)     (LatentEncoder)
    h -> Lambda(h)           (LoadingHead)       explicit low-rank factor loadings
    h -> D(h)                (IdiosyncraticScaleHead)
    z  = mu_z + sigma_z * eta_z,  eta_z ~ N(0, I_L)  (K draws)
    surface_change = einsum("btdl, bkl -> bktd", Lambda(h), z) + D(h) * eta_idio
    surface_level  = history[:, -1] + cumsum(surface_change, dim=2)
    return (B, K, T, D)

Stage B (LearnedMarginalHead):  conditional monotone spline head, knot positions
and output values are both learnable (no precomputed empirical quantiles).
Added later via extend_with_marginal_head().
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder


@dataclass
class NeuralFactorConfig:
    # Shape (stack-agnostic — D is whatever the caller passes)
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    latent_dim: int = 8

    # Encoder
    encoder_hidden: int = 64
    bottleneck_dim: int = 128
    encoder_dropout: float = 0.1
    cond_aug_sigma: float = 0.0

    # Heads
    head_hidden: int = 256
    head_layers: int = 2
    head_dropout: float = 0.1

    # Idiosyncratic scale floor/ceiling (softplus-parameterised)
    idio_softplus_shift: float = -3.0
    idio_scale_clip: float = 0.20

    # Latent prior sigma regularisation (KL-free — just a soft bound)
    latent_logsigma_min: float = -4.0
    latent_logsigma_max: float = 1.0

    # Orthogonality regularisation on Lambda (optional, enabled via regulariser flag)
    ortho_reg_weight: float = 0.0

    # Support clamp for emitted surface levels (IV convention; callers can override)
    support_lo: float = 0.01
    support_hi: float = 1.0

    # Stage B knob: attach learned marginal head (default off; enabled in 250b)
    use_marginal_head: bool = False
    marginal_knots: int = 12

    # Stage C knob: replace Gaussian z with flow-matched posterior
    use_latent_fm: bool = False
    latent_fm_steps: int = 4
    latent_fm_hidden: int = 256
    latent_fm_time_embed: int = 32

    # 251a knob: replace static z with time-indexed z_t via neural SDE.
    # z_0 still comes from LatentFM (or Gaussian reparam); z_{t+1} from Euler step
    # of learned drift f_theta + diffusion g_theta. Readout becomes per-step:
    # Lambda(h)[t] @ z_t instead of Lambda(h)[t] @ z.
    use_latent_sde: bool = False
    latent_sde_hidden: int = 256
    latent_sde_time_embed: int = 32
    latent_sde_eps_floor: float = 1e-4

    # 251c knob: add OU prior regularizer ||f_theta + alpha*z_t||^2 as a
    # loss-only nudge toward mean reversion. alpha is a learnable scalar;
    # does not enter LatentSDE.step/sample forward pass (time-homogeneity
    # invariant preserved).
    use_ou_prior: bool = False
    ou_alpha_init: float = -4.0  # softplus(-4) ~ 0.0181

    # H1 (251e): regime-aware modulation of Lambda and D via learned r_slow
    # embedding. Regime classifier is a 2-layer MLP over per-cell history stats
    # (mean, std, RV, skew-proxy). r_slow is concatenated to h before Lambda/D heads.
    use_regime: bool = False
    regime_dim: int = 8
    regime_hidden: int = 64
    regime_feature_windows: int = 30  # history length used for regime stats

    # H2 (251f): Student-t base distribution for LatentFM prior p_0, transferring
    # the 108a tail insight (kurtosis 0.955) into the 251 stationary-SDE framework.
    # Gaussian eps_t preserved to avoid 30-step CLT smoothing documented in 141b/141c.
    latent_fm_base_dist: str = "gaussian"  # "gaussian" | "student_t"
    latent_fm_nu_init: float = -2.0  # softplus(-2) ~ 0.127; nu = 2 + softplus ~ 2.13

    # H4 (251d): variance-preserving OU. Multiply LatentSDE diffusion output by
    # sqrt(2*alpha) so equilibrium variance g^2/(2*alpha) stays constant as alpha
    # grows. Resolves the spread-vs-MR Pareto observed in 251b->251c.
    use_vp_ou: bool = False

    # W3 (251h): surface clamp mode. H2 found that decode()'s clamp([support_lo,
    # support_hi]) truncates heavy-tail amplified factors (kurt(factor)=255, kurt(samples)=3).
    # "hard" = always clamp (250/251a/251b/251c legacy behavior).
    # "none" = skip clamp when model.training=True; keep clamp at eval/inference.
    training_clamp_mode: str = "hard"

    # W3-pattern applied to D_scale (iteration 2): the idio_scale_clip at line 214
    # is another hand-engineered upper bound. "hard" = always clamp (legacy).
    # "none" = skip during training, keep at eval.
    training_idio_clip_mode: str = "hard"


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)]
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


class LatentEncoder(nn.Module):
    """h -> (mu_z, log_sigma_z) for the Gaussian latent factor posterior."""

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            in_dim=cfg.bottleneck_dim,
            out_dim=2 * cfg.latent_dim,
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(h)
        mu, log_sigma = out.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(
            self.cfg.latent_logsigma_min, self.cfg.latent_logsigma_max
        )
        return mu, log_sigma


class LoadingHead(nn.Module):
    """h -> Lambda(h) in R^{B, T, D, L} — explicit factor loadings per (t, d).

    H1 extension: when cfg.use_regime=True, input is [h, r_slow] with regime_dim
    extra columns. First-layer weight is partitioned accordingly.
    """

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.future_len
        self.D = cfg.n_cells
        self.L = cfg.latent_dim
        in_dim = cfg.bottleneck_dim + (cfg.regime_dim if cfg.use_regime else 0)
        self.net = _mlp(
            in_dim=in_dim,
            out_dim=self.T * self.D * self.L,
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )
        # Small-init: factor path starts near zero so idio path dominates early.
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.1)
                last.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        flat = self.net(h)
        return flat.view(B, self.T, self.D, self.L)


class IdiosyncraticScaleHead(nn.Module):
    """h -> D(h) in R^{B, T, D} — positive scales via shifted softplus.

    H1 extension: when cfg.use_regime=True, input is [h, r_slow] with regime_dim extra columns.
    """

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.future_len
        self.D = cfg.n_cells
        in_dim = cfg.bottleneck_dim + (cfg.regime_dim if cfg.use_regime else 0)
        self.net = _mlp(
            in_dim=in_dim,
            out_dim=self.T * self.D,
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        raw = self.net(h).view(B, self.T, self.D)
        raw = raw + self.cfg.idio_softplus_shift
        scale = F.softplus(raw)
        # Soft cap to keep idio path from dominating early training.
        # W3-pattern: training_idio_clip_mode="none" skips the cap at training time,
        # keeps it at eval (for output-validity). "hard" = always clamp (legacy).
        if self.cfg.training_idio_clip_mode == "hard" or not self.training:
            scale = scale.clamp(max=self.cfg.idio_scale_clip)
        return scale


class LatentFM(nn.Module):
    """Stage C — latent flow-matching posterior.

    Replaces the Gaussian reparameterisation `z = mu_z + sigma_z * eta` with an
    ODE-solved trajectory starting from η ~ N(0, I_L):

        z(t=0) = η ~ N(0, I_L)
        dz/dt  = v_θ(z_t, t, h)        (history-conditioned velocity)
        z(t=1) = posterior sample

    4-step Euler integration by default. Small MLP head; all shapes preserve L.
    Trained end-to-end on the same proper-scoring-rule loss as Stage A/B — if the
    optimal posterior is multimodal (tail-driven), LatentFM learns to deform the
    N(0, I) prior into that distribution.
    """

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.L = cfg.latent_dim
        self.n_steps = cfg.latent_fm_steps
        self.time_embed = nn.Linear(1, cfg.latent_fm_time_embed)
        in_dim = self.L + cfg.latent_fm_time_embed + cfg.bottleneck_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.latent_fm_hidden), nn.GELU(),
            nn.Linear(cfg.latent_fm_hidden, cfg.latent_fm_hidden), nn.GELU(),
            nn.Linear(cfg.latent_fm_hidden, self.L),
        )
        # Small-init velocity so early training is ≈ identity flow (η → η).
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.1)
                last.bias.zero_()
        # H2 (251f): Student-t base distribution for p_0. ν = 2 + softplus(nu_raw)
        # ensures ν > 2 for finite variance. Gaussian is the default (nu_raw=None).
        if cfg.latent_fm_base_dist == "student_t":
            self.nu_raw: torch.nn.Parameter | None = nn.Parameter(
                torch.tensor(float(cfg.latent_fm_nu_init))
            )
        else:
            self.nu_raw = None

    def get_nu(self) -> float | None:
        if self.nu_raw is None:
            return None
        return float(2.0 + F.softplus(self.nu_raw).item())

    def velocity(self, z_t: torch.Tensor, t_val: float, h: torch.Tensor) -> torch.Tensor:
        """z_t: (B, K, L). t_val: scalar in [0, 1]. h: (B, bottleneck). -> (B, K, L)."""
        B, K, _ = z_t.shape
        t_tensor = torch.full(
            (B, K, 1), float(t_val), device=z_t.device, dtype=z_t.dtype
        )
        t_emb = self.time_embed(t_tensor)             # (B, K, time_embed)
        h_exp = h.unsqueeze(1).expand(-1, K, -1)      # (B, K, bottleneck)
        inp = torch.cat([z_t, t_emb, h_exp], dim=-1)  # (B, K, L + time_embed + bottleneck)
        return self.net(inp)

    def sample(self, eta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """eta: (B, K, L) ~ N(0, I_L). h: (B, bottleneck). -> z: (B, K, L)."""
        dt = 1.0 / self.n_steps
        z = eta
        for i in range(self.n_steps):
            v = self.velocity(z, i * dt, h)
            z = z + dt * v
        return z


class LatentSDE(nn.Module):
    """251a — dynamical latent factor process.

    Evolves z_t as a learned neural SDE over the T-step horizon:

        z_{t+1} = z_t + f_theta(z_t, t, h) * dt
                      + g_theta(z_t, t, h) * eps_t * sqrt(dt),   eps_t ~ N(0, I_L)

    f_theta and g_theta share a small MLP on (z_t, t_emb, h) -> (drift, log_diff).
    Diffusion has a hard floor via eps_floor to prevent deterministic collapse
    (kill criterion: effective_rank(z_t) >= 2). Initial condition z_0 comes from
    whatever the caller provides (Gaussian reparam or LatentFM output).

    Stack-agnostic: only touches (z_t, t, h); returns trajectory (B, K, T, L).
    """

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.L = cfg.latent_dim
        self.T = cfg.future_len
        self.eps_floor = cfg.latent_sde_eps_floor
        # 251b switch: latent_sde_time_embed == 0 forces time-homogeneity (f_theta,
        # g_theta functions of (z_t, h) only). > 0 keeps 251a behavior byte-for-byte.
        if cfg.latent_sde_time_embed > 0:
            self.time_embed: nn.Linear | None = nn.Linear(1, cfg.latent_sde_time_embed)
            in_dim = self.L + cfg.latent_sde_time_embed + cfg.bottleneck_dim
        else:
            self.time_embed = None
            in_dim = self.L + cfg.bottleneck_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.latent_sde_hidden), nn.SiLU(),
            nn.Linear(cfg.latent_sde_hidden, cfg.latent_sde_hidden), nn.SiLU(),
            nn.Linear(cfg.latent_sde_hidden, 2 * self.L),
        )
        # Small non-zero init so dt=1/T doesn't make early gradients vanish.
        with torch.no_grad():
            last = self.mlp[-1]
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, std=1e-3)
                last.bias.zero_()
        # 251c OU prior: learnable scalar alpha via softplus(alpha_raw). Loss-only;
        # does NOT enter forward pass. When use_ou_prior=False, attribute is None
        # (no params added, state_dict keys stay clean for 250/251a/251b checkpoints).
        if cfg.use_ou_prior:
            self.alpha_raw: torch.nn.Parameter | None = nn.Parameter(
                torch.tensor(float(cfg.ou_alpha_init))
            )
        else:
            self.alpha_raw = None

    def step(
        self, z_t: torch.Tensor, t_idx: int, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (drift, diffusion) at step index t_idx ∈ [0, T-1)."""
        B, K, L = z_t.shape
        h_exp = h.unsqueeze(1).expand(-1, K, -1)   # (B, K, bottleneck)
        if self.time_embed is not None:
            t_val = (t_idx + 0.5) / self.T  # center-of-interval time coord
            t_tensor = torch.full(
                (B, K, 1), float(t_val), device=z_t.device, dtype=z_t.dtype
            )
            t_emb = self.time_embed(t_tensor)      # (B, K, time_embed)
            inp = torch.cat([z_t, t_emb, h_exp], dim=-1)
        else:
            inp = torch.cat([z_t, h_exp], dim=-1)
        out = self.mlp(inp)                        # (B, K, 2L)
        drift = out[..., :L]
        log_diff = out[..., L:]
        diffusion = F.softplus(log_diff) + self.eps_floor
        # H4 (251d): variance-preserving OU scaling. Multiply g by sqrt(2*alpha)
        # so equilibrium variance g_base^2 / (2*alpha) * (2*alpha) = g_base^2
        # stays constant as alpha grows. Requires use_ou_prior + alpha_raw.
        if self.cfg.use_vp_ou and self.alpha_raw is not None:
            alpha = F.softplus(self.alpha_raw)
            diffusion = diffusion * torch.sqrt(2.0 * alpha)
        return drift, diffusion

    def sample(
        self, z_0: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z_0: (B, K, L). h: (B, bottleneck).

        Returns (z_path, drift_path):
          - z_path: (B, K, T, L). z_path[:, :, 0] = z_0; z_path[:, :, t+1] = Euler step.
          - drift_path: (B, K, T-1, L). f_theta(z_t, t, h) at each integration step.

        drift_path is exposed for loss-side use (251c OU regularizer). Callers that
        only need z_path should unpack and discard drift_path.
        """
        dt = 1.0 / self.T
        sqrt_dt = dt ** 0.5
        z_t = z_0
        z_list = [z_0]
        drift_list: list[torch.Tensor] = []
        for t_idx in range(self.T - 1):
            drift, diffusion = self.step(z_t, t_idx, h)
            drift_list.append(drift)
            eps = torch.randn_like(z_t)
            z_t = z_t + drift * dt + diffusion * eps * sqrt_dt
            z_list.append(z_t)
        z_path = torch.stack(z_list, dim=2)       # (B, K, T, L)
        drift_path = torch.stack(drift_list, dim=2)  # (B, K, T-1, L)
        return z_path, drift_path


class RegimeEncoder(nn.Module):
    """H1 (251e): compute soft regime embedding r_slow from history statistics.

    Bitter-lesson compliant: regime is LEARNED end-to-end from per-cell running
    statistics of the history window. No hand-labeled vol states.

    Output concatenated to h before LoadingHead and IdiosyncraticScaleHead. Entire
    model still (B, K, T, D) stack-agnostic — no reshape into H x W grid.
    """

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.n_features = 4  # stats per cell: mean, std, RV, skew-proxy
        self.input_dim = self.n_features * cfg.n_cells
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, cfg.regime_hidden), nn.SiLU(),
            nn.Linear(cfg.regime_hidden, cfg.regime_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (B, T_hist, D) with D = cfg.n_cells
        B = history.shape[0]
        diff = history.diff(dim=1)  # (B, T_hist-1, D)
        stats = torch.stack([
            history.mean(dim=1),                         # level per cell
            history.std(dim=1) + 1e-6,                   # scale per cell
            (diff ** 2).mean(dim=1).clamp(min=1e-12).sqrt(),  # realized-vol proxy
            (diff ** 3).mean(dim=1),                     # skew-like per cell
        ], dim=-1)  # (B, D, 4)
        flat = stats.reshape(B, -1)  # (B, D*4)
        return self.mlp(flat)  # (B, regime_dim)


class LearnedMarginalHead(nn.Module):
    """Stage B — monotone spline head with fully learnable knot positions and values.

    Parameterisation (Choice A, conditional):
        input  knot offsets: delta_x(h, t)   via softplus -> positive
        output knot offsets: delta_y(h, t)   via softplus -> positive
        knot_x[k] = x_start + cumsum_k(delta_x)
        knot_y[k] = y_start + cumsum_k(delta_y)
    Monotone by construction; both axes learned end-to-end. No precomputed empirical
    quantiles anywhere — Bitter-Lesson compliant.
    """

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.marginal_knots
        self.T = cfg.future_len
        self.D = cfg.n_cells
        # Shared head that produces per-(t, d) knot offsets conditioned on h.
        # Output: 2 * (K - 1) delta values per (t, d) slot -> split into dx, dy.
        self.net = _mlp(
            in_dim=cfg.bottleneck_dim,
            out_dim=self.T * self.D * 2 * (self.K - 1),
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )
        # Anchor points — small learned bias; does NOT depend on training-set quantiles.
        self.x_start = nn.Parameter(torch.tensor(-4.0))
        self.y_start = nn.Parameter(torch.tensor(-4.0))
        # Identity init: start with x = y on the spline.
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.zero_()
                last.bias.zero_()

    def _build_knots(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = h.shape[0]
        raw = self.net(h).view(B, self.T, self.D, 2, self.K - 1)
        # Small bias so softplus produces a reasonable initial spacing (~1.0).
        delta_x = F.softplus(raw[..., 0, :] + 0.5413)  # softplus(0.5413) ~= 1.0
        delta_y = F.softplus(raw[..., 1, :] + 0.5413)
        # Cumulative knot positions
        knot_x_inner = torch.cumsum(delta_x, dim=-1)
        knot_y_inner = torch.cumsum(delta_y, dim=-1)
        # Prepend the start anchor
        start_x = self.x_start.expand(B, self.T, self.D, 1)
        start_y = self.y_start.expand(B, self.T, self.D, 1)
        knot_x = torch.cat([start_x, start_x + knot_x_inner], dim=-1)
        knot_y = torch.cat([start_y, start_y + knot_y_inner], dim=-1)
        return knot_x, knot_y  # (B, T, D, K)

    def forward(self, y_raw: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        y_raw: (B, K_ens, T, D)
        h:     (B, bottleneck_dim)
        returns: (B, K_ens, T, D) — monotone-mapped
        """
        knot_x, knot_y = self._build_knots(h)  # (B, T, D, K)
        B, K_ens, T, D = y_raw.shape
        # Broadcast knots across ensemble: (B, 1, T, D, K) vs (B, K_ens, T, D, 1)
        kx = knot_x.unsqueeze(1)
        ky = knot_y.unsqueeze(1)
        y = y_raw.unsqueeze(-1)

        # Find bucket index: largest k where y >= kx[k]
        # cmp shape: (B, K_ens, T, D, K) — True where y >= knot_x
        cmp = y >= kx
        # Count of True along last dim -> bucket index in [0, K]
        idx = cmp.to(torch.long).sum(dim=-1)
        # Clamp so we always have a valid pair (idx-1, idx)
        idx_lo = (idx - 1).clamp(0, self.K - 2)
        idx_hi = idx_lo + 1

        # Gather knot positions/values at the bucket endpoints
        x_lo = torch.gather(kx.expand(B, K_ens, T, D, self.K), -1, idx_lo.unsqueeze(-1)).squeeze(-1)
        x_hi = torch.gather(kx.expand(B, K_ens, T, D, self.K), -1, idx_hi.unsqueeze(-1)).squeeze(-1)
        y_lo = torch.gather(ky.expand(B, K_ens, T, D, self.K), -1, idx_lo.unsqueeze(-1)).squeeze(-1)
        y_hi = torch.gather(ky.expand(B, K_ens, T, D, self.K), -1, idx_hi.unsqueeze(-1)).squeeze(-1)

        # Piecewise-linear interp with linear extrapolation (slope at endpoints).
        denom = (x_hi - x_lo).clamp(min=1e-6)
        slope = (y_hi - y_lo) / denom
        y_mapped = y_lo + slope * (y_raw - x_lo)
        return y_mapped


class NeuralFactorModel(nn.Module):
    """Stage A non-AR neural factor generator.

    Call convention:
        history: (B, T_hist, D)         -- caller pre-flattens any spatial layout
        returns (B, K, T_future, D)

    For IV use with history shape (B, T_hist, 5, 5), callers should reshape to
    (B, T_hist, D=25) BEFORE calling this model, and reshape returned (B, K, T, D)
    to (B, K, T, 5, 5) for IV-grid scoring. No 5x5 semantics live inside the model.
    """

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        encoder_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            extra_features=0,
            gru_hidden_dim=cfg.encoder_hidden,
            bottleneck_dim=cfg.bottleneck_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=cfg.cond_aug_sigma,
        )
        self.encoder = GRUEncoder(encoder_cfg)
        self.latent_encoder = LatentEncoder(cfg)
        self.loading_head = LoadingHead(cfg)
        self.idio_head = IdiosyncraticScaleHead(cfg)
        if cfg.use_marginal_head:
            self.marginal_head: Optional[LearnedMarginalHead] = LearnedMarginalHead(cfg)
        else:
            self.marginal_head = None
        if cfg.use_latent_fm:
            self.latent_fm: Optional[LatentFM] = LatentFM(cfg)
        else:
            self.latent_fm = None
        if cfg.use_latent_sde:
            self.latent_sde: Optional[LatentSDE] = LatentSDE(cfg)
        else:
            self.latent_sde = None
        if cfg.use_regime:
            self.regime_encoder: Optional[RegimeEncoder] = RegimeEncoder(cfg)
        else:
            self.regime_encoder = None

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, T_hist, D) -> h: (B, bottleneck_dim)."""
        # GRUEncoder's reshape (B, T, -1) is a no-op when last dim is already D.
        return self.encoder(history)

    def decode(
        self,
        history: torch.Tensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        history: (B, T_hist, D) in normalized [-1, 1] convention (same as 183c family).
        Returns:
          samples: (B, K, T_future, D) in IV [0, 1] space.
          aux:     dict with intermediate tensors for diagnostics / losses.
        """
        B = history.shape[0]
        D = history.shape[-1]
        K = n_samples
        assert D == self.cfg.n_cells, (
            f"history last dim {D} != cfg.n_cells {self.cfg.n_cells}"
        )

        h = self.encode_history(history)  # (B, bottleneck_dim)
        mu_z, log_sigma_z = self.latent_encoder(h)  # (B, L), (B, L)
        # H1 (251e): concat regime embedding before LoadingHead / IdiosyncraticScaleHead
        if self.cfg.use_regime and self.regime_encoder is not None:
            r_slow = self.regime_encoder(history)  # (B, regime_dim)
            h_ext = torch.cat([h, r_slow], dim=-1)  # (B, bottleneck_dim + regime_dim)
        else:
            r_slow = None
            h_ext = h
        Lambda = self.loading_head(h_ext)  # (B, T, D, L)
        D_scale = self.idio_head(h_ext)    # (B, T, D)

        # Draw K latent samples. Default eta_z ~ N(0, I_L).
        # H2 (251f): if latent_fm uses student_t base, sample from Student-t(ν) via
        # torch.distributions.StudentT (internally a Gaussian/Chi² scale-mixture).
        # Disable autocast for stability at low ν (Gamma rsample is fp32-sensitive).
        sigma_z = torch.exp(log_sigma_z).unsqueeze(1)  # (B, 1, L)
        use_t = (
            self.latent_fm is not None
            and self.cfg.latent_fm_base_dist == "student_t"
            and self.latent_fm.nu_raw is not None
        )
        if use_t:
            nu = 2.0 + F.softplus(self.latent_fm.nu_raw).float()
            with torch.autocast(device_type=history.device.type, enabled=False):
                dist = torch.distributions.StudentT(df=nu)
                eta_z = dist.rsample(
                    (B, K, self.cfg.latent_dim)
                ).to(device=history.device, dtype=history.dtype)
        else:
            eta_z = torch.randn(
                B, K, self.cfg.latent_dim, device=history.device, dtype=history.dtype
            )
        if self.latent_fm is not None:
            # Stage C: ODE-solve from η ~ N(0, I) to posterior sample, history-conditioned.
            z = self.latent_fm.sample(eta_z, h)
        else:
            # Stage A/B: parametric Gaussian reparameterisation
            z = mu_z.unsqueeze(1) + sigma_z * eta_z  # (B, K, L)

        # 251a: if SDE is active, evolve z_t over T steps; else keep static z.
        # Explicit branch preserves 250a/b/c einsum byte-for-byte when SDE off.
        if self.latent_sde is not None:
            z_path, drift_path = self.latent_sde.sample(z, h)   # (B, K, T, L), (B, K, T-1, L)
            factor = torch.einsum("btdl, bktl -> bktd", Lambda, z_path)
        else:
            z_path = None
            drift_path = None
            # Factor path: (B, T, D, L) x (B, K, L) -> (B, K, T, D)
            factor = torch.einsum("btdl, bkl -> bktd", Lambda, z)

        # Idiosyncratic residual
        eta_idio = torch.randn(
            B, K, self.cfg.future_len, D, device=history.device, dtype=history.dtype
        )
        idio = D_scale.unsqueeze(1) * eta_idio  # (B, K, T, D)

        surface_change = factor + idio  # (B, K, T, D)
        # Denormalize history's last frame to [0, 1] IV space for cumsum base.
        last_iv = 0.5 * (history[:, -1:, :] + 1.0)
        last_iv = last_iv.clamp(self.cfg.support_lo, self.cfg.support_hi)
        last = last_iv.unsqueeze(1)  # (B, 1, 1, D)
        surface_level = last + torch.cumsum(surface_change, dim=2)

        # W3 (251h): conditional surface clamp. Only applied if training_clamp_mode=="hard"
        # OR model is in eval mode. When "none" and training=True, skip clamp so heavy-tail
        # factor amplification survives to the loss.
        def _maybe_clamp(x: torch.Tensor) -> torch.Tensor:
            if self.cfg.training_clamp_mode == "hard" or not self.training:
                return x.clamp(self.cfg.support_lo, self.cfg.support_hi)
            return x

        pre_head = _maybe_clamp(surface_level)

        if self.marginal_head is not None:
            surface_level = self.marginal_head(surface_level, h)

        surface_level = _maybe_clamp(surface_level)

        aux = {
            "h": h,
            "mu_z": mu_z,
            "log_sigma_z": log_sigma_z,
            "Lambda": Lambda,
            "D_scale": D_scale,
            "z": z,
            "z_path": z_path,   # (B, K, T, L) if SDE active; None otherwise
            "drift_path": drift_path,  # (B, K, T-1, L) if SDE active; None otherwise (251c OU reg reads this)
            "r_slow": r_slow,   # (B, regime_dim) if H1 regime active; None otherwise
            "factor": factor,
            "idio": idio,
            "pre_head": pre_head,  # clamped pre-marginal-head samples (B, K, T, D)
        }
        return surface_level, aux

    def forward(
        self,
        history: torch.Tensor,
        n_samples: int = 8,
        **_ignored_kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.decode(history, n_samples)

    # ------------------------------------------------------------------
    # Inference convenience — signature-compatible with sample_batched callers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: Optional[int] = None,
        chunk_size: Optional[int] = None,
        **_ignored_kwargs,
    ) -> torch.Tensor:
        """Inference sampler.

        history must be in normalized [-1, 1] convention (same as 183c family).
        Accepts either (B, T_hist, 5, 5) or (B, T_hist, D) shape.
        Returns (B, K, T_future, 5, 5) if input was 4-D, else (B, K, T_future, D).

        n_steps / chunk_size / max_residual / max_global_residual / extra_hist are
        accepted for API compatibility with the 11-suite evaluator and the
        conditionality harness but are unused (non-AR single-pass generator).
        """
        was_5d = (history.dim() == 4)
        if was_5d:
            B, T_hist, H, W = history.shape
            history_flat = history.reshape(B, T_hist, H * W)
        else:
            history_flat = history
            B, T_hist, D = history.shape

        samples, _ = self.decode(history_flat, n_samples=n_samples)  # (B, K, T, D)

        if was_5d:
            B, K, T, D = samples.shape
            samples = samples.reshape(B, K, T, H, W)
        return samples

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: Optional[int] = None,
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Alias — matches the 220h evaluator's native-path interface.

        Absorbs max_residual / max_global_residual / extra_hist from the
        conditionality test, which assumes an AR kernel signature.
        """
        return self.sample(
            history, n_samples=n_samples, n_steps=n_steps, chunk_size=chunk_size,
        )

    # ------------------------------------------------------------------
    # Regularisers (optional)
    # ------------------------------------------------------------------
    def orthogonality_penalty(self, Lambda: torch.Tensor) -> torch.Tensor:
        """|| Lambda^T Lambda - diag ||_F^2 averaged over (B, T), for use if
        full-rank leak is detected. Shape: (B, T, D, L) -> scalar.
        """
        B, T, D, L = Lambda.shape
        Lm = Lambda.reshape(B * T, D, L)
        gram = torch.einsum("bdl, bdm -> blm", Lm, Lm)  # (B*T, L, L)
        eye = torch.eye(L, device=Lambda.device, dtype=Lambda.dtype).expand(B * T, L, L)
        diag_target = gram * eye  # keep diag, zero off-diag
        off = gram - diag_target
        return off.pow(2).mean()


# ----------------------------------------------------------------------
# Checkpoint helpers — parallels the load_one_day_kernel contract
# ----------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[NeuralFactorModel, dict]:
    """Load a 250-series checkpoint. Returns (model, payload)."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = payload["config"]
    cfg = NeuralFactorConfig(**cfg_dict)
    model = NeuralFactorModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def config_to_dict(cfg: NeuralFactorConfig) -> dict:
    return asdict(cfg)
