"""
Block-AR Conditional DDPM with MCVD Multi-Task Training + Diffusion Forcing.

Combines:
- GRU encoder for variable-length conditioning
- BiGRU denoiser for per-frame noise prediction
- MCVD 4-task masking (forward/backward/interpolation/unconditional)
- Task-adaptive per-frame noise schedules (Diffusion Forcing)
- PYoCo correlated noise for temporal coherence (ICCV 2023)
- Pyramid sampling schedule following the actual DF paper
- Block-autoregressive generation
"""

from dataclasses import dataclass
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import GRUEncoder, CausalConv3dEncoder, EncoderConfig
from diffusion.block_ar.bigru_denoiser import BiGRUDenoiser, DenoiserConfig
from diffusion.block_ar.masking import MCVDTask, get_task_types, sample_mcvd_masks
from diffusion.block_ar.noise_schedules import sample_batch_task_adaptive_noise
from diffusion.ddpm_scheduler import DDPMScheduler, FlowMatchingScheduler


# IV normalization constants — must match train_ddpm_poc.py
IV_MIN = 0.0
IV_MAX = 1.0


def denormalize_iv(iv_norm: torch.Tensor) -> torch.Tensor:
    """Denormalize IV from [-1, 1] to [0, 1]."""
    return (iv_norm + 1.0) / 2.0 * (IV_MAX - IV_MIN) + IV_MIN


def normalize_iv(iv: torch.Tensor) -> torch.Tensor:
    """Normalize IV from [0, 1] to [-1, 1]."""
    return (iv - IV_MIN) / (IV_MAX - IV_MIN) * 2.0 - 1.0


_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)  # ≈ 0.5642


def crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise CRPS for Gaussian predictive distribution N(mu, sigma^2).

    CRPS(mu, sigma, y) = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma, Phi = normal CDF, phi = normal PDF.

    Returns element-wise CRPS (same shape as inputs). Lower is better.
    Gradient w.r.t. sigma creates direct condition→uncertainty learning signal.
    """
    sigma = sigma.clamp(min=1e-6)  # numerical stability
    z = (y - mu) / sigma
    phi_z = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)  # standard normal PDF
    Phi_z = 0.5 * (1 + torch.erf(z / math.sqrt(2)))  # standard normal CDF
    return sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - _INV_SQRT_PI)


def sample_pyoco_noise(shape: tuple, rho: float, device: torch.device) -> torch.Tensor:
    """Sample PYoCo temporally-correlated noise (Ge et al., ICCV 2023).

    Each frame's noise is marginal N(0, I), but adjacent frames share a
    common noise component controlled by rho:
        epsilon_t = rho * epsilon_shared + sqrt(1 - rho^2) * epsilon_t_indep

    Args:
        shape: (B, T, H, W) — target noise shape
        rho: correlation coefficient in [0, 1]. 0 = independent, 1 = fully shared.
        device: torch device

    Returns:
        noise: (B, T, H, W) with each frame marginal N(0, I)
    """
    B, T, H, W = shape
    if rho == 0.0:
        return torch.randn(shape, device=device)
    eps_shared = torch.randn(B, 1, H, W, device=device)
    eps_indep = torch.randn(B, T, H, W, device=device)
    return rho * eps_shared + math.sqrt(1.0 - rho ** 2) * eps_indep


@dataclass
class BlockARConfig:
    # Data
    history_len: int = 30
    future_len: int = 30
    surface_h: int = 5
    surface_w: int = 5
    block_size: int = 10

    # Encoder
    gru_hidden_dim: int = 64
    bottleneck_dim: int = 64
    cond_aug_sigma: float = 0.0
    encoder_dropout: float = 0.1

    # Denoiser
    denoiser_type: str = "bigru"  # "bigru", "conv3d", "causal_conv3d", or "dit"
    bigru_hidden_dim: int = 128
    pos_embed_dim: int = 16
    noise_embed_dim: int = 16
    denoiser_dropout: float = 0.1

    # Conv3D denoiser params (only used when denoiser_type="conv3d")
    conv3d_base_channels: int = 32
    conv3d_n_res_blocks: int = 4
    conv3d_groups: int = 8
    conv3d_noise_embed_dim: int = 64

    # DiT denoiser params (only used when denoiser_type="dit")
    dit_d_model: int = 64
    dit_n_layers: int = 6
    dit_n_heads: int = 4
    dit_mlp_ratio: float = 2.0
    dit_noise_embed_dim: int = 64

    # Diffusion
    n_steps: int = 100
    schedule: str = "cosine"

    # Flow Matching (Lipman et al. 2023): replace DDPM with learned ODE transport.
    # x_t = (1-t)*x_0 + t*eps, velocity v = eps - x_0, Euler ODE solve at inference.
    # No Gaussian noise schedule assumption; per-cell calibration from learned velocity.
    use_flow_matching: bool = False
    fm_n_inference_steps: int = 100  # Euler steps at inference (more = better quality)

    # MCVD
    p_mask: float = 0.2  # legacy Bernoulli (used if mcvd_task_probs all zero)
    jitter_std: float = 0.15
    forward_only: bool = False  # disable MCVD: always FORWARD task (past visible, future masked)
    # Explicit MCVD task probabilities — overrides p_mask when any are nonzero.
    # Order: (forward, backward, interpolation, unconditional), must sum to 1.0.
    mcvd_p_forward: float = 0.0
    mcvd_p_backward: float = 0.0
    mcvd_p_interpolation: float = 0.0
    mcvd_p_unconditional: float = 0.0
    # Down-weight interpolation task gradient (1.0 = no change, 0.3 = 30% gradient).
    # Decouples task exposure from gradient pressure: model still sees interpolation
    # tasks but they contribute less to learning, preserving tail/kurtosis behavior.
    interp_loss_weight: float = 1.0

    # Uniform-t noise (one scalar t per block instead of per-frame task-adaptive)
    use_uniform_noise: bool = False

    # Sampling mode: "pyramid" (DF staggered) or "uniform" (standard DDPM reverse)
    sampling_mode: str = "pyramid"

    # PYoCo correlated noise (0.0 = independent, 1.0 = fully shared)
    noise_rho: float = 0.5

    # Loss function: "mse" or "huber" (Huber/SmoothL1 preserves tails better)
    loss_type: str = "mse"
    huber_delta: float = 0.1  # Huber threshold — smaller = more L1-like for large errors

    # Regime-weighted loss: turb windows get higher training weight
    turb_loss_weight: float = 0.0  # 0.0=off, >0 = turb_weight (calm=1.0)

    # Regime conditioning (hierarchical sampling)
    n_regimes: int = 5
    regime_embed_dim: int = 32
    regime_loss_weight: float = 1.0
    use_regime_conditioning: bool = False

    # Sampling
    max_residual_timestep: int = 20

    # Global horizon-dependent residual noise for growing uncertainty.
    # Frame at global horizon h stops denoising at t_min = max_global_residual * h / (future_len - 1).
    # 0 = denoise all frames to t=0 (no residual, backward compat).
    # 10 = recommended starting point (~0.09 IV std residual at h=29).
    max_global_residual: int = 0

    # Posterior noise temperature: scale posterior variance during reverse diffusion.
    # >1.0 = wider ensemble spread (amplified by vol_scale → regime-adaptive).
    # Applied in z-space before denormalization, so multiplicative in ratio-space.
    noise_temperature: float = 1.0

    # Clamp final output to [0, 1] after denormalization.
    # True = legacy behavior (clips extreme IV values).
    # False = no clamp (preserves full output distribution for fair metric comparison).
    clamp_output: bool = True

    # Encoder type: "gru" (default, flat spatial) or "conv3d" (spatial-aware CausalConv3d)
    encoder_type: str = "gru"

    # Learned uncertainty head: per-horizon scaling of sample spread.
    # Trained separately (generator frozen) with CRPS loss.
    use_uncertainty_head: bool = False
    uncertainty_hidden_dim: int = 64

    # Heteroscedastic forward noise: scale diffusion noise by condition IV level.
    # High-IV conditions get proportionally more noise → wider CIs.
    # σ_cond = (mean_IV / global_mean_IV)^power, applied to forward noise and reverse sampling.
    # power=0.5: variance ∝ IV (conservative). power=1.0: std ∝ IV (matches multiplicative GT).
    heteroscedastic_noise: bool = False
    global_mean_iv: float = 0.2154  # precomputed from training data (denormalized [0,1])
    heteroscedastic_power: float = 0.5  # exponent for noise scaling

    # Per-cell heteroscedastic forward noise: static (5,5) scale from training data.
    # Each cell gets a different noise magnitude proportional to its normalized target std.
    # cell_noise_scale[r,c] = (cell_std / median_cell_std)^power, clamped.
    # This makes the forward process match the actual per-cell variance structure.
    # Denoiser predicts UNSCALED noise; cell_noise_scale applied to forward and reverse noise.
    cell_heteroscedastic: bool = False
    cell_noise_power: float = 0.5
    cell_noise_clamp_min: float = 0.5
    cell_noise_clamp_max: float = 2.0

    # SPADE: per-position learned scale/shift in denoiser AdaGN layers
    use_spade: bool = False

    # Spatial self-attention: multi-head attention over 5x5 grid in denoiser
    use_spatial_attention: bool = False
    spatial_attn_heads: int = 4

    # Per-cell residual head: condition-dependent noise prediction correction
    # per cell. Learns to fix mean bias for specific cells in specific regimes.
    use_percell_head: bool = False
    percell_head_hidden: int = 64
    percell_regime_input: bool = False  # add vol_of_vol scalar to percell_head input

    # Learned variance head (Diffusion2-style): model predicts per-sample log-variance
    # alongside noise. Trained with heteroscedastic NLL + beta-NLL stabilization.
    # The head sees ONLY the condition vector (not x_t), so it can't compensate.
    # At inference, predicted σ scales posterior noise for condition-dependent uncertainty.
    learned_variance: bool = False
    variance_beta_nll: float = 0.5  # beta-NLL weight (0.5 recommended by Seitzer et al.)

    # Ratio-space target: diffusion operates on transformed ratios instead of
    # absolute IV levels. Uncertainty naturally scales with IV level after
    # denormalization, achieving condition-dependent CIs without special noise
    # or heads. Encoder still sees absolute history.
    ratio_target: bool = False
    # "log" = log(future/baseline) with exp() inversion (original, can overflow)
    # "logit" = logit(future) - logit(baseline) with sigmoid() inversion (bounded)
    # "vol_scaled" = log(future/baseline) / vol_scale with exp(sample*vol_scale)*baseline inversion
    #   vol_scale = std(daily_mean_IV_changes) / global_mean_vol — condition-dependent scaling
    #   that creates wider CIs for turbulent conditions (high vol_of_vol).
    # "vol_scaled_percell" = per-cell vol normalization: log(future/baseline) / vol_scale[r,c]
    #   vol_scale[r,c] = std(cell[r,c] daily changes) / global_mean_cell_vol[r,c]
    #   Captures spatial heterogeneity: ATM cells respond 2-5x more to vol_of_vol than deep cells
    # "nsdiff" = NSDiff-inspired learned endpoint: log(future/baseline) / sigma_c where
    #   sigma_c = exp(log_std_head(condition)) is LEARNED from data with Gaussian NLL loss.
    #   Replaces hand-coded vol_scale with end-to-end learned scaling.
    #   Bitter Lesson: let the model learn the optimal standardization from data.
    # "e2e_nll" = End-to-end learned sigma with NLL + non-detached diffusion gradient.
    #   Unlike nsdiff, sigma is NOT detached — gradient flows from diffusion loss through sigma.
    #   NLL provides baseline signal, diffusion gradient adds condition-dependent pressure.
    # "vol_scaled_learned" = Hybrid: vol_scale * exp(learned_correction).
    #   Starts from hand-coded vol_scale (which already achieves Q5/Q1=1.21) and learns
    #   a small multiplicative correction. Correction initialized near 0 (identity).
    #   Combines proven vol_scale signal with end-to-end fine-tuning.
    ratio_target_mode: str = "log"
    global_mean_vol: float = 0.0187  # mean of vol_scale across training data (precomputed)
    vol_scale_power: float = 1.0  # exponent on vol_scale: 0.5=sqrt dampening, 1.0=full
    vol_scale_min: float = 0.5  # min clamp for vol_scale (higher = wider calm CIs)
    vol_scale_max: float = 2.0  # max clamp for vol_scale
    baseline_window: int = 1  # number of history days to average for baseline (1 = last day only)
    use_median_baseline: bool = False  # use median(full history) instead of mean(last K)
    # Learned mean head: MLP predicts per-cell conditional mean μ(condition).
    # Diffusion operates on z_residual = z - μ (zero-mean). At inference: z = μ + denoised.
    # Bitter Lesson: adds capacity for the model to learn regime-specific mean shifts
    # that the iterative diffusion process struggles to capture (e.g., calm-regime bias).
    use_mean_head: bool = False
    mean_head_lambda: float = 1.0  # weight for mean prediction loss
    # Learned drift head: predicts baseline shift in IV-space (drift OUTSIDE exp).
    # IV = (baseline + drift) × exp(z × vol_scale). Different from mean_head which
    # operates in z-space INSIDE exp: IV = baseline × exp((z+μ) × vol_scale).
    use_drift_head: bool = False
    drift_hidden_dim: int = 64
    drift_loss_weight: float = 1.0
    drift_max: float = 0.05  # max drift in IV space
    nsdiff_sigma_lambda: float = 0.1  # NLL/reg loss weight for learned sigma
    e2e_sigma_reg: float = 0.01  # L2 regularization weight for (sigma - 1.0)^2

    # Learned sigma (Nichol & Dhariwal 2021): denoiser predicts per-element
    # variance alongside noise. Variance = exp(v * log(beta_tilde_t) +
    # (1-v) * log(beta_t)), bounded between posterior variance and forward
    # variance. Trained with L_simple + lambda_vlb * L_vlb (KL divergence).
    learn_sigma: bool = False
    lambda_vlb: float = 0.001  # VLB loss weight (small so L_simple dominates)
    spatial_pos_encoding: bool = False  # CoordConv: add row/col coords to denoiser input
    baseline_channel: bool = False  # add baseline surface as extra denoiser input channel

    # Learned per-cell vol_scale correction: static (5,5) parameter trained by diffusion loss.
    # Multiplicative correction on scalar vol_scale: effective = vol_scale * exp(log_cell_scale).
    # Clamped to [-cell_scale_clamp, +cell_scale_clamp] → bounded correction factor.
    # Bitter Lesson: let the model learn per-cell scaling instead of hyperparameter tuning.
    learn_cell_scale: bool = False
    cell_scale_clamp: float = 0.2  # max abs value: 0.2 → [0.82x, 1.22x] correction range

    # Low-rank per-cell vol_scale correction: rank-1 outer product U×V
    # Smooth by construction (no neighbor discontinuities), 10 params total
    low_rank_cell_scale: bool = False
    low_rank_cell_scale_clamp: float = 0.5  # per-component clamp: [-0.5, 0.5]

    # Fixed per-cell vol_scale correction: from calibration head analysis.
    # Multiplicative (5,5) tensor applied to scalar vol_scale. Registered as buffer (frozen).
    # Derived from output-space pinball loss on validation samples.
    # None = disabled, list of 25 floats = (5,5) correction grid.
    cell_scale_values: list = None

    # Per-cell structural normalization: cell_norm[r,c] = (gmcv[r,c]/gmcv.mean())^power
    # Applied multiplicatively with scalar vol_scale: effective_scale = vol_scale * cell_norm
    # Accounts for structural per-cell volatility differences (28× range) that the model
    # only recovers ~85% of. 0.0 = off (current behavior), >0 = per-cell correction.
    cell_norm_power: float = 0.0

    # Per-cell loss weighting: w[r,c] = (gmcv.mean()/gmcv[r,c])^power
    # Inversely proportional to per-cell volatility. High-vol cells get lower weight →
    # less accurate noise predictions → more diversity → wider CIs (fix undercoverage).
    # Low-vol cells get higher weight → more accurate → narrower CIs (fix overcoverage).
    cell_loss_weight_power: float = 0.0

    # Classifier-Free Guidance (CFG): randomly drop conditioning during training
    # to learn both conditional and unconditional noise prediction. At inference,
    # guidance amplifies the effect of conditioning. For turbulent conditions where
    # conditioning is less informative, ε_cond ≈ ε_uncond → guidance has little effect
    # → naturally wider sample spread. For predictable conditions, guidance focuses
    # the output → narrower spread.
    cond_drop_prob: float = 0.0  # prob of dropping condition (0.0 = no CFG)
    guidance_scale: float = 1.0  # inference guidance scale (1.0 = no guidance)

    # Auxiliary regime features: feed explicit regime statistics (vol_of_vol, IV level)
    # as additional conditioning input. Computed from history, projected, and ADDED to
    # the encoder condition vector. Gives the denoiser direct access to regime info
    # that the GRU encoder may not extract into the bottleneck.
    # Bitter Lesson: give the model more data/features, let it learn.
    aux_regime_features: bool = False

    # CRPS variance head: separate module predicting per-sample σ from (condition, t).
    # Trained with CRPS on x₀ predictions alongside standard MSE noise prediction.
    # σ(condition, t) scales posterior noise at inference for condition-dependent CIs.
    # Unlike learn_sigma (Nichol-Dhariwal), this is a SEPARATE head with its own
    # parameters — decoupled from the denoiser to prevent σ collapse.
    crps_variance_head: bool = False
    lambda_crps: float = 0.1  # weight of CRPS auxiliary loss
    crps_n_cells: int = 1  # 1=scalar σ (original), 25=per-cell σ for 5×5 grid
    crps_sigma_clamp: float = 0.0  # max deviation from 1.0 in normalized σ (0=unlimited)
    crps_pos_embed_dim: int = 0  # frame position embedding dim for horizon-dependent σ (0=off)
    crps_boost_only: bool = False  # boost-only mode: σ = max(1.0, σ_norm) — widen, never narrow


class UncertaintyHead(nn.Module):
    """Per-horizon learned scaling for diffusion sample spread.

    Takes encoder condition → monotonically increasing scale factors.
    Applied as: scaled = mean + scale * (sample - mean).

    Architecture: base(condition) + cumsum(softplus(increments(condition)))
    - base: learned scalar log-scale (can be < 0 or > 0, so scale can be < 1 or > 1)
    - increments: monotonically increasing growth from base

    This allows both widening and narrowing at h=0, with guaranteed monotonic
    growth over the horizon.
    """

    def __init__(self, cond_dim: int, future_len: int, hidden_dim: int = 64):
        super().__init__()
        self.future_len = future_len
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, future_len + 1),  # +1 for base
        )
        # Initialize so base ≈ 0 (scale ≈ 1.0) and increments ≈ 0 (no growth)
        nn.init.zeros_(self.mlp[-1].weight)
        bias = torch.full((future_len + 1,), -5.0)
        bias[0] = 0.0  # base starts at 0 → exp(0) = 1.0
        self.mlp[-1].bias = nn.Parameter(bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Predict per-horizon scaling factors.

        Args:
            condition: (B, cond_dim) encoder output

        Returns:
            scale: (B, future_len) positive, monotonically increasing
        """
        raw = self.mlp(condition)  # (B, future_len + 1)
        base = raw[:, :1]  # (B, 1) — log-scale at h=0
        increments = F.softplus(raw[:, 1:])  # (B, future_len), all positive
        cumul = torch.cumsum(increments, dim=1)  # monotonically increasing
        cumul = cumul - cumul[:, :1]  # starts at 0
        log_scale = base + cumul  # base + monotonic growth
        return torch.exp(log_scale)  # (B, future_len) positive, monotonic


class VarianceHead(nn.Module):
    """Per-sample learned variance for condition-dependent uncertainty.

    Predicts log_σ² from condition vector ONLY (not x_t or t).
    This prevents the model from compensating by observing noise levels.

    Used with heteroscedastic NLL (beta-NLL stabilized):
        L = exp(-log_σ²)/2 * ||ε_pred - ε||² + log_σ²/2

    At inference, exp(log_σ²/2) = σ scales posterior noise.
    """

    def __init__(self, cond_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize to log_σ² ≈ 0 → σ ≈ 1 (no scaling initially)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Predict per-sample log-variance.

        Args:
            condition: (B, cond_dim) encoder output

        Returns:
            log_var: (B, 1) log-variance (unbounded)
        """
        return self.net(condition)  # (B, 1)


class CRPSVarianceHead(nn.Module):
    """Condition-dependent variance trained with CRPS on x_0 prediction.

    Takes (condition, t) → per-element log_sigma for posterior noise scaling.
    Unlike VarianceHead, this includes timestep so the model can learn that
    variance is only meaningful at moderate noise levels.

    Trained with CRPS_Gaussian(x_0_pred.detach(), sigma, x_0) — the
    gradient from CRPS creates an explicit condition→uncertainty signal:
    for turbulent conditions where x_0_pred is worse, CRPS rewards larger sigma.
    """

    def __init__(self, cond_dim: int, n_steps: int, hidden_dim: int = 64,
                 n_cells: int = 1, pos_embed_dim: int = 0):
        super().__init__()
        self.n_cells = n_cells
        self.pos_embed_dim = pos_embed_dim
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
        )
        # Optional frame position embedding (for horizon-dependent sigma)
        if pos_embed_dim > 0:
            self.pos_embed = nn.Sequential(
                nn.Linear(1, pos_embed_dim),
                nn.SiLU(),
            )
        else:
            self.pos_embed = None
        # Wider hidden layer when outputting per-cell sigma
        input_dim = cond_dim + hidden_dim + pos_embed_dim
        out_hidden = hidden_dim * 2 if n_cells > 1 else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_hidden),
            nn.SiLU(),
            nn.Linear(out_hidden, n_cells),
        )
        self.n_steps = n_steps
        # Initialize to log_sigma ≈ 0 → sigma ≈ 1 (neutral scaling)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: torch.Tensor, t: torch.Tensor,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict per-sample log-sigma.

        Args:
            condition: (B, cond_dim)
            t: (B,) or (B, T) integer timesteps
            positions: (B, T) frame positions (0-indexed) for horizon-dependent σ.
                       Only used if pos_embed_dim > 0. Ignored otherwise.

        Returns:
            log_sigma: (B, n_cells) if t is (B,), else (B, T, n_cells)
        """
        # Normalize t to [0, 1]
        if t.dim() == 1:
            t_norm = t.float().unsqueeze(-1) / self.n_steps  # (B, 1)
        else:
            # (B, T) → flatten, process, unflatten
            B, T = t.shape
            t_norm = t.float().reshape(-1, 1) / self.n_steps  # (B*T, 1)
            condition = condition.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        t_emb = self.t_embed(t_norm)  # (B, H) or (B*T, H)

        if self.pos_embed is not None and positions is not None:
            # positions: (B, T) → normalize to [0, 1] and flatten
            if positions.dim() == 1:
                pos_norm = positions.float().unsqueeze(-1) / 30.0  # (B, 1)
            else:
                B_pos, T_pos = positions.shape
                pos_norm = positions.float().reshape(-1, 1) / 30.0  # (B*T, 1)
            pos_emb = self.pos_embed(pos_norm)  # (B*T, pos_dim)
            h = torch.cat([condition, t_emb, pos_emb], dim=-1)
        else:
            h = torch.cat([condition, t_emb], dim=-1)

        log_sigma = self.net(h)  # (B, n_cells) or (B*T, n_cells)

        if t.dim() > 1:
            log_sigma = log_sigma.reshape(B, T, self.n_cells)

        return log_sigma


class _DenoiserAdapter(nn.Module):
    """Wraps denoiser to match DDPMScheduler's model(x_t, t, condition) interface."""

    def __init__(
        self,
        denoiser: nn.Module,
        condition: torch.Tensor,
        positions: torch.Tensor,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.condition = condition
        self.positions = positions

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition_unused: torch.Tensor
    ) -> torch.Tensor:
        # x_t: (B, T, 5, 5) -> flatten to (B, T, 25)
        B, T, H, W = x_t.shape
        x_flat = x_t.reshape(B, T, H * W)

        # Expand scalar t (B,) to per-frame (B, T) for denoiser interface
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(B, T)

        # Call denoiser
        noise_pred_flat = self.denoiser(
            x_flat, self.condition, self.positions, t
        )  # (B, T, 25)

        # Reshape back to (B, T, 5, 5)
        return noise_pred_flat.reshape(B, T, H, W)


class ConditionalBlockARDDPM(nn.Module):
    """Block-AR Conditional DDPM with MCVD multi-task training."""

    def __init__(self, config: BlockARConfig):
        super().__init__()
        self.config = config

        enc_cfg = EncoderConfig(
            input_dim=config.surface_h * config.surface_w,
            gru_hidden_dim=config.gru_hidden_dim,
            bottleneck_dim=config.bottleneck_dim,
            cond_aug_sigma=config.cond_aug_sigma,
            dropout=config.encoder_dropout,
        )
        if getattr(config, 'encoder_type', 'gru') == "conv3d":
            self.encoder = CausalConv3dEncoder(enc_cfg)
        else:
            self.encoder = GRUEncoder(enc_cfg)

        denoiser_type = getattr(config, 'denoiser_type', 'bigru')
        if denoiser_type == "dit":
            from diffusion.block_ar.dit_denoiser import (
                DiTBlockDenoiser, DiTDenoiserConfig,
            )
            dit_config = DiTDenoiserConfig(
                frame_dim=config.surface_h * config.surface_w,
                surface_h=config.surface_h,
                surface_w=config.surface_w,
                bottleneck_dim=config.bottleneck_dim,
                pos_embed_dim=config.pos_embed_dim,
                noise_embed_dim=getattr(config, 'dit_noise_embed_dim', 64),
                n_steps=config.n_steps,
                d_model=getattr(config, 'dit_d_model', 64),
                n_layers=getattr(config, 'dit_n_layers', 6),
                n_heads=getattr(config, 'dit_n_heads', 4),
                mlp_ratio=getattr(config, 'dit_mlp_ratio', 2.0),
                learn_sigma=getattr(config, 'learn_sigma', False),
            )
            self.denoiser = DiTBlockDenoiser(dit_config)
        elif denoiser_type in ("conv3d", "causal_conv3d"):
            from diffusion.block_ar.conv3d_denoiser import (
                Conv3DBlockDenoiser, CausalConv3DBlockDenoiser, Conv3DDenoiserConfig,
            )
            conv3d_config = Conv3DDenoiserConfig(
                frame_dim=config.surface_h * config.surface_w,
                surface_h=config.surface_h,
                surface_w=config.surface_w,
                bottleneck_dim=config.bottleneck_dim,
                pos_embed_dim=config.pos_embed_dim,
                noise_embed_dim=config.conv3d_noise_embed_dim,
                n_steps=config.n_steps,
                base_channels=config.conv3d_base_channels,
                n_res_blocks=config.conv3d_n_res_blocks,
                groups=config.conv3d_groups,
                learn_sigma=getattr(config, 'learn_sigma', False),
                spatial_pos_encoding=getattr(config, 'spatial_pos_encoding', False),
                use_spade=config.use_spade,
                use_percell_head=config.use_percell_head,
                percell_head_hidden=config.percell_head_hidden,
                percell_regime_input=getattr(config, 'percell_regime_input', False),
                baseline_channel=getattr(config, 'baseline_channel', False),
                use_spatial_attention=getattr(config, 'use_spatial_attention', False),
                spatial_attn_heads=getattr(config, 'spatial_attn_heads', 4),
            )
            if denoiser_type == "causal_conv3d":
                self.denoiser = CausalConv3DBlockDenoiser(conv3d_config)
            else:
                self.denoiser = Conv3DBlockDenoiser(conv3d_config)
        else:
            self.denoiser = BiGRUDenoiser(
                DenoiserConfig(
                    frame_dim=config.surface_h * config.surface_w,
                    surface_h=config.surface_h,
                    surface_w=config.surface_w,
                    bottleneck_dim=config.bottleneck_dim,
                    pos_embed_dim=config.pos_embed_dim,
                    noise_embed_dim=config.noise_embed_dim,
                    gru_hidden_dim=config.bigru_hidden_dim,
                    dropout=config.denoiser_dropout,
                )
            )

        self.scheduler = DDPMScheduler(
            n_steps=config.n_steps,
            schedule=config.schedule,
            device="cpu",
        )

        # Flow Matching scheduler (used when use_flow_matching=True)
        self.fm_scheduler = None
        if getattr(config, 'use_flow_matching', False):
            self.fm_scheduler = FlowMatchingScheduler(
                n_steps=getattr(config, 'fm_n_inference_steps', config.n_steps),
                device="cpu",
            )

        # Per-cell global mean vol (precomputed from training data)
        # Used by vol_scaled_percell mode AND cell_norm_power > 0
        _need_gmcv = (
            getattr(config, 'ratio_target_mode', 'log') == 'vol_scaled_percell'
            or getattr(config, 'cell_norm_power', 0.0) != 0
            or getattr(config, 'cell_loss_weight_power', 0.0) > 0
        )
        if _need_gmcv:
            _gmcv = torch.tensor([
                [0.156073, 0.043923, 0.014329, 0.024547, 0.090272],
                [0.080897, 0.018450, 0.009556, 0.008812, 0.059856],
                [0.032432, 0.012188, 0.007496, 0.006284, 0.038562],
                [0.018790, 0.007964, 0.005952, 0.005125, 0.007265],
                [0.022400, 0.007741, 0.005296, 0.004725, 0.008441],
            ], dtype=torch.float32)
            _is_percell_mode = getattr(config, 'ratio_target_mode', 'log') == 'vol_scaled_percell'
            self.register_buffer('global_mean_cell_vol', _gmcv, persistent=_is_percell_mode)
            # Precompute static per-cell normalization factor (non-persistent: recomputed from config)
            # Positive power: high-vol cells get wider CIs. Negative power: low-vol cells get wider CIs.
            if getattr(config, 'cell_norm_power', 0.0) != 0:
                cell_norm = (_gmcv / _gmcv.mean()).pow(config.cell_norm_power)
                self.register_buffer('cell_norm_factor', cell_norm, persistent=False)  # (5, 5)
            # Per-cell loss weights: inversely proportional to per-cell volatility
            if getattr(config, 'cell_loss_weight_power', 0.0) > 0:
                cell_weights = (_gmcv.mean() / _gmcv).pow(config.cell_loss_weight_power)
                # Normalize so mean weight = 1.0 (doesn't change effective learning rate)
                cell_weights = cell_weights / cell_weights.mean()
                self.register_buffer('cell_loss_weights', cell_weights, persistent=False)  # (5, 5)

        # NSDiff learned sigma head: condition → log_sigma (scalar per sample)
        # Trained with Gaussian NLL loss: 0.5 * log(sigma^2) + 0.5 * z^2
        # where z = log_ratio / sigma. Model learns optimal standardization.
        if getattr(config, 'ratio_target_mode', 'log') in ('nsdiff', 'e2e_nll'):
            self.log_std_head = nn.Sequential(
                nn.Linear(config.bottleneck_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
        else:
            self.log_std_head = None

        # Learned per-cell sigma: condition → (5,5) per-cell correction
        # Bitter Lesson approach: NN learns per-cell uncertainty from condition vector.
        # sigma[r,c] = vol_scale_percell[r,c] * exp(correction[r,c])
        # where vol_scale_percell is the hand-coded per-cell ratio (proven to work),
        # and correction is learned. Initialized at 0 (identity correction).
        if getattr(config, 'ratio_target_mode', 'log') == 'learned_percell':
            self.percell_sigma_head = nn.Sequential(
                nn.Linear(config.bottleneck_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, config.surface_h * config.surface_w),
            )
            # Initialize near zero (no correction) — let NN learn deviations
            nn.init.zeros_(self.percell_sigma_head[-1].weight)
            nn.init.zeros_(self.percell_sigma_head[-1].bias)
            # Pre-computed per-cell global mean vol for base scaling
            _gmcv = torch.tensor([
                [0.156073, 0.043923, 0.014329, 0.024547, 0.090272],
                [0.080897, 0.018450, 0.009556, 0.008812, 0.059856],
                [0.032432, 0.012188, 0.007496, 0.006284, 0.038562],
                [0.018790, 0.007964, 0.005952, 0.005125, 0.007265],
                [0.022400, 0.007741, 0.005296, 0.004725, 0.008441],
            ], dtype=torch.float32)
            self.register_buffer('percell_global_mean_vol', _gmcv)
        else:
            self.percell_sigma_head = None

        # Per-cell RevIN (Reversible Instance Normalization):
        # Foundation model consensus approach (Chronos, Moirai, TimesFM).
        # Each cell (r,c) normalized independently by its own std from history.
        # Learned per-cell floor replaces vol_scale_min hyperparameter.
        # log_revin_floor: (5,5) learnable parameter, initialized from global_mean_cell_vol.
        self.log_revin_floor = None
        if getattr(config, 'ratio_target_mode', 'log') == 'percell_revin':
            # Initialize floor from global_mean_cell_vol * baseline scale
            # These are typical per-cell daily change stds from training data
            _gmcv = torch.tensor([
                [0.156073, 0.043923, 0.014329, 0.024547, 0.090272],
                [0.080897, 0.018450, 0.009556, 0.008812, 0.059856],
                [0.032432, 0.012188, 0.007496, 0.006284, 0.038562],
                [0.018790, 0.007964, 0.005952, 0.005125, 0.007265],
                [0.022400, 0.007741, 0.005296, 0.004725, 0.008441],
            ], dtype=torch.float32)
            # Floor = 0.3 * global_mean_cell_vol (conservative initial floor)
            # Model learns optimal per-cell floor from training signal
            init_floor = 0.3 * _gmcv
            self.log_revin_floor = nn.Parameter(torch.log(init_floor))  # (5, 5)

        # Learned per-cell vol_scale correction: static (5,5) parameter, not a NN head.
        # Trained directly by MSE diffusion loss gradient. Bounded by cell_scale_clamp.
        # When cell needs wider CIs: gradient pushes correction UP → larger vol_scale for that cell.
        # Bitter Lesson: replaces vol_scale_min hyperparameter with learned capacity.
        self.log_cell_scale = None
        if getattr(config, 'learn_cell_scale', False):
            # Initialize at 0.0 (identity: no correction)
            self.log_cell_scale = nn.Parameter(torch.zeros(config.surface_h, config.surface_w))

        # Low-rank per-cell vol_scale correction: rank-1 outer product
        # correction(r,c) = exp(log_u[r] + log_v[c]) — smooth by construction
        # 10 params total (5 row + 5 col), max neighbor ratio bounded by clamp
        self.log_u_cell = None
        self.log_v_cell = None
        if getattr(config, 'low_rank_cell_scale', False):
            self.log_u_cell = nn.Parameter(torch.zeros(config.surface_h))  # (5,)
            self.log_v_cell = nn.Parameter(torch.zeros(config.surface_w))  # (5,)

        # Fixed per-cell vol_scale correction: registered as BUFFER (not trained by MSE).
        # Derived from calibration head analysis (output-space pinball loss on val set).
        # Applied identically to learn_cell_scale but frozen — denoiser adapts to it.
        # The per-cell pattern is structural (same for calm/turb), not regime-dependent.
        _cell_scale_values = getattr(config, 'cell_scale_values', None)
        if _cell_scale_values is not None:
            _csv = torch.tensor(_cell_scale_values, dtype=torch.float32)
            if _csv.shape == (config.surface_h * config.surface_w,):
                _csv = _csv.reshape(config.surface_h, config.surface_w)
            self.register_buffer('fixed_cell_scale', _csv)  # (5, 5) non-trainable
        else:
            self.fixed_cell_scale = None

        # Hybrid vol_scaled_learned: condition → small correction on top of vol_scale
        # sigma = vol_scale * exp(correction), initialized near correction=0
        if getattr(config, 'ratio_target_mode', 'log') == 'vol_scaled_learned':
            self.log_correction_head = nn.Sequential(
                nn.Linear(config.bottleneck_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
            # Initialize output layer near zero so correction starts at ~identity
            nn.init.zeros_(self.log_correction_head[-1].weight)
            nn.init.zeros_(self.log_correction_head[-1].bias)
        else:
            self.log_correction_head = None

        # Learned mean head: predicts per-cell conditional mean of z (ratio target).
        # Bitter Lesson approach: add capacity for regime-specific mean shifts.
        # Diffusion model handles zero-mean residual; mean head captures the drift.
        if getattr(config, 'use_mean_head', False) and config.ratio_target:
            self.mean_head = nn.Sequential(
                nn.Linear(config.bottleneck_dim, 64),
                nn.SiLU(),
                nn.Linear(64, config.surface_h * config.surface_w),
            )
            # Initialize near zero — model starts by predicting "no drift"
            nn.init.zeros_(self.mean_head[-1].weight)
            nn.init.zeros_(self.mean_head[-1].bias)
        else:
            self.mean_head = None

        # Learned drift head: predicts baseline shift in IV-space.
        # IV = (baseline + drift) × exp(z × vol_scale) — drift OUTSIDE exp.
        # Different from mean_head which operates in z-space INSIDE exp.
        if getattr(config, 'use_drift_head', False) and config.ratio_target:
            _drift_hidden = getattr(config, 'drift_hidden_dim', 64)
            self.drift_head = nn.Sequential(
                nn.Linear(config.bottleneck_dim, _drift_hidden),
                nn.SiLU(),
                nn.Linear(_drift_hidden, config.surface_h * config.surface_w),
            )
            # Zero-init: starts as no-drift (identical to current behavior)
            nn.init.zeros_(self.drift_head[-1].weight)
            nn.init.zeros_(self.drift_head[-1].bias)
        else:
            self.drift_head = None

        # Auxiliary regime features: project vol_of_vol + IV level to cond space
        if getattr(config, 'aux_regime_features', False):
            # 2 scalar inputs (vol_of_vol, mean_iv_level) → bottleneck_dim
            self.regime_feature_proj = nn.Sequential(
                nn.Linear(2, 64),
                nn.SiLU(),
                nn.Linear(64, config.bottleneck_dim),
            )
            # Initialize output near zero so regime features start as no-op
            nn.init.zeros_(self.regime_feature_proj[-1].weight)
            nn.init.zeros_(self.regime_feature_proj[-1].bias)
        else:
            self.regime_feature_proj = None

        # Regime conditioning (hierarchical sampling)
        if config.use_regime_conditioning:
            self.regime_classifier = nn.Sequential(
                nn.Linear(config.bottleneck_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, config.n_regimes),
            )
            self.regime_embed = nn.Embedding(config.n_regimes, config.regime_embed_dim)
            self.regime_proj = nn.Sequential(
                nn.Linear(config.bottleneck_dim + config.regime_embed_dim, config.bottleneck_dim),
                nn.SiLU(),
            )
        else:
            self.regime_classifier = None
            self.regime_embed = None
            self.regime_proj = None

        # Learned uncertainty head (trained separately, generator frozen)
        if config.use_uncertainty_head:
            self.uncertainty_head = UncertaintyHead(
                cond_dim=config.bottleneck_dim,
                future_len=config.future_len,
                hidden_dim=config.uncertainty_hidden_dim,
            )
        else:
            self.uncertainty_head = None

        # Learned variance head (Diffusion2-style, trained jointly)
        if config.learned_variance:
            self.variance_head = VarianceHead(
                cond_dim=config.bottleneck_dim,
                hidden_dim=64,
            )
        else:
            self.variance_head = None

        # CFG: learnable null embedding for unconditional generation
        if config.cond_drop_prob > 0:
            self.null_condition = nn.Parameter(torch.zeros(1, config.bottleneck_dim))
        else:
            self.null_condition = None

        # CRPS variance head: separate from denoiser, trained with CRPS on x₀
        if getattr(config, 'crps_variance_head', False):
            n_cells = getattr(config, 'crps_n_cells', 1)
            pos_embed_dim = getattr(config, 'crps_pos_embed_dim', 0)
            self.crps_var_head = CRPSVarianceHead(
                cond_dim=config.bottleneck_dim,
                n_steps=config.n_steps,
                hidden_dim=64,
                n_cells=n_cells,
                pos_embed_dim=pos_embed_dim,
            )
        else:
            self.crps_var_head = None

        # Per-cell heteroscedastic forward noise: static (5,5) noise scale
        # from training data. Initialized to ones; overwritten by
        # set_cell_noise_scale() before training starts.
        if getattr(config, 'cell_heteroscedastic', False):
            self.register_buffer(
                'cell_noise_scale',
                torch.ones(config.surface_h, config.surface_w),
            )

    def _augment_condition(
        self, condition: torch.Tensor, regime_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Augment condition with regime embedding if enabled.

        Projects concat(condition, regime_embed) back to bottleneck_dim
        so denoiser interface stays unchanged.
        """
        if self.regime_proj is not None and regime_ids is not None:
            r_emb = self.regime_embed(regime_ids)  # (B, regime_embed_dim)
            return self.regime_proj(torch.cat([condition, r_emb], dim=-1))
        return condition

    def _compute_baseline(self, surfaces: torch.Tensor) -> torch.Tensor:
        """Compute baseline surface for ratio target.

        Args:
            surfaces: (B, T, 5, 5) normalized history surfaces in [-1, 1]

        Returns:
            baseline: (B, 1, 5, 5) denormalized baseline in [0, 1]
        """
        if getattr(self.config, 'use_median_baseline', False):
            # Median over full history — robust, stable anchor
            baseline = denormalize_iv(surfaces).median(dim=1).values  # (B, 5, 5)
        else:
            K = min(self.config.baseline_window, surfaces.shape[1])
            baseline = denormalize_iv(surfaces[:, -K:]).mean(dim=1)  # (B, 5, 5)
        return baseline.clamp(min=0.01).unsqueeze(1)  # (B, 1, 5, 5)

    def _compute_regime_features(self, history: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary regime features from history for conditioning.

        Features:
          - vol_of_vol: std of daily mean-IV changes (regime indicator)
          - mean_iv_level: mean IV level of last 5 days (level indicator)

        Args:
            history: (B, T, 5, 5) normalized history surfaces in [-1, 1]

        Returns:
            (B, 2) regime feature vector
        """
        B = history.shape[0]
        # Mean IV per timestep: (B, T)
        mean_iv = history.mean(dim=(-2, -1))
        # Daily changes: (B, T-1)
        daily_diff = mean_iv[:, 1:] - mean_iv[:, :-1]
        # Vol of vol: std of daily changes: (B,)
        vol_of_vol = daily_diff.std(dim=1)
        # Mean IV level of last 5 days: (B,)
        mean_level = mean_iv[:, -5:].mean(dim=1)
        # Stack: (B, 2)
        return torch.stack([vol_of_vol, mean_level], dim=1)

    def _add_regime_features(
        self, condition: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        """Add projected regime features to condition vector if enabled."""
        if self.regime_feature_proj is not None:
            feats = self._compute_regime_features(history)
            regime_emb = self.regime_feature_proj(feats)  # (B, bottleneck_dim)
            return condition + regime_emb
        return condition

    def _mcvd_task_probs(self):
        """Return explicit task probs tuple, or None for legacy p_mask mode."""
        c = self.config
        s = c.mcvd_p_forward + c.mcvd_p_backward + c.mcvd_p_interpolation + c.mcvd_p_unconditional
        if s == 0.0:
            return None  # legacy: use p_mask Bernoulli
        return (c.mcvd_p_forward, c.mcvd_p_backward, c.mcvd_p_interpolation, c.mcvd_p_unconditional)

    def _ensure_scheduler_device(self, device: torch.device) -> None:
        """Recreate scheduler on the correct device if needed."""
        if str(self.scheduler.device) != str(device):
            self.scheduler = DDPMScheduler(
                n_steps=self.config.n_steps,
                schedule=self.config.schedule,
                device=device,
            )
        if self.fm_scheduler is not None and str(self.fm_scheduler.device) != str(device):
            self.fm_scheduler = FlowMatchingScheduler(
                n_steps=getattr(self.config, 'fm_n_inference_steps', self.config.n_steps),
                device=device,
            )

    def _compute_learned_variance(
        self, v_pred: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute learned variance from model output v (Nichol & Dhariwal 2021).

        Variance = exp(v * log(beta_tilde_t) + (1-v) * log(beta_t))
        where v is clamped to [0, 1] for interpolation between posterior
        variance (beta_tilde, the lower bound) and forward variance (beta, the upper bound).

        Args:
            v_pred: (B, T, H, W) raw model output (unconstrained)
            t: (B, T) per-frame diffusion timesteps

        Returns:
            log_variance: (B, T, H, W) log of learned variance
        """
        B, T, H, W = v_pred.shape
        t_flat = t.flatten()  # (B*T,)

        # Get schedule variances
        log_beta = torch.log(self.scheduler.betas.clamp(min=1e-20))
        log_beta_tilde = self.scheduler.posterior_log_variance

        log_beta_t = log_beta[t_flat].view(B, T, 1, 1)  # upper bound
        log_beta_tilde_t = log_beta_tilde[t_flat].view(B, T, 1, 1)  # lower bound

        # Clamp v to [0, 1] for interpolation
        v = torch.sigmoid(v_pred)  # smooth clamping to [0, 1]

        # Log-interpolation: v=0 → beta_tilde (narrow), v=1 → beta (wide)
        log_variance = v * log_beta_t + (1 - v) * log_beta_tilde_t

        return log_variance

    def _compute_vlb_loss(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
        v_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VLB (variational lower bound) loss for learned variance.

        KL(q(x_{t-1}|x_t, x_0) || p_theta(x_{t-1}|x_t))

        Both are Gaussians, so KL has closed form:
        KL = 0.5 * [log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2) / sigma2^2 - 1]

        The mean prediction (noise_pred) is DETACHED — the VLB loss only
        trains the variance head, not the noise prediction.

        Args:
            x_0: (B, T, H, W) clean target
            x_t: (B, T, H, W) noisy input
            t: (B, T) per-frame timesteps
            noise_pred: (B, T, H, W) DETACHED noise prediction
            v_pred: (B, T, H, W) variance fraction (has gradient)

        Returns:
            vlb_loss: scalar
        """
        B, T, H, W = x_0.shape
        t_flat = t.flatten()

        # True posterior (q) parameters
        # Mean: mu_q = (sqrt(alpha_bar_prev) * beta / (1-alpha_bar)) * x_0
        #            + (sqrt(alpha) * (1-alpha_bar_prev) / (1-alpha_bar)) * x_t
        alpha_bar_t = self.scheduler.alpha_bar[t_flat].view(B, T, 1, 1)
        alpha_bar_prev_t = self.scheduler.alpha_bar_prev[t_flat].view(B, T, 1, 1)
        alpha_t = self.scheduler.alphas[t_flat].view(B, T, 1, 1)
        beta_t = self.scheduler.betas[t_flat].view(B, T, 1, 1)

        # True posterior mean
        coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
        mu_q = coef_x0 * x_0 + coef_xt * x_t

        # True posterior log-variance
        log_var_q = self.scheduler.posterior_log_variance[t_flat].view(B, T, 1, 1)

        # Model posterior mean (from detached noise_pred)
        sqrt_recip = self.scheduler.sqrt_recip_alpha_bar[t_flat].view(B, T, 1, 1)
        sqrt_recip_m1 = self.scheduler.sqrt_recip_alpha_bar_minus_one[t_flat].view(B, T, 1, 1)
        mu_theta = sqrt_recip * x_t - sqrt_recip_m1 * noise_pred

        # Model posterior log-variance (from v_pred)
        log_var_theta = self._compute_learned_variance(v_pred, t)

        # KL divergence between two Gaussians
        # KL = 0.5 * [log(var_theta/var_q) + var_q/var_theta + (mu_q - mu_theta)^2/var_theta - 1]
        kl = 0.5 * (
            log_var_theta - log_var_q
            + torch.exp(log_var_q - log_var_theta)
            + (mu_q - mu_theta).pow(2) * torch.exp(-log_var_theta)
            - 1.0
        )

        # Average over all dimensions
        return kl.mean()

    def _compute_noise_scale(self, context: torch.Tensor) -> torch.Tensor:
        """Compute per-sample noise scale from conditioning context IV level.

        For heteroscedastic forward noise: high-IV conditions get more noise,
        low-IV conditions get less. Scale = (mean_IV / global_mean_IV)^power.

        power=0.5: variance ∝ IV level (conservative)
        power=1.0: std ∝ IV level (matches multiplicative GT heteroskedasticity)

        Args:
            context: (B, T, 5, 5) in [-1, 1] normalized IV space

        Returns:
            noise_scale: (B, 1, 1, 1) positive scalar per sample
        """
        last_frame = context[:, -1]  # (B, 5, 5) in [-1, 1]
        iv_denorm = denormalize_iv(last_frame)  # (B, 5, 5) in [0, 1]
        iv_level = iv_denorm.mean(dim=(1, 2))  # (B,)
        iv_level = iv_level.clamp(min=0.01)  # avoid zero/negative
        ratio = iv_level / self.config.global_mean_iv
        noise_scale = ratio.pow(self.config.heteroscedastic_power)
        return noise_scale[:, None, None, None]  # (B, 1, 1, 1)

    def compute_cell_noise_scale(self, train_loader, device='cpu'):
        """Compute per-cell noise scale from training data statistics.

        Iterates training data, computes vol_scaled normalized targets,
        measures per-cell std, and stores (cell_std / median)^power as
        the cell_noise_scale buffer.

        Must be called before training when cell_heteroscedastic=True.
        """
        assert hasattr(self, 'cell_noise_scale'), \
            "cell_heteroscedastic must be True to compute cell_noise_scale"

        eps_iv = 1e-4
        H, W = self.config.surface_h, self.config.surface_w
        all_targets = []

        with torch.no_grad():
            for batch in train_loader:
                history = batch['history'].to(device)
                future = batch['future'].to(device)

                # Compute vol_scaled targets (same as forward())
                baseline = self._compute_baseline(history)

                past_abs = denormalize_iv(history)
                mean_iv = past_abs.mean(dim=(-1, -2))
                daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
                vol = daily_chg.std(dim=1, keepdim=True)
                vol_scale = (vol / self.config.global_mean_vol).clamp(
                    self.config.vol_scale_min, self.config.vol_scale_max
                )
                vol_scale = vol_scale.pow(self.config.vol_scale_power)
                vol_scale = vol_scale.unsqueeze(-1).unsqueeze(-1)

                # Denormalize future for all blocks
                bs = self.config.block_size
                n_blocks = self.config.future_len // bs
                for bi in range(n_blocks):
                    block = future[:, bi*bs:(bi+1)*bs]
                    target_abs = denormalize_iv(block).clamp(min=eps_iv, max=1-eps_iv)
                    log_ratio = torch.log(target_abs / baseline)
                    target_norm = log_ratio / vol_scale
                    all_targets.append(target_norm.cpu().numpy())

        import numpy as np
        all_targets = np.concatenate(all_targets, axis=0)  # (N*n_blocks, bs, H, W)
        cell_std = all_targets.std(axis=(0, 1))  # (H, W)
        median_std = np.median(cell_std)
        power = self.config.cell_noise_power
        cell_scale = (cell_std / median_std) ** power
        cell_scale = np.clip(
            cell_scale, self.config.cell_noise_clamp_min, self.config.cell_noise_clamp_max
        )

        self.cell_noise_scale.copy_(torch.tensor(cell_scale, dtype=torch.float32))

        print(f"Cell noise scale (power={power}, "
              f"clamp=[{self.config.cell_noise_clamp_min}, {self.config.cell_noise_clamp_max}]):")
        for r in range(H):
            print(f"  {' '.join(f'{cell_scale[r,c]:.3f}' for c in range(W))}")
        print(f"  Range: [{cell_scale.min():.3f}, {cell_scale.max():.3f}], "
              f"Median: {np.median(cell_scale):.3f}")

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Training forward pass with MCVD multi-task block-AR training.

        Args:
            history: (B, history_len, 5, 5) in [-1, 1]
            future: (B, future_len, 5, 5) in [-1, 1]
            regime_ids: (B,) regime labels for hierarchical conditioning

        Returns:
            dict with 'loss' scalar, optionally 'regime_loss', 'regime_acc'
        """
        B = history.shape[0]
        device = history.device
        bs = self.config.block_size
        n_blocks = self.config.future_len // bs

        self._ensure_scheduler_device(device)

        total_loss = 0.0

        for block_idx in range(n_blocks):
            start = block_idx * bs
            end = start + bs

            # Target block (absolute normalized [-1, 1])
            target_block = future[:, start:end]  # (B, bs, 5, 5)

            # Build past context: history + previous GT blocks (teacher forcing)
            if block_idx > 0:
                past_ctx = torch.cat(
                    [history, future[:, :start]], dim=1
                )
            else:
                past_ctx = history

            # Build future context: GT blocks after current
            if end < self.config.future_len:
                future_ctx = future[:, end:]
            else:
                future_ctx = None

            # Sample MCVD masks
            if self.config.forward_only:
                # FORWARD task only: past visible, future always masked
                mask_past = torch.zeros(B, dtype=torch.bool, device=device)
                mask_future = torch.ones(B, dtype=torch.bool, device=device)
            else:
                mask_past, mask_future = sample_mcvd_masks(
                    B, self.config.p_mask, device=device,
                    task_probs=self._mcvd_task_probs(),
                )
                # Last block: force mask_future=True (no future context available)
                if future_ctx is None:
                    mask_future = torch.ones(B, dtype=torch.bool, device=device)

            # Encode past and future separately
            past_cond = self.encoder(past_ctx, mask=mask_past)  # (B, bottleneck_dim)

            if future_ctx is not None and future_ctx.shape[1] > 0:
                future_cond = self.encoder(future_ctx, mask=mask_future)
            else:
                # No future context (last block) — use learned null embedding
                # to match what encoder returns when mask=True
                future_cond = self.encoder.null_embedding.expand(B, -1)

            # Additive conditioning
            condition = past_cond + future_cond  # (B, bottleneck_dim)

            # Regime conditioning: augment with regime embedding
            condition = self._augment_condition(condition, regime_ids)

            # Auxiliary regime features (vol_of_vol, IV level)
            condition = self._add_regime_features(condition, history)

            # CFG: randomly replace condition with null embedding during training
            if self.training and self.null_condition is not None and self.config.cond_drop_prob > 0:
                drop_mask = (torch.rand(B, device=device) < self.config.cond_drop_prob)
                condition = torch.where(
                    drop_mask.unsqueeze(-1),
                    self.null_condition.expand(B, -1),
                    condition,
                )

            # Convert target to ratio space if ratio_target is enabled.
            # baseline = anchor surface in denormalized [0,1] space.
            drift_loss = None  # may be set inside vol_scaled branch
            if self.config.ratio_target:
                eps_iv = 1e-4
                baseline = self._compute_baseline(past_ctx)  # (B, 1, 5, 5)
                target_abs = denormalize_iv(target_block)  # (B, bs, 5, 5) in [0, 1]
                target_abs = target_abs.clamp(min=eps_iv, max=1.0 - eps_iv)
                if self.config.ratio_target_mode == "logit":
                    # logit(future) - logit(baseline): bounded inversion via sigmoid
                    baseline_c = baseline.clamp(min=eps_iv, max=1.0 - eps_iv)
                    target_block = (torch.logit(target_abs) - torch.logit(baseline_c))
                    target_block = target_block.clamp(-1.0, 1.0)
                elif self.config.ratio_target_mode == "vol_scaled":
                    # log(future/baseline) / vol_scale: doubly-normalized for
                    # condition-dependent uncertainty on BOTH baseline_iv AND vol_of_vol.
                    # vol_scale = std(daily changes) / global_mean, clipped to [0.5, 2.0].
                    past_abs = denormalize_iv(past_ctx)  # (B, T_past, 5, 5)
                    mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_past)
                    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]  # (B, T_past-1)
                    vol = daily_chg.std(dim=1, keepdim=True)  # (B, 1)
                    vol_scale = (vol / self.config.global_mean_vol).clamp(self.config.vol_scale_min, self.config.vol_scale_max)  # (B, 1)
                    vol_scale = vol_scale.pow(self.config.vol_scale_power)  # dampening
                    vol_scale = vol_scale.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
                    # Per-cell structural normalization (if enabled)
                    if getattr(self, 'cell_norm_factor', None) is not None:
                        vol_scale = vol_scale * self.cell_norm_factor  # (B, 1, 1, 1) * (5, 5) → (B, 1, 5, 5)
                    # Learned per-cell correction (if enabled)
                    if self.log_cell_scale is not None:
                        clamp = getattr(self.config, 'cell_scale_clamp', 0.2)
                        correction = torch.exp(self.log_cell_scale.clamp(-clamp, clamp))  # (5, 5)
                        vol_scale = vol_scale * correction  # broadcast: (B, 1, 1, 1) or (B, 1, 5, 5) * (5, 5)
                    # Low-rank per-cell correction (if enabled)
                    if self.log_u_cell is not None:
                        lr_clamp = getattr(self.config, 'low_rank_cell_scale_clamp', 0.5)
                        log_u = self.log_u_cell.clamp(-lr_clamp, lr_clamp)  # (5,)
                        log_v = self.log_v_cell.clamp(-lr_clamp, lr_clamp)  # (5,)
                        lr_correction = torch.exp(log_u.unsqueeze(1) + log_v.unsqueeze(0))  # (5, 5)
                        vol_scale = vol_scale * lr_correction
                    # Fixed per-cell correction from calibration head (frozen buffer)
                    if self.fixed_cell_scale is not None:
                        vol_scale = vol_scale * self.fixed_cell_scale  # (B, 1, 1, 1) * (5, 5)
                    # Learned drift: shift baseline in IV-space before computing log-ratio
                    effective_baseline = baseline  # (B, 1, 5, 5)
                    if self.drift_head is not None:
                        drift_raw = self.drift_head(condition.detach())  # (B, 25)
                        drift_raw = drift_raw.reshape(B, self.config.surface_h, self.config.surface_w)  # (B, 5, 5)
                        drift_max = getattr(self.config, 'drift_max', 0.05)
                        drift = torch.tanh(drift_raw) * drift_max  # (B, 5, 5) in [-max, +max]
                        # Supervision target: mean direction of future IV movement
                        drift_target = (target_abs.mean(dim=1) - baseline.squeeze(1)).detach()  # (B, 5, 5)
                        drift_loss = F.mse_loss(drift, drift_target)
                        effective_baseline = (baseline + drift.unsqueeze(1)).clamp(min=0.01)  # (B, 1, 5, 5)
                    log_ratio = torch.log(target_abs / effective_baseline)
                    target_block = (log_ratio / vol_scale).clamp(-1.0, 1.0)
                elif self.config.ratio_target_mode == "additive_scaled":
                    # Additive formulation: (future - baseline) / (vol_scale * baseline)
                    # Symmetric denormalization: baseline + z * vol_scale * baseline
                    # Eliminates exp() asymmetry (11.8x at vs=1.5 → 1.0x).
                    past_abs = denormalize_iv(past_ctx)  # (B, T_past, 5, 5)
                    mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_past)
                    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]  # (B, T_past-1)
                    vol = daily_chg.std(dim=1, keepdim=True)  # (B, 1)
                    vol_scale = (vol / self.config.global_mean_vol).clamp(self.config.vol_scale_min, self.config.vol_scale_max)  # (B, 1)
                    vol_scale = vol_scale.pow(self.config.vol_scale_power)
                    vol_scale = vol_scale.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
                    if getattr(self, 'cell_norm_factor', None) is not None:
                        vol_scale = vol_scale * self.cell_norm_factor
                    # Additive target: (future - baseline) / (vol_scale * baseline)
                    diff = target_abs - baseline  # (B, T_block, 5, 5)
                    scale = vol_scale * baseline  # (B, 1, 5, 5) * (B, 1, 5, 5) → broadcast
                    target_block = (diff / scale.clamp(min=1e-6)).clamp(-3.0, 3.0)
                elif self.config.ratio_target_mode == "additive_whitened":
                    # True additive whitening with data-derived per-cell σ:
                    # σ[r,c] = max(cell_std[r,c], vol_scale × σ_base)
                    # - cell_std: per-cell daily-change std from history (per-cell uncertainty)
                    # - σ_floor: regime-dependent minimum (per-regime uncertainty)
                    # - Per-horizon: implicit (deviation grows with h, denoiser learns)
                    # Denorm: IV = baseline + z × σ  (additive, symmetric, no exp bias)
                    past_abs = denormalize_iv(past_ctx)  # (B, T_past, 5, 5)
                    # Per-cell daily-change std from history
                    daily_chg_cell = past_abs[:, 1:] - past_abs[:, :-1]  # (B, T-1, 5, 5)
                    cell_std = daily_chg_cell.std(dim=1)  # (B, 5, 5)
                    # Regime-dependent floor from vol_of_vol
                    mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_past)
                    mean_chg = mean_iv[:, 1:] - mean_iv[:, :-1]  # (B, T-1)
                    vol = mean_chg.std(dim=1, keepdim=True)  # (B, 1)
                    vol_scale = (vol / self.config.global_mean_vol).clamp(
                        self.config.vol_scale_min, self.config.vol_scale_max)
                    vol_scale = vol_scale.pow(self.config.vol_scale_power)
                    sigma_base = self.config.global_mean_vol * (self.config.block_size ** 0.5)
                    sigma_floor = (vol_scale * sigma_base).unsqueeze(-1)  # (B, 1, 1)
                    # Per-cell σ: max of data-derived cell std and regime floor
                    sigma = torch.maximum(
                        cell_std,  # (B, 5, 5)
                        sigma_floor.expand_as(cell_std)  # (B, 5, 5)
                    ).unsqueeze(1)  # (B, 1, 5, 5)
                    diff = target_abs - baseline  # (B, T_block, 5, 5)
                    target_block = (diff / sigma.clamp(min=1e-6)).clamp(-3.0, 3.0)
                elif self.config.ratio_target_mode == "vol_scaled_learned":
                    # Hybrid: vol_scale * exp(learned_correction)
                    # vol_scale is the proven hand-coded scaling; correction is learned
                    past_abs = denormalize_iv(past_ctx)  # (B, T_past, 5, 5)
                    mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_past)
                    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]  # (B, T_past-1)
                    vol = daily_chg.std(dim=1, keepdim=True)  # (B, 1)
                    vol_scale = (vol / self.config.global_mean_vol).clamp(self.config.vol_scale_min, self.config.vol_scale_max)  # (B, 1)
                    vol_scale = vol_scale.pow(self.config.vol_scale_power)

                    # Learned correction: small multiplicative factor
                    log_corr = self.log_correction_head(condition)  # (B, 1)
                    log_corr_clamped = log_corr.clamp(-0.5, 0.5)  # limit correction to [0.6, 1.65]x
                    sigma = vol_scale * torch.exp(log_corr_clamped)  # (B, 1)
                    sigma_4d = sigma.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

                    log_ratio = torch.log(target_abs / baseline)
                    # NLL auxiliary loss for sigma training
                    z_nll = log_ratio / sigma_4d
                    nsdiff_nll = (torch.log(sigma_4d) + 0.5 * z_nll ** 2).mean()
                    # L2 regularization on correction to keep near identity
                    nsdiff_nll = nsdiff_nll + self.config.e2e_sigma_reg * (log_corr ** 2).mean()

                    # Detach sigma for diffusion target (correction trained only by NLL)
                    sigma_det = sigma_4d.detach()
                    target_block = (log_ratio / sigma_det).clamp(-1.0, 1.0)
                elif self.config.ratio_target_mode == "vol_scaled_percell":
                    # Per-cell vol normalization: each cell scaled by its own vol_of_vol
                    # vol_scale[r,c] = std(cell[r,c] daily changes) / global_mean_cell_vol[r,c]
                    past_abs = denormalize_iv(past_ctx)  # (B, T_past, 5, 5)
                    daily_chg = past_abs[:, 1:] - past_abs[:, :-1]  # (B, T-1, 5, 5)
                    cell_vol = daily_chg.std(dim=1)  # (B, 5, 5)
                    vol_scale = (cell_vol / self.global_mean_cell_vol).clamp(self.config.vol_scale_min, self.config.vol_scale_max)  # (B, 5, 5)
                    vol_scale = vol_scale.unsqueeze(1)  # (B, 1, 5, 5) broadcast over time
                    log_ratio = torch.log(target_abs / baseline)
                    target_block = (log_ratio / vol_scale).clamp(-1.0, 1.0)
                elif self.config.ratio_target_mode == "learned_percell":
                    # Magnitude × Shape decomposition:
                    # sigma[r,c] = vol_scale (scalar, proven) * pattern[r,c] (NN-learned)
                    # pattern is normalized to mean=1.0 → preserves total uncertainty budget
                    # Magnitude from scalar vol_of_vol (preserves kurtosis)
                    past_abs = denormalize_iv(past_ctx)  # (B, T_past, 5, 5)
                    mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_past)
                    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]  # (B, T_past-1)
                    vol = daily_chg.std(dim=1, keepdim=True)  # (B, 1)
                    vol_scale = (vol / self.config.global_mean_vol).clamp(self.config.vol_scale_min, self.config.vol_scale_max)  # (B, 1)
                    vol_scale = vol_scale.pow(self.config.vol_scale_power)

                    # Shape: NN-learned per-cell relative pattern
                    log_pattern = self.percell_sigma_head(condition)  # (B, 25)
                    log_pattern = log_pattern.reshape(-1, 5, 5)  # (B, 5, 5)
                    pattern = torch.exp(log_pattern)  # (B, 5, 5) all positive
                    # Normalize to mean=1.0 per sample → preserves total budget
                    pattern = pattern / pattern.mean(dim=(-1, -2), keepdim=True)

                    # Combined sigma: scalar magnitude × spatial pattern
                    sigma_4d = vol_scale.unsqueeze(-1).unsqueeze(-1) * pattern.unsqueeze(1)
                    # (B, 1, 5, 5)

                    log_ratio = torch.log(target_abs / baseline)  # (B, bs, 5, 5)

                    # Per-cell NLL for training the pattern head
                    z_nll = log_ratio / sigma_4d
                    log_sigma_4d = torch.log(sigma_4d)
                    nsdiff_nll = (log_sigma_4d + 0.5 * z_nll ** 2).mean()

                    # Standardize target (DETACH sigma: head trained by NLL only)
                    sigma_det = sigma_4d.detach()
                    target_block = (log_ratio / sigma_det).clamp(-3.0, 3.0)
                elif self.config.ratio_target_mode == "nsdiff":
                    # NSDiff-inspired learned standardization:
                    # sigma_c = exp(log_std_head(condition)) is learned from data
                    # target = log(future/baseline) / sigma_c (standardized residual)
                    # sigma_c trained with Gaussian NLL: 0.5*log(sigma^2) + 0.5*z^2
                    log_ratio = torch.log(target_abs / baseline)  # (B, bs, 5, 5)
                    log_sigma = self.log_std_head(condition)  # (B, 1)
                    log_sigma_clamped = log_sigma.clamp(-2, 2)
                    sigma_c = torch.exp(log_sigma_clamped)  # (B, 1)
                    sigma_4d = sigma_c.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

                    # NLL auxiliary loss for sigma training
                    z_nll = log_ratio / sigma_4d  # (B, bs, 5, 5)
                    nsdiff_nll = (log_sigma_clamped.unsqueeze(-1).unsqueeze(-1)
                                  + 0.5 * z_nll ** 2).mean()

                    # Standardize target (detach sigma so diffusion loss only trains denoiser)
                    sigma_det = sigma_4d.detach()
                    target_block = (log_ratio / sigma_det).clamp(-3.0, 3.0)
                elif self.config.ratio_target_mode == "e2e_nll":
                    # End-to-end learned sigma: like nsdiff but sigma is NOT detached.
                    # Diffusion loss gradient flows through sigma → condition-dependent scaling.
                    # NLL provides baseline signal, L2 reg prevents collapse.
                    log_ratio = torch.log(target_abs / baseline)  # (B, bs, 5, 5)
                    log_sigma = self.log_std_head(condition)  # (B, 1)
                    log_sigma_clamped = log_sigma.clamp(-2, 2)
                    sigma_c = torch.exp(log_sigma_clamped)  # (B, 1)
                    sigma_4d = sigma_c.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

                    # NLL auxiliary loss
                    z_nll = log_ratio / sigma_4d  # (B, bs, 5, 5)
                    nsdiff_nll = (log_sigma_clamped.unsqueeze(-1).unsqueeze(-1)
                                  + 0.5 * z_nll ** 2).mean()
                    # L2 regularization: keep sigma near 1.0
                    nsdiff_nll = nsdiff_nll + self.config.e2e_sigma_reg * (log_sigma ** 2).mean()

                    # NON-DETACHED: gradient flows from diffusion loss through sigma
                    target_block = (log_ratio / sigma_4d).clamp(-3.0, 3.0)
                elif self.config.ratio_target_mode == "percell_revin":
                    # Per-Cell RevIN: foundation model consensus approach.
                    # Each cell normalized by its own std from history window.
                    # target = (future - baseline) / cell_std
                    # cell_std has learned per-cell floor (replaces vol_scale_min).
                    past_abs = denormalize_iv(past_ctx)  # (B, T_past, 5, 5)
                    # Per-cell std of daily changes in history
                    daily_chg = past_abs[:, 1:] - past_abs[:, :-1]  # (B, T-1, 5, 5)
                    cell_std_data = daily_chg.std(dim=1)  # (B, 5, 5)
                    # Apply learned per-cell floor
                    floor = torch.exp(self.log_revin_floor)  # (5, 5)
                    cell_std = torch.maximum(cell_std_data, floor.unsqueeze(0))  # (B, 5, 5)
                    cell_std = cell_std.unsqueeze(1)  # (B, 1, 5, 5)
                    # Additive normalization: (future - baseline) / cell_std
                    diff = target_abs - baseline  # (B, T_block, 5, 5)
                    target_block = (diff / cell_std).clamp(-3.0, 3.0)
                else:
                    # log(future / baseline): original log-ratio mode
                    target_block = torch.log(target_abs / baseline)
                    target_block = target_block.clamp(-1.0, 1.0)

            # Mean head: subtract predicted mean before diffusion, train mean head separately.
            # This gives the model a direct path to predict regime-specific mean shifts
            # without going through 100 steps of iterative denoising.
            mean_loss = None
            if self.mean_head is not None and self.config.ratio_target:
                mu = self.mean_head(condition)  # (B, 25)
                mu = mu.reshape(B, self.config.surface_h, self.config.surface_w)  # (B, 5, 5)
                mu = mu.unsqueeze(1)  # (B, 1, 5, 5) broadcast over time
                # Mean head loss: predict the average z across time dimension
                target_mean = target_block.mean(dim=1, keepdim=True)  # (B, 1, 5, 5)
                mean_loss = F.mse_loss(mu, target_mean.detach())
                # Subtract mean (detached) from target — diffusion learns zero-mean residual
                target_block = target_block - mu.detach()

            # Positions for this block (shared by FM and DDPM paths)
            positions = (
                torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)
                + block_idx * bs
            )  # (B, bs)

            # Regime scalar for percell_head (shared)
            regime_scalar = None
            if getattr(self.config, 'percell_regime_input', False) and self.config.use_percell_head:
                past_abs = denormalize_iv(past_ctx)  # (B, T_past, 5, 5)
                _mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_past)
                _daily_chg = _mean_iv[:, 1:] - _mean_iv[:, :-1]
                regime_scalar = _daily_chg.std(dim=1, keepdim=True)  # (B, 1)

            # Baseline surface for denoiser input channel (shared)
            baseline_surf = None
            if getattr(self.config, 'baseline_channel', False):
                baseline_surf = self._compute_baseline(past_ctx).squeeze(1)  # (B, 5, 5)

            if getattr(self.config, 'use_flow_matching', False):
                # === Flow Matching training path ===
                # Sample continuous t ~ U(0, 1), one per sample (uniform across block)
                t_cont = torch.rand(B, 1, device=device)  # (B, 1) in [0, 1]
                t_cont = t_cont.expand(B, bs)  # (B, bs)

                # Sample noise
                noise = sample_pyoco_noise(
                    target_block.shape, self.config.noise_rho, device
                )

                # Forward interpolation: x_t = (1-t)*x_0 + t*noise
                t_4d = t_cont.unsqueeze(-1).unsqueeze(-1)  # (B, bs, 1, 1)
                x_t = (1 - t_4d) * target_block + t_4d * noise  # (B, bs, 5, 5)

                # Velocity target: v = noise - x_0
                velocity_target = noise - target_block  # (B, bs, 5, 5)

                # Denoiser input: scale t to integer range for embedding compatibility
                # t=0 → 0, t=1 → n_steps-1 (matches sinusoidal embedding frequency range)
                noise_levels = (t_cont * (self.config.n_steps - 1)).float()  # (B, bs)

                x_flat = x_t.reshape(B, bs, -1)  # (B, bs, 25)
                v_pred_flat = self.denoiser(
                    x_flat, condition, positions, noise_levels,
                    regime_scalar=regime_scalar,
                    baseline_surface=baseline_surf,
                )  # (B, bs, 25) — now predicts velocity, not noise
                if isinstance(v_pred_flat, tuple):
                    v_pred_flat = v_pred_flat[0]  # ignore learned sigma channel

                # Loss: MSE(v_pred, velocity_target)
                velocity_flat = velocity_target.reshape(B, bs, -1)  # (B, bs, 25)
                block_loss = F.mse_loss(v_pred_flat, velocity_flat)
            else:
                # === Standard DDPM training path ===
                # Sample noise levels
                if self.config.use_uniform_noise:
                    # One scalar t per sample, replicated across block frames
                    k = torch.randint(0, self.config.n_steps, (B, 1), device=device)
                    k = k.expand(B, bs).contiguous()  # (B, bs)
                else:
                    # Per-frame task-adaptive noise (DF)
                    k = sample_batch_task_adaptive_noise(
                        mask_past, mask_future, bs, self.config.n_steps, self.config.jitter_std
                    )  # (B, bs)

                # Forward diffusion with PYoCo correlated noise
                noise_unscaled = sample_pyoco_noise(
                    target_block.shape, self.config.noise_rho, device
                )

                # Heteroscedastic noise: scale forward noise by condition IV level,
                # but keep unscaled ε as loss target (model learns noise direction,
                # σ handles magnitude). At inference, multiply ε_θ by σ.
                if self.config.heteroscedastic_noise:
                    noise_scale = self._compute_noise_scale(past_ctx)
                    noise_forward = noise_unscaled * noise_scale  # σε for forward process
                else:
                    noise_forward = noise_unscaled

                # Per-cell heteroscedastic noise: static (5,5) scale from training data.
                # Model predicts SCALED noise (what was actually added).
                if getattr(self, 'cell_noise_scale', None) is not None and self.config.cell_heteroscedastic:
                    _cns = self.cell_noise_scale.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    noise_forward = noise_forward * _cns

                noisy_block, _ = self.scheduler.q_sample_per_frame(
                    target_block, k, noise_forward
                )  # (B, bs, 5, 5)

                # Denoise
                noisy_flat = noisy_block.reshape(B, bs, -1)  # (B, bs, 25)

                v_pred = None
                if getattr(self.config, 'learn_sigma', False):
                    noise_pred, v_pred = self.denoiser(
                        noisy_flat, condition, positions, k,
                        regime_scalar=regime_scalar,
                        baseline_surface=baseline_surf,
                    )  # (B, bs, 25) each
                else:
                    noise_pred = self.denoiser(
                        noisy_flat, condition, positions, k,
                        regime_scalar=regime_scalar,
                        baseline_surface=baseline_surf,
                    )  # (B, bs, 25)

                # Loss target: unscaled ε when using per-sample heteroscedastic noise
                # (model predicts ε, σ applied at x_0 recovery).
                # For cell_heteroscedastic: model predicts SCALED noise (noise_forward)
                # because cell_noise_scale is static — denoiser CAN learn it.
                if self.config.heteroscedastic_noise and not self.config.cell_heteroscedastic:
                    noise_flat = noise_unscaled.reshape(B, bs, -1)  # (B, bs, 25)
                else:
                    noise_flat = noise_forward.reshape(B, bs, -1)  # (B, bs, 25)

                if v_pred is not None and self.config.loss_type == "crps":
                    # CRPS loss: train noise prediction with proper scoring rule.
                    # v_pred is raw log-sigma (unbounded), NOT Nichol-Dhariwal interpolation.
                    # CRPS provides direct gradient from conditioning → predicted uncertainty.
                    sigma_pred = torch.exp(v_pred.clamp(-10, 5))  # (B, bs, 25)
                    crps_vals = crps_gaussian(noise_pred, sigma_pred, noise_flat)
                    block_loss = crps_vals.mean()
                elif v_pred is not None:
                    # Nichol & Dhariwal learned variance:
                    # L_simple (noise MSE) + lambda_vlb * L_vlb (KL for variance)
                    block_loss = F.mse_loss(noise_pred, noise_flat)

                    # VLB loss: KL(q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t))
                    # Mean prediction is DETACHED — variance doesn't affect noise learning.
                    vlb = self._compute_vlb_loss(
                        target_block, noisy_block, k,
                        noise_pred.detach().reshape(B, bs, self.config.surface_h, self.config.surface_w),
                        v_pred.reshape(B, bs, self.config.surface_h, self.config.surface_w),
                    )
                    block_loss = block_loss + self.config.lambda_vlb * vlb
                elif self.variance_head is not None:
                    # Heteroscedastic NLL with beta-NLL stabilization.
                    # Variance head predicts per-sample log_σ² from condition only.
                    log_var = self.variance_head(condition)  # (B, 1)

                    # Per-sample squared error: ||ε_pred - ε||²
                    sq_err = (noise_pred - noise_flat).pow(2).mean(dim=(1, 2))  # (B,)

                    # Heteroscedastic NLL: exp(-log_var)/2 * sq_err + log_var/2
                    precision = torch.exp(-log_var.squeeze(1))  # (B,)
                    nll = 0.5 * precision * sq_err + 0.5 * log_var.squeeze(1)  # (B,)

                    # beta-NLL: reweight by detached variance^beta to prevent collapse
                    beta = self.config.variance_beta_nll
                    if beta > 0:
                        var_detached = torch.exp(log_var.squeeze(1)).detach()
                        nll = nll * var_detached.pow(beta)

                    block_loss = nll.mean()
                else:
                    alpha = self.config.interp_loss_weight
                    if alpha < 1.0 and not self.config.forward_only:
                        # Per-sample loss with interpolation down-weighting
                        if self.config.loss_type == "huber":
                            per_sample = F.smooth_l1_loss(
                                noise_pred, noise_flat, beta=self.config.huber_delta,
                                reduction='none',
                            ).mean(dim=(1, 2))  # (B,)
                        else:
                            per_sample = F.mse_loss(
                                noise_pred, noise_flat, reduction='none'
                            ).mean(dim=(1, 2))  # (B,)
                        tasks = get_task_types(mask_past, mask_future)
                        weights = torch.ones(B, device=device)
                        weights[tasks == MCVDTask.INTERPOLATION] = alpha
                        block_loss = (per_sample * weights).mean()
                    else:
                        if self.config.loss_type == "huber":
                            block_loss = F.smooth_l1_loss(
                                noise_pred, noise_flat, beta=self.config.huber_delta
                            )
                        elif getattr(self, 'cell_loss_weights', None) is not None:
                            # Per-cell weighted MSE: modulate loss per spatial position
                            H, W = self.config.surface_h, self.config.surface_w
                            sq_err = (noise_pred - noise_flat).pow(2)  # (B, bs, 25)
                            sq_err = sq_err.reshape(B, bs, H, W)
                            block_loss = (sq_err * self.cell_loss_weights).mean()
                        else:
                            block_loss = F.mse_loss(noise_pred, noise_flat)

            # DDPM-only auxiliary losses (skip for Flow Matching)
            if not getattr(self.config, 'use_flow_matching', False):
                # Regime-weighted loss: turb windows get higher weight
                if getattr(self.config, 'turb_loss_weight', 0.0) > 0:
                    per_sample = F.mse_loss(
                        noise_pred, noise_flat, reduction='none'
                    ).mean(dim=(1, 2))  # (B,)
                    _past_abs_w = denormalize_iv(past_ctx)
                    _miv = _past_abs_w.mean(dim=(-1, -2))  # (B, T)
                    _dchg = _miv[:, 1:] - _miv[:, :-1]
                    _vov = _dchg.std(dim=1)  # (B,)
                    _med = _vov.median()
                    _w = torch.where(
                        _vov > _med,
                        torch.tensor(self.config.turb_loss_weight, device=device),
                        torch.tensor(1.0, device=device),
                    )
                    _w = _w / _w.mean()  # normalize to mean=1 (preserve loss scale)
                    block_loss = (per_sample * _w).mean()

                # NSDiff/learned sigma NLL: auxiliary loss for learned standardization
                if (self.log_std_head is not None or self.log_correction_head is not None or self.percell_sigma_head is not None) and self.config.ratio_target:
                    # nsdiff_nll was computed above when building target_block
                    block_loss = block_loss + self.config.nsdiff_sigma_lambda * nsdiff_nll

                # CRPS variance head: auxiliary loss for condition-dependent σ
                if self.crps_var_head is not None:
                    H, W = self.config.surface_h, self.config.surface_w
                    with torch.no_grad():
                        # Recover x₀ prediction from noise prediction
                        t_flat_crps = k.flatten()
                        sqrt_recip = self.scheduler.sqrt_recip_alpha_bar[t_flat_crps].view(B, bs, 1, 1)
                        sqrt_recip_m1 = self.scheduler.sqrt_recip_alpha_bar_minus_one[t_flat_crps].view(B, bs, 1, 1)
                        noise_pred_4d = noise_pred.reshape(B, bs, H, W)
                        x_0_pred = (sqrt_recip * noisy_block - sqrt_recip_m1 * noise_pred_4d).clamp(-1, 1)

                    # Predict σ from condition (detached), timestep, and frame position
                    # Global frame positions for horizon-dependent σ
                    frame_positions = torch.arange(start, end, device=device).unsqueeze(0).expand(B, -1)  # (B, bs)
                    log_sigma = self.crps_var_head(condition.detach(), k, positions=frame_positions)  # (B, bs, n_cells)
                    sigma = torch.exp(log_sigma.clamp(-10, 5))
                    if self.crps_var_head.n_cells == 1:
                        sigma_4d = sigma.unsqueeze(-1).expand(B, bs, H, W)  # (B, bs, H, W)
                    else:
                        sigma_4d = sigma.view(B, bs, H, W)  # (B, bs, H, W) per-cell

                    crps_loss = crps_gaussian(x_0_pred, sigma_4d, target_block).mean()
                    block_loss = block_loss + getattr(self.config, 'lambda_crps', 0.1) * crps_loss

            # Mean head loss: auxiliary MSE for learned mean prediction
            if mean_loss is not None:
                block_loss = block_loss + self.config.mean_head_lambda * mean_loss

            # Drift head loss: auxiliary MSE for baseline shift prediction
            if drift_loss is not None:
                block_loss = block_loss + getattr(self.config, 'drift_loss_weight', 1.0) * drift_loss

            total_loss = total_loss + block_loss

        result = {"loss": total_loss / n_blocks}

        # Regime classification loss (if labels provided and classifier exists)
        if regime_ids is not None and self.regime_classifier is not None:
            # no_grad on encoder: classifier adapts to encoder features,
            # encoder is trained only by diffusion loss
            with torch.no_grad():
                regime_cond = self.encoder(history, mask=None)
            regime_logits = self.regime_classifier(regime_cond)
            regime_loss = F.cross_entropy(regime_logits, regime_ids)
            result["loss"] = result["loss"] + self.config.regime_loss_weight * regime_loss
            result["regime_loss"] = regime_loss.item()
            result["regime_acc"] = (
                (regime_logits.argmax(dim=1) == regime_ids).float().mean().item()
            )

        # Log drift loss for monitoring
        if drift_loss is not None:
            result["drift_loss"] = drift_loss.item()

        return result

    def _pyramid_timesteps(
        self, T: int, n_steps: int, device: torch.device,
        t_min: Optional[torch.Tensor] = None,
    ) -> list:
        """Build the DF pyramid scheduling matrix.

        Returns a list of (B, T)-broadcastable per-frame timestep tensors,
        one per denoising iteration. Earlier frames are denoised faster.

        The pyramid has (n_steps + T - 1) total iterations:
          - Iteration 0: all frames at t = n_steps - 1  (fully noisy)
          - Each iteration: frame i's timestep decreases by 1 once the
            "denoising wave" reaches it (wave front arrives at frame i
            after i iterations)
          - Final iteration: frame i at t = t_min[i]

        Args:
            T: number of frames
            n_steps: number of diffusion timesteps
            device: torch device
            t_min: (T,) per-frame minimum timestep. Frames stop denoising
                   at their t_min instead of 0. None = all zeros.

        Returns:
            List of (T,) long tensors, one per iteration where at least
            one frame's timestep changes.
        """
        if t_min is None:
            t_min = torch.zeros(T, dtype=torch.long, device=device)

        # Build schedule: iteration k, frame i has t = clamp(n_steps-1-k+i, t_min[i], n_steps-1)
        # Total raw iterations: n_steps + T - 1, but many are duplicates
        # when frames are clamped. We deduplicate to avoid wasted compute.
        n_iters = n_steps + T - 1
        frame_idx = torch.arange(T, device=device)  # (T,)
        schedule = []
        prev = None
        for k in range(n_iters):
            t_frame = torch.max((n_steps - 1 - k + frame_idx), t_min).clamp(max=n_steps - 1)  # (T,)
            if prev is None or not torch.equal(t_frame, prev):
                schedule.append(t_frame)
                prev = t_frame
        return schedule

    @torch.no_grad()
    def _sample_block_pyramid(
        self,
        condition: torch.Tensor,
        positions: torch.Tensor,
        shape: tuple,
        t_min: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
        scale_x0_recovery: bool = True,
        regime_scalar: Optional[torch.Tensor] = None,
        baseline_surface: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample one block using the DF pyramid schedule.

        All frames are denoised jointly across iterations. Earlier frames
        clean up faster, so the BiGRU hidden state propagates clean info
        from early frames to noisy later frames — exactly the DF paper's
        "soft causality" mechanism.

        Args:
            condition: (B, bottleneck_dim) encoder output
            positions: (B, T) absolute frame positions
            shape: (B, T, H, W)
            t_min: (T,) per-frame minimum timestep. Frames stop denoising
                   at their t_min, retaining residual noise at that level.
                   None = all zeros (fully denoise, backward compat).
            noise_scale: (B, 1, 1, 1) per-sample noise scaling for
                   heteroscedastic diffusion. Scales initial noise and
                   posterior noise. None = standard isotropic noise.
            scale_x0_recovery: if True, multiply noise_pred by noise_scale
                   in x_0 recovery (for heteroscedastic forward noise).
                   If False, x_0 recovery is standard (for learned variance).

        Returns:
            block: (B, T, H, W) in [-1, 1]
        """
        B, T, H, W = shape
        device = condition.device
        n_steps = self.config.n_steps

        if t_min is None:
            t_min = torch.zeros(T, dtype=torch.long, device=device)

        # (T,) -> (B, T) for broadcasting
        t_min_expanded = t_min.unsqueeze(0).expand(B, -1)

        # Build pyramid schedule (frames clamp at their t_min instead of 0)
        schedule = self._pyramid_timesteps(T, n_steps, device, t_min=t_min)

        # Start from pure noise (scaled for heteroscedastic diffusion)
        x_t = torch.randn(shape, device=device)
        if noise_scale is not None:
            x_t = x_t * noise_scale

        # Run pyramid denoising
        for iter_idx in range(len(schedule) - 1):
            t_current = schedule[iter_idx].unsqueeze(0).expand(B, -1)    # (B, T)
            t_next = schedule[iter_idx + 1].unsqueeze(0).expand(B, -1)   # (B, T)

            # Skip if no frame needs updating (all at their t_min)
            active = (t_current > t_min_expanded)  # (B, T)
            if not active.any():
                break

            # Flatten spatial dims for denoiser: (B, T, H, W) -> (B, T, H*W)
            x_flat = x_t.reshape(B, T, H * W)

            # Predict noise (and optionally variance fraction)
            learned_log_var = None
            if getattr(self.config, 'learn_sigma', False):
                noise_pred_flat, v_pred_flat = self.denoiser(
                    x_flat, condition, positions, t_current,
                    regime_scalar=regime_scalar,
                    baseline_surface=baseline_surface,
                )
                noise_pred = noise_pred_flat.reshape(B, T, H, W)
                v_pred = v_pred_flat.reshape(B, T, H, W)
                learned_log_var = self._compute_learned_variance(v_pred, t_current)
            else:
                noise_pred_flat = self.denoiser(
                    x_flat, condition, positions, t_current,
                    regime_scalar=regime_scalar,
                    baseline_surface=baseline_surface,
                )
                noise_pred = noise_pred_flat.reshape(B, T, H, W)

            # DDPM posterior step: for each frame, go from t_current to t_next
            # Predict x_0 from noise prediction
            t_flat = t_current.flatten()
            sqrt_recip = self.scheduler.sqrt_recip_alpha_bar[t_flat].view(B, T, 1, 1)
            sqrt_recip_m1 = self.scheduler.sqrt_recip_alpha_bar_minus_one[t_flat].view(B, T, 1, 1)
            if noise_scale is not None and scale_x0_recovery:
                # Heteroscedastic forward noise: model predicts ε, forward used σε
                x_0_pred = sqrt_recip * x_t - sqrt_recip_m1 * noise_scale * noise_pred
            else:
                # Standard or learned-variance: model predicts ε, forward used ε
                x_0_pred = sqrt_recip * x_t - sqrt_recip_m1 * noise_pred
            # NSDiff standardized residuals can be in [-3, 3]; standard targets in [-1, 1]
            x0_clamp = 3.0 if (self.config.ratio_target and
                               self.config.ratio_target_mode in ("nsdiff", "e2e_nll")) else 1.0
            x_0_pred = x_0_pred.clamp(-x0_clamp, x0_clamp)

            # Compute posterior mean: mu = coef_x0 * x_0_pred + coef_xt * x_t
            alpha_t = self.scheduler.alphas[t_flat].view(B, T, 1, 1)
            alpha_bar_t = self.scheduler.alpha_bar[t_flat].view(B, T, 1, 1)
            alpha_bar_prev_t = self.scheduler.alpha_bar_prev[t_flat].view(B, T, 1, 1)
            beta_t = self.scheduler.betas[t_flat].view(B, T, 1, 1)

            coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
            coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
            mean = coef_x0 * x_0_pred + coef_xt * x_t

            # Add noise (except when t_current is at absolute 0 where posterior_var=0)
            z = torch.randn_like(x_t)
            nonzero = (t_current > 0).float().unsqueeze(-1).unsqueeze(-1)
            if learned_log_var is not None:
                x_new = mean + nonzero * torch.exp(0.5 * learned_log_var) * z
            elif noise_scale is not None:
                posterior_var = self.scheduler.posterior_variance[t_flat].view(B, T, 1, 1)
                x_new = mean + nonzero * noise_scale * torch.sqrt(posterior_var) * z
            else:
                posterior_var = self.scheduler.posterior_variance[t_flat].view(B, T, 1, 1)
                x_new = mean + nonzero * torch.sqrt(posterior_var) * z

            # Only update active frames (those above their t_min)
            active_mask = active.float().unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
            x_t = active_mask * x_new + (1 - active_mask) * x_t

        return x_t

    @torch.no_grad()
    def _sample_block_uniform(
        self,
        condition: torch.Tensor,
        positions: torch.Tensor,
        shape: tuple,
        t_min: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
        scale_x0_recovery: bool = True,
        uncond_condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        regime_scalar: Optional[torch.Tensor] = None,
        baseline_surface: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample one block using uniform DDPM reverse (all frames at same t).

        Standard DDPM reverse diffusion where all frames share the same
        global timestep at each step. Supports per-frame t_min for growing
        uncertainty (frames stop denoising at their t_min).

        When guidance_scale > 1.0 and uncond_condition is provided, uses
        classifier-free guidance: ε = ε_uncond + w * (ε_cond - ε_uncond).

        Per-cell noise scaling: if self._cell_noise_scale is set (H, W tensor),
        applies per-cell scaling to both initial and posterior noise injection.
        This modulates CI width per spatial location without affecting x_0
        recovery (preserves denoiser's mean prediction).

        Args:
            condition: (B, bottleneck_dim) encoder output
            positions: (B, T) absolute frame positions
            shape: (B, T, H, W)
            t_min: (T,) per-frame minimum timestep, or None (all zeros)
            noise_scale: (B, 1, 1, 1) per-sample noise scaling for
                   heteroscedastic diffusion. None = standard isotropic noise.
            scale_x0_recovery: if True, multiply noise_pred by noise_scale
                   in x_0 recovery (for heteroscedastic forward noise).
                   If False, x_0 recovery is standard (for learned variance).
            uncond_condition: (B, bottleneck_dim) null condition for CFG.
            guidance_scale: CFG guidance weight. 1.0 = no guidance.

        Returns:
            block: (B, T, H, W) in [-1, 1]
        """
        B, T, H, W = shape
        device = condition.device
        n_steps = self.config.n_steps

        if t_min is None:
            t_min = torch.zeros(T, dtype=torch.long, device=device)

        # (T,) -> (B, T) for broadcasting
        t_min_expanded = t_min.unsqueeze(0).expand(B, -1)

        # Per-cell noise scale: (1, 1, H, W) for broadcasting.
        # Priority: registered buffer (cell_heteroscedastic training) > external attr (Exp 46 hack)
        _cns = None
        if getattr(self.config, 'cell_heteroscedastic', False) and hasattr(self, 'cell_noise_scale'):
            _cns = self.cell_noise_scale.to(device)
            if _cns.dim() == 2:
                _cns = _cns.unsqueeze(0).unsqueeze(0)
        elif getattr(self, '_cell_noise_scale', None) is not None:
            _cns = self._cell_noise_scale.to(device)
            if _cns.dim() == 2:
                _cns = _cns.unsqueeze(0).unsqueeze(0)

        # Start from pure noise (scaled for heteroscedastic diffusion)
        x_t = torch.randn(shape, device=device)
        if noise_scale is not None:
            x_t = x_t * noise_scale
        if _cns is not None:
            x_t = x_t * _cns

        # Standard DDPM reverse: t = n_steps-1, n_steps-2, ..., 0
        for t_global in reversed(range(n_steps)):
            # All frames at same t, clamped to per-frame t_min
            t_current = torch.full((B, T), t_global, device=device, dtype=torch.long)
            t_current = torch.maximum(t_current, t_min_expanded)

            # Skip if all frames have reached their t_min
            active = (t_current > t_min_expanded)  # (B, T)
            if not active.any():
                break

            # Flatten spatial dims for denoiser: (B, T, H, W) -> (B, T, H*W)
            x_flat = x_t.reshape(B, T, H * W)

            # Predict noise (and optionally variance fraction)
            learned_log_var = None
            crps_log_sigma = None
            if getattr(self.config, 'learn_sigma', False):
                noise_pred_flat, v_pred_flat = self.denoiser(
                    x_flat, condition, positions, t_current,
                    regime_scalar=regime_scalar,
                    baseline_surface=baseline_surface,
                )
                noise_pred = noise_pred_flat.reshape(B, T, H, W)
                v_pred = v_pred_flat.reshape(B, T, H, W)
                if self.config.loss_type == "crps":
                    # CRPS mode: v_pred is raw log_sigma, use as posterior noise scale
                    crps_log_sigma = v_pred.clamp(-10, 5)
                else:
                    # Nichol-Dhariwal mode: v_pred is interpolation fraction
                    learned_log_var = self._compute_learned_variance(v_pred, t_current)
            else:
                noise_pred_flat = self.denoiser(
                    x_flat, condition, positions, t_current,
                    regime_scalar=regime_scalar,
                    baseline_surface=baseline_surface,
                )
                noise_pred = noise_pred_flat.reshape(B, T, H, W)

            # CFG: classifier-free guidance
            if uncond_condition is not None and guidance_scale != 1.0:
                noise_uncond_flat = self.denoiser(
                    x_flat, uncond_condition, positions, t_current,
                    regime_scalar=regime_scalar,
                    baseline_surface=baseline_surface,
                )
                if isinstance(noise_uncond_flat, tuple):
                    noise_uncond_flat = noise_uncond_flat[0]
                noise_uncond = noise_uncond_flat.reshape(B, T, H, W)
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            # DDPM posterior step (same math as _sample_block_pyramid)
            t_flat = t_current.flatten()
            sqrt_recip = self.scheduler.sqrt_recip_alpha_bar[t_flat].view(B, T, 1, 1)
            sqrt_recip_m1 = self.scheduler.sqrt_recip_alpha_bar_minus_one[t_flat].view(B, T, 1, 1)
            if noise_scale is not None and scale_x0_recovery:
                # Heteroscedastic forward noise: model predicts ε, forward used σε
                x_0_pred = sqrt_recip * x_t - sqrt_recip_m1 * noise_scale * noise_pred
            else:
                # Standard or learned-variance: model predicts ε, forward used ε
                x_0_pred = sqrt_recip * x_t - sqrt_recip_m1 * noise_pred
            x0_clamp = 3.0 if (self.config.ratio_target and
                               self.config.ratio_target_mode in ("nsdiff", "e2e_nll")) else 1.0
            x_0_pred = x_0_pred.clamp(-x0_clamp, x0_clamp)

            alpha_t = self.scheduler.alphas[t_flat].view(B, T, 1, 1)
            alpha_bar_t = self.scheduler.alpha_bar[t_flat].view(B, T, 1, 1)
            alpha_bar_prev_t = self.scheduler.alpha_bar_prev[t_flat].view(B, T, 1, 1)
            beta_t = self.scheduler.betas[t_flat].view(B, T, 1, 1)

            coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
            coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
            mean = coef_x0 * x_0_pred + coef_xt * x_t

            # Posterior variance: use learned variance if available
            z = torch.randn_like(x_t)
            # Noise temperature: scale posterior noise for wider/narrower CIs
            _noise_temp = getattr(self.config, 'noise_temperature', 1.0)
            if _noise_temp != 1.0:
                z = z * (_noise_temp ** 0.5)
            nonzero = (t_current > 0).float().unsqueeze(-1).unsqueeze(-1)
            if self.crps_var_head is not None:
                # CRPS variance head: standard DDPM posterior noise (σ applied post-hoc)
                posterior_var = self.scheduler.posterior_variance[t_flat].view(B, T, 1, 1)
                x_new = mean + nonzero * torch.sqrt(posterior_var) * z
            elif crps_log_sigma is not None:
                # CRPS-learned per-element sigma: condition-dependent posterior noise
                posterior_var = self.scheduler.posterior_variance[t_flat].view(B, T, 1, 1)
                # Scale the fixed posterior std by the learned sigma ratio
                crps_sigma = torch.exp(crps_log_sigma)  # (B, T, H, W)
                x_new = mean + nonzero * crps_sigma * torch.sqrt(posterior_var) * z
            elif learned_log_var is not None:
                # Learned variance from Nichol & Dhariwal
                x_new = mean + nonzero * torch.exp(0.5 * learned_log_var) * z
            elif noise_scale is not None:
                posterior_var = self.scheduler.posterior_variance[t_flat].view(B, T, 1, 1)
                _ns_combined = noise_scale * _cns if _cns is not None else noise_scale
                x_new = mean + nonzero * _ns_combined * torch.sqrt(posterior_var) * z
            else:
                posterior_var = self.scheduler.posterior_variance[t_flat].view(B, T, 1, 1)
                _post_scale = _cns if _cns is not None else 1.0
                x_new = mean + nonzero * _post_scale * torch.sqrt(posterior_var) * z

            # Only update active frames (those above their t_min)
            active_mask = active.float().unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
            x_t = active_mask * x_new + (1 - active_mask) * x_t

        return x_t

    @torch.no_grad()
    def _sample_block_flow_matching(
        self,
        condition: torch.Tensor,
        positions: torch.Tensor,
        shape: tuple,
        n_steps: Optional[int] = None,
        uncond_condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        regime_scalar: Optional[torch.Tensor] = None,
        baseline_surface: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample one block using Flow Matching Euler ODE solve.

        Solves the ODE dx/dt = -v_θ(x_t, t) from t=1 (noise) to t=0 (clean)
        using N Euler steps. The denoiser predicts velocity v = noise - x_0.

        Args:
            condition: (B, bottleneck_dim) encoder output
            positions: (B, T) absolute frame positions
            shape: (B, T, H, W)
            n_steps: number of Euler steps (None = use config)
            uncond_condition: (B, bottleneck_dim) null condition for CFG
            guidance_scale: CFG weight (1.0 = no guidance)
            regime_scalar: (B, 1) optional regime indicator
            baseline_surface: (B, 5, 5) baseline IV surface

        Returns:
            block: (B, T, H, W) in [-1, 1]
        """
        B, T, H, W = shape
        device = condition.device
        N = n_steps or getattr(self.config, 'fm_n_inference_steps', self.config.n_steps)
        dt = 1.0 / N

        # Start from pure noise at t=1
        x_t = torch.randn(shape, device=device)

        # Euler steps from t=1 to t=0
        for i in range(N):
            t_val = 1.0 - i * dt  # current t (decreasing from 1 to dt)

            # Scale to integer-equivalent range for sinusoidal embedding
            noise_levels = torch.full(
                (B, T), t_val * (self.config.n_steps - 1),
                device=device, dtype=torch.float32,
            )

            # Predict velocity
            x_flat = x_t.reshape(B, T, H * W)
            v_pred_flat = self.denoiser(
                x_flat, condition, positions, noise_levels,
                regime_scalar=regime_scalar,
                baseline_surface=baseline_surface,
            )
            if isinstance(v_pred_flat, tuple):
                v_pred_flat = v_pred_flat[0]
            v_pred = v_pred_flat.reshape(B, T, H, W)

            # CFG: classifier-free guidance
            if uncond_condition is not None and guidance_scale != 1.0:
                v_uncond_flat = self.denoiser(
                    x_flat, uncond_condition, positions, noise_levels,
                    regime_scalar=regime_scalar,
                    baseline_surface=baseline_surface,
                )
                if isinstance(v_uncond_flat, tuple):
                    v_uncond_flat = v_uncond_flat[0]
                v_uncond = v_uncond_flat.reshape(B, T, H, W)
                v_pred = v_uncond + guidance_scale * (v_pred - v_uncond)

            # Euler step: x_{t-dt} = x_t - dt * v_pred
            x_t = x_t - dt * v_pred

        # Clamp output
        x0_clamp = 3.0 if (self.config.ratio_target and
                           self.config.ratio_target_mode in ("nsdiff", "e2e_nll")) else 1.0
        x_t = x_t.clamp(-x0_clamp, x0_clamp)

        return x_t

    def _compute_block_t_min(
        self, block_idx: int, block_size: int, future_len: int,
        max_global_residual: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Compute per-frame t_min for a block based on global horizon.

        t_min(h) = max_global_residual * h / (future_len - 1)

        where h is the global forecast horizon (0 to future_len-1).

        Args:
            block_idx: which block (0-indexed)
            block_size: frames per block
            future_len: total future frames
            max_global_residual: max residual timestep at h=future_len-1
            device: torch device

        Returns:
            (block_size,) long tensor, or None if max_global_residual == 0
        """
        if max_global_residual == 0:
            return None
        frame_indices = torch.arange(block_size, device=device)
        global_horizons = block_idx * block_size + frame_indices  # 0..future_len-1
        t_min = (max_global_residual * global_horizons.float() / (future_len - 1)).long()
        return t_min

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        n_samples: int = 1,
        max_residual: int = 20,
        max_global_residual: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Block-AR generation using DF pyramid sampling.

        Each block is denoised using the pyramid schedule where earlier
        frames clean up faster, enabling the BiGRU to propagate clean
        information to noisy later frames (DF "soft causality").

        With max_global_residual > 0, frames retain horizon-dependent
        residual noise: frame at global horizon h stops denoising at
        t_min = max_global_residual * h / (future_len - 1). This creates
        naturally growing uncertainty with forecast horizon.

        When regime conditioning is enabled, samples regime once per
        trajectory and uses it consistently across all blocks.

        Args:
            history: (B, history_len, 5, 5) in [-1, 1]
            n_samples: number of independent samples per history
            max_residual: unused (kept for API compatibility)
            max_global_residual: override config.max_global_residual.
                None = use config value.
            temperature: regime sampling temperature (higher = more diverse)

        Returns:
            (B, n_samples, future_len, 5, 5) denormalized to [0, 1]
        """
        B = history.shape[0]
        device = history.device
        bs = self.config.block_size
        n_blocks = self.config.future_len // bs
        mgr = max_global_residual if max_global_residual is not None else self.config.max_global_residual

        self._ensure_scheduler_device(device)

        all_samples = []

        for _ in range(n_samples):
            current_cond_surfaces = history  # (B, history_len, 5, 5) in [-1, 1]
            blocks = []

            # Sample regime once per trajectory (consistent across blocks)
            regime_ids = None
            if self.regime_classifier is not None:
                init_cond = self.encoder(history, mask=None)
                regime_logits = self.regime_classifier(init_cond)
                regime_probs = F.softmax(regime_logits / temperature, dim=-1)
                regime_ids = torch.multinomial(regime_probs, num_samples=1).squeeze(-1)

            for block_idx in range(n_blocks):
                # Encode growing context (no masking at inference)
                condition = self.encoder(
                    current_cond_surfaces, mask=None
                )  # (B, bottleneck_dim)

                # Forward-only training adds null_embedding as future_cond;
                # match at inference to avoid train/infer mismatch.
                if self.config.forward_only:
                    condition = condition + self.encoder.null_embedding.expand(B, -1)

                # Augment with regime embedding
                condition = self._augment_condition(condition, regime_ids)

                # Auxiliary regime features (vol_of_vol, IV level)
                condition = self._add_regime_features(condition, history)

                # Create positions for this block
                positions = (
                    torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)
                    + block_idx * bs
                )

                # Compute per-frame t_min for this block
                t_min = self._compute_block_t_min(
                    block_idx, bs, self.config.future_len, mgr, device
                )

                # Compute noise scale for posterior noise
                ns = None
                sx0 = True  # scale x_0 recovery (only for heteroscedastic forward noise)
                if self.config.heteroscedastic_noise:
                    ns = self._compute_noise_scale(current_cond_surfaces)
                elif self.variance_head is not None:
                    # Learned variance: σ from variance head (condition only)
                    log_var = self.variance_head(condition)  # (B, 1)
                    sigma = torch.exp(0.5 * log_var)  # (B, 1) -> std dev
                    ns = sigma[:, :, None, None]  # (B, 1, 1, 1)
                    sx0 = False  # forward process was standard, no x_0 recovery scaling

                # CFG: compute unconditional condition for guided sampling
                uncond_cond = None
                cfg_scale = self.config.guidance_scale
                if self.null_condition is not None and cfg_scale != 1.0:
                    uncond_cond = self.null_condition.expand(B, -1)

                # Regime scalar for percell_head (vol_of_vol from history)
                regime_scalar = None
                if getattr(self.config, 'percell_regime_input', False) and self.config.use_percell_head:
                    past_abs = denormalize_iv(current_cond_surfaces)
                    _mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T)
                    _daily_chg = _mean_iv[:, 1:] - _mean_iv[:, :-1]
                    regime_scalar = _daily_chg.std(dim=1, keepdim=True)  # (B, 1)

                # Baseline surface for denoiser input channel
                baseline_surf = None
                if getattr(self.config, 'baseline_channel', False):
                    baseline_surf = self._compute_baseline(current_cond_surfaces).squeeze(1)  # (B, 5, 5)

                # Generate block
                shape = (B, bs, self.config.surface_h, self.config.surface_w)
                if getattr(self.config, 'use_flow_matching', False):
                    block = self._sample_block_flow_matching(
                        condition, positions, shape,
                        uncond_condition=uncond_cond, guidance_scale=cfg_scale,
                        regime_scalar=regime_scalar,
                        baseline_surface=baseline_surf,
                    )
                elif self.config.sampling_mode == "uniform":
                    block = self._sample_block_uniform(
                        condition, positions, shape, t_min=t_min,
                        noise_scale=ns, scale_x0_recovery=sx0,
                        uncond_condition=uncond_cond, guidance_scale=cfg_scale,
                        regime_scalar=regime_scalar,
                        baseline_surface=baseline_surf,
                    )
                else:
                    block = self._sample_block_pyramid(
                        condition, positions, shape, t_min=t_min,
                        noise_scale=ns, scale_x0_recovery=sx0,
                        regime_scalar=regime_scalar,
                        baseline_surface=baseline_surf,
                    )
                # block: (B, bs, 5, 5) in [-1, 1]

                # Add learned mean back to residual (mean head bias correction)
                if self.mean_head is not None and self.config.ratio_target:
                    with torch.no_grad():
                        mu = self.mean_head(condition)  # (B, 25)
                        mu = mu.reshape(B, self.config.surface_h, self.config.surface_w)
                        mu = mu.unsqueeze(1)  # (B, 1, 5, 5)
                        block = block + mu

                # Convert from ratio space to absolute normalized space
                if self.config.ratio_target:
                    eps_iv = 1e-4
                    baseline = self._compute_baseline(current_cond_surfaces)  # (B, 1, 5, 5)
                    if self.config.ratio_target_mode == "logit":
                        # sigmoid(logit_diff + logit(baseline)) — bounded in (0, 1)
                        baseline_c = baseline.clamp(min=eps_iv, max=1.0 - eps_iv)
                        block_abs = torch.sigmoid(block + torch.logit(baseline_c))
                    elif self.config.ratio_target_mode == "vol_scaled":
                        # exp(sample * vol_scale) * baseline
                        past_abs = denormalize_iv(current_cond_surfaces)
                        mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_past)
                        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
                        vol = daily_chg.std(dim=1, keepdim=True)  # (B, 1)
                        vol_scale = (vol / self.config.global_mean_vol).clamp(self.config.vol_scale_min, self.config.vol_scale_max)
                        vol_scale = vol_scale.pow(self.config.vol_scale_power)
                        vol_scale = vol_scale.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
                        # Per-cell structural normalization (if enabled)
                        if getattr(self, 'cell_norm_factor', None) is not None:
                            vol_scale = vol_scale * self.cell_norm_factor  # (B, 1, 1, 1) * (5, 5) → (B, 1, 5, 5)
                        # Learned per-cell correction (if enabled)
                        if self.log_cell_scale is not None:
                            clamp = getattr(self.config, 'cell_scale_clamp', 0.2)
                            correction = torch.exp(self.log_cell_scale.clamp(-clamp, clamp))
                            vol_scale = vol_scale * correction
                        # Low-rank per-cell correction (if enabled)
                        if self.log_u_cell is not None:
                            lr_clamp = getattr(self.config, 'low_rank_cell_scale_clamp', 0.5)
                            log_u = self.log_u_cell.clamp(-lr_clamp, lr_clamp)
                            log_v = self.log_v_cell.clamp(-lr_clamp, lr_clamp)
                            lr_correction = torch.exp(log_u.unsqueeze(1) + log_v.unsqueeze(0))
                            vol_scale = vol_scale * lr_correction
                        # Fixed per-cell correction from calibration head (frozen buffer)
                        if self.fixed_cell_scale is not None:
                            vol_scale = vol_scale * self.fixed_cell_scale
                        ratio = torch.exp(block * vol_scale)
                        # Learned drift: shift baseline in IV-space (drift OUTSIDE exp)
                        effective_baseline = baseline  # (B, 1, 5, 5)
                        if self.drift_head is not None:
                            with torch.no_grad():
                                drift_raw = self.drift_head(condition)  # (B, 25)
                                drift_raw = drift_raw.reshape(B, self.config.surface_h, self.config.surface_w)
                                drift_max = getattr(self.config, 'drift_max', 0.05)
                                drift = torch.tanh(drift_raw) * drift_max  # (B, 5, 5)
                                effective_baseline = (baseline + drift.unsqueeze(1)).clamp(min=0.01)
                        block_abs = (ratio * effective_baseline).clamp(0.001, 1.0)
                    elif self.config.ratio_target_mode == "additive_scaled":
                        # Additive: baseline + z * vol_scale * baseline
                        past_abs = denormalize_iv(current_cond_surfaces)
                        mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_past)
                        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
                        vol = daily_chg.std(dim=1, keepdim=True)  # (B, 1)
                        vol_scale = (vol / self.config.global_mean_vol).clamp(self.config.vol_scale_min, self.config.vol_scale_max)
                        vol_scale = vol_scale.pow(self.config.vol_scale_power)
                        vol_scale = vol_scale.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
                        if getattr(self, 'cell_norm_factor', None) is not None:
                            vol_scale = vol_scale * self.cell_norm_factor
                        predicted_diff = block * vol_scale * baseline  # (B, T_block, 5, 5)
                        block_abs = (baseline + predicted_diff).clamp(0.001, 1.0)
                    elif self.config.ratio_target_mode == "additive_whitened":
                        # Additive whitening: IV = baseline + z × σ
                        # σ = max(cell_std, vol_scale × σ_base)
                        past_abs = denormalize_iv(current_cond_surfaces)
                        daily_chg_cell = past_abs[:, 1:] - past_abs[:, :-1]  # (B, T-1, 5, 5)
                        cell_std = daily_chg_cell.std(dim=1)  # (B, 5, 5)
                        mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_past)
                        mean_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
                        vol = mean_chg.std(dim=1, keepdim=True)  # (B, 1)
                        vol_scale = (vol / self.config.global_mean_vol).clamp(
                            self.config.vol_scale_min, self.config.vol_scale_max)
                        vol_scale = vol_scale.pow(self.config.vol_scale_power)
                        sigma_base = self.config.global_mean_vol * (self.config.block_size ** 0.5)
                        sigma_floor = (vol_scale * sigma_base).unsqueeze(-1)  # (B, 1, 1)
                        sigma = torch.maximum(
                            cell_std,
                            sigma_floor.expand_as(cell_std)
                        ).unsqueeze(1)  # (B, 1, 5, 5)
                        block_abs = (baseline + block * sigma).clamp(0.001, 1.0)
                    elif self.config.ratio_target_mode == "vol_scaled_percell":
                        # Per-cell: exp(sample * vol_scale[r,c]) * baseline
                        past_abs = denormalize_iv(current_cond_surfaces)
                        daily_chg = past_abs[:, 1:] - past_abs[:, :-1]  # (B, T-1, 5, 5)
                        cell_vol = daily_chg.std(dim=1)  # (B, 5, 5)
                        vol_scale = (cell_vol / self.global_mean_cell_vol).clamp(self.config.vol_scale_min, self.config.vol_scale_max)
                        vol_scale = vol_scale.unsqueeze(1)  # (B, 1, 5, 5)
                        ratio = torch.exp(block * vol_scale)
                        block_abs = (ratio * baseline).clamp(0.001, 1.0)
                    elif self.config.ratio_target_mode == "learned_percell":
                        # Magnitude × Shape: scalar vol_scale * NN pattern
                        past_abs = denormalize_iv(current_cond_surfaces)
                        mean_iv = past_abs.mean(dim=(-1, -2))
                        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
                        vol = daily_chg.std(dim=1, keepdim=True)
                        vol_scale = (vol / self.config.global_mean_vol).clamp(self.config.vol_scale_min, self.config.vol_scale_max)
                        vol_scale = vol_scale.pow(self.config.vol_scale_power)
                        log_pattern = self.percell_sigma_head(condition)
                        log_pattern = log_pattern.reshape(-1, 5, 5)
                        pattern = torch.exp(log_pattern)
                        pattern = pattern / pattern.mean(dim=(-1, -2), keepdim=True)
                        sigma_4d = vol_scale.unsqueeze(-1).unsqueeze(-1) * pattern.unsqueeze(1)
                        log_ratio = block * sigma_4d
                        ratio = torch.exp(log_ratio)
                        block_abs = (ratio * baseline).clamp(0.001, 1.0)
                    elif self.config.ratio_target_mode == "vol_scaled_learned":
                        # Hybrid: vol_scale * exp(correction), then exp(sample * sigma) * baseline
                        past_abs = denormalize_iv(current_cond_surfaces)
                        mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_past)
                        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
                        vol = daily_chg.std(dim=1, keepdim=True)  # (B, 1)
                        vol_scale = (vol / self.config.global_mean_vol).clamp(self.config.vol_scale_min, self.config.vol_scale_max)
                        vol_scale = vol_scale.pow(self.config.vol_scale_power)
                        log_corr = self.log_correction_head(condition)  # (B, 1)
                        log_corr_clamped = log_corr.clamp(-0.5, 0.5)
                        sigma = vol_scale * torch.exp(log_corr_clamped)  # (B, 1)
                        sigma_4d = sigma.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
                        log_ratio = block * sigma_4d  # unstandardize
                        ratio = torch.exp(log_ratio)
                        block_abs = (ratio * baseline).clamp(0.001, 1.0)
                    elif self.config.ratio_target_mode in ("nsdiff", "e2e_nll"):
                        # NSDiff/e2e: destandardize with learned sigma, then exp
                        log_sigma = self.log_std_head(condition)  # (B, 1)
                        sigma_c = torch.exp(log_sigma.clamp(-2, 2))
                        sigma_4d = sigma_c.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
                        log_ratio = block * sigma_4d  # unstandardize
                        ratio = torch.exp(log_ratio)
                        block_abs = (ratio * baseline).clamp(0.001, 1.0)
                    elif self.config.ratio_target_mode == "percell_revin":
                        # Per-Cell RevIN: prediction = sample * cell_std + baseline
                        past_abs = denormalize_iv(current_cond_surfaces)
                        daily_chg = past_abs[:, 1:] - past_abs[:, :-1]  # (B, T-1, 5, 5)
                        cell_std_data = daily_chg.std(dim=1)  # (B, 5, 5)
                        floor = torch.exp(self.log_revin_floor)  # (5, 5)
                        cell_std = torch.maximum(cell_std_data, floor.unsqueeze(0))  # (B, 5, 5)
                        cell_std = cell_std.unsqueeze(1)  # (B, 1, 5, 5)
                        block_abs = (block * cell_std + baseline).clamp(0.001, 1.0)
                    else:
                        # exp(log_ratio) * baseline — original mode
                        ratio = torch.exp(block)
                        block_abs = (ratio * baseline).clamp(0.001, 1.0)
                    block = normalize_iv(block_abs)  # back to [-1, 1] absolute

                blocks.append(block)

                # Grow conditioning surfaces
                current_cond_surfaces = torch.cat(
                    [current_cond_surfaces, block], dim=1
                )

            # Concatenate all blocks
            full_trajectory = torch.cat(blocks, dim=1)  # (B, future_len, 5, 5)
            all_samples.append(full_trajectory)

        # Stack samples: (B, n_samples, future_len, 5, 5)
        samples = torch.stack(all_samples, dim=1)

        # Apply learned uncertainty scaling (in normalized space, before denorm)
        if self.uncertainty_head is not None:
            # Get condition from initial history (same for all samples)
            uh_cond = self.encoder(history, mask=None)
            if self.config.forward_only:
                uh_cond = uh_cond + self.encoder.null_embedding.expand(B, -1)
            uh_cond = self._augment_condition(uh_cond, None)
            scale = self.uncertainty_head(uh_cond)  # (B, future_len)
            scale = scale[:, None, :, None, None]  # (B, 1, future_len, 1, 1)
            mean = samples.mean(dim=1, keepdim=True)
            samples = mean + scale * (samples - mean)

        # CRPS variance head: per-cell, per-horizon spread redistribution.
        # σ from head captures both per-cell structure AND horizon growth (via pos embed).
        # Normalize over all cells+frames together to preserve relative magnitudes.
        if self.crps_var_head is not None and self.crps_var_head.n_cells > 1:
            crps_cond = self.encoder(history, mask=None)
            if self.config.forward_only:
                crps_cond = crps_cond + self.encoder.null_embedding.expand(B, -1)
            crps_cond = self._augment_condition(crps_cond, None)
            # Use t=0 (clean, most informative for spread calibration)
            t_zero = torch.zeros(B, self.config.future_len, dtype=torch.long, device=history.device)
            positions = torch.arange(self.config.future_len, device=history.device).unsqueeze(0).expand(B, -1)
            log_sigma = self.crps_var_head(crps_cond, t_zero, positions=positions)  # (B, future_len, n_cells)
            sigma_raw = torch.exp(log_sigma.clamp(-10, 5))
            H, W = self.config.surface_h, self.config.surface_w
            sigma_raw = sigma_raw.view(B, self.config.future_len, H, W)
            # Normalize over ALL cells AND frames → mean=1 globally.
            # Preserves both per-cell structure AND horizon growth.
            sigma = sigma_raw / sigma_raw.mean(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-6)
            # Boost-only mode: only WIDEN cells, never narrow.
            # Fixes zero-sum problem where borderline cells get pushed below threshold.
            if getattr(self.config, 'crps_boost_only', False):
                sigma = sigma.clamp(min=1.0)
            # Optional clamp to limit correction range
            clamp_val = getattr(self.config, 'crps_sigma_clamp', 0.0)
            if clamp_val > 0:
                sigma = sigma.clamp(1.0 - clamp_val, 1.0 + clamp_val)
            sigma = sigma[:, None, :, :, :]  # (B, 1, future_len, H, W) for broadcast
            # Scale deviations from ensemble mean per-cell, per-horizon
            mean = samples.mean(dim=1, keepdim=True)
            samples = mean + sigma * (samples - mean)

        # Denormalize to [0, 1]
        samples = denormalize_iv(samples)
        if self.config.clamp_output:
            samples = samples.clamp(0.0, 1.0)

        return samples

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 1,
        max_residual: int = 20,
        max_global_residual: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Block-AR generation with all samples batched in parallel.

        Folds n_samples into the batch dimension so the GPU processes
        B*n_samples items per denoiser call instead of B. Same algorithm
        as sample(), just parallelized.

        Args:
            history: (B, history_len, 5, 5) in [-1, 1]
            n_samples: number of independent samples per history
            max_residual: unused (kept for API compatibility)
            max_global_residual: override config.max_global_residual.
                None = use config value.
            temperature: regime sampling temperature (higher = more diverse)

        Returns:
            (B, n_samples, future_len, 5, 5) denormalized to [0, 1]
        """
        B = history.shape[0]
        device = history.device
        bs = self.config.block_size
        n_blocks = self.config.future_len // bs
        S = n_samples
        B_eff = B * S
        mgr = max_global_residual if max_global_residual is not None else self.config.max_global_residual

        self._ensure_scheduler_device(device)

        # Expand history: (B, T, H, W) → (B*S, T, H, W)
        # repeat_interleave gives [h0,h0,...,h0, h1,h1,...,h1, ...]
        # so .view(B, S, ...) later correctly groups samples per history
        history_expanded = history.repeat_interleave(S, dim=0)
        current_cond_surfaces = history_expanded.clone()

        # Sample regimes for all B*S items (once, reuse across blocks)
        regime_ids = None
        if self.regime_classifier is not None:
            init_cond = self.encoder(current_cond_surfaces, mask=None)
            regime_logits = self.regime_classifier(init_cond)
            regime_probs = F.softmax(regime_logits / temperature, dim=-1)
            regime_ids = torch.multinomial(regime_probs, num_samples=1).squeeze(-1)

        blocks = []

        for block_idx in range(n_blocks):
            # Encode growing context: (B*S, T_growing, 5, 5) → (B*S, bottleneck_dim)
            condition = self.encoder(
                current_cond_surfaces, mask=None
            )

            # Forward-only training adds null_embedding as future_cond;
            # match at inference to avoid train/infer mismatch.
            if self.config.forward_only:
                condition = condition + self.encoder.null_embedding.expand(B_eff, -1)

            # Augment with regime embedding
            condition = self._augment_condition(condition, regime_ids)

            # Auxiliary regime features (use original history, expanded)
            condition = self._add_regime_features(condition, history_expanded)

            # Positions: (B*S, bs)
            positions = (
                torch.arange(bs, device=device).unsqueeze(0).expand(B_eff, -1)
                + block_idx * bs
            )

            # Compute per-frame t_min for this block
            t_min = self._compute_block_t_min(
                block_idx, bs, self.config.future_len, mgr, device
            )

            # Compute noise scale for posterior noise
            ns = None
            sx0 = True  # scale x_0 recovery (only for heteroscedastic forward noise)
            if self.config.heteroscedastic_noise:
                ns = self._compute_noise_scale(current_cond_surfaces)
            elif self.variance_head is not None:
                # Learned variance: σ from variance head (condition only)
                log_var = self.variance_head(condition)  # (B_eff, 1)
                sigma = torch.exp(0.5 * log_var)  # (B_eff, 1) -> std dev
                ns = sigma[:, :, None, None]  # (B_eff, 1, 1, 1)
                sx0 = False  # forward process was standard, no x_0 recovery scaling

            # CFG: compute unconditional condition for guided sampling
            uncond_cond = None
            cfg_scale = self.config.guidance_scale
            if self.null_condition is not None and cfg_scale != 1.0:
                uncond_cond = self.null_condition.expand(B_eff, -1)

            # Regime scalar for percell_head (vol_of_vol from history)
            regime_scalar = None
            if getattr(self.config, 'percell_regime_input', False) and self.config.use_percell_head:
                past_abs = denormalize_iv(current_cond_surfaces)
                _mean_iv = past_abs.mean(dim=(-1, -2))  # (B_eff, T)
                _daily_chg = _mean_iv[:, 1:] - _mean_iv[:, :-1]
                regime_scalar = _daily_chg.std(dim=1, keepdim=True)  # (B_eff, 1)

            # Baseline surface for denoiser input channel
            baseline_surf = None
            if getattr(self.config, 'baseline_channel', False):
                baseline_surf = self._compute_baseline(current_cond_surfaces).squeeze(1)  # (B_eff, 5, 5)

            # Generate block for all samples in parallel
            shape = (B_eff, bs, self.config.surface_h, self.config.surface_w)
            if self.config.sampling_mode == "uniform":
                block = self._sample_block_uniform(
                    condition, positions, shape, t_min=t_min,
                    noise_scale=ns, scale_x0_recovery=sx0,
                    uncond_condition=uncond_cond, guidance_scale=cfg_scale,
                    regime_scalar=regime_scalar,
                    baseline_surface=baseline_surf,
                )
            else:
                block = self._sample_block_pyramid(
                    condition, positions, shape, t_min=t_min,
                    noise_scale=ns, scale_x0_recovery=sx0,
                    regime_scalar=regime_scalar,
                    baseline_surface=baseline_surf,
                )
            # block: (B*S, bs, 5, 5)

            # Convert from ratio space to absolute normalized space
            if self.config.ratio_target:
                eps_iv = 1e-4
                baseline = self._compute_baseline(current_cond_surfaces)  # (B_eff, 1, 5, 5)
                if self.config.ratio_target_mode == "logit":
                    # sigmoid(logit_diff + logit(baseline)) — bounded in (0, 1)
                    baseline_c = baseline.clamp(min=eps_iv, max=1.0 - eps_iv)
                    block_abs = torch.sigmoid(block + torch.logit(baseline_c))
                else:
                    # exp(log_ratio) * baseline — original mode
                    ratio = torch.exp(block)
                    block_abs = (ratio * baseline).clamp(0.001, 1.0)
                block = normalize_iv(block_abs)  # back to [-1, 1] absolute

            blocks.append(block)

            # Grow conditioning surfaces
            current_cond_surfaces = torch.cat(
                [current_cond_surfaces, block], dim=1
            )

        # Concatenate blocks: (B*S, future_len, 5, 5)
        full_trajectory = torch.cat(blocks, dim=1)

        # Reshape: (B*S, future_len, 5, 5) → (B, S, future_len, 5, 5)
        samples = full_trajectory.view(
            B, S, self.config.future_len,
            self.config.surface_h, self.config.surface_w,
        )

        # Apply learned uncertainty scaling (in normalized space, before denorm)
        if self.uncertainty_head is not None:
            uh_cond = self.encoder(history, mask=None)
            if self.config.forward_only:
                uh_cond = uh_cond + self.encoder.null_embedding.expand(B, -1)
            uh_cond = self._augment_condition(uh_cond, None)
            scale = self.uncertainty_head(uh_cond)  # (B, future_len)
            scale = scale[:, None, :, None, None]  # (B, 1, future_len, 1, 1)
            mean = samples.mean(dim=1, keepdim=True)
            samples = mean + scale * (samples - mean)

        # CRPS variance head: per-cell, per-horizon spread redistribution
        if self.crps_var_head is not None and self.crps_var_head.n_cells > 1:
            crps_cond = self.encoder(history, mask=None)
            if self.config.forward_only:
                crps_cond = crps_cond + self.encoder.null_embedding.expand(B, -1)
            crps_cond = self._augment_condition(crps_cond, None)
            t_zero = torch.zeros(B, self.config.future_len, dtype=torch.long, device=history.device)
            positions = torch.arange(self.config.future_len, device=history.device).unsqueeze(0).expand(B, -1)
            log_sigma = self.crps_var_head(crps_cond, t_zero, positions=positions)
            sigma = torch.exp(log_sigma.clamp(-10, 5))
            H, W = self.config.surface_h, self.config.surface_w
            sigma = sigma.view(B, self.config.future_len, H, W)
            # Normalize over ALL cells AND frames → mean=1 globally
            sigma = sigma / sigma.mean(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-6)
            # Boost-only mode: only WIDEN cells, never narrow
            if getattr(self.config, 'crps_boost_only', False):
                sigma = sigma.clamp(min=1.0)
            clamp_val = getattr(self.config, 'crps_sigma_clamp', 0.0)
            if clamp_val > 0:
                sigma = sigma.clamp(1.0 - clamp_val, 1.0 + clamp_val)
            sigma = sigma[:, None, :, :, :]
            mean = samples.mean(dim=1, keepdim=True)
            samples = mean + sigma * (samples - mean)

        # Denormalize to [0, 1]
        samples = denormalize_iv(samples)
        if self.config.clamp_output:
            samples = samples.clamp(0.0, 1.0)

        return samples
