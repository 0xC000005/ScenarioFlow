"""
Configuration for Block-AR DDPM on Volatility Surfaces.

Block-AR Diffusion with MCVD multi-task training and Diffusion Forcing
noise schedules. Targets improved CI coverage over the DDPM POC baseline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BlockARPOCConfig:
    """POC configuration for Block-AR DDPM on volatility surfaces."""

    # === Data Configuration ===
    data_path: str = "data/vol_surface_with_ret.npz"
    history_len: int = 30
    future_len: int = 30
    surface_h: int = 5
    surface_w: int = 5
    block_size: int = 10

    # === Encoder ===
    gru_hidden_dim: int = 64
    bottleneck_dim: int = 64
    cond_aug_sigma: float = 0.0
    encoder_dropout: float = 0.1

    # === Encoder ===
    encoder_type: str = "gru"  # "gru" (flat spatial) or "conv3d" (spatial-aware CausalConv3d)

    # === Denoiser ===
    denoiser_type: str = "bigru"  # "bigru" or "conv3d"
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

    # === Diffusion Process ===
    n_steps: int = 100
    schedule: str = "cosine"

    # === Flow Matching ===
    use_flow_matching: bool = False  # replace DDPM with learned ODE transport
    fm_n_inference_steps: int = 100  # Euler steps at inference

    # === MCVD ===
    p_mask: float = 0.2  # legacy Bernoulli (used if mcvd task probs all zero)
    jitter_std: float = 0.15
    forward_only: bool = False  # disable MCVD: always FORWARD task
    # Explicit MCVD task probabilities — overrides p_mask when any nonzero
    mcvd_p_forward: float = 0.0
    mcvd_p_backward: float = 0.0
    mcvd_p_interpolation: float = 0.0
    mcvd_p_unconditional: float = 0.0
    interp_loss_weight: float = 1.0  # down-weight interpolation loss (1.0=full, 0.3=30%)

    # === Noise Schedule ===
    use_uniform_noise: bool = False  # one scalar t per block instead of per-frame task-adaptive
    sampling_mode: str = "pyramid"   # "pyramid" (DF staggered) or "uniform" (standard DDPM)

    # === PYoCo correlated noise ===
    noise_rho: float = 0.5

    # === Sampling ===
    max_residual_timestep: int = 20
    max_global_residual: int = 0  # growing uncertainty: 0=off, 10=recommended
    noise_temperature: float = 1.0  # posterior noise temperature: >1.0 widens CIs
    clamp_output: bool = True  # clamp samples to [0,1] after denorm

    # === Regime Conditioning (hierarchical sampling) ===
    n_regimes: int = 5
    regime_embed_dim: int = 32
    regime_loss_weight: float = 1.0
    use_regime_conditioning: bool = False

    # === Learned Uncertainty Head (trained separately, Phase 3) ===
    use_uncertainty_head: bool = False
    uncertainty_hidden_dim: int = 64

    # === Heteroscedastic Forward Noise ===
    heteroscedastic_noise: bool = False
    global_mean_iv: float = 0.2154  # precomputed from training data (denormalized [0,1])
    heteroscedastic_power: float = 0.5  # 0.5=var∝IV, 1.0=std∝IV (multiplicative)

    # === Learned Variance (Diffusion2-style) ===
    learned_variance: bool = False
    variance_beta_nll: float = 0.5  # beta-NLL weight (0.5 recommended by Seitzer et al.)

    # === Ratio-Space Target ===
    ratio_target: bool = False  # diffusion on transformed ratios for conditional uncertainty
    ratio_target_mode: str = "log"  # "log", "logit", "vol_scaled", "additive_scaled", "additive_whitened", "vol_scaled_percell", "nsdiff", "e2e_nll", "vol_scaled_learned", "learned_percell", or "percell_revin"
    global_mean_vol: float = 0.0187  # mean vol_scale across training data
    vol_scale_power: float = 1.0  # exponent on vol_scale: 0.5=sqrt dampening, 1.0=full
    vol_scale_min: float = 0.5  # min clamp for vol_scale (higher = wider calm CIs)
    vol_scale_max: float = 2.0  # max clamp for vol_scale
    baseline_window: int = 1  # number of history days to average for baseline (1 = last day only)
    use_median_baseline: bool = False  # use median(full history) instead of mean(last K)
    nsdiff_sigma_lambda: float = 0.1  # NLL loss weight for learned sigma (nsdiff/e2e_nll mode)
    e2e_sigma_reg: float = 0.01  # L2 regularization on log_sigma for e2e_nll mode

    # Learned sigma (Nichol & Dhariwal 2021): denoiser predicts variance
    learn_sigma: bool = False
    lambda_vlb: float = 0.001  # VLB loss weight

    # CoordConv: add row/col coordinate channels to denoiser input
    spatial_pos_encoding: bool = False

    # Baseline surface as extra denoiser input channel (per-cell spatial context)
    baseline_channel: bool = False

    # Learned per-cell vol_scale correction: static (5,5) parameter trained by diffusion loss
    # Replaces vol_scale_min hyperparameter with learned capacity (Bitter Lesson)
    learn_cell_scale: bool = False
    cell_scale_clamp: float = 0.2  # max abs value for log correction: 0.2 → [0.82x, 1.22x]

    # Low-rank per-cell vol_scale correction: rank-1 outer product U×V
    # Smooth by construction (no neighbor discontinuities), 10 params total
    low_rank_cell_scale: bool = False
    low_rank_cell_scale_clamp: float = 0.5  # per-component clamp: [-0.5, 0.5]

    # Fixed per-cell vol_scale correction from calibration head analysis
    cell_scale_values: list = None

    # Per-cell structural normalization
    cell_norm_power: float = 0.0  # 0.0=off, >0 = per-cell vol normalization

    # Per-cell loss weighting
    cell_loss_weight_power: float = 0.0  # 0.0=off, >0 = weight inversely to per-cell vol

    # Classifier-Free Guidance (CFG)
    cond_drop_prob: float = 0.0  # prob of dropping condition (0.0 = no CFG)
    guidance_scale: float = 1.0  # inference guidance scale (1.0 = no guidance)

    # Mean prediction head (bias correction via learned capacity)
    use_mean_head: bool = False
    mean_head_lambda: float = 1.0  # weight for mean prediction loss

    # Learned drift head: predicts baseline shift in IV-space
    # IV = (baseline + drift) × exp(z × vol_scale) instead of baseline × exp(...)
    use_drift_head: bool = False
    drift_hidden_dim: int = 64
    drift_loss_weight: float = 1.0  # weight for drift supervision MSE loss
    drift_max: float = 0.05  # max drift in IV space (±5% of [0,1] range)

    # Auxiliary regime features (vol_of_vol + IV level as explicit conditioning)
    aux_regime_features: bool = False

    # CRPS variance head
    crps_variance_head: bool = False
    lambda_crps: float = 0.1  # CRPS auxiliary loss weight
    crps_n_cells: int = 1  # 1=scalar σ (original), 25=per-cell σ for 5×5 grid
    crps_sigma_clamp: float = 0.0  # max deviation from 1.0 in normalized σ (0=unlimited)
    crps_pos_embed_dim: int = 0  # frame position embedding dim for horizon-dependent σ (0=off)
    crps_boost_only: bool = False  # boost-only mode: σ = max(1.0, σ_norm) — widen, never narrow

    # Per-cell heteroscedastic forward noise: static (5,5) noise scale from
    # training data. Each cell gets noise proportional to its normalized target std.
    cell_heteroscedastic: bool = False
    cell_noise_power: float = 0.5  # power for scale: (cell_std / median)^power
    cell_noise_clamp_min: float = 0.5
    cell_noise_clamp_max: float = 2.0

    # SPADE: per-position learned scale/shift in denoiser AdaGN layers
    use_spade: bool = False

    # Spatial self-attention: multi-head attention over 5x5 grid in denoiser
    use_spatial_attention: bool = False
    spatial_attn_heads: int = 4

    # Per-cell residual head: condition-dependent noise prediction correction
    use_percell_head: bool = False
    percell_head_hidden: int = 64
    percell_regime_input: bool = False  # add vol_of_vol scalar to percell_head input

    # === Loss ===
    loss_type: str = "mse"  # "mse" or "huber"
    huber_delta: float = 0.1  # Huber threshold (smaller = more L1-like)

    # Regime-weighted loss: scale per-sample loss by vol_of_vol quantile
    # turb windows (high vol_of_vol) get higher weight → model allocates more capacity to turb
    turb_loss_weight: float = 0.0  # 0.0=off, >0 = turb_weight/calm_weight ratio (e.g. 2.0)

    # === Training ===
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    # Data split (same as DDPM POC for fair comparison)
    train_end: int = 4040
    val_start: int = 4040
    val_end: int = 4540
    test_start: int = 4540

    # === Evaluation ===
    n_eval_samples: int = 10
    eval_horizons: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    ci_levels: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.9, 0.95])

    # === Output ===
    output_dir: str = "models/backfill/block_ar"
    checkpoint_every: int = 10

    # === Device ===
    device: str = "cuda"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> str:
        return f"{self.output_dir}/best_model.pt"

    @property
    def results_dir(self) -> str:
        return "results/block_ar"


def get_default_config() -> BlockARPOCConfig:
    return BlockARPOCConfig()


def get_fast_test_config() -> BlockARPOCConfig:
    return BlockARPOCConfig(
        n_steps=20,
        batch_size=32,
        epochs=5,
        n_eval_samples=5,
        bigru_hidden_dim=64,
        gru_hidden_dim=32,
    )


if __name__ == "__main__":
    config = get_default_config()
    print("Block-AR POC Default Configuration:")
    print("-" * 40)
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        print(f"  {field_name}: {value}")
