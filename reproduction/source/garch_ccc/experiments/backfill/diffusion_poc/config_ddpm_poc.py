"""
Configuration for DDPM POC on Volatility Surfaces.

This is a minimal configuration for validating that diffusion can achieve
better CI coverage than VAE (33%) on vol surface forecasting.

Key design choices:
- 30 days history -> 30 days future (not 60->60 yet)
- 100 diffusion steps (not 1000)
- Small model (~100K params)
- No temporal attention yet
- No regime classifier yet
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class DDPMPOCConfig:
    """POC configuration for DDPM on volatility surfaces."""

    # === Data Configuration ===
    data_path: str = "data/vol_surface_with_ret.npz"
    history_len: int = 30
    future_len: int = 30
    surface_h: int = 5
    surface_w: int = 5

    # === Model Architecture ===
    base_channels: int = 32  # Small for POC
    n_res_blocks: int = 4
    condition_dim: int = 128
    time_embed_dim: int = 64
    groups: int = 8
    dropout: float = 0.0  # Keep 0.0 for baseline; cross-attention mode collapse is fundamental, not dropout-related

    # === Diffusion Process ===
    n_steps: int = 100  # Fast for POC (vs 1000 in production)
    schedule: str = 'cosine'
    noise_schedule: str = 'uniform'  # 'uniform', 'independent', 'structured_causal', 'erdm_progressive'

    # === Structured Causal Noise (Option G) ===
    structured_spread_scale: float = 50.0  # Total spread across frames (should be < n_steps)

    # === ERDM Progressive (Option H) ===
    erdm_sigma_min: float = 0.002
    erdm_sigma_max: float = 80.0  # Should be < 100 for n_steps=100
    erdm_rho: float = -10.0

    # === Classifier-Free Guidance (CFG) ===
    cond_drop_prob: float = 0.0  # Probability of dropping condition during training (0.1 recommended for CFG)

    # === Training ===
    batch_size: int = 64
    epochs: int = 50  # Quick validation
    lr: float = 1e-3  # Higher for faster convergence
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 5

    # Data split (using same as original VAE for fair comparison)
    train_end: int = 4040  # ~16 years training
    val_start: int = 4040
    val_end: int = 4540   # 500 days validation
    test_start: int = 4540

    # === Evaluation ===
    n_eval_samples: int = 100  # Samples per history for CI coverage
    eval_horizons: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    ci_levels: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.9, 0.95])

    # === Hierarchical Regime Sampling (Option J) ===
    n_regimes: int = 5
    regime_embed_dim: int = 32
    regime_loss_weight: float = 1.0
    regime_labels_path: str = "data/regime_labels.npz"
    use_regime_conditioning: bool = False  # Enable with --use_regime flag

    # === Output ===
    output_dir: str = "models/backfill/ddpm_poc"
    checkpoint_every: int = 10
    log_every: int = 50

    # === Device ===
    device: str = "cuda"

    def __post_init__(self):
        """Create output directory if needed."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> str:
        return f"{self.output_dir}/best_model.pt"

    @property
    def results_dir(self) -> str:
        return "results/ddpm_poc"


def get_default_config() -> DDPMPOCConfig:
    """Get default POC configuration."""
    return DDPMPOCConfig()


def get_fast_test_config() -> DDPMPOCConfig:
    """Get configuration for fast testing (smaller, fewer epochs)."""
    return DDPMPOCConfig(
        base_channels=16,
        n_res_blocks=2,
        n_steps=20,
        batch_size=32,
        epochs=5,
        n_eval_samples=10,
    )


def get_full_config() -> DDPMPOCConfig:
    """Get configuration for full training (if POC succeeds)."""
    return DDPMPOCConfig(
        history_len=60,
        future_len=60,
        base_channels=64,
        n_res_blocks=6,
        n_steps=200,
        batch_size=32,
        epochs=200,
        lr=5e-4,
        n_eval_samples=200,
    )


if __name__ == "__main__":
    # Print default config
    config = get_default_config()
    print("DDPM POC Default Configuration:")
    print("-" * 40)
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        print(f"  {field_name}: {value}")
