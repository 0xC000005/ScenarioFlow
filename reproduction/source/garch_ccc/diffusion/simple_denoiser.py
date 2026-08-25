"""
Simple 3D Denoiser for Volatility Surface Diffusion.

This is a minimal denoiser for the POC that:
1. Encodes history context using CausalConv3d
2. Conditions on diffusion timestep
3. Predicts noise in the future surfaces

Reuses building blocks from vae/causal_3d_blocks.py.
No temporal attention yet - just proves diffusion can work on vol surfaces.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.causal_3d_blocks import (
    CausalConv3d,
    ResnetBlockCausal3D,
)
from diffusion.time_embedding import TimeEmbedding, AdaptiveGroupNorm

# IV normalization constants (maps [IV_MIN, IV_MAX] ↔ [-1, 1])
# Following standard DDPM practice from Ho et al. 2020
# Empirical range from SPX data: min=0.01, max=0.9957
# Use slightly wider range [0.0, 1.0] to handle edge cases
IV_MIN = 0.0
IV_MAX = 1.0


def denormalize_iv(iv_norm: torch.Tensor) -> torch.Tensor:
    """Denormalize IV from [-1, 1] to [0.05, 1.0]."""
    return (iv_norm + 1.0) / 2.0 * (IV_MAX - IV_MIN) + IV_MIN


@dataclass
class DenoiserConfig:
    """Configuration for SimpleDenoiser3D."""
    # Data dimensions
    history_len: int = 30
    future_len: int = 30
    surface_h: int = 5
    surface_w: int = 5

    # Model architecture
    base_channels: int = 32
    n_res_blocks: int = 4
    condition_dim: int = 128
    time_embed_dim: int = 64
    groups: int = 8
    dropout: float = 0.0

    # Diffusion
    n_steps: int = 100
    noise_schedule: str = 'uniform'  # 'uniform', 'independent', 'structured_causal', 'erdm_progressive'

    # Structured Causal Noise (Option G) parameters
    structured_spread_scale: float = 50.0  # Total spread across frames (should be < n_steps)

    # ERDM Progressive (Option H) parameters
    erdm_sigma_min: float = 0.002
    erdm_sigma_max: float = 80.0  # Should be < 100 for n_steps=100
    erdm_rho: float = -10.0

    # Classifier-Free Guidance (CFG)
    cond_drop_prob: float = 0.0  # Probability of dropping condition during training

    # Hierarchical Regime Sampling (Option J)
    n_regimes: int = 5
    regime_embed_dim: int = 32
    regime_loss_weight: float = 1.0
    use_regime_conditioning: bool = False  # If True, add regime embedding to condition


class HistoryEncoder(nn.Module):
    """
    Encodes history of volatility surfaces into a conditioning vector.

    Uses CausalConv3d to process temporal information, then pools to a vector.
    The conditioning vector is used for FiLM modulation in the denoiser.
    """

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config

        # Initial projection: (B, 1, T_hist, 5, 5) -> (B, C, T_hist, 5, 5)
        self.conv_in = CausalConv3d(1, config.base_channels, kernel_size=3)

        # Stack of ResNet blocks
        self.blocks = nn.ModuleList([
            ResnetBlockCausal3D(
                config.base_channels,
                config.base_channels,
                groups=config.groups,
                dropout=config.dropout,
            )
            for _ in range(2)
        ])

        # Pool to condition vector
        # Global average pool over T, H, W -> (B, C)
        # Then project to condition_dim
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Linear(config.base_channels, config.condition_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Encode history surfaces.

        Args:
            history: (B, T_hist, H, W) past volatility surfaces

        Returns:
            condition: (B, condition_dim) conditioning vector for FiLM
        """
        # Add channel dim: (B, T_hist, H, W) -> (B, 1, T_hist, H, W)
        x = history.unsqueeze(1)

        # Conv + ResNet blocks
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        # x: (B, C, T_hist, H, W) e.g., (B, 32, 30, 5, 5)

        # Global condition for FiLM
        x_pooled = self.pool(x)  # (B, C, 1, 1, 1)
        x_flat = x_pooled.view(x_pooled.shape[0], -1)  # (B, C)
        condition = self.proj(x_flat)  # (B, condition_dim)

        return condition


class RegimeClassifier(nn.Module):
    """Predicts market regime from history context.

    Used in hierarchical sampling (Option J) to:
    1. Classify the likely future regime from history encoding
    2. Enable regime-conditioned diffusion generation
    """

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config
        self.classifier = nn.Sequential(
            nn.Linear(config.condition_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, config.n_regimes),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Predict regime logits from history encoding.

        Args:
            condition: (B, condition_dim) history encoding

        Returns:
            logits: (B, n_regimes) unnormalized regime probabilities
        """
        return self.classifier(condition)


class SimpleDenoiser3D(nn.Module):
    """
    Minimal 3D denoiser for volatility surface diffusion POC.

    Architecture:
    1. History encoder -> condition vector
    2. Time embedding -> t_emb vector
    3. Future encoder: noisy_future -> features
    4. Condition injection: features + condition + t_emb
    5. ResNet blocks for denoising
    6. Output: predicted noise

    This is intentionally simple - no U-Net skip connections,
    no temporal attention. Just enough to validate the approach.
    """

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config
        C = config.base_channels

        # History encoder
        self.history_encoder = HistoryEncoder(config)

        # CFG: learnable null embedding for unconditional generation
        self.null_condition = nn.Parameter(torch.zeros(1, config.condition_dim))

        # Time embedding
        self.time_embed = TimeEmbedding(
            n_steps=config.n_steps,
            embed_dim=config.time_embed_dim,
        )

        # Regime embedding (only if regime conditioning enabled)
        if config.use_regime_conditioning:
            self.regime_embed = nn.Embedding(config.n_regimes, config.regime_embed_dim)
            cond_input_dim = config.condition_dim + config.regime_embed_dim + config.time_embed_dim
        else:
            self.regime_embed = None
            cond_input_dim = config.condition_dim + config.time_embed_dim

        # Project condition (+ optional regime) + time to channels
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, C * 2),
            nn.SiLU(),
            nn.Linear(C * 2, C),
        )

        # Input conv: (B, 1, T_fut, 5, 5) -> (B, C, T_fut, 5, 5)
        self.conv_in = CausalConv3d(1, C, kernel_size=3)

        # Stack of ResNet blocks with adaptive norm for time/condition
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(config.n_res_blocks):
            self.blocks.append(
                ResnetBlockCausal3D(C, C, groups=config.groups, dropout=config.dropout)
            )
            self.norms.append(
                AdaptiveGroupNorm(C, num_groups=config.groups, embed_dim=C)
            )

        # Output conv: (B, C, T_fut, 5, 5) -> (B, 1, T_fut, 5, 5)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(config.groups, C),
            nn.SiLU(),
            CausalConv3d(C, 1, kernel_size=3),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        history: torch.Tensor,
        regime_id: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """
        Predict noise in noisy future surfaces.

        Args:
            x_noisy: (B, T_fut, H, W) noisy future surfaces
            t: (B,) uniform timesteps OR (B, T_fut) per-frame timesteps (Diffusion Forcing)
            history: (B, T_hist, H, W) past surfaces for conditioning
            regime_id: (B,) regime indices for hierarchical sampling (Option J)
            force_uncond: If True, use null condition (for CFG unconditional prediction)

        Returns:
            noise_pred: (B, T_fut, H, W) predicted noise
        """
        B = x_noisy.shape[0]
        T_fut = x_noisy.shape[1]
        per_frame = (t.dim() == 2)

        # CFG: handle conditioning
        if force_uncond:
            # Unconditional prediction: use null embedding
            condition = self.null_condition.expand(B, -1)
        elif self.training and self.config.cond_drop_prob > 0:
            # Training with CFG: randomly drop conditioning
            real_condition = self.history_encoder(history)
            null_condition = self.null_condition.expand(B, -1)

            # Create drop mask: which samples use null condition
            drop_mask = torch.rand(B, device=x_noisy.device) < self.config.cond_drop_prob
            drop_mask = drop_mask.unsqueeze(-1)  # (B, 1) for broadcasting

            # Apply mask: drop_mask=True -> null, drop_mask=False -> real
            condition = torch.where(drop_mask, null_condition, real_condition)
        else:
            # Standard: encode history
            condition = self.history_encoder(history)

        # Time embedding - handles both (B,) and (B, T) shapes
        t_emb = self.time_embed(t)  # (B, time_embed_dim) or (B, T, time_embed_dim)

        # Get regime embedding if enabled
        if self.regime_embed is not None and regime_id is not None:
            regime_emb = self.regime_embed(regime_id)  # (B, regime_embed_dim)
        elif self.regime_embed is not None:
            # Regime conditioning enabled but no regime_id provided - use zeros
            regime_emb = torch.zeros(B, self.config.regime_embed_dim, device=x_noisy.device)
        else:
            regime_emb = None

        # Combine condition (+ optional regime) and time
        if per_frame:
            # Per-frame (Diffusion Forcing): broadcast condition to each frame
            # condition: (B, condition_dim) -> (B, T, condition_dim)
            condition_expanded = condition.unsqueeze(1).expand(-1, T_fut, -1)
            # t_emb: (B, T, time_embed_dim)
            if regime_emb is not None:
                regime_expanded = regime_emb.unsqueeze(1).expand(-1, T_fut, -1)
                cond_combined = torch.cat([condition_expanded, regime_expanded, t_emb], dim=2)
            else:
                cond_combined = torch.cat([condition_expanded, t_emb], dim=2)

            # Project per-frame: (B, T, ...) -> (B, T, C)
            cond_combined_flat = cond_combined.view(B * T_fut, -1)
            cond_emb_flat = self.cond_proj(cond_combined_flat)
            cond_emb = cond_emb_flat.view(B, T_fut, -1)  # (B, T, C)
        else:
            # Uniform timestep (standard DDPM)
            if regime_emb is not None:
                cond_combined = torch.cat([condition, regime_emb, t_emb], dim=1)
            else:
                cond_combined = torch.cat([condition, t_emb], dim=1)
            cond_emb = self.cond_proj(cond_combined)  # (B, C)

        # Process noisy future
        x = x_noisy.unsqueeze(1)  # (B, 1, T_fut, H, W)
        x = self.conv_in(x)  # (B, C, T_fut, H, W)

        # ResNet blocks with adaptive norm (FiLM conditioning)
        # AdaptiveGroupNorm handles both (B, C) and (B, T, C) embeddings
        for block, norm in zip(self.blocks, self.norms):
            x = block(x)
            x = norm(x, cond_emb)

        # Output
        x = self.conv_out(x)  # (B, 1, T_fut, H, W)
        noise_pred = x.squeeze(1)  # (B, T_fut, H, W)

        return noise_pred


class ConditionalDDPM(nn.Module):
    """
    Full conditional DDPM model combining scheduler and denoiser.

    This is a convenience wrapper for training and sampling.
    """

    def __init__(self, config: DenoiserConfig, scheduler_config: Optional[dict] = None):
        super().__init__()
        self.config = config
        self.denoiser = SimpleDenoiser3D(config)

        # Regime classifier for hierarchical sampling (Option J)
        if config.use_regime_conditioning:
            self.regime_classifier = RegimeClassifier(config)
        else:
            self.regime_classifier = None

        # Import scheduler here to avoid circular import
        from diffusion.ddpm_scheduler import DDPMScheduler

        scheduler_config = scheduler_config or {}
        self.scheduler = DDPMScheduler(
            n_steps=config.n_steps,
            schedule=scheduler_config.get('schedule', 'cosine'),
            device=scheduler_config.get('device', 'cpu'),
        )

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Training forward pass.

        Args:
            history: (B, T_hist, H, W) past surfaces
            future: (B, T_fut, H, W) future surfaces (clean)
            regime_ids: (B,) regime labels for hierarchical sampling (Option J)

        Returns:
            dict with 'loss', 'noise_pred', 'noise', and optionally:
            - 'snr' for Diffusion Forcing
            - 'regime_loss', 'regime_acc' for regime conditioning
        """
        B = history.shape[0]
        T_fut = future.shape[1]
        device = history.device

        # Generate noise
        noise = torch.randn_like(future)

        # Sample timesteps and add noise based on noise schedule
        if self.config.noise_schedule == 'independent':
            # Diffusion Forcing: independent timestep per frame
            # Each frame gets its own noise level τ_i ~ Uniform(0, T)
            t = self.scheduler.sample_independent_timesteps(B, T_fut, device)  # (B, T)
            x_noisy, _ = self.scheduler.q_sample_per_frame(future, t, noise)

        elif self.config.noise_schedule == 'structured_causal':
            # Structured Causal Noise (Option G): shared base + progressive spread
            # Adjacent frames differ by ~spread_scale/n_frames, preserving constraints
            t = self.scheduler.sample_structured_causal_timesteps(
                B, T_fut, device,
                spread_scale=self.config.structured_spread_scale,
            )
            x_noisy, _ = self.scheduler.q_sample_per_frame(future, t, noise)

        elif self.config.noise_schedule == 'erdm_progressive':
            # ERDM Progressive Schedule (Option H): position-dependent noise
            # Frame 0 gets less noise, Frame T-1 gets more noise
            t = self.scheduler.sample_erdm_timesteps(
                B, T_fut, device,
                sigma_min=self.config.erdm_sigma_min,
                sigma_max=self.config.erdm_sigma_max,
                rho=self.config.erdm_rho,
            )
            x_noisy, _ = self.scheduler.q_sample_per_frame(future, t, noise)

        else:
            # Standard DDPM: uniform timestep for all frames
            t = torch.randint(0, self.config.n_steps, (B,), device=device)
            x_noisy, _ = self.scheduler.q_sample(future, t, noise)

        # Predict noise - denoiser handles both (B,) and (B, T) timesteps
        # Pass regime_ids if provided (for regime-conditioned generation)
        noise_pred = self.denoiser(x_noisy, t, history, regime_id=regime_ids)

        # MSE loss on noise
        loss = F.mse_loss(noise_pred, noise)

        result = {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise': noise,
        }

        # Add SNR info for per-frame schedules analysis
        if self.config.noise_schedule in ['independent', 'structured_causal', 'erdm_progressive']:
            result['snr'] = self.scheduler.get_per_frame_snr(t)

        # Regime classification loss (if labels provided and classifier exists)
        if regime_ids is not None and self.regime_classifier is not None:
            # Encode history for classifier
            # NOTE: no_grad() prevents regime loss from affecting history encoder
            # The encoder is trained only by diffusion loss; classifier adapts to encoder features
            with torch.no_grad():
                history_cond = self.denoiser.history_encoder(history)
            regime_logits = self.regime_classifier(history_cond)
            regime_loss = F.cross_entropy(regime_logits, regime_ids)
            loss = loss + self.config.regime_loss_weight * regime_loss
            result['loss'] = loss
            result['regime_loss'] = regime_loss.item()
            result['regime_acc'] = (regime_logits.argmax(dim=1) == regime_ids).float().mean().item()

        return result

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        n_samples: int = 1,
        sampler: str = 'ddpm',
        n_inference_steps: int = 20,
        max_residual_timestep: int = 20,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate future surface samples.

        Args:
            history: (B, T_hist, H, W) past surfaces
            n_samples: Number of samples to generate per history
            sampler: Sampling method:
                - 'ddpm': Standard DDPM (all steps, uniform timesteps)
                - 'ddim': Fast DDIM (step-skipping, uniform timesteps)
                - 'ddpm_staggered': Gold standard for per-frame training (exact DDPM
                  with staggered timesteps - each frame denoises to its own t_min)
                - 'ddim_staggered': Fast approximation for per-frame training
            n_inference_steps: For DDIM/staggered, number of denoising steps (default: 20)
            max_residual_timestep: For staggered, t_min for last frame (default: 20)
                Controls uncertainty growth - higher = more noise in later frames
            guidance_scale: CFG guidance scale (1.0 = no guidance, >1.0 = stronger conditioning)

        Returns:
            samples: (B, n_samples, T_fut, H, W) generated futures
        """
        B = history.shape[0]
        device = history.device
        T_fut = self.config.future_len
        H, W = self.config.surface_h, self.config.surface_w

        # Update scheduler device if needed
        if self.scheduler.device != device:
            self.scheduler = self._move_scheduler_to_device(device)

        # Auto-select appropriate sampler for per-frame training schedules
        # Models trained with per-frame timesteps need staggered sampling to match
        per_frame_schedules = ['independent', 'structured_causal', 'erdm_progressive']
        if self.config.noise_schedule in per_frame_schedules and sampler in ['ddim', 'ddpm']:
            import warnings
            # Prefer ddpm_staggered (exact) over ddim_staggered (approximation)
            new_sampler = 'ddpm_staggered'
            warnings.warn(
                f"Model trained with '{self.config.noise_schedule}' uses per-frame timesteps. "
                f"Switching sampler from '{sampler}' to '{new_sampler}' for correct inference."
            )
            sampler = new_sampler

        samples = []
        for _ in range(n_samples):
            if sampler == 'ddpm_staggered':
                # Staggered DDPM: exact reverse diffusion for per-frame training
                # Frame 0 → t=0 (clean), Frame T-1 → t=max_residual (noisy)
                # This is the gold standard - mathematically exact unlike DDIM
                shape = (B, T_fut, H, W)
                x_0 = self.scheduler.sample_ddpm_staggered(
                    self.denoiser, history, shape,
                    max_residual_timestep=max_residual_timestep,
                )
                samples.append(x_0)
            elif sampler == 'ddim_staggered':
                # Staggered DDIM: fast approximation for per-frame training
                # Frame 0 → t=0 (clean), Frame T-1 → t=max_residual (noisy)
                shape = (B, T_fut, H, W)
                x_0 = self.scheduler.sample_ddim_staggered(
                    self.denoiser, history, shape,
                    n_inference_steps=n_inference_steps,
                    max_residual_timestep=max_residual_timestep,
                )
                samples.append(x_0)
            elif sampler == 'ddim':
                # Fast DDIM sampling with step-skipping (uniform timesteps)
                shape = (B, T_fut, H, W)
                x_0 = self.scheduler.sample_ddim(
                    self.denoiser, history, shape,
                    n_inference_steps=n_inference_steps,
                    guidance_scale=guidance_scale,
                )
                samples.append(x_0)
            else:
                # Standard DDPM sampling (all steps)
                x_t = torch.randn(B, T_fut, H, W, device=device)

                # Reverse diffusion
                for t_val in reversed(range(self.config.n_steps)):
                    t = torch.full((B,), t_val, device=device, dtype=torch.long)
                    x_t = self.scheduler.p_sample(self.denoiser, x_t, t, history)

                samples.append(x_t)

        # Stack samples: (B, n_samples, T_fut, H, W)
        samples = torch.stack(samples, dim=1)

        # Denormalize from [-1, 1] to [0.05, 1.0] (standard DDPM practice)
        samples = denormalize_iv(samples)

        # For staggered sampling, clamp to valid IV range [0, 1]
        # (Staggered leaves residual noise that can produce values outside range)
        if sampler in ['ddim_staggered', 'ddpm_staggered']:
            samples = samples.clamp(0.0, 1.0)

        return samples

    def _move_scheduler_to_device(self, device):
        """Create new scheduler on correct device."""
        from diffusion.ddpm_scheduler import DDPMScheduler
        return DDPMScheduler(
            n_steps=self.config.n_steps,
            schedule='cosine',
            device=device,
        )

    @torch.no_grad()
    def sample_hierarchical(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_inference_steps: int = 20,
        temperature: float = 1.0,
    ) -> tuple:
        """Hierarchical sampling: sample regime first, then trajectory (Option J).

        This enables generating diverse trajectories that explicitly sample from
        different market regimes (calm, crisis, spike, etc.) rather than averaging
        them out.

        Args:
            history: (B, T_hist, H, W) past surfaces
            n_samples: Number of samples to generate per history
            n_inference_steps: Number of DDIM denoising steps
            temperature: Temperature for regime sampling (higher = more diverse)

        Returns:
            samples: (B, n_samples, T_fut, H, W) generated futures
            regimes: (B, n_samples) sampled regime for each trajectory
        """
        if not self.config.use_regime_conditioning:
            raise ValueError("Hierarchical sampling requires use_regime_conditioning=True")

        if self.regime_classifier is None:
            raise ValueError("Regime classifier not initialized")

        B = history.shape[0]
        device = history.device
        T_fut = self.config.future_len
        H, W = self.config.surface_h, self.config.surface_w

        # Update scheduler device if needed
        if self.scheduler.device != device:
            self.scheduler = self._move_scheduler_to_device(device)

        # Encode history for regime classification
        history_cond = self.denoiser.history_encoder(history)  # (B, condition_dim)

        # Get regime probabilities from classifier
        regime_logits = self.regime_classifier(history_cond)  # (B, n_regimes)
        regime_probs = F.softmax(regime_logits / temperature, dim=-1)

        # Sample regimes for each trajectory
        # Expand probs: (B, K) -> (B*n_samples, K), sample, reshape to (B, n_samples)
        probs_expanded = regime_probs.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, self.config.n_regimes)
        regimes = torch.multinomial(probs_expanded, num_samples=1).reshape(B, n_samples)  # (B, n_samples)

        # Generate trajectories conditioned on sampled regimes
        samples = []
        for s in range(n_samples):
            regime_id = regimes[:, s]  # (B,)

            # Standard DDPM sampling with regime conditioning
            x_t = torch.randn(B, T_fut, H, W, device=device)

            # Reverse diffusion with regime conditioning
            for t_val in reversed(range(self.config.n_steps)):
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                # Get noise prediction with regime conditioning
                noise_pred = self.denoiser(x_t, t, history, regime_id=regime_id)
                x_t = self.scheduler.p_sample_with_pred(x_t, t, noise_pred)

            samples.append(x_t)

        # Stack samples: (B, n_samples, T_fut, H, W)
        samples = torch.stack(samples, dim=1)

        # Denormalize from [-1, 1] to [0, 1]
        samples = denormalize_iv(samples)

        return samples, regimes

    @torch.no_grad()
    def sample_variable_length(
        self,
        history: torch.Tensor,
        n_chunks: int = 3,
        n_samples: int = 50,
        n_inference_steps: int = 20,
        temperature: float = 1.0,
    ) -> tuple:
        """Generate variable-length trajectories via chunk-and-shift (Option J).

        Generates multi-chunk trajectories by:
        1. Generate first 30-day chunk with hierarchical regime sampling
        2. Use end of chunk as history for next chunk
        3. Repeat with regime resampling each chunk

        This enables regime transitions within a single trajectory.

        Args:
            history: (B, T_hist, H, W) past surfaces
            n_chunks: Number of 30-day chunks to generate
            n_samples: Number of trajectory samples per history
            n_inference_steps: DDIM steps per chunk
            temperature: Temperature for regime sampling

        Returns:
            trajectories: (B, n_samples, n_chunks * T_fut, H, W)
            regimes: (B, n_samples, n_chunks) regime per chunk
        """
        if not self.config.use_regime_conditioning:
            raise ValueError("Variable-length generation requires use_regime_conditioning=True")

        B = history.shape[0]
        chunks = []
        all_regimes = []
        current_history = history

        for chunk_idx in range(n_chunks):
            # Sample this chunk with hierarchical regime
            chunk_samples, chunk_regimes = self.sample_hierarchical(
                current_history, n_samples, n_inference_steps, temperature
            )
            chunks.append(chunk_samples)
            all_regimes.append(chunk_regimes)

            # Use end of this chunk as history for next
            # Take last history_len days of first sample as new history
            # Note: chunk_samples is in [0, 1] after denormalize, need to re-normalize for next iteration
            from experiments.backfill.diffusion_poc.train_ddpm_poc import normalize_iv
            current_history = normalize_iv(chunk_samples[:, 0, -self.config.history_len:, :, :])

        # Concatenate chunks along time dimension
        trajectories = torch.cat(chunks, dim=2)  # (B, n_samples, n_chunks*T_fut, H, W)
        regimes = torch.stack(all_regimes, dim=2)  # (B, n_samples, n_chunks)

        return trajectories, regimes


def test_simple_denoiser():
    """Unit tests for SimpleDenoiser3D."""
    print("Testing SimpleDenoiser3D...")

    device = 'cpu'
    config = DenoiserConfig(
        history_len=30,
        future_len=30,
        surface_h=5,
        surface_w=5,
        base_channels=16,  # Small for testing
        n_res_blocks=2,
    )

    model = SimpleDenoiser3D(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Test forward pass
    B = 4
    history = torch.randn(B, config.history_len, 5, 5)
    x_noisy = torch.randn(B, config.future_len, 5, 5)
    t = torch.randint(0, config.n_steps, (B,))

    noise_pred = model(x_noisy, t, history)
    assert noise_pred.shape == (B, config.future_len, 5, 5), f"Wrong output shape: {noise_pred.shape}"
    print(f"  Forward pass: OK (output shape {noise_pred.shape})")

    # Test gradient flow
    loss = noise_pred.mean()
    loss.backward()
    # Note: null_condition doesn't get gradients unless force_uncond=True or cond_drop_prob>0
    params_with_grad = [
        (name, p) for name, p in model.named_parameters()
        if p.requires_grad and 'null_condition' not in name
    ]
    has_grad = all(p.grad is not None for name, p in params_with_grad)
    assert has_grad, "Some parameters have no gradient"
    print("  Gradient flow: OK")

    # Test full DDPM model
    print("\nTesting ConditionalDDPM...")
    ddpm = ConditionalDDPM(config)

    # Training forward
    future = torch.randn(B, config.future_len, 5, 5)
    result = ddpm(history, future)
    assert 'loss' in result, "Missing loss in output"
    print(f"  Training forward: OK (loss={result['loss'].item():.6f})")

    # Test sampling (just 2 steps to verify it runs)
    config_fast = DenoiserConfig(
        history_len=30,
        future_len=30,
        surface_h=5,
        surface_w=5,
        base_channels=16,
        n_res_blocks=2,
        n_steps=5,  # Very few steps for fast test
    )
    ddpm_fast = ConditionalDDPM(config_fast)
    samples = ddpm_fast.sample(history[:2], n_samples=2)  # 2 history, 2 samples each
    assert samples.shape == (2, 2, config.future_len, 5, 5), f"Wrong sample shape: {samples.shape}"
    print(f"  Sampling: OK (shape {samples.shape})")

    # Check sample diversity
    sample_std = samples.std(dim=1).mean()
    print(f"  Sample diversity (std across samples): {sample_std:.6f}")

    # === Per-frame (Diffusion Forcing) tests ===
    print("\nTesting Per-Frame Mode (Diffusion Forcing)...")

    # Test per-frame forward pass
    model.zero_grad()
    t_per_frame = torch.randint(0, config.n_steps, (B, config.future_len))
    noise_pred_pf = model(x_noisy, t_per_frame, history)
    assert noise_pred_pf.shape == (B, config.future_len, 5, 5), f"Wrong per-frame shape: {noise_pred_pf.shape}"
    print(f"  Per-frame forward: OK (shape {noise_pred_pf.shape})")

    # Test gradient flow with per-frame
    loss_pf = noise_pred_pf.mean()
    loss_pf.backward()
    # Note: null_condition doesn't get gradients unless force_uncond=True or cond_drop_prob>0
    params_with_grad_pf = [
        (name, p) for name, p in model.named_parameters()
        if p.requires_grad and 'null_condition' not in name
    ]
    has_grad_pf = all(p.grad is not None for name, p in params_with_grad_pf)
    assert has_grad_pf, "Some parameters have no gradient in per-frame mode"
    print("  Per-frame gradient flow: OK")

    # Verify outputs differ when using different per-frame timesteps
    model.eval()
    with torch.no_grad():
        t_low = torch.zeros(B, config.future_len, dtype=torch.long)  # All low noise
        t_high = torch.full((B, config.future_len), config.n_steps - 1, dtype=torch.long)  # All high noise
        pred_low = model(x_noisy, t_low, history)
        pred_high = model(x_noisy, t_high, history)
        assert not torch.allclose(pred_low, pred_high, atol=1e-3), \
            "Different timesteps should give different predictions"
    print("  Per-frame timestep differentiation: OK")

    # Test ConditionalDDPM with Diffusion Forcing (noise_schedule='independent')
    print("\nTesting ConditionalDDPM with Diffusion Forcing...")
    config_df = DenoiserConfig(
        history_len=30,
        future_len=30,
        surface_h=5,
        surface_w=5,
        base_channels=16,
        n_res_blocks=2,
        n_steps=20,
        noise_schedule='independent',  # Enable Diffusion Forcing
    )
    ddpm_df = ConditionalDDPM(config_df)

    # Training forward with Diffusion Forcing
    future = torch.randn(B, config_df.future_len, 5, 5)
    result_df = ddpm_df(history, future)
    assert 'loss' in result_df, "Missing loss in Diffusion Forcing output"
    assert 'snr' in result_df, "Missing SNR in Diffusion Forcing output"
    assert result_df['snr'].shape == (B, config_df.future_len), \
        f"Wrong SNR shape: {result_df['snr'].shape}, expected ({B}, {config_df.future_len})"
    print(f"  Diffusion Forcing forward: OK (loss={result_df['loss'].item():.6f})")
    print(f"  SNR shape: {result_df['snr'].shape} (min={result_df['snr'].min():.4f}, max={result_df['snr'].max():.4f})")

    print("\nSimpleDenoiser3D tests passed!\n")


if __name__ == "__main__":
    test_simple_denoiser()
