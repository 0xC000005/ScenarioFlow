"""
DDPM Scheduler for Volatility Surface Diffusion.

Implements the forward and reverse diffusion processes following:
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)

The scheduler handles:
1. Forward process q(x_t | x_0): adding noise according to a schedule
2. Reverse process p(x_{t-1} | x_t): denoising given model predictions
3. Full sampling loop from x_T ~ N(0,I) to x_0
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class DDPMScheduler:
    """
    DDPM diffusion scheduler.

    Supports linear and cosine noise schedules.
    """

    def __init__(
        self,
        n_steps: int = 100,
        schedule: str = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        device: str = 'cpu',
    ):
        """
        Args:
            n_steps: Number of diffusion timesteps
            schedule: 'linear' or 'cosine' beta schedule
            beta_start: Starting beta for linear schedule
            beta_end: Ending beta for linear schedule
            clip_sample: Whether to clip samples during sampling
            clip_sample_range: Clip range (for normalized IV surfaces, use ~1.0)
            device: Device to place tensors on
        """
        self.n_steps = n_steps
        self.schedule = schedule
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.device = device

        # Compute beta schedule
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, n_steps)
        elif schedule == 'cosine':
            betas = self._cosine_beta_schedule(n_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas.to(device)

        # Pre-compute alpha and related quantities
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bar[:-1]])

        # Pre-compute quantities for q(x_t | x_0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # Pre-compute quantities for p(x_{t-1} | x_t)
        # Posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        # Clamp to avoid log(0)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance = torch.log(self.posterior_variance)

        # Coefficients for x_0 prediction from noise prediction
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recip_alpha_bar_minus_one = torch.sqrt(1.0 / self.alpha_bar - 1.0)

    def _cosine_beta_schedule(self, n_steps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine beta schedule from Nichol & Dhariwal (2021).

        More gradual noise addition than linear schedule, especially at the start.
        """
        steps = n_steps + 1
        t = torch.linspace(0, n_steps, steps)
        f_t = torch.cos(((t / n_steps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(betas, 0.0001, 0.999)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: sample x_t given x_0.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x_0: Clean samples (B, ...)
            t: Timesteps (B,) integers in [0, n_steps)
            noise: Optional pre-sampled noise

        Returns:
            x_t: Noisy samples (B, ...)
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get coefficients for each sample in batch
        sqrt_alpha_bar_t = self._gather(self.sqrt_alpha_bar, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = self._gather(self.sqrt_one_minus_alpha_bar, t, x_0.shape)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)

        Args:
            x_t: Noisy samples (B, ...)
            t: Timesteps (B,)
            noise_pred: Predicted noise (B, ...)

        Returns:
            x_0_pred: Predicted clean samples (B, ...)
        """
        sqrt_recip_alpha_bar_t = self._gather(self.sqrt_recip_alpha_bar, t, x_t.shape)
        sqrt_recip_alpha_bar_minus_one_t = self._gather(
            self.sqrt_recip_alpha_bar_minus_one, t, x_t.shape
        )

        x_0_pred = sqrt_recip_alpha_bar_t * x_t - sqrt_recip_alpha_bar_minus_one_t * noise_pred

        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -self.clip_sample_range, self.clip_sample_range)

        return x_0_pred

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single denoising step: sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            model: Denoiser network that predicts noise
            x_t: Current noisy samples (B, ...)
            t: Current timesteps (B,)
            condition: Conditioning context (e.g., history encoding)

        Returns:
            x_{t-1}: Slightly less noisy samples (B, ...)
        """
        # Predict noise
        noise_pred = model(x_t, t, condition)

        # Predict x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

        # Compute mean of p(x_{t-1} | x_t, x_0)
        # mu = (sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)) * x_0
        #    + (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x_t
        alpha_t = self._gather(self.alphas, t, x_t.shape)
        alpha_bar_t = self._gather(self.alpha_bar, t, x_t.shape)
        alpha_bar_prev_t = self._gather(self.alpha_bar_prev, t, x_t.shape)
        beta_t = self._gather(self.betas, t, x_t.shape)

        coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)

        mean = coef_x0 * x_0_pred + coef_xt * x_t

        # Sample (except at t=0, where we return the mean)
        noise = torch.randn_like(x_t)
        posterior_variance_t = self._gather(self.posterior_variance, t, x_t.shape)

        # Mask: at t=0, we don't add noise
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))

        x_prev = mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

        return x_prev

    def p_sample_with_pred(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single denoising step with pre-computed noise prediction.

        This is useful when the caller needs to compute the noise prediction
        with additional conditioning (e.g., regime_id) that the scheduler
        doesn't know about.

        Args:
            x_t: Current noisy samples (B, ...)
            t: Current timesteps (B,)
            noise_pred: Pre-computed noise prediction (B, ...)

        Returns:
            x_{t-1}: Slightly less noisy samples (B, ...)
        """
        # Predict x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

        # Compute mean of p(x_{t-1} | x_t, x_0)
        alpha_t = self._gather(self.alphas, t, x_t.shape)
        alpha_bar_t = self._gather(self.alpha_bar, t, x_t.shape)
        alpha_bar_prev_t = self._gather(self.alpha_bar_prev, t, x_t.shape)
        beta_t = self._gather(self.betas, t, x_t.shape)

        coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)

        mean = coef_x0 * x_0_pred + coef_xt * x_t

        # Sample (except at t=0, where we return the mean)
        noise = torch.randn_like(x_t)
        posterior_variance_t = self._gather(self.posterior_variance, t, x_t.shape)

        # Mask: at t=0, we don't add noise
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))

        x_prev = mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Full reverse diffusion: sample from x_T ~ N(0,I) to x_0.

        Args:
            model: Denoiser network
            condition: Conditioning context (B, ...)
            shape: Shape of samples to generate (B, T, H, W) for vol surfaces
            return_intermediates: If True, return all intermediate x_t

        Returns:
            x_0: Generated clean samples (B, ...)
            intermediates: (optional) List of intermediate samples
        """
        device = condition.device
        B = condition.shape[0]

        # Start from pure noise
        x_t = torch.randn(B, *shape[1:], device=device)

        intermediates = [] if return_intermediates else None

        # Reverse diffusion: t = T-1, T-2, ..., 0
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t_batch, condition)

            if return_intermediates:
                intermediates.append(x_t.clone())

        if return_intermediates:
            return x_t, intermediates
        return x_t

    def _gather(self, values: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Gather values at timestep t and reshape for broadcasting.

        Args:
            values: 1D tensor of values indexed by timestep
            t: Timestep indices (B,)
            shape: Target tensor shape for broadcasting

        Returns:
            values_t: Values at timesteps t, reshaped for broadcasting (B, 1, 1, ...)
        """
        out = values.gather(0, t)
        # Reshape to (B, 1, 1, ...) for broadcasting
        return out.view(-1, *([1] * (len(shape) - 1)))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-noise ratio at timestep t.

        SNR(t) = alpha_bar_t / (1 - alpha_bar_t)

        Useful for weighting losses (e.g., arbitrage penalty at high SNR).

        Args:
            t: Timesteps (B,)

        Returns:
            snr_t: SNR values (B,)
        """
        alpha_bar_t = self.alpha_bar[t]
        return alpha_bar_t / (1.0 - alpha_bar_t)

    # =========================================================================
    # Diffusion Forcing: Per-frame independent noise levels
    # =========================================================================

    def sample_independent_timesteps(
        self,
        batch_size: int,
        n_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample independent timesteps for each frame (Diffusion Forcing).

        Each frame gets its own noise level, sampled uniformly from [0, n_steps).
        This teaches the model to denoise frames at different noise levels
        simultaneously, enabling natural uncertainty growth across horizons.

        Args:
            batch_size: Number of samples in batch
            n_frames: Number of frames (temporal dimension)
            device: Device to place tensor on

        Returns:
            t: Independent timesteps (B, T) where each t[b, i] ~ Uniform(0, n_steps)
        """
        return torch.randint(0, self.n_steps, (batch_size, n_frames), device=device)

    def sample_structured_causal_timesteps(
        self,
        batch_size: int,
        n_frames: int,
        device: torch.device,
        spread_scale: float = 50.0,
    ) -> torch.Tensor:
        """
        Sample structured causal timesteps (Option G).

        Each batch item shares a base timestep with progressive spread across frames.
        Adjacent frames differ by only ~spread_scale/n_frames noise steps, preserving
        temporal constraints while teaching uncertainty growth.

        Args:
            batch_size: Number of samples in batch
            n_frames: Number of frames (temporal dimension)
            device: Device to place tensor on
            spread_scale: Total spread across all frames (default: 50)

        Returns:
            t: Structured timesteps (B, T) where t[b, i] = base_t[b] + spread * (i / (n_frames - 1))
        """
        # Warn if spread_scale is too large for n_steps
        if spread_scale >= self.n_steps:
            import warnings
            warnings.warn(
                f"spread_scale ({spread_scale}) >= n_steps ({self.n_steps}). "
                f"This will cause base_t to always be 0. Consider spread_scale < n_steps * 0.5"
            )

        # Sample base timestep for each batch item
        # Use range that allows full spread: [0, n_steps - spread_scale - 1]
        max_base = max(1, self.n_steps - int(spread_scale) - 1)
        base_t = torch.randint(0, max_base, (batch_size, 1), device=device).float()

        # Create frame indices
        frame_idx = torch.arange(n_frames, device=device).float()

        # Compute spread per frame: spread_scale * (i / (n_frames - 1))
        spread = spread_scale * frame_idx / (n_frames - 1)  # (T,)

        # Add spread to base: t[b, i] = base_t[b] + spread[i]
        t = base_t + spread.unsqueeze(0)  # (B, T)

        # Clamp to valid range and convert to long
        t = t.clamp(0, self.n_steps - 1).long()

        return t

    def sample_erdm_timesteps(
        self,
        batch_size: int,
        n_frames: int,
        device: torch.device,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = -10.0,
    ) -> torch.Tensor:
        """
        Sample ERDM progressive timesteps (Option H).

        Implements the snapshot-dependent noise schedule from ERDM (arXiv:2506.20024).
        Each frame gets a noise level based on its position in the forecast window.
        Frame 0 (nearest) gets lower noise, frame T-1 (furthest) gets higher noise.

        ERDM formula:
            sigma_bar(w) = (sigma_max^(1/rho) + t_w * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
            where t_w = w / (W - 1) is frame position in [0, 1]

        Args:
            batch_size: Number of samples in batch
            n_frames: Number of frames (window size W)
            device: Device to place tensor on
            sigma_min: Minimum noise level for near frames (default: 0.002)
            sigma_max: Maximum noise level for far frames (default: 80)
            rho: Schedule curvature (default: -10, negative unlike EDM's +7)

        Returns:
            t: ERDM timesteps (B, T) with position-dependent noise levels
        """
        # Warn if sigma_max is too large for typical n_steps
        if sigma_max > 100:
            import warnings
            warnings.warn(
                f"sigma_max ({sigma_max}) is large for n_steps={self.n_steps}. "
                f"This may cause extreme timestep mappings. Consider sigma_max < 100 for n_steps=100"
            )

        # Compute position-dependent sigma for each frame
        # frame_positions: 0 = near (low noise), 1 = far (high noise)
        frame_positions = torch.arange(n_frames, device=device).float() / (n_frames - 1)

        # ERDM formula (adapted): sigma increases with frame position
        # Original ERDM: sigma_bar = (sigma_max^(1/rho) + t * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        # where t=0 gives sigma_max (noisy) and t=1 gives sigma_min (clean)
        # We invert by using (1 - frame_positions) so frame 0 is clean and frame T-1 is noisy
        sigma_max_pow = sigma_max ** (1.0 / rho)
        sigma_min_pow = sigma_min ** (1.0 / rho)
        inverted_positions = 1.0 - frame_positions  # frame 0 -> 1 (clean), frame T-1 -> 0 (noisy)

        sigma_bar = (sigma_max_pow + inverted_positions * (sigma_min_pow - sigma_max_pow)) ** rho  # (T,)

        # Convert sigma to alpha_bar: alpha_bar = 1 / (1 + sigma^2)
        alpha_bar_target = 1.0 / (1.0 + sigma_bar ** 2)  # (T,)

        # Find closest timestep t for each frame's alpha_bar
        # self.alpha_bar is (n_steps,), alpha_bar_target is (T,)
        alpha_bar_diff = torch.abs(self.alpha_bar.unsqueeze(1) - alpha_bar_target.unsqueeze(0))  # (n_steps, T)
        t_frame = alpha_bar_diff.argmin(dim=0)  # (T,) - closest timestep per frame

        # Expand to batch and add small random offset for stochasticity
        t_frame_expanded = t_frame.unsqueeze(0).expand(batch_size, -1)  # (B, T)

        # Add small random offset (-10 to +10) for training diversity
        # Use shared offset per batch item to preserve progressive structure
        offset = torch.randint(-10, 11, (batch_size, 1), device=device)
        t = (t_frame_expanded + offset).clamp(0, self.n_steps - 1)

        return t

    def q_sample_per_frame(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion with per-frame timesteps (Diffusion Forcing).

        Each frame can have a different noise level, allowing the model to learn
        to denoise "anchor" frames (low noise) while predicting "uncertain" frames
        (high noise) in the same forward pass.

        Args:
            x_0: Clean samples (B, T, H, W) - 4D tensor with temporal dimension
            t: Per-frame timesteps (B, T) - each frame has its own timestep
            noise: Optional pre-sampled noise (B, T, H, W)

        Returns:
            x_t: Noisy samples with per-frame noise levels (B, T, H, W)
            noise: The noise that was added (B, T, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        B, T, H, W = x_0.shape
        assert t.shape == (B, T), f"Expected t shape (B, T)=({B}, {T}), got {t.shape}"

        # Gather coefficients for each (batch, frame) pair
        # sqrt_alpha_bar has shape (n_steps,)
        # t has shape (B, T)
        # We need output shape (B, T, 1, 1) for broadcasting with (B, T, H, W)

        # Flatten t to index into 1D arrays, then reshape
        t_flat = t.flatten()  # (B * T,)
        sqrt_alpha_bar_flat = self.sqrt_alpha_bar[t_flat]  # (B * T,)
        sqrt_one_minus_alpha_bar_flat = self.sqrt_one_minus_alpha_bar[t_flat]  # (B * T,)

        # Reshape to (B, T, 1, 1) for broadcasting
        sqrt_alpha_bar_t = sqrt_alpha_bar_flat.view(B, T, 1, 1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_flat.view(B, T, 1, 1)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def get_per_frame_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get SNR for per-frame timesteps.

        Args:
            t: Per-frame timesteps (B, T)

        Returns:
            snr: Per-frame SNR values (B, T)
        """
        B, T = t.shape
        t_flat = t.flatten()
        alpha_bar_flat = self.alpha_bar[t_flat]
        snr_flat = alpha_bar_flat / (1.0 - alpha_bar_flat)
        return snr_flat.view(B, T)

    def p_sample_per_frame(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        t_min: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single DDPM denoising step with per-frame timesteps (B, T).

        Uses the exact DDPM posterior: p(x_{t-1} | x_t, x_0) = N(μ, σ²)
        Unlike DDIM (which is an approximation), this is mathematically exact.

        Args:
            model: Denoiser network that predicts noise
            x_t: Current noisy samples (B, T, H, W)
            t: Current per-frame timesteps (B, T)
            condition: History for conditioning (B, T_hist, H, W)
            t_min: Optional per-frame minimum timesteps (T,) - frames at t_min freeze

        Returns:
            x_{t-1}: Slightly less noisy samples (B, T, H, W)
        """
        B, T_frames, H, W = x_t.shape

        # 1. Predict noise with per-frame timesteps
        noise_pred = model(x_t, t, condition)

        # 2. Predict x_0 per-frame
        sqrt_recip_alpha_bar = self._gather_per_frame(self.sqrt_recip_alpha_bar, t, x_t.shape)
        sqrt_recip_alpha_bar_m1 = self._gather_per_frame(self.sqrt_recip_alpha_bar_minus_one, t, x_t.shape)
        x_0_pred = sqrt_recip_alpha_bar * x_t - sqrt_recip_alpha_bar_m1 * noise_pred

        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -self.clip_sample_range, self.clip_sample_range)

        # 3. Compute DDPM posterior mean coefficients
        alpha_t = self._gather_per_frame(self.alphas, t, x_t.shape)
        alpha_bar_t = self._gather_per_frame(self.alpha_bar, t, x_t.shape)
        alpha_bar_prev_t = self._gather_per_frame(self.alpha_bar_prev, t, x_t.shape)
        beta_t = self._gather_per_frame(self.betas, t, x_t.shape)

        # Posterior mean: μ = coef_x0 * x_0_pred + coef_xt * x_t
        coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
        mean = coef_x0 * x_0_pred + coef_xt * x_t

        # 4. Sample with posterior variance
        noise = torch.randn_like(x_t)
        posterior_variance_t = self._gather_per_frame(self.posterior_variance, t, x_t.shape)

        # Mask: at t=0, we don't add noise
        nonzero_mask = (t != 0).float().unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)

        x_prev = mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

        # 5. Freeze frames that have reached their t_min (if provided)
        if t_min is not None:
            t_min_expanded = t_min.view(1, T_frames, 1, 1).expand(B, -1, 1, 1)
            should_update = (t.unsqueeze(-1).unsqueeze(-1) > t_min_expanded)  # (B, T, 1, 1)
            x_prev = torch.where(should_update, x_prev, x_t)

        return x_prev

    @torch.no_grad()
    def sample_ddpm_staggered(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        max_residual_timestep: int = 20,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Full DDPM reverse diffusion with staggered per-frame timesteps.

        Unlike standard DDPM where all frames share the same timestep, staggered
        sampling gives each frame its own denoising schedule:
        - Frame 0: Fully denoised (t → 0), producing clean output
        - Frame T-1: Partially denoised (t → max_residual_timestep), retaining noise

        This is the mathematically exact version of staggered sampling (vs DDIM
        which is an approximation). Use this as the gold standard for models
        trained with per-frame noise schedules (structured_causal, erdm_progressive).

        Args:
            model: Denoiser network (must support per-frame timesteps)
            condition: History conditioning (B, T_hist, H, W)
            shape: Output shape (B, T_fut, H, W)
            max_residual_timestep: t_min for last frame (controls uncertainty growth)
            return_intermediates: Whether to return intermediate states

        Returns:
            x_0: Generated samples (B, T_fut, H, W)
                 - Frame 0 is clean (denoised to t=0)
                 - Frame T-1 has residual noise (stopped at t=max_residual_timestep)
        """
        device = condition.device
        B = condition.shape[0]
        T_fut = shape[1]

        # Step 1: Compute per-frame minimum timesteps
        # t_min[i] = max_residual * (i / (T-1))
        # Frame 0 → t_min=0, Frame T-1 → t_min=max_residual
        # Clamp max_residual to n_steps-1 to prevent out-of-bounds
        effective_max_residual = min(max_residual_timestep, self.n_steps - 1)
        frame_idx = torch.arange(T_fut, dtype=torch.float32, device=device)
        t_min = (effective_max_residual * frame_idx / (T_fut - 1)).long()  # (T,)

        # Step 2: Initialize from pure noise
        x_t = torch.randn(B, *shape[1:], device=device)

        intermediates = [] if return_intermediates else None

        # Step 3: Full DDPM reverse diffusion (all n_steps)
        for t_global in reversed(range(self.n_steps)):
            # Per-frame current timestep: max(global_t, t_min[i])
            # Frames that have reached t_min stay at t_min
            t_current = torch.full((B, T_fut), t_global, device=device, dtype=torch.long)
            t_current = torch.maximum(t_current, t_min.unsqueeze(0).expand(B, -1))

            # DDPM step with per-frame timesteps
            x_t = self.p_sample_per_frame(model, x_t, t_current, condition, t_min)

            if return_intermediates:
                intermediates.append(x_t.clone())

        if return_intermediates:
            return x_t, intermediates
        return x_t

    def ddim_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single DDIM denoising step from timestep t to t_prev.

        DDIM is deterministic given the same initial noise, and can skip
        multiple timesteps (e.g., go from t=80 to t=60 directly).

        Args:
            model: Denoiser network that predicts noise
            x_t: Current noisy samples (B, ...)
            t: Current timesteps (B,)
            t_prev: Target timesteps (B,), can be -1 for final step
            condition: Conditioning context

        Returns:
            x_{t_prev}: Less noisy samples (B, ...)
        """
        # Predict noise
        noise_pred = model(x_t, t, condition)

        # Predict x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

        # Get alpha_bar values
        alpha_bar_t = self._gather(self.alpha_bar, t, x_t.shape)

        # Handle t_prev = -1 case (final step to x_0)
        # When t_prev < 0, alpha_bar_prev = 1.0 (no noise)
        t_prev_clamped = t_prev.clamp(min=0)
        alpha_bar_t_prev = self._gather(self.alpha_bar, t_prev_clamped, x_t.shape)
        # Set alpha_bar_prev = 1.0 where t_prev < 0
        alpha_bar_t_prev = torch.where(
            t_prev.view(-1, *([1] * (x_t.dim() - 1))) >= 0,
            alpha_bar_t_prev,
            torch.ones_like(alpha_bar_t_prev)
        )

        # DDIM update (deterministic)
        # x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred + sqrt(1 - alpha_bar_{t-1}) * noise_pred
        x_prev = (
            torch.sqrt(alpha_bar_t_prev) * x_0_pred +
            torch.sqrt(1.0 - alpha_bar_t_prev) * noise_pred
        )

        return x_prev

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        n_inference_steps: int = 20,
        return_intermediates: bool = False,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        DDIM sampling with step-skipping for faster inference.

        Unlike DDPM which requires all n_steps, DDIM can use fewer steps
        (e.g., 20 instead of 100) by skipping intermediate timesteps.

        Args:
            model: Denoiser network
            condition: Conditioning context (B, ...) - typically history tensor
            shape: Shape of samples to generate (B, T, H, W)
            n_inference_steps: Number of denoising steps (can be << n_steps)
            return_intermediates: If True, return all intermediate x_t
            guidance_scale: CFG guidance scale (1.0 = no guidance, >1.0 = stronger conditioning)

        Returns:
            x_0: Generated clean samples (B, ...)
        """
        device = condition.device
        B = condition.shape[0]

        # Create evenly-spaced timestep schedule
        # e.g., if n_steps=100 and n_inference_steps=20: [95, 90, 85, ..., 5, 0]
        step_ratio = self.n_steps / n_inference_steps
        timesteps = [int((self.n_steps - 1) - i * step_ratio) for i in range(n_inference_steps)]
        timesteps = [max(0, t) for t in timesteps]  # Ensure non-negative

        # Start from pure noise
        x_t = torch.randn(B, *shape[1:], device=device)

        intermediates = [] if return_intermediates else None

        # DDIM loop
        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Next timestep (or -1 if this is the last step)
            if i + 1 < len(timesteps):
                t_prev_val = timesteps[i + 1]
            else:
                t_prev_val = -1  # Signal final step
            t_prev = torch.full((B,), t_prev_val, device=device, dtype=torch.long)

            # CFG: Classifier-Free Guidance
            if guidance_scale != 1.0:
                # Dual prediction for CFG
                noise_pred_cond = model(x_t, t, condition, force_uncond=False)
                noise_pred_uncond = model(x_t, t, condition, force_uncond=True)
                # CFG formula: ε = ε_uncond + guidance_scale * (ε_cond - ε_uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                # Custom DDIM step with pre-computed noise_pred
                x_t = self._ddim_step_with_noise(x_t, t, t_prev, noise_pred)
            else:
                # Standard DDIM step
                x_t = self.ddim_sample(model, x_t, t, t_prev, condition)

            if return_intermediates:
                intermediates.append(x_t.clone())

        if return_intermediates:
            return x_t, intermediates
        return x_t

    def _ddim_step_with_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single DDIM step using pre-computed noise prediction (for CFG).

        Args:
            x_t: Current noisy samples (B, ...)
            t: Current timesteps (B,)
            t_prev: Target timesteps (B,), can be -1 for final step
            noise_pred: Pre-computed noise prediction (B, ...)

        Returns:
            x_{t_prev}: Less noisy samples (B, ...)
        """
        # Predict x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

        # Get alpha_bar values
        alpha_bar_t = self._gather(self.alpha_bar, t, x_t.shape)

        # Handle t_prev = -1 case (final step to x_0)
        t_prev_clamped = t_prev.clamp(min=0)
        alpha_bar_t_prev = self._gather(self.alpha_bar, t_prev_clamped, x_t.shape)
        # Set alpha_bar_prev = 1.0 where t_prev < 0
        alpha_bar_t_prev = torch.where(
            t_prev.view(-1, *([1] * (x_t.dim() - 1))) >= 0,
            alpha_bar_t_prev,
            torch.ones_like(alpha_bar_t_prev)
        )

        # DDIM update (deterministic)
        x_prev = (
            torch.sqrt(alpha_bar_t_prev) * x_0_pred +
            torch.sqrt(1.0 - alpha_bar_t_prev) * noise_pred
        )

        return x_prev

    # =========================================================================
    # Staggered DDIM Sampling (Causal Reverse Diffusion)
    # =========================================================================

    def _gather_per_frame(
        self,
        values: torch.Tensor,
        t: torch.Tensor,
        x_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Gather values at per-frame timesteps and reshape for broadcasting.

        Unlike _gather() which handles (B,) timesteps, this handles (B, T)
        per-frame timesteps for staggered sampling.

        Args:
            values: 1D tensor of values indexed by timestep (n_steps,)
            t: Per-frame timestep indices (B, T)
            x_shape: Target tensor shape (B, T, H, W) for broadcasting

        Returns:
            values_t: Values at timesteps, reshaped to (B, T, 1, 1)
        """
        B, T_frames = t.shape
        t_flat = t.flatten()  # (B * T,)
        values_flat = values[t_flat]  # (B * T,)
        # Reshape to (B, T, 1, 1) for broadcasting with (B, T, H, W)
        return values_flat.view(B, T_frames, 1, 1)

    def ddim_sample_per_frame(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        condition: torch.Tensor,
        t_min: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single DDIM step with per-frame timesteps (staggered sampling).

        Unlike ddim_sample() which uses uniform timesteps, this handles
        per-frame timesteps for causal reverse diffusion. Frames that have
        reached their minimum timestep (t_min) are frozen and not updated.

        Args:
            model: Denoiser network that predicts noise (must support per-frame t)
            x_t: Current noisy samples (B, T, H, W)
            t: Current per-frame timesteps (B, T)
            t_prev: Target per-frame timesteps (B, T)
            condition: History for conditioning (B, T_hist, H, W)
            t_min: Per-frame minimum timesteps (T,) - frames freeze at their t_min

        Returns:
            x_prev: Updated samples (B, T, H, W), with frozen frames unchanged
        """
        B, T_frames, H, W = x_t.shape

        # Predict noise with per-frame timesteps
        noise_pred = model(x_t, t, condition)

        # Predict x_0 per-frame
        sqrt_recip_alpha_bar = self._gather_per_frame(self.sqrt_recip_alpha_bar, t, x_t.shape)
        sqrt_recip_alpha_bar_m1 = self._gather_per_frame(self.sqrt_recip_alpha_bar_minus_one, t, x_t.shape)
        x_0_pred = sqrt_recip_alpha_bar * x_t - sqrt_recip_alpha_bar_m1 * noise_pred

        # Clip x_0 prediction
        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -self.clip_sample_range, self.clip_sample_range)

        # Get alpha_bar for t_prev (handle t_prev < 0 as final step)
        t_prev_clamped = t_prev.clamp(min=0)
        alpha_bar_prev = self._gather_per_frame(self.alpha_bar, t_prev_clamped, x_t.shape)

        # Where t_prev < 0, set alpha_bar_prev = 1.0 (fully clean)
        final_step_mask = (t_prev < 0).unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        alpha_bar_prev = torch.where(final_step_mask, torch.ones_like(alpha_bar_prev), alpha_bar_prev)

        # DDIM update formula
        x_updated = (
            torch.sqrt(alpha_bar_prev) * x_0_pred +
            torch.sqrt(1.0 - alpha_bar_prev) * noise_pred
        )

        # Freeze frames that have reached their t_min
        # t_min is (T,), expand to (B, T, 1, 1)
        t_min_expanded = t_min.view(1, T_frames, 1, 1).expand(B, -1, 1, 1)
        # should_update: True if current t > t_min (frame hasn't reached minimum)
        should_update = (t.unsqueeze(-1).unsqueeze(-1) > t_min_expanded)  # (B, T, 1, 1)

        x_prev = torch.where(should_update, x_updated, x_t)

        return x_prev

    @torch.no_grad()
    def sample_ddim_staggered(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        n_inference_steps: int = 20,
        max_residual_timestep: int = 20,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        DDIM sampling with staggered per-frame timesteps (causal reverse diffusion).

        Unlike standard DDIM where all frames share the same timestep, staggered
        sampling gives each frame its own denoising schedule:
        - Frame 0: Fully denoised (t → 0), producing clean output
        - Frame T-1: Partially denoised (t → max_residual_timestep), retaining noise

        This matches Diffusion Forcing training where each frame had independent
        noise levels, fixing the train/inference mismatch.

        Args:
            model: Denoiser network (must support per-frame timesteps)
            condition: History conditioning (B, T_hist, H, W)
            shape: Output shape (B, T_fut, H, W)
            n_inference_steps: Number of DDIM steps
            max_residual_timestep: t_min for last frame (controls uncertainty growth)
            return_intermediates: Whether to return intermediate states

        Returns:
            x_0: Generated samples (B, T_fut, H, W)
                 - Frame 0 is clean (denoised to t=0)
                 - Frame T-1 has residual noise (stopped at t=max_residual_timestep)
        """
        device = condition.device
        B = condition.shape[0]
        T_fut = shape[1]

        # Step 1: Compute per-frame minimum timesteps
        # t_min[i] = max_residual * (i / (T-1))
        # Frame 0 → t_min=0, Frame T-1 → t_min=max_residual
        # Clamp max_residual to n_steps-1 to prevent out-of-bounds
        effective_max_residual = min(max_residual_timestep, self.n_steps - 1)
        frame_idx = torch.arange(T_fut, dtype=torch.float32, device=device)
        t_min = (effective_max_residual * frame_idx / (T_fut - 1)).long()  # (T,)

        # Step 2: Create global timestep schedule (same as standard DDIM)
        step_ratio = self.n_steps / n_inference_steps
        global_timesteps = [int((self.n_steps - 1) - i * step_ratio) for i in range(n_inference_steps)]
        global_timesteps = [max(0, t) for t in global_timesteps]

        # Step 3: Start from pure noise
        x_t = torch.randn(B, *shape[1:], device=device)

        intermediates = [] if return_intermediates else None

        # Step 4: Staggered DDIM loop
        for step_idx, global_t in enumerate(global_timesteps):
            # Per-frame current timestep: max(global_t, t_min[i])
            # Frames that have reached t_min stay at t_min
            t_current = torch.full((B, T_fut), global_t, device=device, dtype=torch.long)
            t_current = torch.maximum(t_current, t_min.unsqueeze(0).expand(B, -1))

            # Per-frame target timestep
            if step_idx + 1 < len(global_timesteps):
                global_t_next = global_timesteps[step_idx + 1]
                t_next = torch.full((B, T_fut), global_t_next, device=device, dtype=torch.long)
                # Clamp to t_min (frames can't go below their minimum)
                t_next = torch.maximum(t_next, t_min.unsqueeze(0).expand(B, -1))
            else:
                # Final step: frames go to t=-1 (clean) or stay at t_min
                # Frame i with t_min[i]=0 goes to t=-1 (fully clean)
                # Frame i with t_min[i]>0 stays at t_min[i] (retains noise)
                t_next = torch.where(
                    t_min.unsqueeze(0).expand(B, -1) == 0,
                    torch.full((B, T_fut), -1, device=device, dtype=torch.long),
                    t_min.unsqueeze(0).expand(B, -1)
                )

            # DDIM step with per-frame timesteps
            x_t = self.ddim_sample_per_frame(model, x_t, t_current, t_next, condition, t_min)

            if return_intermediates:
                intermediates.append(x_t.clone())

        if return_intermediates:
            return x_t, intermediates
        return x_t


def test_ddpm_scheduler():
    """Unit tests for DDPM scheduler."""
    print("Testing DDPM Scheduler...")

    device = 'cpu'
    scheduler = DDPMScheduler(n_steps=100, schedule='cosine', device=device)

    # Test q_sample (forward process)
    B, T, H, W = 4, 30, 5, 5
    x_0 = torch.randn(B, T, H, W)
    t = torch.randint(0, 100, (B,))

    x_t, noise = scheduler.q_sample(x_0, t)
    assert x_t.shape == x_0.shape, f"q_sample shape mismatch: {x_t.shape} vs {x_0.shape}"
    print(f"  q_sample: OK (shape {x_t.shape})")

    # Test noise recovery
    x_0_recovered = scheduler.predict_x0_from_noise(x_t, t, noise)
    mse = ((x_0 - x_0_recovered) ** 2).mean()
    # Note: clipping may affect this
    print(f"  x_0 recovery MSE: {mse:.6f} (should be small if no clipping)")

    # Test SNR
    snr_0 = scheduler.snr(torch.tensor([0]))
    snr_50 = scheduler.snr(torch.tensor([50]))
    snr_99 = scheduler.snr(torch.tensor([99]))
    print(f"  SNR at t=0: {snr_0.item():.2f} (high - clean)")
    print(f"  SNR at t=50: {snr_50.item():.4f}")
    print(f"  SNR at t=99: {snr_99.item():.6f} (low - noisy)")
    assert snr_0 > snr_50 > snr_99, "SNR should decrease with t"
    print("  SNR monotonicity: OK")

    # Test alpha_bar bounds
    assert scheduler.alpha_bar[0] > 0.9, f"alpha_bar_0 should be close to 1, got {scheduler.alpha_bar[0]}"
    assert scheduler.alpha_bar[-1] < 0.1, f"alpha_bar_T should be close to 0, got {scheduler.alpha_bar[-1]}"
    print(f"  alpha_bar bounds: OK (alpha_bar_0={scheduler.alpha_bar[0]:.4f}, alpha_bar_T={scheduler.alpha_bar[-1]:.4f})")

    # =========================================================================
    # Test Diffusion Forcing methods (per-frame noise)
    # =========================================================================
    print("\nTesting Diffusion Forcing methods...")

    # Test sample_independent_timesteps
    t_per_frame = scheduler.sample_independent_timesteps(B, T, device)
    assert t_per_frame.shape == (B, T), f"Expected shape ({B}, {T}), got {t_per_frame.shape}"
    assert t_per_frame.min() >= 0 and t_per_frame.max() < 100, "Timesteps out of range"
    print(f"  sample_independent_timesteps: OK (shape {t_per_frame.shape})")

    # Test q_sample_per_frame
    x_t_pf, noise_pf = scheduler.q_sample_per_frame(x_0, t_per_frame)
    assert x_t_pf.shape == x_0.shape, f"q_sample_per_frame shape mismatch: {x_t_pf.shape} vs {x_0.shape}"
    print(f"  q_sample_per_frame: OK (shape {x_t_pf.shape})")

    # Verify different frames have different noise levels
    # Frame with t=0 should be almost clean, frame with t=99 should be very noisy
    t_test = torch.zeros(1, T, dtype=torch.long)
    t_test[0, 0] = 0   # First frame: clean
    t_test[0, -1] = 99  # Last frame: noisy
    x_0_test = torch.ones(1, T, H, W)
    x_t_test, _ = scheduler.q_sample_per_frame(x_0_test, t_test)

    # Clean frame should be close to original
    clean_diff = (x_t_test[0, 0] - x_0_test[0, 0]).abs().mean()
    # Noisy frame should be far from original
    noisy_diff = (x_t_test[0, -1] - x_0_test[0, -1]).abs().mean()
    assert noisy_diff > clean_diff, f"Noisy frame should differ more: clean={clean_diff:.4f}, noisy={noisy_diff:.4f}"
    print(f"  Per-frame noise levels: OK (clean_diff={clean_diff:.4f}, noisy_diff={noisy_diff:.4f})")

    # Test get_per_frame_snr
    snr_pf = scheduler.get_per_frame_snr(t_per_frame)
    assert snr_pf.shape == (B, T), f"Expected SNR shape ({B}, {T}), got {snr_pf.shape}"
    print(f"  get_per_frame_snr: OK (shape {snr_pf.shape})")

    # =========================================================================
    # Test Structured Causal Noise (Option G)
    # =========================================================================
    print("\nTesting Structured Causal Noise (Option G)...")

    t_structured = scheduler.sample_structured_causal_timesteps(B, T, device, spread_scale=50.0)
    assert t_structured.shape == (B, T), f"Expected shape ({B}, {T}), got {t_structured.shape}"
    assert t_structured.min() >= 0 and t_structured.max() < 100, "Timesteps out of range"
    print(f"  sample_structured_causal_timesteps: OK (shape {t_structured.shape})")

    # Verify monotonic structure: later frames should have >= timesteps (shared base + spread)
    for b in range(B):
        diffs = t_structured[b, 1:] - t_structured[b, :-1]
        assert (diffs >= 0).all(), "Structured causal: later frames should have >= timesteps"
    print("  Monotonic structure: OK")
    print(f"    Sample: t[0,0]={t_structured[0,0].item()}, t[0,-1]={t_structured[0,-1].item()}, spread={t_structured[0,-1].item() - t_structured[0,0].item()}")

    # Test q_sample_per_frame with structured timesteps
    x_t_struct, _ = scheduler.q_sample_per_frame(x_0, t_structured)
    assert x_t_struct.shape == x_0.shape, f"Shape mismatch: {x_t_struct.shape} vs {x_0.shape}"
    print("  q_sample_per_frame with structured: OK")

    # =========================================================================
    # Test ERDM Progressive Schedule (Option H)
    # =========================================================================
    print("\nTesting ERDM Progressive Schedule (Option H)...")

    t_erdm = scheduler.sample_erdm_timesteps(B, T, device, sigma_min=0.002, sigma_max=80.0, rho=-10.0)
    assert t_erdm.shape == (B, T), f"Expected shape ({B}, {T}), got {t_erdm.shape}"
    assert t_erdm.min() >= 0 and t_erdm.max() < 100, "Timesteps out of range"
    print(f"  sample_erdm_timesteps: OK (shape {t_erdm.shape})")

    # Verify progressive structure: expect mean to increase with frame index
    mean_t_per_frame = t_erdm.float().mean(dim=0)  # Average across batch
    assert mean_t_per_frame[-1] > mean_t_per_frame[0], \
        f"ERDM: later frames should have higher avg timestep (got frame0={mean_t_per_frame[0]:.1f} vs frame{T-1}={mean_t_per_frame[-1]:.1f})"
    print(f"  Progressive structure: OK (mean t: frame0={mean_t_per_frame[0]:.1f}, frame{T-1}={mean_t_per_frame[-1]:.1f})")

    # Test q_sample_per_frame with ERDM timesteps
    x_t_erdm, _ = scheduler.q_sample_per_frame(x_0, t_erdm)
    assert x_t_erdm.shape == x_0.shape, f"Shape mismatch: {x_t_erdm.shape} vs {x_0.shape}"
    print("  q_sample_per_frame with ERDM: OK")

    # =========================================================================
    # Test DDPM Staggered Sampling (Gold Standard)
    # =========================================================================
    print("\nTesting DDPM Staggered Sampling...")

    # Create a simple mock model for testing
    class MockDenoiser(nn.Module):
        def forward(self, x_t, t, condition):
            """Mock denoiser that predicts zero noise (returns input unchanged)."""
            return torch.zeros_like(x_t)

    mock_model = MockDenoiser()

    # Test p_sample_per_frame
    t_per_frame = scheduler.sample_structured_causal_timesteps(B, T, device, spread_scale=50.0)
    x_t_test = torch.randn(B, T, H, W)
    condition_test = torch.randn(B, T, H, W)  # Mock history

    x_prev = scheduler.p_sample_per_frame(mock_model, x_t_test, t_per_frame, condition_test)
    assert x_prev.shape == x_t_test.shape, f"p_sample_per_frame shape mismatch: {x_prev.shape} vs {x_t_test.shape}"
    print(f"  p_sample_per_frame: OK (shape {x_prev.shape})")

    # Test with t_min freezing
    t_min = torch.arange(T, dtype=torch.long, device=device) * 2  # Frame i freezes at t=2*i
    x_prev_frozen = scheduler.p_sample_per_frame(mock_model, x_t_test, t_per_frame, condition_test, t_min=t_min)
    assert x_prev_frozen.shape == x_t_test.shape, "p_sample_per_frame with t_min shape mismatch"
    print("  p_sample_per_frame with t_min: OK")

    # Test sample_ddpm_staggered (use very few steps for test)
    scheduler_fast = DDPMScheduler(n_steps=5, schedule='cosine', device=device)
    shape_test = (2, T, H, W)  # B=2
    condition_small = torch.randn(2, T, H, W)

    x_0_staggered = scheduler_fast.sample_ddpm_staggered(
        mock_model, condition_small, shape_test,
        max_residual_timestep=2,
    )
    assert x_0_staggered.shape == (2, T, H, W), f"sample_ddpm_staggered shape mismatch: {x_0_staggered.shape}"
    print(f"  sample_ddpm_staggered: OK (shape {x_0_staggered.shape})")

    # Test with return_intermediates
    x_0_stag, intermediates = scheduler_fast.sample_ddpm_staggered(
        mock_model, condition_small, shape_test,
        max_residual_timestep=2,
        return_intermediates=True,
    )
    assert len(intermediates) == 5, f"Expected 5 intermediates for n_steps=5, got {len(intermediates)}"
    print(f"  sample_ddpm_staggered with intermediates: OK ({len(intermediates)} steps)")

    print("\nDDPM Scheduler tests passed!\n")


class FlowMatchingScheduler:
    """
    Conditional Flow Matching (CFM) scheduler.

    Implements the optimal transport (OT) conditional flow matching from
    Lipman et al. (2023) "Flow Matching for Generative Modeling".

    Forward path: x_t = (1 - t) * x_0 + t * eps  (linear interpolation)
    Velocity target: v = eps - x_0
    Inference: Euler ODE solve from t=1 (noise) to t=0 (clean)

    No noise schedule assumption — the velocity field is learned end-to-end.
    """

    def __init__(
        self,
        n_steps: int = 100,
        sigma_min: float = 1e-4,
        device: str = 'cpu',
    ):
        """
        Args:
            n_steps: Number of Euler steps for inference (more = higher quality)
            sigma_min: Minimum noise level to avoid singularity at t=0
            device: Device to place tensors on
        """
        self.n_steps = n_steps
        self.sigma_min = sigma_min
        self.device = device

    def to(self, device):
        """Move scheduler to device."""
        self.device = device
        return self

    def interpolate(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward interpolation: x_t = (1 - t) * x_0 + t * noise.

        Args:
            x_0: Clean samples (B, T, H, W)
            noise: Gaussian noise (B, T, H, W)
            t: Continuous timesteps (B, T) in [0, 1]

        Returns:
            x_t: Interpolated samples (B, T, H, W)
        """
        t_expanded = t.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        return (1 - t_expanded) * x_0 + t_expanded * noise

    def get_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity target: v = noise - x_0.

        Args:
            x_0: Clean samples (B, T, H, W)
            noise: Gaussian noise (B, T, H, W)

        Returns:
            v: Velocity target (B, T, H, W)
        """
        return noise - x_0

    def euler_step(
        self,
        x_t: torch.Tensor,
        v_pred: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Single Euler ODE step: x_{t-dt} = x_t - dt * v_pred.

        Args:
            x_t: Current state (B, T, H, W)
            v_pred: Predicted velocity (B, T, H, W)
            dt: Step size (positive, going from t=1 toward t=0)

        Returns:
            x_{t-dt}: Next state (B, T, H, W)
        """
        return x_t - dt * v_pred


if __name__ == "__main__":
    test_ddpm_scheduler()
