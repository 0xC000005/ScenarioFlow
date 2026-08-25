"""
Diffusion Model Components for Volatility Surface Generation.

This package implements a DDPM (Denoising Diffusion Probabilistic Model) approach
to conditional generation of future volatility surfaces.

Key insight: VAEs produce conditional means, not diverse samples. Diffusion models
can sample from the full conditional distribution, enabling proper CI coverage.

Components:
- ddpm_scheduler.py: Forward/reverse diffusion process
- time_embedding.py: Sinusoidal timestep encoding
- simple_denoiser.py: 3D U-Net denoiser using CausalConv3d blocks
"""

from .ddpm_scheduler import DDPMScheduler
from .time_embedding import SinusoidalTimeEmbedding, TimeEmbedding
from .simple_denoiser import SimpleDenoiser3D

__all__ = [
    'DDPMScheduler',
    'SinusoidalTimeEmbedding',
    'TimeEmbedding',
    'SimpleDenoiser3D',
]
