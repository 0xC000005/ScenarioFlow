"""
BiGRU Denoiser for Block-AR Diffusion.

Bidirectional GRU noise predictor that takes noisy frames, conditioning,
positional embeddings, and per-frame noise level embeddings to predict
the noise component for each frame.

Uses FiLM conditioning (scale + shift) to modulate features via the
conditioning vector, preventing the optimizer from ignoring conditioning.

Dual-path architecture:
- BiGRU path: temporal modeling with FiLM conditioning
- Spatial stream: per-frame 5x5 conv with AdaGN (noise-level-aware spatial processing)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from diffusion.time_embedding import SinusoidalTimeEmbedding


@dataclass
class DenoiserConfig:
    frame_dim: int = 25
    surface_h: int = 5
    surface_w: int = 5
    bottleneck_dim: int = 64
    pos_embed_dim: int = 16
    noise_embed_dim: int = 16
    gru_hidden_dim: int = 128
    spatial_mid_channels: int = 16
    dropout: float = 0.1


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scale and shift features by conditioning."""

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feature_dim * 2)
        # Initialize near-identity: scale=1, shift=0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) features to modulate
            cond: (B, cond_dim) conditioning vector
        Returns:
            (B, T, D) modulated features
        """
        params = self.proj(cond)  # (B, 2*D)
        scale, shift = params.unsqueeze(1).chunk(2, dim=-1)  # each (B, 1, D)
        return x * (1 + scale) + shift


class SpatialStream(nn.Module):
    """Parallel spatial CNN stream with AdaGN conditioning.

    Processes each frame as a 5x5 grid through 3 conv layers with
    Adaptive Group Normalization (AdaGN) after the first two layers.
    AdaGN erases noise-level statistics via GroupNorm then re-injects
    them via FiLM-style scale/shift from the conditioning vector.

    Architecture:
        conv1 (1 -> mid_ch) -> AdaGN -> SiLU
        conv2 (mid_ch -> mid_ch) -> AdaGN -> SiLU
        conv3 (mid_ch -> mid_ch) -> SiLU
    """

    def __init__(self, h: int, w: int, mid_channels: int, cond_dim: int):
        """
        Args:
            h: spatial height (5)
            w: spatial width (5)
            mid_channels: number of conv channels (16)
            cond_dim: conditioning dimension (noise_embed_dim + bottleneck_dim = 80)
        """
        super().__init__()
        self.h = h
        self.w = w
        self.mid_channels = mid_channels

        # Conv layers
        self.conv1 = nn.Conv2d(1, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        # GroupNorm (affine=False: no learnable params, AdaGN provides scale/shift)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=mid_channels, affine=False)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=mid_channels, affine=False)

        # AdaGN projections: conditioning -> (scale, shift) for each GroupNorm
        self.ada_proj1 = nn.Linear(cond_dim, mid_channels * 2)
        self.ada_proj2 = nn.Linear(cond_dim, mid_channels * 2)

        # Initialize AdaGN near-identity
        nn.init.zeros_(self.ada_proj1.weight)
        nn.init.zeros_(self.ada_proj1.bias)
        nn.init.zeros_(self.ada_proj2.weight)
        nn.init.zeros_(self.ada_proj2.bias)

        self.act = nn.SiLU()

    def forward(
        self,
        frames: torch.Tensor,
        noise_emb: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 25) flattened noisy IV surfaces
            noise_emb: (B, T, noise_embed_dim) per-frame noise level embeddings
            condition: (B, bottleneck_dim) conditioning from encoder
        Returns:
            (B, T, mid_channels * H * W) spatial features
        """
        B, T, D = frames.shape

        # Build per-frame conditioning: noise_emb (B,T,16) + condition (B,64) -> (B,T,80)
        cond_expanded = condition.unsqueeze(1).expand(-1, T, -1)  # (B, T, bottleneck_dim)
        cond_vec = torch.cat([noise_emb, cond_expanded], dim=-1)  # (B, T, cond_dim)
        cond_flat = cond_vec.reshape(B * T, -1)  # (B*T, cond_dim)

        # Reshape frames to spatial grid
        h = frames.reshape(B * T, 1, self.h, self.w)  # (B*T, 1, 5, 5)

        # Layer 1: conv -> GroupNorm -> AdaGN -> SiLU
        h = self.conv1(h)  # (B*T, mid_ch, 5, 5)
        h = self.gn1(h)
        scale1, shift1 = self.ada_proj1(cond_flat).chunk(2, dim=-1)  # each (B*T, mid_ch)
        h = h * (1 + scale1[:, :, None, None]) + shift1[:, :, None, None]
        h = self.act(h)

        # Layer 2: conv -> GroupNorm -> AdaGN -> SiLU
        h = self.conv2(h)  # (B*T, mid_ch, 5, 5)
        h = self.gn2(h)
        scale2, shift2 = self.ada_proj2(cond_flat).chunk(2, dim=-1)  # each (B*T, mid_ch)
        h = h * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        h = self.act(h)

        # Layer 3: conv -> SiLU (no AdaGN)
        h = self.conv3(h)  # (B*T, mid_ch, 5, 5)
        h = self.act(h)

        # Flatten spatial dims: (B*T, mid_ch, 5, 5) -> (B, T, mid_ch * 25)
        h = h.reshape(B, T, self.mid_channels * self.h * self.w)

        return h


class BiGRUDenoiser(nn.Module):
    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config

        self.pos_embed = SinusoidalTimeEmbedding(dim=config.pos_embed_dim)
        self.noise_embed = SinusoidalTimeEmbedding(dim=config.noise_embed_dim)

        # Input projection WITHOUT conditioning (FiLM applied after)
        input_dim = (
            config.frame_dim
            + config.pos_embed_dim
            + config.noise_embed_dim
        )
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.gru_hidden_dim),
            nn.SiLU(),
        )
        self.input_dropout = nn.Dropout(config.dropout)

        # FiLM conditioning: applied before and after GRU
        self.film_pre = FiLMLayer(config.bottleneck_dim, config.gru_hidden_dim)
        self.film_post = FiLMLayer(config.bottleneck_dim, config.gru_hidden_dim * 2)

        self.bigru = nn.GRU(
            input_size=config.gru_hidden_dim,
            hidden_size=config.gru_hidden_dim,
            bidirectional=True,
            batch_first=True,
        )

        # Spatial stream: parallel path with AdaGN
        spatial_cond_dim = config.noise_embed_dim + config.bottleneck_dim
        self.spatial_stream = SpatialStream(
            h=config.surface_h,
            w=config.surface_w,
            mid_channels=config.spatial_mid_channels,
            cond_dim=spatial_cond_dim,
        )

        # Output projection: BiGRU output (256) + spatial stream (mid_ch * 25)
        bigru_out_dim = config.gru_hidden_dim * 2
        spatial_out_dim = config.spatial_mid_channels * config.surface_h * config.surface_w
        self.output_proj = nn.Linear(bigru_out_dim + spatial_out_dim, config.frame_dim)

    def forward(
        self,
        noisy_frames: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise for each frame.

        Args:
            noisy_frames: (B, T_block, 25) flattened noisy IV surfaces
            condition: (B, bottleneck_dim) conditioning from encoder
            positions: (B, T_block) long - absolute frame positions
            noise_levels: (B, T_block) long - per-frame diffusion timesteps

        Returns:
            noise_pred: (B, T_block, 25) predicted noise
        """
        # Embeddings (computed early for both paths)
        pos_emb = self.pos_embed(positions)  # (B, T_block, pos_embed_dim)
        noise_emb = self.noise_embed(noise_levels)  # (B, T_block, noise_embed_dim)

        # === Spatial stream (parallel path) ===
        spatial_out = self.spatial_stream(
            noisy_frames, noise_emb, condition,
        )  # (B, T_block, mid_ch * 25)

        # === BiGRU path (UNCHANGED) ===
        # Concatenate frame + position + noise embeddings
        x = torch.cat(
            [noisy_frames, pos_emb, noise_emb], dim=-1
        )  # (B, T_block, input_dim)

        # Project and apply dropout
        x = self.input_proj(x)  # (B, T_block, gru_hidden_dim)
        x = self.input_dropout(x)

        # FiLM conditioning before GRU
        x = self.film_pre(x, condition)

        # Bidirectional GRU
        x, _ = self.bigru(x)  # (B, T_block, gru_hidden_dim * 2)

        # FiLM conditioning after GRU
        x = self.film_post(x, condition)

        # === Concatenate both paths ===
        x = torch.cat([x, spatial_out], dim=-1)  # (B, T_block, 256 + mid_ch*25)

        # Output projection to noise prediction
        noise_pred = self.output_proj(x)  # (B, T_block, frame_dim)

        return noise_pred
