"""
Conv3D Denoiser for Block-AR Diffusion.

3D convolutional noise predictor that processes blocks as (B, 1, T, 5, 5) volumes.

Two variants:
- Conv3DBlockDenoiser: Standard (non-causal) Conv3d for MCVD bidirectional tasks
- CausalConv3DBlockDenoiser: CausalConv3d for forward-only, enables skewness

Architecture: Conv3d(1, C) -> N x [ResBlock3D + AdaptiveGroupNorm] -> Conv3d(C, 1)
Conditioning via AdaptiveGroupNorm (GroupNorm + FiLM scale/shift from embeddings).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from vae.causal_3d_blocks import CausalConv3d, ResnetBlockCausal3D
from diffusion.time_embedding import (
    AdaptiveGroupNorm,
    SpatialAdaptiveGroupNorm,
    SinusoidalTimeEmbedding,
    TimeEmbedding,
)


@dataclass
class Conv3DDenoiserConfig:
    frame_dim: int = 25
    surface_h: int = 5
    surface_w: int = 5
    bottleneck_dim: int = 64
    pos_embed_dim: int = 16
    noise_embed_dim: int = 64  # TimeEmbedding with MLP (matches SimpleDenoiser3D)
    n_steps: int = 100
    base_channels: int = 32
    n_res_blocks: int = 4
    groups: int = 8
    learn_sigma: bool = False  # output 2 channels (noise + variance fraction)
    spatial_pos_encoding: bool = False  # add row/col coordinate channels (CoordConv)
    use_spade: bool = False  # per-position learned scale/shift in AdaGN (SPADE)
    use_percell_head: bool = False  # condition-dependent per-cell noise correction
    percell_head_hidden: int = 64
    percell_regime_input: bool = False  # add vol_of_vol scalar to percell_head input
    baseline_channel: bool = False  # add baseline surface as extra input channel
    use_spatial_attention: bool = False  # spatial self-attention over 5x5 grid
    spatial_attn_heads: int = 4


class ResBlock3D(nn.Module):
    """Standard 3D residual block with non-causal Conv3d."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        # Initialize conv2 near-zero for residual near-identity at init
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> (B, C, T, H, W)"""
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over the spatial grid (5x5 = 25 positions).

    Each time step is processed independently — attention is across spatial
    positions only. This lets the denoiser learn position-dependent noise
    prediction patterns conditioned on the current spatial context.

    Bitter Lesson: attention breaks Conv3D weight sharing, allowing each
    cell to learn unique noise dynamics without hand-designed features.
    """

    def __init__(self, channels: int, n_heads: int = 4, groups: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        # Zero-init output projection → residual is identity at init
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> (B, C, T, H, W) with spatial attention."""
        B, C, T, H, W = x.shape
        S = H * W  # 25 spatial positions

        # Process each time step independently: (B, C, T, H, W) -> (B*T, C, S)
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, S)

        # Normalize
        h = self.norm(x_flat)  # (B*T, C, S)

        # QKV projection
        qkv = self.qkv(h)  # (B*T, 3C, S)
        q, k, v = qkv.chunk(3, dim=1)  # each (B*T, C, S)

        # Reshape for multi-head attention: (B*T, n_heads, S, head_dim)
        head_dim = C // self.n_heads
        q = q.reshape(B * T, self.n_heads, head_dim, S).permute(0, 1, 3, 2)
        k = k.reshape(B * T, self.n_heads, head_dim, S).permute(0, 1, 3, 2)
        v = v.reshape(B * T, self.n_heads, head_dim, S).permute(0, 1, 3, 2)

        # Scaled dot-product attention
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # Reshape back: (B*T, n_heads, S, head_dim) -> (B*T, C, S)
        out = out.permute(0, 1, 3, 2).reshape(B * T, C, S)
        out = self.proj(out)

        # Reshape to original: (B*T, C, S) -> (B, C, T, H, W)
        out = out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return x + out  # residual connection


class Conv3DBlockDenoiser(nn.Module):
    """Conv3D denoiser for Block-AR diffusion.

    Drop-in replacement for BiGRUDenoiser with identical forward interface.
    """

    def __init__(self, config: Conv3DDenoiserConfig):
        super().__init__()
        self.config = config
        C = config.base_channels

        # Noise level embedding: TimeEmbedding with MLP (64-dim)
        self.noise_embed = TimeEmbedding(
            n_steps=config.n_steps, embed_dim=config.noise_embed_dim
        )

        # Position embedding: lightweight sinusoidal (16-dim)
        self.pos_embed = SinusoidalTimeEmbedding(dim=config.pos_embed_dim)

        # Conditioning projection: concat(noise_emb, pos_emb, condition) -> C
        cond_input_dim = config.noise_embed_dim + config.pos_embed_dim + config.bottleneck_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, C * 4),
            nn.SiLU(),
            nn.Linear(C * 4, C),
        )

        # Conv3D backbone
        # CoordConv: add row/col coordinate channels if enabled
        self.spatial_pos_encoding = config.spatial_pos_encoding
        self.baseline_channel = getattr(config, 'baseline_channel', False)
        in_channels = 1
        if config.spatial_pos_encoding:
            in_channels += 2  # row/col coordinate channels
        if self.baseline_channel:
            in_channels += 1  # baseline surface channel
        self.conv_in = nn.Conv3d(in_channels, C, kernel_size=3, padding=1)

        # Pre-compute coordinate grids (registered as buffers, not parameters)
        if config.spatial_pos_encoding:
            H, W = config.surface_h, config.surface_w
            row_coords = torch.linspace(-1, 1, H).view(1, 1, 1, H, 1).expand(1, 1, 1, H, W).clone()
            col_coords = torch.linspace(-1, 1, W).view(1, 1, 1, 1, W).expand(1, 1, 1, H, W).clone()
            self.register_buffer('row_coords', row_coords)  # (1, 1, 1, H, W)
            self.register_buffer('col_coords', col_coords)  # (1, 1, 1, H, W)

        self.res_blocks = nn.ModuleList()
        self.ada_norms = nn.ModuleList()
        NormClass = SpatialAdaptiveGroupNorm if config.use_spade else AdaptiveGroupNorm
        for _ in range(config.n_res_blocks):
            self.res_blocks.append(ResBlock3D(C, groups=config.groups))
            if config.use_spade:
                self.ada_norms.append(NormClass(
                    C, config.groups, embed_dim=C,
                    surface_h=config.surface_h, surface_w=config.surface_w,
                ))
            else:
                self.ada_norms.append(NormClass(C, config.groups, embed_dim=C))

        # Spatial self-attention inserted after middle ResBlock
        self.use_spatial_attention = getattr(config, 'use_spatial_attention', False)
        self.spatial_attn_idx = config.n_res_blocks // 2 - 1  # after block 1 (0-indexed) for 4 blocks
        if self.use_spatial_attention:
            n_heads = getattr(config, 'spatial_attn_heads', 4)
            self.spatial_attn = SpatialSelfAttention(C, n_heads=n_heads, groups=config.groups)

        self.final_norm = nn.GroupNorm(config.groups, C)
        self.final_act = nn.SiLU()
        out_channels = 2 if config.learn_sigma else 1
        self.conv_out = nn.Conv3d(C, out_channels, kernel_size=3, padding=1)
        self.learn_sigma = config.learn_sigma

        # Per-cell residual head: condition-dependent noise prediction correction.
        # Learns a per-cell additive correction from the condition embedding,
        # enabling regime-dependent mean bias correction for each cell.
        self.use_percell_head = config.use_percell_head
        self.percell_regime_input = getattr(config, 'percell_regime_input', False)
        if config.use_percell_head:
            frame_dim = config.surface_h * config.surface_w
            H_pc = config.percell_head_hidden
            # Input dim: condition + optional regime scalar (vol_of_vol)
            pc_input_dim = config.bottleneck_dim + (1 if self.percell_regime_input else 0)
            self.percell_head = nn.Sequential(
                nn.Linear(pc_input_dim, H_pc),
                nn.SiLU(),
                nn.Linear(H_pc, frame_dim),
            )
            # Zero-init last layer so head starts as identity (no correction)
            nn.init.zeros_(self.percell_head[-1].weight)
            nn.init.zeros_(self.percell_head[-1].bias)

        # Initialize conv_out near-zero
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        noisy_frames: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
        noise_levels: torch.Tensor,
        regime_scalar: torch.Tensor = None,
        baseline_surface: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict noise for each frame, optionally with variance fraction.

        Args:
            noisy_frames: (B, T_block, 25) flattened noisy IV surfaces
            condition: (B, bottleneck_dim) conditioning from encoder
            positions: (B, T_block) long - absolute frame positions
            noise_levels: (B, T_block) long - per-frame diffusion timesteps
            regime_scalar: (B, 1) optional regime indicator (e.g. vol_of_vol)
            baseline_surface: (B, 5, 5) baseline IV surface (history[-1])

        Returns:
            noise_pred: (B, T_block, 25) predicted noise
            If learn_sigma: returns (noise_pred, v_pred) where v_pred is
                (B, T_block, 25) variance fraction for log-interpolation
        """
        B, T, D = noisy_frames.shape
        H, W = self.config.surface_h, self.config.surface_w

        # Embeddings
        noise_emb = self.noise_embed(noise_levels)  # (B, T, noise_embed_dim)
        pos_emb = self.pos_embed(positions)          # (B, T, pos_embed_dim)
        cond_expanded = condition.unsqueeze(1).expand(-1, T, -1)  # (B, T, bottleneck_dim)

        # Combined conditioning: (B, T, C)
        cond_cat = torch.cat([noise_emb, pos_emb, cond_expanded], dim=-1)  # (B, T, cond_input_dim)
        cond = self.cond_proj(cond_cat.reshape(B * T, -1)).reshape(B, T, -1)  # (B, T, C)

        # Reshape input to volume: (B, T, 25) -> (B, 1, T, 5, 5)
        x = noisy_frames.reshape(B, T, H, W).unsqueeze(1)  # (B, 1, T, H, W)

        # CoordConv: concatenate spatial coordinate channels
        if self.spatial_pos_encoding:
            row_grid = self.row_coords.expand(B, 1, T, H, W)  # (B, 1, T, H, W)
            col_grid = self.col_coords.expand(B, 1, T, H, W)  # (B, 1, T, H, W)
            x = torch.cat([x, row_grid, col_grid], dim=1)     # (B, 3, T, H, W)

        # Baseline channel: provide per-cell spatial context directly to denoiser
        if self.baseline_channel and baseline_surface is not None:
            # baseline_surface: (B, 5, 5) → (B, 1, 1, 5, 5) → expand to (B, 1, T, 5, 5)
            bl = baseline_surface.unsqueeze(1).unsqueeze(1).expand(B, 1, T, H, W)
            x = torch.cat([x, bl], dim=1)  # add baseline as extra channel

        # Conv3D backbone
        x = self.conv_in(x)  # (B, C, T, H, W)

        for i, (res_block, ada_norm) in enumerate(zip(self.res_blocks, self.ada_norms)):
            x = res_block(x)           # (B, C, T, H, W)
            x = ada_norm(x, cond)      # AdaGN with per-frame conditioning
            # Spatial self-attention after middle ResBlock
            if self.use_spatial_attention and i == self.spatial_attn_idx:
                x = self.spatial_attn(x)

        x = self.final_act(self.final_norm(x))
        x = self.conv_out(x)  # (B, out_ch, T, H, W)

        # Per-cell residual head
        if self.use_percell_head:
            if self.percell_regime_input and regime_scalar is not None:
                pc_input = torch.cat([condition, regime_scalar], dim=-1)  # (B, bottleneck_dim+1)
            else:
                pc_input = condition  # (B, bottleneck_dim)
            delta = self.percell_head(pc_input)  # (B, 25)

        if self.learn_sigma:
            noise_pred = x[:, 0].reshape(B, T, D)  # (B, T, 25)
            v_pred = x[:, 1].reshape(B, T, D)  # (B, T, 25)
            if self.use_percell_head:
                noise_pred = noise_pred + delta.unsqueeze(1)  # broadcast across T
            return noise_pred, v_pred
        else:
            noise_pred = x.squeeze(1).reshape(B, T, D)
            if self.use_percell_head:
                noise_pred = noise_pred + delta.unsqueeze(1)  # broadcast across T
            return noise_pred


class CausalConv3DBlockDenoiser(nn.Module):
    """CausalConv3D denoiser for Block-AR diffusion.

    Uses CausalConv3d (asymmetric temporal padding) to preserve temporal
    causality. This creates structural asymmetry in the reverse diffusion
    process that preserves skewness — matching DDPM POC's SimpleDenoiser3D.

    Drop-in replacement for Conv3DBlockDenoiser with identical forward interface.
    """

    def __init__(self, config: Conv3DDenoiserConfig):
        super().__init__()
        self.config = config
        C = config.base_channels

        # Noise level embedding: TimeEmbedding with MLP (64-dim)
        self.noise_embed = TimeEmbedding(
            n_steps=config.n_steps, embed_dim=config.noise_embed_dim
        )

        # Position embedding: lightweight sinusoidal (16-dim)
        self.pos_embed = SinusoidalTimeEmbedding(dim=config.pos_embed_dim)

        # Conditioning projection: concat(noise_emb, pos_emb, condition) -> C
        cond_input_dim = config.noise_embed_dim + config.pos_embed_dim + config.bottleneck_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, C * 4),
            nn.SiLU(),
            nn.Linear(C * 4, C),
        )

        # CausalConv3D backbone (matches SimpleDenoiser3D architecture)
        # CoordConv: add row/col coordinate channels if enabled
        self.spatial_pos_encoding = config.spatial_pos_encoding
        in_channels = 3 if config.spatial_pos_encoding else 1
        self.conv_in = CausalConv3d(in_channels, C, kernel_size=3)

        # Pre-compute coordinate grids (registered as buffers, not parameters)
        if config.spatial_pos_encoding:
            H, W = config.surface_h, config.surface_w
            row_coords = torch.linspace(-1, 1, H).view(1, 1, 1, H, 1).expand(1, 1, 1, H, W).clone()
            col_coords = torch.linspace(-1, 1, W).view(1, 1, 1, 1, W).expand(1, 1, 1, H, W).clone()
            self.register_buffer('row_coords', row_coords)
            self.register_buffer('col_coords', col_coords)

        self.res_blocks = nn.ModuleList()
        self.ada_norms = nn.ModuleList()
        for _ in range(config.n_res_blocks):
            self.res_blocks.append(
                ResnetBlockCausal3D(C, C, groups=config.groups)
            )
            self.ada_norms.append(
                AdaptiveGroupNorm(C, config.groups, embed_dim=C)
            )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(config.groups, C),
            nn.SiLU(),
            CausalConv3d(C, 1, kernel_size=3),
        )

    def forward(
        self,
        noisy_frames: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise for each frame (same interface as Conv3DBlockDenoiser).

        Args:
            noisy_frames: (B, T_block, 25) flattened noisy IV surfaces
            condition: (B, bottleneck_dim) conditioning from encoder
            positions: (B, T_block) long - absolute frame positions
            noise_levels: (B, T_block) long - per-frame diffusion timesteps

        Returns:
            noise_pred: (B, T_block, 25) predicted noise
        """
        B, T, D = noisy_frames.shape
        H, W = self.config.surface_h, self.config.surface_w

        # Embeddings
        noise_emb = self.noise_embed(noise_levels)  # (B, T, noise_embed_dim)
        pos_emb = self.pos_embed(positions)          # (B, T, pos_embed_dim)
        cond_expanded = condition.unsqueeze(1).expand(-1, T, -1)  # (B, T, bottleneck_dim)

        # Combined conditioning: (B, T, C)
        cond_cat = torch.cat([noise_emb, pos_emb, cond_expanded], dim=-1)
        cond = self.cond_proj(cond_cat.reshape(B * T, -1)).reshape(B, T, -1)

        # Reshape input to volume: (B, T, 25) -> (B, 1, T, 5, 5)
        x = noisy_frames.reshape(B, T, H, W).unsqueeze(1)

        # CoordConv: concatenate spatial coordinate channels
        if self.spatial_pos_encoding:
            row_grid = self.row_coords.expand(B, 1, T, H, W)
            col_grid = self.col_coords.expand(B, 1, T, H, W)
            x = torch.cat([x, row_grid, col_grid], dim=1)  # (B, 3, T, H, W)

        # CausalConv3D backbone
        x = self.conv_in(x)  # (B, C, T, H, W)

        for res_block, ada_norm in zip(self.res_blocks, self.ada_norms):
            x = res_block(x)           # (B, C, T, H, W)
            x = ada_norm(x, cond)      # AdaGN with per-frame conditioning

        x = self.conv_out(x)  # (B, 1, T, H, W)

        # Reshape back: (B, 1, T, H, W) -> (B, T, 25)
        noise_pred = x.squeeze(1).reshape(B, T, D)

        return noise_pred
