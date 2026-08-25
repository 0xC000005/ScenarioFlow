"""
Diffusion Transformer (DiT) Denoiser for Block-AR Diffusion.

Treats each cell in the 5x5 IV grid as an independent token (25 tokens per timestep).
Uses AdaLN-Zero conditioning (Peebles & Xie 2023) — the standard conditioning approach
for diffusion transformers.

No weight sharing between cells: each token has unique attention patterns and can learn
position-specific noise dynamics. Bitter Lesson: transformers > convolutional weight sharing.

Architecture: Input_proj(1, d) + PosEmbed -> N x [DiTBlock(AdaLN-Zero)] -> FinalLayer -> 1
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from diffusion.time_embedding import SinusoidalTimeEmbedding, TimeEmbedding


@dataclass
class DiTDenoiserConfig:
    frame_dim: int = 25
    surface_h: int = 5
    surface_w: int = 5
    bottleneck_dim: int = 64
    pos_embed_dim: int = 16
    noise_embed_dim: int = 64
    n_steps: int = 100
    d_model: int = 64
    n_layers: int = 6
    n_heads: int = 4
    mlp_ratio: float = 2.0
    learn_sigma: bool = False


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning.

    AdaLN-Zero (from DiT paper): modulates LayerNorm with learned scale/shift/gate
    from the condition embedding. Gates initialized to zero so the block starts as
    near-identity, enabling stable training from random init.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 2.0, cond_dim: int = None):
        super().__init__()
        if cond_dim is None:
            cond_dim = d_model

        # Self-attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # FFN
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

        # AdaLN-Zero: 6 modulation params (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        # Zero-init so gates start at 0 (block = identity at init)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (BT, S, d_model) where S=25 spatial tokens
            cond: (BT, cond_dim) conditioning vector

        Returns:
            x: (BT, S, d_model)
        """
        # Get 6 modulation parameters
        mod = self.adaLN_modulation(cond)  # (BT, 6*d_model)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        # Attention block with AdaLN-Zero
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + gate1.unsqueeze(1) * h

        # MLP block with AdaLN-Zero
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h

        return x


class DiTBlockDenoiser(nn.Module):
    """Diffusion Transformer denoiser for Block-AR diffusion.

    Drop-in replacement for Conv3DBlockDenoiser with identical forward interface.
    Processes 5x5 IV grid as 25 independent tokens per timestep.
    """

    def __init__(self, config: DiTDenoiserConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        S = config.surface_h * config.surface_w  # 25

        # Input projection: per-cell scalar -> d_model
        self.input_proj = nn.Linear(1, d)

        # Learned spatial position embeddings (25 positions in the 5x5 grid)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, S, d))
        nn.init.normal_(self.spatial_pos_embed, std=0.02)

        # Noise level embedding
        self.noise_embed = TimeEmbedding(
            n_steps=config.n_steps, embed_dim=config.noise_embed_dim
        )

        # Temporal position embedding
        self.pos_embed = SinusoidalTimeEmbedding(dim=config.pos_embed_dim)

        # Conditioning projection: concat(noise_emb, pos_emb, condition) -> d_model
        cond_input_dim = config.noise_embed_dim + config.pos_embed_dim + config.bottleneck_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, d * 4),
            nn.SiLU(),
            nn.Linear(d * 4, d),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d, config.n_heads, config.mlp_ratio, cond_dim=d)
            for _ in range(config.n_layers)
        ])

        # Final layer with AdaLN-Zero
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 2 * d),
        )
        out_channels = 2 if config.learn_sigma else 1
        self.output_proj = nn.Linear(d, out_channels)
        self.learn_sigma = config.learn_sigma

        # Zero-init output projection for residual near-identity at start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

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
        Predict noise for each frame (same interface as Conv3DBlockDenoiser).

        Args:
            noisy_frames: (B, T_block, 25) flattened noisy IV surfaces
            condition: (B, bottleneck_dim) conditioning from encoder
            positions: (B, T_block) long - absolute frame positions
            noise_levels: (B, T_block) long - per-frame diffusion timesteps
            regime_scalar: unused (kept for interface compatibility)
            baseline_surface: unused (kept for interface compatibility)

        Returns:
            noise_pred: (B, T_block, 25) predicted noise
            If learn_sigma: returns (noise_pred, v_pred) tuple
        """
        B, T, D = noisy_frames.shape
        S = D  # 25 spatial tokens

        # Conditioning: per-frame (B, T, cond_dim) -> (B*T, d_model)
        noise_emb = self.noise_embed(noise_levels)  # (B, T, noise_embed_dim)
        pos_emb = self.pos_embed(positions)          # (B, T, pos_embed_dim)
        cond_expanded = condition.unsqueeze(1).expand(-1, T, -1)  # (B, T, bottleneck_dim)
        cond_cat = torch.cat([noise_emb, pos_emb, cond_expanded], dim=-1)
        cond = self.cond_proj(cond_cat.reshape(B * T, -1))  # (B*T, d_model)

        # Tokenize: (B, T, 25) -> (B*T, 25, 1) -> (B*T, 25, d_model)
        x = noisy_frames.reshape(B * T, S, 1)
        x = self.input_proj(x)  # (B*T, 25, d_model)
        x = x + self.spatial_pos_embed  # add learned spatial positions

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond)  # (B*T, 25, d_model)

        # Final layer with AdaLN
        h = self.final_norm(x)
        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.output_proj(h)  # (B*T, 25, out_channels)

        if self.learn_sigma:
            noise_pred = out[..., 0].reshape(B, T, D)  # (B, T, 25)
            v_pred = out[..., 1].reshape(B, T, D)       # (B, T, 25)
            return noise_pred, v_pred
        else:
            noise_pred = out[..., 0].reshape(B, T, D)   # (B, T, 25)
            return noise_pred
