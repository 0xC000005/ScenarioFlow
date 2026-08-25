"""
Encoders for Block-AR Diffusion.

Encodes variable-length IV surface sequences into a fixed-size bottleneck
conditioning vector. Supports per-item masking for MCVD task types and
conditioning augmentation to mitigate exposure bias in autoregressive generation.

Two encoder variants:
  - GRUEncoder: Flattens spatial dims, processes with GRU. Default.
  - CausalConv3dEncoder: Processes history as 3D volume preserving spatial structure.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.causal_3d_blocks import CausalConv3d, ResnetBlockCausal3D


@dataclass
class EncoderConfig:
    input_dim: int = 25
    extra_features: int = 0  # additional input features (e.g. returns)
    gru_hidden_dim: int = 64
    bottleneck_dim: int = 64
    cond_aug_sigma: float = 0.0
    dropout: float = 0.1


class GRUEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config.input_dim + config.extra_features,
            hidden_size=config.gru_hidden_dim,
            batch_first=True,
        )
        # Attention pooling over all GRU hidden states (instead of last-only)
        self.attn_proj = nn.Linear(config.gru_hidden_dim, 1)
        self.bottleneck = nn.Linear(config.gru_hidden_dim, config.bottleneck_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.null_embedding = nn.Parameter(torch.zeros(1, config.bottleneck_dim))
        self.cond_aug_sigma = config.cond_aug_sigma

    def forward(
        self,
        surfaces: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        extra: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode IV surface sequence into bottleneck conditioning vector.

        Uses attention pooling over ALL GRU hidden states, preventing
        the exponential information decay of last-hidden-state extraction.

        Args:
            surfaces: (B, T, 5, 5) IV surfaces in [-1, 1]
            mask: (B,) bool tensor or None. True = replace with null_embedding.
            extra: (B, T, F) optional extra features (e.g. bounded returns).
                   If None and extra_features > 0, zero-padded automatically.

        Returns:
            (B, bottleneck_dim) conditioning vector
        """
        B = surfaces.shape[0]

        # Flatten spatial dims: (B, T, 5, 5) -> (B, T, 25)
        x = surfaces.reshape(B, surfaces.shape[1], -1)

        # Concatenate extra features if provided
        if extra is not None:
            if extra.dim() == 2:
                extra = extra.unsqueeze(-1)  # (B, T) → (B, T, 1)
            x = torch.cat([x, extra], dim=-1)  # (B, T, 25+F)
        elif self.config.extra_features > 0:
            x = F.pad(x, (0, self.config.extra_features))  # zero-pad

        # GRU forward — use ALL hidden states, not just final
        output, _ = self.gru(x)  # output: (B, T, gru_hidden_dim)

        # Attention pooling: learn which timesteps matter most
        attn_logits = self.attn_proj(output).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T)
        h = (attn_weights.unsqueeze(-1) * output).sum(dim=1)  # (B, gru_hidden_dim)

        # Bottleneck projection
        output = self.bottleneck(h)  # (B, bottleneck_dim)

        # Dropout
        output = self.dropout(output)

        # Conditioning augmentation (training only)
        if self.training:
            output = output + self.cond_aug_sigma * torch.randn_like(output)

        # Apply mask: replace masked items with null_embedding
        if mask is not None:
            output = torch.where(
                mask.unsqueeze(-1),
                self.null_embedding.expand(B, -1),
                output,
            )

        return output


class CausalConv3dEncoder(nn.Module):
    """Spatial-aware encoder using CausalConv3d blocks.

    Processes history as a (B, 1, T, 5, 5) 3D volume, preserving spatial
    relationships between moneyness/tenor grid points. Drop-in replacement
    for GRUEncoder with same interface (surfaces, mask) → (B, bottleneck_dim).

    Architecture matches the DDPM POC's HistoryEncoder:
      (B, T, 5, 5) → unsqueeze → CausalConv3d → 2× ResnetBlockCausal3D
      → global avg pool → Linear → (B, bottleneck_dim)
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        base_ch = 32  # matches DDPM POC

        self.conv_in = CausalConv3d(1, base_ch, kernel_size=3)
        self.blocks = nn.ModuleList([
            ResnetBlockCausal3D(base_ch, base_ch, groups=8, dropout=config.dropout)
            for _ in range(2)
        ])
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Linear(base_ch, config.bottleneck_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.null_embedding = nn.Parameter(torch.zeros(1, config.bottleneck_dim))
        self.cond_aug_sigma = config.cond_aug_sigma

    def forward(
        self, surfaces: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode history surfaces preserving spatial structure.

        Args:
            surfaces: (B, T, 5, 5) IV surfaces in [-1, 1]
            mask: (B,) bool tensor or None. True = replace with null_embedding.

        Returns:
            (B, bottleneck_dim) conditioning vector
        """
        B = surfaces.shape[0]

        # (B, T, 5, 5) → (B, 1, T, 5, 5) — add channel dim
        x = surfaces.unsqueeze(1)

        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)

        # Global avg pool → (B, C, 1, 1, 1) → (B, C)
        x = self.pool(x).view(B, -1)
        output = self.proj(x)
        output = self.dropout(output)

        if self.training:
            output = output + self.cond_aug_sigma * torch.randn_like(output)

        if mask is not None:
            output = torch.where(
                mask.unsqueeze(-1),
                self.null_embedding.expand(B, -1),
                output,
            )

        return output
