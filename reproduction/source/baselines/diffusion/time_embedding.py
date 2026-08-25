"""
Time Embedding for Diffusion Models.

Provides sinusoidal position encoding for diffusion timesteps,
following the standard approach from transformer literature adapted for diffusion.

Reference: Vaswani et al., "Attention Is All You Need" (2017)
Adapted for diffusion in: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding.

    Maps integer timesteps to continuous embeddings using sin/cos encoding.
    This gives the model a smooth, continuous representation of time.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        """
        Args:
            dim: Embedding dimension (should be even)
            max_period: Maximum period for sin/cos (higher = slower frequency variation)
        """
        super().__init__()
        assert dim % 2 == 0, f"Embedding dim must be even, got {dim}"
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal embedding for timesteps.

        Args:
            t: Timesteps of shape (B,) or (B, T) for per-frame (Diffusion Forcing)

        Returns:
            emb: Embeddings (B, dim) or (B, T, dim) matching input shape
        """
        device = t.device
        input_shape = t.shape
        half_dim = self.dim // 2

        # Flatten to 1D for computation
        t_flat = t.flatten().float()  # (B,) or (B*T,)

        # Compute frequencies: exp(-log(max_period) * i / (half_dim - 1))
        # This gives frequencies ranging from 1 to 1/max_period
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / (half_dim - 1)
        )

        # Compute arguments: t * freq
        # t_flat: (N,) -> (N, 1)
        # freqs: (half_dim,) -> (1, half_dim)
        args = t_flat.unsqueeze(1) * freqs.unsqueeze(0)

        # Concatenate sin and cos embeddings
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (N, dim)

        # Reshape back to match input: (B, dim) or (B, T, dim)
        emb = emb.view(*input_shape, self.dim)

        return emb


class TimeEmbedding(nn.Module):
    """
    Full time embedding with learnable MLP projection.

    Maps timesteps to embeddings via sinusoidal encoding + MLP.
    This is the standard approach in diffusion U-Nets.
    """

    def __init__(
        self,
        n_steps: int,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            n_steps: Number of diffusion timesteps (for reference, not used directly)
            embed_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension (default: 4 * embed_dim)
        """
        super().__init__()
        self.n_steps = n_steps
        self.embed_dim = embed_dim
        hidden_dim = hidden_dim or 4 * embed_dim

        # Sinusoidal embedding (outputs embed_dim)
        self.sinusoidal = SinusoidalTimeEmbedding(embed_dim)

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time embedding with MLP projection.

        Args:
            t: Timesteps (B,) or (B, T) for per-frame (Diffusion Forcing)

        Returns:
            emb: Embeddings (B, embed_dim) or (B, T, embed_dim) matching input
        """
        input_shape = t.shape

        # Sinusoidal encoding - handles (B,) or (B, T)
        emb = self.sinusoidal(t)  # (B, dim) or (B, T, dim)

        # Flatten for MLP if per-frame
        if t.dim() == 2:
            B, T = input_shape
            emb = emb.view(B * T, -1)  # (B*T, dim)
            emb = self.mlp(emb)
            emb = emb.view(B, T, -1)  # (B, T, embed_dim)
        else:
            emb = self.mlp(emb)

        return emb


class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Normalization conditioned on time embedding.

    Applies GroupNorm then scales/shifts based on time embedding.
    This is a lightweight version of FiLM (Feature-wise Linear Modulation).

    Used in diffusion U-Nets to condition on timestep.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        embed_dim: int,
        eps: float = 1e-6,
    ):
        """
        Args:
            num_channels: Number of input channels
            num_groups: Number of groups for GroupNorm
            embed_dim: Time embedding dimension
            eps: Small constant for numerical stability
        """
        super().__init__()

        # Adjust groups to be compatible with channel count
        num_groups = min(num_groups, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        num_groups = max(1, num_groups)

        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)

        # Project time embedding to scale and shift
        self.proj = nn.Linear(embed_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive normalization.

        Args:
            x: Input tensor (B, C, T, H, W) for 3D or (B, C, ...) for other dims
            t_emb: Time embedding (B, embed_dim) or (B, T, embed_dim) for per-frame

        Returns:
            Normalized and scaled/shifted tensor (B, C, ...)
        """
        # Normalize
        x = self.norm(x)

        # Handle per-frame embeddings (Diffusion Forcing)
        if t_emb.dim() == 3:
            # Per-frame: t_emb is (B, T, embed_dim)
            # x is (B, C, T, H, W)
            B, T, _ = t_emb.shape

            # Project each frame's embedding: (B, T, embed_dim) -> (B, T, 2*C)
            params = self.proj(t_emb.view(B * T, -1)).view(B, T, -1)  # (B, T, 2*C)
            scale, shift = params.chunk(2, dim=2)  # (B, T, C), (B, T, C)

            # Reshape for broadcasting: (B, T, C) -> (B, C, T, 1, 1)
            scale = scale.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
            shift = shift.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
        else:
            # Standard: t_emb is (B, embed_dim)
            # Get scale and shift from time embedding
            params = self.proj(t_emb)  # (B, 2*C)
            scale, shift = params.chunk(2, dim=1)  # (B, C), (B, C)

            # Reshape for broadcasting: (B, C) -> (B, C, 1, 1, ...) for spatial dims
            n_spatial = x.dim() - 2
            scale = scale.view(scale.shape[0], scale.shape[1], *([1] * n_spatial))
            shift = shift.view(shift.shape[0], shift.shape[1], *([1] * n_spatial))

        # Apply scale and shift
        return x * (1 + scale) + shift


class SpatialAdaptiveGroupNorm(nn.Module):
    """
    AdaptiveGroupNorm + learned per-position scale/shift (SPADE-style).

    After GroupNorm + FiLM (condition-dependent uniform scale/shift), applies
    learned spatial modulation: y * (1 + gamma_s[c,h,w]) + beta_s[c,h,w].

    This gives each spatial position independent learned capacity to modulate
    the normalized features. The multiplicative interaction with FiLM means
    the spatial correction scales with the condition (e.g. larger in turbulent
    regimes when FiLM scale is large).

    Extra parameters per layer: 2 * C * H * W (e.g. 2*32*5*5 = 1600).
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        embed_dim: int,
        surface_h: int = 5,
        surface_w: int = 5,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Adjust groups to be compatible with channel count
        num_groups = min(num_groups, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        num_groups = max(1, num_groups)

        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)

        # FiLM: project condition embedding to per-channel scale and shift
        self.proj = nn.Linear(embed_dim, num_channels * 2)

        # SPADE: per-position learned scale and shift (zero-init → identity at start)
        self.gamma_spatial = nn.Parameter(
            torch.zeros(1, num_channels, 1, surface_h, surface_w)
        )
        self.beta_spatial = nn.Parameter(
            torch.zeros(1, num_channels, 1, surface_h, surface_w)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive normalization with spatial modulation.

        Args:
            x: Input tensor (B, C, T, H, W)
            t_emb: Time/condition embedding (B, embed_dim) or (B, T, embed_dim)

        Returns:
            Normalized, FiLM-modulated, spatially-modulated tensor
        """
        # Normalize
        x = self.norm(x)

        # FiLM: condition-dependent scale/shift (same as AdaptiveGroupNorm)
        if t_emb.dim() == 3:
            B, T, _ = t_emb.shape
            params = self.proj(t_emb.view(B * T, -1)).view(B, T, -1)
            scale, shift = params.chunk(2, dim=2)
            scale = scale.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            shift = shift.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        else:
            params = self.proj(t_emb)
            scale, shift = params.chunk(2, dim=1)
            n_spatial = x.dim() - 2
            scale = scale.view(scale.shape[0], scale.shape[1], *([1] * n_spatial))
            shift = shift.view(shift.shape[0], shift.shape[1], *([1] * n_spatial))

        # Apply FiLM then spatial SPADE:
        # y = [GroupNorm(x) * (1+scale_t) + shift_t] * (1+gamma_s) + beta_s
        # Cross-term: scale_t * gamma_s provides regime × position interaction
        y = x * (1 + scale) + shift
        return y * (1 + self.gamma_spatial) + self.beta_spatial


def test_time_embedding():
    """Unit tests for time embedding modules."""
    print("Testing Time Embedding...")

    device = 'cpu'
    B = 4
    embed_dim = 64

    # Test sinusoidal embedding
    sin_emb = SinusoidalTimeEmbedding(embed_dim)
    t = torch.tensor([0, 25, 50, 99])
    emb = sin_emb(t)
    assert emb.shape == (B, embed_dim), f"Wrong shape: {emb.shape}"
    print(f"  Sinusoidal embedding: OK (shape {emb.shape})")

    # Check that different timesteps give different embeddings
    assert not torch.allclose(emb[0], emb[1]), "Different timesteps should give different embeddings"
    print("  Embedding uniqueness: OK")

    # Check smoothness: adjacent timesteps should have similar embeddings
    t_seq = torch.arange(100)
    emb_seq = sin_emb(t_seq)
    diffs = (emb_seq[1:] - emb_seq[:-1]).norm(dim=1)
    assert diffs.std() < diffs.mean(), "Embeddings should vary smoothly"
    print(f"  Embedding smoothness: OK (diff std={diffs.std():.4f} < mean={diffs.mean():.4f})")

    # Test full time embedding with MLP
    time_emb = TimeEmbedding(n_steps=100, embed_dim=embed_dim)
    t = torch.randint(0, 100, (B,))
    emb = time_emb(t)
    assert emb.shape == (B, embed_dim), f"Wrong shape: {emb.shape}"
    print(f"  TimeEmbedding with MLP: OK (shape {emb.shape})")

    # Test adaptive group norm
    C, T, H, W = 32, 10, 5, 5
    agn = AdaptiveGroupNorm(C, num_groups=8, embed_dim=embed_dim)
    x = torch.randn(B, C, T, H, W)
    t_emb = time_emb(torch.randint(0, 100, (B,)))
    out = agn(x, t_emb)
    assert out.shape == x.shape, f"AdaptiveGroupNorm changed shape: {out.shape} vs {x.shape}"
    print(f"  AdaptiveGroupNorm: OK (shape {out.shape})")

    # === Per-frame (Diffusion Forcing) tests ===
    print("\nTesting Per-Frame Mode (Diffusion Forcing)...")

    # Test sinusoidal with per-frame timesteps
    T_frames = 30
    t_per_frame = torch.randint(0, 100, (B, T_frames))
    emb_per_frame = sin_emb(t_per_frame)
    assert emb_per_frame.shape == (B, T_frames, embed_dim), f"Wrong per-frame shape: {emb_per_frame.shape}"
    print(f"  Sinusoidal per-frame: OK (shape {emb_per_frame.shape})")

    # Verify per-frame consistency: embedding for same timestep should match
    t_single = torch.tensor([42])
    t_batch = torch.tensor([[42, 42, 42]])  # Same timestep repeated
    emb_single = sin_emb(t_single)  # (1, dim)
    emb_batch = sin_emb(t_batch)  # (1, 3, dim)
    assert torch.allclose(emb_single[0], emb_batch[0, 0], atol=1e-6), "Per-frame should match single"
    assert torch.allclose(emb_single[0], emb_batch[0, 1], atol=1e-6), "Per-frame should match single"
    print("  Per-frame consistency: OK")

    # Test TimeEmbedding with per-frame
    t_per_frame = torch.randint(0, 100, (B, T_frames))
    emb_per_frame = time_emb(t_per_frame)
    assert emb_per_frame.shape == (B, T_frames, embed_dim), f"Wrong per-frame shape: {emb_per_frame.shape}"
    print(f"  TimeEmbedding per-frame: OK (shape {emb_per_frame.shape})")

    # Test AdaptiveGroupNorm with per-frame embeddings
    x = torch.randn(B, C, T_frames, H, W)  # Note: T_frames must match
    t_emb_per_frame = time_emb(torch.randint(0, 100, (B, T_frames)))
    out = agn(x, t_emb_per_frame)
    assert out.shape == x.shape, f"AdaptiveGroupNorm per-frame changed shape: {out.shape} vs {x.shape}"
    print(f"  AdaptiveGroupNorm per-frame: OK (shape {out.shape})")

    # Verify per-frame modulation is different across frames
    # Create input with different timesteps per frame
    t_varied = torch.tensor([[0, 50, 99]])  # Very different timesteps
    t_emb_varied = time_emb(t_varied)  # (1, 3, embed_dim)
    x_small = torch.ones(1, C, 3, H, W)  # Uniform input
    agn_small = AdaptiveGroupNorm(C, num_groups=8, embed_dim=embed_dim)
    out_small = agn_small(x_small, t_emb_varied)
    # Different frames should have different outputs due to different timestep modulation
    assert not torch.allclose(out_small[0, :, 0], out_small[0, :, 1], atol=1e-3), \
        "Different timesteps should produce different modulation"
    print("  Per-frame differentiation: OK")

    print("\nTime Embedding tests passed!\n")


if __name__ == "__main__":
    test_time_embedding()
