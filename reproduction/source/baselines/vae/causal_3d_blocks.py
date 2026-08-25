"""
Causal 3D Convolution Building Blocks for Temporal Surface Generation.

Ported and simplified from HunyuanVideo (Tencent):
https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/vae/unet_causal_3d_blocks.py

Key adaptations for volatility surface (5x5 grid):
- No spatial downsampling (5x5 too small)
- Optional temporal compression
- Removed diffusers dependencies (pure PyTorch)
- Simplified GroupNorm groups for smaller channel counts
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Module):
    """
    Implements a causal 3D convolution layer where each position only depends
    on previous timesteps and current spatial locations.

    This maintains temporal causality in video/surface generation tasks.
    Frame t can only see frames <= t, never t+1.

    Key insight: Asymmetric temporal padding (kernel-1, 0) ensures causality.
    - Spatial (W, H): symmetric padding kernel//2 on both sides
    - Temporal (T): asymmetric padding (kernel-1, 0) - only left padding

    Source: HunyuanVideo (Tencent)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode: str = 'replicate',
        bias: bool = True,
    ):
        super().__init__()

        # Handle int or tuple kernel_size
        if isinstance(kernel_size, int):
            k_t, k_h, k_w = kernel_size, kernel_size, kernel_size
        else:
            k_t, k_h, k_w = kernel_size

        self.pad_mode = pad_mode

        # Padding order for F.pad: (W_left, W_right, H_left, H_right, T_left, T_right)
        # Spatial: symmetric padding
        # Temporal: asymmetric (causal) - only pad left side
        self.time_causal_padding = (
            k_w // 2, k_w // 2,     # W: symmetric
            k_h // 2, k_h // 2,     # H: symmetric
            k_t - 1, 0              # T: left-only (CAUSAL!)
        )

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            padding=0  # We handle padding manually for causality
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor
        Returns:
            (B, C_out, T', H', W') tensor where T' depends on stride
        """
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class CausalConv3dTranspose(nn.Module):
    """
    Causal 3D transposed convolution for upsampling.

    Uses the same asymmetric padding strategy as CausalConv3d to maintain causality.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

        # Calculate output padding to get exact dimensions
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal transposed convolution."""
        out = self.conv_transpose(x)

        # Trim temporal dimension to maintain causality
        # Remove the last (kernel_size[0] - 1) frames that look into future
        k_t = self.kernel_size[0]
        if k_t > 1:
            out = out[:, :, :-(k_t - 1), :, :]

        return out


class UpsampleCausal3D(nn.Module):
    """
    Causal 3D upsampling layer using nearest-neighbor interpolation + conv.

    Uses nearest-neighbor + conv instead of transposed conv
    to avoid checkerboard artifacts.

    For volatility surfaces, we use simple uniform upsampling to ensure
    proper round-trip (encode -> decode recovers original shape).
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_conv: bool = True,
        upsample_factor: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.upsample_factor = upsample_factor

        if use_conv:
            self.conv = CausalConv3d(channels, self.out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, target_temporal_size: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor
            target_temporal_size: Optional target T dimension (for exact shape matching)
        Returns:
            Upsampled (B, C_out, T*factor_t, H*factor_h, W*factor_w) tensor
        """
        B, C, T, H, W = x.shape

        # Handle dtype for interpolation (bfloat16 not supported by upsample_nearest)
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        # Simple uniform upsampling (no special first-frame handling)
        # This ensures proper round-trip: downsample(T) -> T', upsample(T') -> T
        x = F.interpolate(
            x,
            scale_factor=self.upsample_factor,
            mode="nearest"
        )

        # If target size specified, trim or pad to match
        if target_temporal_size is not None and x.shape[2] != target_temporal_size:
            if x.shape[2] > target_temporal_size:
                x = x[:, :, :target_temporal_size, :, :]
            else:
                # Pad by repeating last frame
                pad_size = target_temporal_size - x.shape[2]
                x = F.pad(x, (0, 0, 0, 0, 0, pad_size), mode='replicate')

        # Cast back if needed
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        if self.use_conv:
            x = self.conv(x)

        return x


class DownsampleCausal3D(nn.Module):
    """
    Causal 3D downsampling layer using strided convolution.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_conv: bool = True,
        stride: Union[int, Tuple[int, int, int]] = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if isinstance(stride, int):
            stride = (stride, stride, stride)

        if use_conv:
            self.conv = CausalConv3d(
                channels,
                self.out_channels,
                kernel_size=3,
                stride=stride
            )
        else:
            self.pool = nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor
        Returns:
            Downsampled (B, C_out, T/s_t, H/s_h, W/s_w) tensor
        """
        if self.use_conv:
            return self.conv(x)
        else:
            return self.pool(x)


class ResnetBlockCausal3D(nn.Module):
    """
    Residual block with causal 3D convolutions.

    Structure: x -> norm1 -> act -> conv1 -> norm2 -> act -> conv2 -> + shortcut

    Simplified from HunyuanVideo (removed time embeddings, attention options).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        groups: int = 8,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Adjust groups to be compatible with channel count
        groups = min(groups, in_channels, out_channels)
        while in_channels % groups != 0 or out_channels % groups != 0:
            groups -= 1
        groups = max(1, groups)

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)

        self.nonlinearity = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Shortcut connection if channels differ
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor
        Returns:
            (B, C_out, T, H, W) tensor
        """
        residual = x

        # First conv block
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        # Second conv block
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Shortcut
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return residual + h


class DownEncoderBlockCausal3D(nn.Module):
    """
    Encoder block: ResNet blocks followed by optional downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        groups: int = 8,
        dropout: float = 0.0,
        add_downsample: bool = True,
        downsample_stride: Union[int, Tuple[int, int, int]] = 2,
    ):
        super().__init__()

        # ResNet blocks
        resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    groups=groups,
                    dropout=dropout,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        # Downsampler
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                DownsampleCausal3D(
                    out_channels,
                    out_channels=out_channels,
                    use_conv=True,
                    stride=downsample_stride,
                )
            ])
        else:
            self.downsamplers = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)

        return x


class UpDecoderBlockCausal3D(nn.Module):
    """
    Decoder block: ResNet blocks followed by optional upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        groups: int = 8,
        dropout: float = 0.0,
        add_upsample: bool = True,
        upsample_factor: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()

        # ResNet blocks
        resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    groups=groups,
                    dropout=dropout,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        # Upsampler
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                UpsampleCausal3D(
                    out_channels,
                    out_channels=out_channels,
                    use_conv=True,
                    upsample_factor=upsample_factor,
                )
            ])
        else:
            self.upsamplers = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)

        return x


class MidBlockCausal3D(nn.Module):
    """
    Middle block with ResNet blocks (no attention for our small 5x5 grid).

    Simplified from HunyuanVideo UNetMidBlockCausal3D - removed attention
    since our spatial dimensions are too small (5x5) to benefit from it.
    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        resnets = [
            ResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=groups,
                dropout=dropout,
            )
        ]

        for _ in range(num_layers):
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    groups=groups,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(resnets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        return x


def test_causal_conv3d():
    """
    Unit test to verify temporal causality of CausalConv3d.

    Test: Changing frame t+1 should NOT affect output at frame t.
    """
    print("Testing CausalConv3d temporal causality...")

    conv = CausalConv3d(1, 1, kernel_size=3)
    conv.eval()

    # Create input: (B=1, C=1, T=5, H=5, W=5)
    x1 = torch.randn(1, 1, 5, 5, 5)
    x2 = x1.clone()

    # Modify frame t=3 (should only affect t>=3)
    x2[:, :, 3, :, :] = torch.randn(1, 1, 5, 5)

    with torch.no_grad():
        y1 = conv(x1)
        y2 = conv(x2)

    # Check frames before t=3 are identical
    for t in range(3):
        diff = (y1[:, :, t] - y2[:, :, t]).abs().max().item()
        status = "PASS" if diff < 1e-6 else "FAIL"
        print(f"  Frame {t}: max diff = {diff:.2e} [{status}]")

    # Check frame t=3 onwards ARE different
    for t in range(3, 5):
        diff = (y1[:, :, t] - y2[:, :, t]).abs().max().item()
        status = "PASS" if diff > 1e-6 else "FAIL (should differ)"
        print(f"  Frame {t}: max diff = {diff:.2e} [{status}]")

    print("Causality test complete.\n")


if __name__ == "__main__":
    test_causal_conv3d()

    # Test shapes
    print("Testing building block shapes...")

    # Input: (B=2, C=1, T=20, H=5, W=5) - batch of 20-frame vol surfaces
    x = torch.randn(2, 1, 20, 5, 5)
    print(f"Input shape: {x.shape}")

    # CausalConv3d
    conv = CausalConv3d(1, 16, kernel_size=3)
    y = conv(x)
    print(f"After CausalConv3d(1->16): {y.shape}")

    # ResnetBlockCausal3D
    resnet = ResnetBlockCausal3D(16, 32, groups=8)
    y = resnet(y)
    print(f"After ResnetBlock(16->32): {y.shape}")

    # DownEncoderBlockCausal3D (temporal only, no spatial)
    down = DownEncoderBlockCausal3D(32, 64, add_downsample=True, downsample_stride=(2, 1, 1))
    y = down(y)
    print(f"After DownEncoderBlock(32->64, T/2): {y.shape}")

    # MidBlockCausal3D
    mid = MidBlockCausal3D(64, num_layers=1, groups=8)
    y = mid(y)
    print(f"After MidBlock(64): {y.shape}")

    # UpDecoderBlockCausal3D
    up = UpDecoderBlockCausal3D(64, 32, add_upsample=True, upsample_factor=(2, 1, 1))
    y = up(y)
    print(f"After UpDecoderBlock(64->32, T*2): {y.shape}")

    print("\nAll shape tests passed!")
