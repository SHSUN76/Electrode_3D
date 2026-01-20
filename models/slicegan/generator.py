"""
3D Generator for SliceGAN.

Generates 3D volumes from latent vectors using transposed 3D convolutions.
Input: (batch, nz, 4, 4, 4)
Output: (batch, nc, 64, 64, 64)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class Generator3D(nn.Module):
    """
    3D Generator for SliceGAN.

    Architecture:
        - 5 transposed convolution layers
        - BatchNorm after each layer except the last
        - ReLU activation for hidden layers
        - Softmax (n-phase) or Tanh (grayscale/color) for output

    Args:
        nz: Latent vector dimension (default: 512)
        ngf: Base number of feature maps (default: 64)
        nc: Number of output channels (phases) (default: 3)
        imtype: Image type - "nphase", "grayscale", or "color" (default: "nphase")
    """

    def __init__(
        self,
        nz: int = 512,
        ngf: int = 64,
        nc: int = 3,
        imtype: str = "nphase",
    ):
        super().__init__()

        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.imtype = imtype

        # Feature map sizes: nz -> 512 -> 256 -> 128 -> 64 -> nc
        self.gf: List[int] = [nz, 512, 256, 128, 64, nc]

        # Convolution parameters
        self.gk: List[int] = [4, 4, 4, 4, 4]  # kernel sizes
        self.gs: List[int] = [2, 2, 2, 2, 2]  # strides
        self.gp: List[int] = [1, 1, 1, 1, 1]  # padding

        # Build layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(len(self.gk)):
            self.convs.append(
                nn.ConvTranspose3d(
                    self.gf[i],
                    self.gf[i + 1],
                    self.gk[i],
                    self.gs[i],
                    self.gp[i],
                    bias=False,
                )
            )
            if i < len(self.gk) - 1:  # No BN on last layer
                self.bns.append(nn.BatchNorm3d(self.gf[i + 1]))

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights using normal distribution."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor, imtype: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, nz, 4, 4, 4)
            imtype: Override image type for this forward pass

        Returns:
            Generated 3D volume of shape (batch, nc, 64, 64, 64)
        """
        imtype = imtype or self.imtype

        # Apply transposed convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            x = self.bns[i](x)
            x = torch.relu(x)

        # Last layer without batch norm
        x = self.convs[-1](x)

        # Final activation based on image type
        if imtype == "nphase":
            x = torch.softmax(x, dim=1)  # One-hot encoding for phases
        else:
            x = torch.tanh(x)  # Grayscale or color

        return x

    def generate(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate random 3D volumes.

        Args:
            batch_size: Number of volumes to generate
            device: Device to use

        Returns:
            Generated volumes of shape (batch, nc, 64, 64, 64)
        """
        device = device or next(self.parameters()).device
        noise = torch.randn(batch_size, self.nz, 4, 4, 4, device=device)
        return self.forward(noise)

    def generate_large(
        self,
        size: tuple = (128, 128, 128),
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate larger volumes by tiling and blending.

        Args:
            size: Desired output size (D, H, W)
            batch_size: Number of volumes to generate
            device: Device to use

        Returns:
            Generated volumes of shape (batch, nc, D, H, W)
        """
        device = device or next(self.parameters()).device

        # For now, use interpolation for larger sizes
        # TODO: Implement proper tiling with blending
        base = self.generate(batch_size, device)

        if size != (64, 64, 64):
            base = torch.nn.functional.interpolate(
                base,
                size=size,
                mode="trilinear",
                align_corners=False,
            )

        return base


class GeneratorWithAdaIN(Generator3D):
    """
    Generator with Adaptive Instance Normalization for feature disentanglement.

    Allows control over style attributes (e.g., particle size, porosity).

    Based on: "Feature disentanglement in generating a three-dimensional
    structure from a two-dimensional slice with sliceGAN"
    """

    def __init__(
        self,
        nz: int = 512,
        ngf: int = 64,
        nc: int = 3,
        style_dim: int = 64,
        imtype: str = "nphase",
    ):
        super().__init__(nz, ngf, nc, imtype)

        self.style_dim = style_dim

        # AdaIN layers
        self.adain_layers = nn.ModuleList()
        for i, feat_size in enumerate(self.gf[1:-1]):  # Skip first and last
            self.adain_layers.append(AdaIN(feat_size, style_dim))

    def forward(
        self,
        x: torch.Tensor,
        style: Optional[torch.Tensor] = None,
        imtype: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional style control.

        Args:
            x: Input tensor of shape (batch, nz, 4, 4, 4)
            style: Style vector of shape (batch, style_dim)
            imtype: Override image type

        Returns:
            Generated 3D volume
        """
        imtype = imtype or self.imtype

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            if style is not None and i < len(self.adain_layers):
                x = self.adain_layers[i](x, style)
            else:
                x = self.bns[i](x)
            x = torch.relu(x)

        x = self.convs[-1](x)

        if imtype == "nphase":
            x = torch.softmax(x, dim=1)
        else:
            x = torch.tanh(x)

        return x


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer.

    Normalizes content features and applies style-dependent
    scaling and shifting.
    """

    def __init__(self, num_features: int, style_dim: int):
        super().__init__()

        self.norm = nn.InstanceNorm3d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive instance normalization.

        Args:
            x: Content features (batch, C, D, H, W)
            style: Style vector (batch, style_dim)

        Returns:
            Stylized features
        """
        # Generate gamma and beta from style
        style_params = self.fc(style)
        gamma, beta = style_params.chunk(2, dim=1)

        # Reshape for broadcasting
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Normalize and apply style
        out = self.norm(x)
        out = gamma * out + beta

        return out
