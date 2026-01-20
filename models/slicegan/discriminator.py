"""
2D Discriminator (Critic) for SliceGAN.

Uses WGAN-GP (Wasserstein GAN with Gradient Penalty) for stable training.
Input: 2D slices from generated 3D volumes
Output: Wasserstein distance score
"""

import torch
import torch.nn as nn
from typing import List


class Critic2D(nn.Module):
    """
    2D Discriminator/Critic for SliceGAN using WGAN-GP.

    Architecture:
        - 5 convolutional layers
        - LeakyReLU activation
        - No batch normalization (for WGAN-GP)
        - No sigmoid (outputs raw score for Wasserstein distance)

    Args:
        nc: Number of input channels (phases) (default: 3)
        ndf: Base number of feature maps (default: 64)
    """

    def __init__(self, nc: int = 3, ndf: int = 64):
        super().__init__()

        self.nc = nc
        self.ndf = ndf

        # Feature map sizes: nc -> 64 -> 128 -> 256 -> 512 -> 1
        self.df: List[int] = [nc, 64, 128, 256, 512, 1]

        # Convolution parameters
        self.dk: List[int] = [4, 4, 4, 4, 4]  # kernel sizes
        self.ds: List[int] = [2, 2, 2, 2, 2]  # strides
        self.dp: List[int] = [1, 1, 1, 1, 0]  # padding

        # Build layers
        self.convs = nn.ModuleList()

        for i in range(len(self.dk)):
            self.convs.append(
                nn.Conv2d(
                    self.df[i],
                    self.df[i + 1],
                    self.dk[i],
                    self.ds[i],
                    self.dp[i],
                    bias=False,
                )
            )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights using normal distribution."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, nc, H, W)

        Returns:
            Critic score of shape (batch, 1)
        """
        batch_size = x.size(0)

        for conv in self.convs[:-1]:
            x = conv(x)
            x = nn.functional.leaky_relu(x, 0.2, inplace=True)

        # Adaptive pooling to handle variable input sizes
        x = nn.functional.adaptive_avg_pool2d(x, (4, 4))

        # Last layer
        x = self.convs[-1](x)

        # Flatten to (batch, 1), preserving batch dimension
        return x.view(batch_size, -1).mean(dim=1, keepdim=True)


class MultiScaleCritic2D(nn.Module):
    """
    Multi-scale 2D Critic for better texture capture.

    Uses multiple discriminators at different scales.
    """

    def __init__(self, nc: int = 3, ndf: int = 64, num_scales: int = 3):
        super().__init__()

        self.num_scales = num_scales

        # Create critics at different scales
        self.critics = nn.ModuleList([
            Critic2D(nc, ndf) for _ in range(num_scales)
        ])

        # Downsampling for multi-scale
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass at multiple scales.

        Args:
            x: Input tensor of shape (batch, nc, H, W)

        Returns:
            List of critic scores at each scale
        """
        outputs = []

        for i, critic in enumerate(self.critics):
            outputs.append(critic(x))
            if i < self.num_scales - 1:
                x = self.downsample(x)

        return outputs


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Calculate gradient penalty for WGAN-GP.

    Enforces Lipschitz constraint by penalizing gradients that deviate from 1.

    Args:
        critic: Discriminator/critic network
        real: Real samples
        fake: Fake (generated) samples
        device: Computation device
        lambda_gp: Gradient penalty coefficient

    Returns:
        Gradient penalty loss
    """
    batch_size = real.size(0)

    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real)

    # Create interpolated samples
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    # Get critic output for interpolated samples
    critic_interpolated = critic(interpolated)

    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten gradients
    gradients = gradients.view(batch_size, -1)

    # Calculate gradient norm
    gradient_norm = gradients.norm(2, dim=1)

    # Gradient penalty: (||grad|| - 1)^2
    gp = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return gp


class WGANGPLoss:
    """
    WGAN-GP loss calculator.

    Computes Wasserstein loss with gradient penalty.
    """

    def __init__(self, lambda_gp: float = 10.0):
        self.lambda_gp = lambda_gp

    def critic_loss(
        self,
        critic: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
        device: torch.device,
    ) -> tuple:
        """
        Calculate critic (discriminator) loss.

        Args:
            critic: Critic network
            real: Real samples
            fake: Fake samples
            device: Computation device

        Returns:
            Tuple of (total_loss, wasserstein_distance, gradient_penalty)
        """
        # Critic outputs
        critic_real = critic(real).mean()
        critic_fake = critic(fake.detach()).mean()

        # Wasserstein distance
        wasserstein_distance = critic_real - critic_fake

        # Gradient penalty
        gp = gradient_penalty(critic, real, fake, device, self.lambda_gp)

        # Total loss (minimize -W_distance + GP)
        total_loss = -wasserstein_distance + gp

        return total_loss, wasserstein_distance.item(), gp.item()

    def generator_loss(
        self,
        critic: nn.Module,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate generator loss.

        Args:
            critic: Critic network
            fake: Generated samples

        Returns:
            Generator loss (negative critic output)
        """
        return -critic(fake).mean()
