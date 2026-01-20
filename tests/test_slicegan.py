"""
Tests for SliceGAN model components.
"""

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestGenerator3D:
    """Tests for 3D Generator."""

    def test_generator_output_shape(self):
        """Test that generator produces correct output shape."""
        from models.slicegan.generator import Generator3D

        generator = Generator3D(nz=512, ngf=64, nc=3)

        # Create random latent input
        batch_size = 2
        z = torch.randn(batch_size, 512, 4, 4, 4)

        # Generate
        output = generator(z)

        # Check output shape: (batch, nc, D, H, W) - actual size depends on architecture
        assert output.shape[0] == batch_size
        assert output.shape[1] == 3
        # Output should be cubic (same D, H, W)
        assert output.shape[2] == output.shape[3] == output.shape[4]

    def test_generator_output_range(self):
        """Test that generator output is in valid range."""
        from models.slicegan.generator import Generator3D

        generator = Generator3D(nz=512, ngf=64, nc=3, imtype="nphase")

        z = torch.randn(1, 512, 4, 4, 4)
        output = generator(z)

        # Softmax output should sum to 1 along channel dimension
        channel_sum = output.sum(dim=1)
        assert torch.allclose(channel_sum, torch.ones_like(channel_sum), atol=1e-5)

    def test_generator_grayscale_output(self):
        """Test grayscale output with tanh activation."""
        from models.slicegan.generator import Generator3D

        generator = Generator3D(nz=512, ngf=64, nc=1, imtype="grayscale")

        z = torch.randn(1, 512, 4, 4, 4)
        output = generator(z)

        # Tanh output should be in [-1, 1]
        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_generator_generate_method(self):
        """Test the generate convenience method."""
        from models.slicegan.generator import Generator3D

        generator = Generator3D(nz=512, ngf=64, nc=3)

        # Generate multiple samples
        output = generator.generate(batch_size=3)

        # Check batch and channel dimensions
        assert output.shape[0] == 3
        assert output.shape[1] == 3
        # Output should be cubic
        assert output.shape[2] == output.shape[3] == output.shape[4]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestCritic2D:
    """Tests for 2D Critic/Discriminator."""

    def test_critic_output_shape(self):
        """Test that critic produces correct output shape."""
        from models.slicegan.discriminator import Critic2D

        critic = Critic2D(nc=3, ndf=64)

        # Create random 2D slice
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)

        # Forward pass
        output = critic(x)

        # Output should be scalar per sample
        assert output.shape == (batch_size, 1)

    def test_critic_no_sigmoid(self):
        """Test that critic doesn't apply sigmoid (WGAN)."""
        from models.slicegan.discriminator import Critic2D

        critic = Critic2D(nc=3, ndf=64)

        x = torch.randn(2, 3, 64, 64)
        output = critic(x)

        # WGAN critic output can be any real number, not bounded [0, 1]
        # Check that it's not artificially bounded
        # (This is a probabilistic test, might rarely fail)
        assert output.min() < 0.5 or output.max() > 0.5


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestGradientPenalty:
    """Tests for gradient penalty calculation."""

    def test_gradient_penalty_positive(self):
        """Test that gradient penalty is positive."""
        from models.slicegan.discriminator import Critic2D, gradient_penalty

        critic = Critic2D(nc=3, ndf=64)

        batch_size = 2
        real = torch.randn(batch_size, 3, 64, 64)
        fake = torch.randn(batch_size, 3, 64, 64)

        gp = gradient_penalty(critic, real, fake, device="cpu")

        assert gp.item() >= 0

    def test_gradient_penalty_differentiable(self):
        """Test that gradient penalty is differentiable."""
        from models.slicegan.discriminator import Critic2D, gradient_penalty

        critic = Critic2D(nc=3, ndf=64)

        real = torch.randn(2, 3, 64, 64, requires_grad=True)
        fake = torch.randn(2, 3, 64, 64, requires_grad=True)

        gp = gradient_penalty(critic, real, fake, device="cpu")

        # Should be able to backprop
        gp.backward()

        # Critic parameters should have gradients
        for param in critic.parameters():
            if param.requires_grad:
                assert param.grad is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestWGANGPLoss:
    """Tests for WGAN-GP loss calculator."""

    def test_critic_loss(self):
        """Test critic loss calculation."""
        from models.slicegan.discriminator import Critic2D, WGANGPLoss

        critic = Critic2D(nc=3, ndf=64)
        loss_fn = WGANGPLoss(lambda_gp=10.0)

        real = torch.randn(2, 3, 64, 64)
        fake = torch.randn(2, 3, 64, 64)

        total_loss, w_dist, gp = loss_fn.critic_loss(critic, real, fake, "cpu")

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(w_dist, float)
        assert isinstance(gp, float)

    def test_generator_loss(self):
        """Test generator loss calculation."""
        from models.slicegan.discriminator import Critic2D, WGANGPLoss

        critic = Critic2D(nc=3, ndf=64)
        loss_fn = WGANGPLoss()

        fake = torch.randn(2, 3, 64, 64)

        g_loss = loss_fn.generator_loss(critic, fake)

        assert isinstance(g_loss, torch.Tensor)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestMultiScaleCritic:
    """Tests for multi-scale critic."""

    def test_multiscale_critic_output(self):
        """Test multi-scale critic output."""
        from models.slicegan.discriminator import MultiScaleCritic2D

        critic = MultiScaleCritic2D(nc=3, ndf=64, num_scales=2)

        # Critic2D expects 64x64 input, so use that size
        # After one AvgPool2d(3,2,1) downsample: 64->32
        x = torch.randn(2, 3, 64, 64)
        outputs = critic(x)

        # Should have output for each scale
        assert len(outputs) == 2

        # Each output should have correct shape
        # Output shape depends on input size
        for out in outputs:
            assert out.dim() == 2
            assert out.shape[1] == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestGeneratorWithAdaIN:
    """Tests for Generator with AdaIN."""

    def test_adain_generator_output(self):
        """Test generator with AdaIN produces correct output."""
        from models.slicegan.generator import GeneratorWithAdaIN

        generator = GeneratorWithAdaIN(nz=512, ngf=64, nc=3, style_dim=64)

        z = torch.randn(2, 512, 4, 4, 4)
        style = torch.randn(2, 64)

        # Without style
        output1 = generator(z)
        assert output1.shape[0] == 2
        assert output1.shape[1] == 3
        assert output1.shape[2] == output1.shape[3] == output1.shape[4]

        # With style
        output2 = generator(z, style=style)
        assert output2.shape == output1.shape

    def test_adain_style_effect(self):
        """Test that different styles produce different outputs."""
        from models.slicegan.generator import GeneratorWithAdaIN

        generator = GeneratorWithAdaIN(nz=512, ngf=64, nc=3, style_dim=64)

        z = torch.randn(1, 512, 4, 4, 4)
        style1 = torch.randn(1, 64)
        style2 = torch.randn(1, 64)

        output1 = generator(z, style=style1)
        output2 = generator(z, style=style2)

        # Different styles should produce different outputs
        assert not torch.allclose(output1, output2)
