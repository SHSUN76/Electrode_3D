"""
SliceGAN implementation for 2D to 3D microstructure generation.

Based on: Kench, S., Cooper, S.J. "Generating three-dimensional structures
from a two-dimensional slice with generative adversarial network-based
dimensionality expansion" Nature Machine Intelligence (2021)

GitHub: https://github.com/stke9/SliceGAN
"""

from models.slicegan.generator import Generator3D
from models.slicegan.discriminator import Critic2D
from models.slicegan.trainer import SliceGAN

__all__ = ["SliceGAN", "Generator3D", "Critic2D"]
