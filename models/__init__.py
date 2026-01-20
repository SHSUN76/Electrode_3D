"""
Models package for Electrode 3D Generator.

Contains:
- SliceGAN: 2D to 3D structure generation
- Segmentation models: U-Net 3D, Swin UNETR
"""

from models.slicegan import SliceGAN, Generator3D, Critic2D

__all__ = ["SliceGAN", "Generator3D", "Critic2D"]
