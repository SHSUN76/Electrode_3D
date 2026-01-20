"""
Electrode 3D Generator
======================

A comprehensive toolkit for generating 3D battery electrode microstructures
using deep learning (SliceGAN) and integrating with simulation tools (COMSOL).

Main Components:
- SliceGAN: 2D to 3D structure generation
- Preprocessing: Image segmentation and data preparation
- Postprocessing: Mesh generation and refinement
- COMSOL Integration: Simulation workflow automation
"""

__version__ = "0.1.0"
__author__ = "SHSUN76"

from electrode_generator.core import ElectrodeGenerator
from electrode_generator.config import Config

__all__ = [
    "ElectrodeGenerator",
    "Config",
    "__version__",
]
