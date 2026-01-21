"""
Preprocessing module for Electrode 3D Generator.

Contains:
- Image loading and preprocessing
- Noise filtering (Gaussian, Bilateral, NLM)
- Multi-phase segmentation
- TIFF sequence loading for micro-CT data
- Training slice extraction for SliceGAN
- Data augmentation
"""

from preprocessing.image_processor import (
    ImagePreprocessor,
    StackProcessor,
    natural_sort_key,
)
from preprocessing.augmentation import DataAugmentor

__all__ = [
    "ImagePreprocessor",
    "StackProcessor",
    "DataAugmentor",
    "natural_sort_key",
]
