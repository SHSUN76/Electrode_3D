"""
Preprocessing module for Electrode 3D Generator.

Contains:
- Image loading and preprocessing
- Noise filtering
- Segmentation
- Data augmentation
"""

from preprocessing.image_processor import ImagePreprocessor
from preprocessing.augmentation import DataAugmentor

__all__ = ["ImagePreprocessor", "DataAugmentor"]
