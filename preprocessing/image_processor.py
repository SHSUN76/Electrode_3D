"""
Image preprocessing for electrode microstructure analysis.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import tifffile
except ImportError:
    tifffile = None

from electrode_generator.config import PreprocessingConfig


class ImagePreprocessor:
    """
    Image preprocessing pipeline for electrode microstructures.

    Handles:
    - Loading various image formats (PNG, TIFF, etc.)
    - Noise filtering
    - Normalization
    - Phase segmentation

    Args:
        config: Preprocessing configuration
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def load(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file.

        Args:
            path: Path to image file

        Returns:
            Image as numpy array
        """
        path = Path(path)

        if path.suffix.lower() in [".tif", ".tiff"]:
            if tifffile is None:
                raise ImportError("tifffile is required for TIFF files. Install with: pip install tifffile")
            return tifffile.imread(str(path))
        else:
            return np.array(Image.open(path))

    def denoise(
        self,
        image: np.ndarray,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply Gaussian denoising.

        Args:
            image: Input image
            sigma: Gaussian sigma (uses config if None)

        Returns:
            Denoised image
        """
        sigma = sigma or self.config.denoise_sigma
        return ndimage.gaussian_filter(image.astype(float), sigma=sigma)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        image = image.astype(np.float32)
        min_val = image.min()
        max_val = image.max()

        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image

    def resize(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image
            target_size: Target (height, width)

        Returns:
            Resized image
        """
        target_size = target_size or self.config.target_size

        if image.shape[:2] == target_size:
            return image

        pil_image = Image.fromarray(image)
        resized = pil_image.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
        return np.array(resized)

    def segment_phases(
        self,
        image: np.ndarray,
        num_classes: Optional[int] = None,
        method: str = "threshold",
    ) -> np.ndarray:
        """
        Segment image into phases.

        Args:
            image: Input image
            num_classes: Number of phases
            method: Segmentation method ("threshold", "kmeans", "otsu")

        Returns:
            Segmented image with integer labels
        """
        num_classes = num_classes or self.config.num_classes

        if method == "threshold":
            return self._segment_threshold(image, num_classes)
        elif method == "kmeans":
            return self._segment_kmeans(image, num_classes)
        elif method == "otsu":
            return self._segment_otsu(image, num_classes)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    def _segment_threshold(
        self,
        image: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        """Multi-threshold segmentation."""
        # Normalize to [0, 1]
        if len(image.shape) == 3:
            image = np.mean(image, axis=-1)

        image = self.normalize(image)

        # Calculate thresholds
        thresholds = np.linspace(0, 1, num_classes + 1)[1:-1]

        # Apply thresholds
        segmented = np.zeros_like(image, dtype=np.uint8)
        for i, thresh in enumerate(thresholds):
            segmented[image > thresh] = i + 1

        return segmented

    def _segment_kmeans(
        self,
        image: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        """K-means segmentation."""
        from sklearn.cluster import KMeans

        # Flatten image
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[-1])
        else:
            pixels = image.reshape(-1, 1)

        # K-means clustering
        kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)

        # Reshape back
        return labels.reshape(image.shape[:2]).astype(np.uint8)

    def _segment_otsu(
        self,
        image: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        """Multi-Otsu thresholding."""
        from skimage.filters import threshold_multiotsu

        if len(image.shape) == 3:
            image = np.mean(image, axis=-1)

        # Calculate Otsu thresholds
        thresholds = threshold_multiotsu(image, classes=num_classes)

        # Apply thresholds
        return np.digitize(image, bins=thresholds).astype(np.uint8)

    def to_one_hot(
        self,
        segmented: np.ndarray,
        num_classes: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert segmented image to one-hot encoding.

        Args:
            segmented: Segmented image with integer labels
            num_classes: Number of classes

        Returns:
            One-hot encoded array (num_classes, H, W)
        """
        if num_classes is None:
            num_classes = int(segmented.max()) + 1

        one_hot = np.zeros((num_classes,) + segmented.shape, dtype=np.float32)

        for i in range(num_classes):
            one_hot[i] = (segmented == i).astype(np.float32)

        return one_hot

    def load_and_preprocess(
        self,
        path: Union[str, Path],
        segment: bool = True,
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.

        Args:
            path: Path to image file
            segment: Whether to segment into phases

        Returns:
            Preprocessed image
        """
        # Load
        image = self.load(path)

        # Denoise
        if self.config.denoise:
            image = self.denoise(image)

        # Resize
        image = self.resize(image)

        # Normalize
        if self.config.normalize:
            image = self.normalize(image)

        # Segment
        if segment:
            image = self.segment_phases(image)

        return image


class StackProcessor:
    """
    Process 3D image stacks (e.g., FIB-SEM data).
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.image_processor = ImagePreprocessor(config)

    def load_stack(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load 3D image stack.

        Args:
            path: Path to stack file (TIFF) or directory of slices

        Returns:
            3D array (D, H, W)
        """
        path = Path(path)

        if path.is_dir():
            # Load from directory of slices
            files = sorted(path.glob("*.tif")) + sorted(path.glob("*.png"))
            slices = [self.image_processor.load(f) for f in files]
            return np.stack(slices, axis=0)
        else:
            # Load from single TIFF stack
            if tifffile is None:
                raise ImportError("tifffile is required for TIFF stacks")
            return tifffile.imread(str(path))

    def preprocess_stack(
        self,
        stack: np.ndarray,
        segment: bool = True,
    ) -> np.ndarray:
        """
        Preprocess 3D stack.

        Args:
            stack: 3D array (D, H, W)
            segment: Whether to segment

        Returns:
            Preprocessed 3D array
        """
        processed = []

        for i in range(stack.shape[0]):
            slice_2d = stack[i]

            # Denoise
            if self.config.denoise:
                slice_2d = self.image_processor.denoise(slice_2d)

            # Normalize
            if self.config.normalize:
                slice_2d = self.image_processor.normalize(slice_2d)

            # Segment
            if segment:
                slice_2d = self.image_processor.segment_phases(slice_2d)

            processed.append(slice_2d)

        return np.stack(processed, axis=0)

    def align_stack(
        self,
        stack: np.ndarray,
        method: str = "phase_correlation",
    ) -> np.ndarray:
        """
        Align 3D stack to correct for drift.

        Args:
            stack: 3D array (D, H, W)
            method: Alignment method

        Returns:
            Aligned stack
        """
        from skimage.registration import phase_cross_correlation

        aligned = np.zeros_like(stack)
        aligned[0] = stack[0]  # Reference slice

        for i in range(1, stack.shape[0]):
            # Calculate shift
            shift, _, _ = phase_cross_correlation(stack[0], stack[i])

            # Apply shift
            aligned[i] = ndimage.shift(stack[i], shift)

        return aligned
