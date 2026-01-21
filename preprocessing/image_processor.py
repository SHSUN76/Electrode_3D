"""
Image preprocessing for electrode microstructure analysis.

Supports:
- TIFF sequence loading from micro-CT data
- Adaptive denoising with bilateral filtering
- Multi-phase segmentation with morphological operations
- Training slice extraction for SliceGAN
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    from skimage import morphology, restoration
    from skimage.filters import threshold_multiotsu
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from electrode_generator.config import PreprocessingConfig


def natural_sort_key(path: Path) -> list:
    """
    Generate a key for natural sorting of file paths.

    Handles filenames like: img_1.tif, img_2.tif, ..., img_10.tif

    Args:
        path: Path object to extract sort key from

    Returns:
        List of alternating text and integer parts for sorting
    """
    name = path.stem
    parts = re.split(r'(\d+)', name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


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
        method: str = "gaussian",
    ) -> np.ndarray:
        """
        Apply denoising filter.

        Args:
            image: Input image
            sigma: Filter strength parameter (uses config if None)
            method: Denoising method ("gaussian", "bilateral", "nlm")

        Returns:
            Denoised image
        """
        sigma = sigma or self.config.denoise_sigma

        if method == "gaussian":
            return ndimage.gaussian_filter(image.astype(float), sigma=sigma)
        elif method == "bilateral":
            return self.denoise_bilateral(image, sigma)
        elif method == "nlm":
            return self.denoise_nlm(image, sigma)
        else:
            raise ValueError(f"Unknown denoising method: {method}")

    def denoise_bilateral(
        self,
        image: np.ndarray,
        sigma_spatial: float = 3.0,
        sigma_color: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving denoising.

        Bilateral filtering smooths noise while preserving edges,
        which is ideal for micro-CT images with distinct phase boundaries.

        Args:
            image: Input image (2D or 3D)
            sigma_spatial: Spatial sigma (controls spatial extent of smoothing)
            sigma_color: Color/range sigma (controls edge sensitivity).
                        If None, estimated from image noise.

        Returns:
            Denoised image
        """
        image = image.astype(np.float64)

        # Normalize to [0, 1] for consistent processing
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = image

        # Estimate sigma_color from image noise if not provided
        if sigma_color is None:
            # Use MAD (Median Absolute Deviation) for robust noise estimation
            sigma_color = np.median(np.abs(normalized - np.median(normalized))) * 1.4826 * 3

        # Use OpenCV if available (faster)
        if HAS_CV2 and image.ndim == 2:
            normalized_8bit = (normalized * 255).astype(np.uint8)
            d = int(sigma_spatial * 2) | 1  # Ensure odd diameter
            denoised = cv2.bilateralFilter(
                normalized_8bit,
                d=d,
                sigmaColor=sigma_color * 255,
                sigmaSpace=sigma_spatial
            )
            result = denoised.astype(np.float64) / 255.0
        elif HAS_SKIMAGE:
            # Use scikit-image denoise_bilateral
            from skimage.restoration import denoise_bilateral as sk_bilateral
            result = sk_bilateral(
                normalized,
                sigma_color=sigma_color,
                sigma_spatial=sigma_spatial,
                channel_axis=None if image.ndim == 2 else -1
            )
        else:
            # Fallback to Gaussian if no advanced filter available
            result = ndimage.gaussian_filter(normalized, sigma=sigma_spatial)

        # Restore original range
        if img_max > img_min:
            return result * (img_max - img_min) + img_min
        return result

    def denoise_nlm(
        self,
        image: np.ndarray,
        h: float = 1.0,
        patch_size: int = 5,
        patch_distance: int = 6,
    ) -> np.ndarray:
        """
        Apply Non-Local Means denoising.

        NLM denoising is excellent for preserving texture details
        while removing noise.

        Args:
            image: Input image
            h: Filter strength (higher = more smoothing)
            patch_size: Size of patches for comparison
            patch_distance: Maximum distance for searching similar patches

        Returns:
            Denoised image
        """
        if not HAS_SKIMAGE:
            raise ImportError(
                "scikit-image is required for NLM denoising. "
                "Install with: pip install scikit-image"
            )

        image = image.astype(np.float64)

        # Normalize to [0, 1]
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = image

        # Estimate noise sigma
        sigma_est = np.median(np.abs(normalized - np.median(normalized))) * 1.4826

        # Apply NLM
        from skimage.restoration import denoise_nl_means, estimate_sigma

        if image.ndim == 2:
            denoised = denoise_nl_means(
                normalized,
                h=h * sigma_est,
                patch_size=patch_size,
                patch_distance=patch_distance,
                fast_mode=True,
                channel_axis=None
            )
        else:
            denoised = denoise_nl_means(
                normalized,
                h=h * sigma_est,
                patch_size=patch_size,
                patch_distance=patch_distance,
                fast_mode=True
            )

        # Restore original range
        if img_max > img_min:
            return denoised * (img_max - img_min) + img_min
        return denoised

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

    def segment_multiphase_improved(
        self,
        image: np.ndarray,
        n_phases: int = 3,
        denoise_first: bool = True,
        morphology_closing: bool = True,
        min_size: int = 64,
    ) -> np.ndarray:
        """
        Improved multi-phase segmentation with morphological post-processing.

        Combines Multi-Otsu thresholding with morphological operations
        to produce cleaner segmentation with fewer noise artifacts.

        Args:
            image: Input image (2D or 3D)
            n_phases: Number of phases to segment
            denoise_first: Apply bilateral denoising before segmentation
            morphology_closing: Apply morphological closing to fill holes
            min_size: Minimum region size (smaller regions are removed)

        Returns:
            Segmented image with integer labels (0 to n_phases-1)
        """
        if not HAS_SKIMAGE:
            raise ImportError(
                "scikit-image is required for improved segmentation. "
                "Install with: pip install scikit-image"
            )

        # Handle grayscale conversion for color images
        if image.ndim == 3 and image.shape[-1] in [3, 4]:
            image = np.mean(image, axis=-1)

        # Step 1: Denoise with bilateral filter for edge preservation
        if denoise_first:
            image = self.denoise_bilateral(image, sigma_spatial=2.0)

        # Step 2: Multi-Otsu thresholding
        thresholds = threshold_multiotsu(image, classes=n_phases)
        segmented = np.digitize(image, bins=thresholds).astype(np.uint8)

        # Step 3: Morphological closing to fill small holes in each phase
        if morphology_closing:
            from skimage.morphology import closing, disk, ball, remove_small_objects

            cleaned = np.zeros_like(segmented)

            for phase in range(n_phases):
                mask = (segmented == phase)

                if image.ndim == 2:
                    selem = disk(2)
                else:
                    selem = ball(2)

                closed = closing(mask, selem)

                # Remove small objects
                # max_size replaces deprecated min_size (removes objects <= max_size)
                if min_size > 0:
                    closed = remove_small_objects(closed, max_size=min_size - 1, connectivity=1)

                cleaned[closed] = phase

        else:
            cleaned = segmented

        return cleaned

    def segment_with_watershed(
        self,
        image: np.ndarray,
        markers: Optional[np.ndarray] = None,
        min_distance: int = 10,
    ) -> np.ndarray:
        """
        Segment image using watershed algorithm.

        Useful for separating touching particles in electrode structures.

        Args:
            image: Input image
            markers: Optional marker array (if None, generated automatically)
            min_distance: Minimum distance between markers

        Returns:
            Labeled segmentation
        """
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image is required for watershed segmentation")

        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from skimage.morphology import disk

        if image.ndim == 3 and image.shape[-1] in [3, 4]:
            image = np.mean(image, axis=-1)

        # Compute distance transform for marker detection
        from scipy import ndimage as ndi

        # Binarize first
        binary = image > np.median(image)
        distance = ndi.distance_transform_edt(binary)

        # Find local maxima as markers
        if markers is None:
            coords = peak_local_max(
                distance,
                min_distance=min_distance,
                footprint=disk(min_distance)
            )
            markers = np.zeros(distance.shape, dtype=int)
            for i, coord in enumerate(coords, start=1):
                markers[tuple(coord)] = i

        # Apply watershed
        labels = watershed(-distance, markers, mask=binary)

        return labels

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
    Process 3D image stacks (e.g., FIB-SEM, micro-CT data).

    Supports:
    - Loading TIFF sequences from directories with natural sorting
    - Multi-TIFF stack loading
    - Training slice extraction for SliceGAN
    - 3D preprocessing and alignment
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.image_processor = ImagePreprocessor(config)

    def load_stack(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load 3D image stack with natural sorting.

        Args:
            path: Path to stack file (TIFF) or directory of slices

        Returns:
            3D array (D, H, W)
        """
        path = Path(path)

        if path.is_dir():
            # Load from directory of slices with natural sorting
            return self.load_tiff_sequence(path)
        else:
            # Load from single TIFF stack
            if tifffile is None:
                raise ImportError("tifffile is required for TIFF stacks")
            return tifffile.imread(str(path))

    def load_tiff_sequence(
        self,
        folder_path: Union[str, Path],
        extensions: Optional[List[str]] = None,
        max_slices: Optional[int] = None,
    ) -> np.ndarray:
        """
        Load micro-CT TIFF sequence as 3D volume with natural sorting.

        Handles filenames with varying digit counts like:
        img_1.tif, img_2.tif, ..., img_10.tif, ..., img_100.tif

        Args:
            folder_path: Path to folder containing TIFF slices
            extensions: List of extensions to include (default: ['.tif', '.tiff', '.png'])
            max_slices: Maximum number of slices to load (None for all)

        Returns:
            3D numpy array (D, H, W) containing the volume
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not folder_path.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        extensions = extensions or ['.tif', '.tiff', '.png', '.TIF', '.TIFF', '.PNG']

        # Collect all matching files
        files = []
        for ext in extensions:
            files.extend(folder_path.glob(f"*{ext}"))

        if not files:
            raise ValueError(f"No image files found in {folder_path}")

        # Natural sort
        files = sorted(set(files), key=natural_sort_key)

        # Limit if specified
        if max_slices is not None:
            files = files[:max_slices]

        # Load slices
        slices = []
        for f in files:
            img = self.image_processor.load(f)
            slices.append(img)

        volume = np.stack(slices, axis=0)
        print(f"Loaded TIFF sequence: {volume.shape} from {len(files)} files")

        return volume

    def extract_training_slices(
        self,
        volume: np.ndarray,
        axis: str = 'z',
        num_slices: Optional[int] = None,
        stride: int = 1,
        random: bool = False,
    ) -> List[np.ndarray]:
        """
        Extract 2D slices from 3D volume for SliceGAN training.

        SliceGAN learns from 2D slices and generates 3D volumes
        that statistically match the input slices.

        Args:
            volume: 3D numpy array (D, H, W) or (D, H, W, C)
            axis: Axis to slice along ('x', 'y', or 'z')
            num_slices: Number of slices to extract (None for all)
            stride: Step between slices when extracting
            random: If True, randomly sample slices instead of uniform stride

        Returns:
            List of 2D numpy arrays
        """
        axis_map = {'z': 0, 'y': 1, 'x': 2}
        if axis.lower() not in axis_map:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got: {axis}")

        axis_idx = axis_map[axis.lower()]
        n_total = volume.shape[axis_idx]

        # Determine which indices to extract
        if random and num_slices:
            indices = np.random.choice(n_total, size=min(num_slices, n_total), replace=False)
            indices = sorted(indices)
        elif num_slices:
            # Evenly spaced slices
            indices = np.linspace(0, n_total - 1, num_slices, dtype=int)
        else:
            # All slices with stride
            indices = range(0, n_total, stride)

        # Extract slices
        slices = []
        for i in indices:
            if axis_idx == 0:
                slices.append(volume[i, :, :])
            elif axis_idx == 1:
                slices.append(volume[:, i, :])
            else:
                slices.append(volume[:, :, i])

        print(f"Extracted {len(slices)} slices along {axis}-axis from volume {volume.shape}")

        return slices

    def extract_orthogonal_slices(
        self,
        volume: np.ndarray,
        num_per_axis: int = 10,
    ) -> List[np.ndarray]:
        """
        Extract training slices from all three orthogonal axes.

        This provides more diverse training data for SliceGAN
        by sampling from XY, XZ, and YZ planes.

        Args:
            volume: 3D numpy array
            num_per_axis: Number of slices to extract per axis

        Returns:
            List of 2D numpy arrays from all axes
        """
        all_slices = []

        for axis in ['z', 'y', 'x']:
            slices = self.extract_training_slices(
                volume,
                axis=axis,
                num_slices=num_per_axis,
                random=True
            )
            all_slices.extend(slices)

        print(f"Extracted {len(all_slices)} total slices from all axes")
        return all_slices

    def volume_to_binary(
        self,
        volume: np.ndarray,
        threshold: Optional[float] = None,
        method: str = "otsu",
    ) -> np.ndarray:
        """
        Convert grayscale volume to binary (solid/pore) segmentation.

        Args:
            volume: 3D grayscale volume
            threshold: Manual threshold (if None, auto-computed)
            method: Thresholding method ("otsu", "mean", "median")

        Returns:
            Binary 3D array (0=pore, 1=solid)
        """
        if threshold is None:
            if method == "otsu":
                from skimage.filters import threshold_otsu
                threshold = threshold_otsu(volume)
            elif method == "mean":
                threshold = volume.mean()
            elif method == "median":
                threshold = np.median(volume)
            else:
                raise ValueError(f"Unknown method: {method}")

        binary = (volume > threshold).astype(np.uint8)
        porosity = 1.0 - binary.mean()
        print(f"Binarized volume with threshold={threshold:.2f}, porosity={porosity:.2%}")

        return binary

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
