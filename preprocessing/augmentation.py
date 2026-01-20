"""
Data augmentation for electrode microstructure images.
"""

from typing import List, Optional, Tuple
import numpy as np

from electrode_generator.config import PreprocessingConfig


class DataAugmentor:
    """
    Data augmentation pipeline for microstructure images.

    Supports:
    - Random rotation
    - Random flipping
    - Random cropping
    - Elastic deformation
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def rotate(
        self,
        image: np.ndarray,
        angle: Optional[float] = None,
    ) -> np.ndarray:
        """
        Random rotation.

        Args:
            image: Input image
            angle: Rotation angle in degrees (random if None)

        Returns:
            Rotated image
        """
        from scipy import ndimage

        if angle is None:
            angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)

        return ndimage.rotate(image, angle, reshape=False, mode="nearest")

    def flip(
        self,
        image: np.ndarray,
        horizontal: bool = True,
        vertical: bool = True,
    ) -> np.ndarray:
        """
        Random flipping.

        Args:
            image: Input image
            horizontal: Allow horizontal flip
            vertical: Allow vertical flip

        Returns:
            Flipped image
        """
        result = image.copy()

        if horizontal and np.random.random() > 0.5:
            result = np.flip(result, axis=-1)

        if vertical and np.random.random() > 0.5:
            result = np.flip(result, axis=-2)

        return result

    def crop(
        self,
        image: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Random crop.

        Args:
            image: Input image
            size: Crop size (H, W)

        Returns:
            Cropped image
        """
        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w = image.shape[-2:]

        crop_h, crop_w = size

        if h <= crop_h or w <= crop_w:
            return image

        y = np.random.randint(0, h - crop_h)
        x = np.random.randint(0, w - crop_w)

        if len(image.shape) == 2:
            return image[y:y + crop_h, x:x + crop_w]
        else:
            return image[..., y:y + crop_h, x:x + crop_w]

    def elastic_deform(
        self,
        image: np.ndarray,
        alpha: float = 36,
        sigma: float = 6,
    ) -> np.ndarray:
        """
        Elastic deformation for realistic microstructure variation.

        Args:
            image: Input image
            alpha: Deformation intensity
            sigma: Gaussian smoothing sigma

        Returns:
            Deformed image
        """
        from scipy.ndimage import map_coordinates, gaussian_filter

        shape = image.shape[-2:]

        # Random displacement fields
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            sigma, mode="constant", cval=0
        ) * alpha
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            sigma, mode="constant", cval=0
        ) * alpha

        # Create coordinate grid
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")

        # Apply deformation
        coords = [y + dy, x + dx]

        if len(image.shape) == 2:
            return map_coordinates(image, coords, order=1, mode="nearest")
        else:
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = map_coordinates(image[i], coords, order=1, mode="nearest")
            return result

    def augment(
        self,
        image: np.ndarray,
        n_augmented: int = 5,
    ) -> List[np.ndarray]:
        """
        Generate augmented versions of an image.

        Args:
            image: Input image
            n_augmented: Number of augmented images to generate

        Returns:
            List of augmented images (including original)
        """
        results = [image]  # Original

        for _ in range(n_augmented):
            aug = image.copy()

            # Random rotation
            if np.random.random() > 0.5:
                aug = self.rotate(aug)

            # Random flip
            if self.config.flip_horizontal or self.config.flip_vertical:
                aug = self.flip(
                    aug,
                    horizontal=self.config.flip_horizontal,
                    vertical=self.config.flip_vertical,
                )

            # Elastic deformation (less frequent)
            if np.random.random() > 0.7:
                aug = self.elastic_deform(aug)

            results.append(aug)

        return results

    def create_patches(
        self,
        image: np.ndarray,
        patch_size: int = 64,
        num_patches: int = 900,
        augment: bool = True,
    ) -> np.ndarray:
        """
        Create training patches from an image.

        Args:
            image: Input image (C, H, W) or (H, W)
            patch_size: Size of each patch
            num_patches: Total number of patches
            augment: Apply augmentation

        Returns:
            Array of patches (N, C, H, W)
        """
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]

        c, h, w = image.shape

        if h < patch_size or w < patch_size:
            raise ValueError(f"Image too small for patches. Got ({h}, {w}), need at least ({patch_size}, {patch_size})")

        patches = []

        for _ in range(num_patches):
            # Random position
            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)

            patch = image[:, y:y + patch_size, x:x + patch_size]

            # Augment
            if augment and self.config.augment:
                if np.random.random() > 0.5:
                    patch = self.rotate(patch)
                if self.config.flip_horizontal and np.random.random() > 0.5:
                    patch = np.flip(patch, axis=-1)
                if self.config.flip_vertical and np.random.random() > 0.5:
                    patch = np.flip(patch, axis=-2)

            patches.append(patch)

        return np.array(patches)
