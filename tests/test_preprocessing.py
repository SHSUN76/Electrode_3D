"""
Tests for preprocessing module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from electrode_generator.config import PreprocessingConfig
from preprocessing.image_processor import ImagePreprocessor, StackProcessor
from preprocessing.augmentation import DataAugmentor


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        config = PreprocessingConfig(
            target_size=(128, 128),
            num_classes=3,
            denoise=True,
            denoise_sigma=1.0,
        )
        return ImagePreprocessor(config)

    @pytest.fixture
    def sample_image(self):
        """Create sample grayscale image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    def test_normalize(self, preprocessor, sample_image):
        """Test image normalization."""
        normalized = preprocessor.normalize(sample_image)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32

    def test_normalize_constant_image(self, preprocessor):
        """Test normalization of constant image."""
        constant_image = np.ones((100, 100), dtype=np.uint8) * 128
        normalized = preprocessor.normalize(constant_image)

        # Constant image should remain constant after normalization
        assert np.allclose(normalized, normalized[0, 0])

    def test_resize(self, preprocessor, sample_image):
        """Test image resizing."""
        resized = preprocessor.resize(sample_image, target_size=(64, 64))

        assert resized.shape == (64, 64)

    def test_resize_same_size(self, preprocessor, sample_image):
        """Test that same-size resize returns original."""
        resized = preprocessor.resize(sample_image, target_size=sample_image.shape[:2])

        np.testing.assert_array_equal(resized, sample_image)

    def test_denoise(self, preprocessor, sample_image):
        """Test image denoising."""
        # Add noise
        noisy = sample_image.astype(float) + np.random.normal(0, 25, sample_image.shape)
        noisy = np.clip(noisy, 0, 255)

        # Denoise
        denoised = preprocessor.denoise(noisy, sigma=2.0)

        # Denoised should be smoother (lower std)
        assert np.std(denoised) < np.std(noisy)

    def test_segment_threshold(self, preprocessor, sample_image):
        """Test threshold segmentation."""
        segmented = preprocessor.segment_phases(sample_image, num_classes=3, method="threshold")

        assert segmented.dtype == np.uint8
        assert set(np.unique(segmented)).issubset({0, 1, 2})

    def test_to_one_hot(self, preprocessor):
        """Test one-hot encoding."""
        segmented = np.array([[0, 1], [1, 2]], dtype=np.uint8)
        one_hot = preprocessor.to_one_hot(segmented, num_classes=3)

        assert one_hot.shape == (3, 2, 2)
        assert one_hot.sum(axis=0).all() == 1  # Sum along classes should be 1

    def test_load_png(self, preprocessor):
        """Test loading PNG image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test PNG
            img = Image.fromarray(np.random.randint(0, 256, (100, 100), dtype=np.uint8))
            path = Path(tmpdir) / "test.png"
            img.save(path)

            # Load
            loaded = preprocessor.load(path)

            assert loaded.shape == (100, 100)


class TestStackProcessor:
    """Tests for StackProcessor class."""

    @pytest.fixture
    def stack_processor(self):
        """Create stack processor instance."""
        config = PreprocessingConfig(
            denoise=True,
            normalize=True,
        )
        return StackProcessor(config)

    @pytest.fixture
    def sample_stack(self):
        """Create sample 3D stack."""
        np.random.seed(42)
        return np.random.randint(0, 256, (10, 64, 64), dtype=np.uint8)

    def test_preprocess_stack_shape(self, stack_processor, sample_stack):
        """Test that preprocessing preserves shape."""
        processed = stack_processor.preprocess_stack(sample_stack, segment=False)

        assert processed.shape == sample_stack.shape

    def test_preprocess_stack_with_segmentation(self, stack_processor, sample_stack):
        """Test preprocessing with segmentation."""
        processed = stack_processor.preprocess_stack(sample_stack, segment=True)

        assert processed.shape == sample_stack.shape
        assert processed.dtype == np.uint8

    def test_load_stack_from_directory(self, stack_processor):
        """Test loading stack from directory of slices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test slices
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8))
                img.save(tmpdir / f"slice_{i:03d}.png")

            # Load
            stack = stack_processor.load_stack(tmpdir)

            assert stack.shape == (5, 64, 64)


class TestDataAugmentor:
    """Tests for DataAugmentor class."""

    @pytest.fixture
    def augmentor(self):
        """Create augmentor instance."""
        config = PreprocessingConfig(
            augment=True,
            flip_horizontal=True,
            flip_vertical=True,
            rotation_range=45.0,
        )
        return DataAugmentor(config)

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        np.random.seed(42)
        return np.random.rand(64, 64).astype(np.float32)

    def test_rotate(self, augmentor, sample_image):
        """Test rotation augmentation."""
        rotated = augmentor.rotate(sample_image, angle=45)

        assert rotated.shape == sample_image.shape

    def test_flip_horizontal(self, augmentor, sample_image):
        """Test horizontal flip."""
        # Force horizontal flip
        np.random.seed(0)  # Seed that triggers flip
        flipped = augmentor.flip(sample_image, horizontal=True, vertical=False)

        # Either flipped or not, shape should be preserved
        assert flipped.shape == sample_image.shape

    def test_crop(self, augmentor, sample_image):
        """Test random crop."""
        cropped = augmentor.crop(sample_image, size=(32, 32))

        assert cropped.shape == (32, 32)

    def test_crop_too_small(self, augmentor, sample_image):
        """Test crop on image smaller than crop size."""
        small_image = sample_image[:30, :30]
        cropped = augmentor.crop(small_image, size=(64, 64))

        # Should return original if too small
        np.testing.assert_array_equal(cropped, small_image)

    def test_elastic_deform(self, augmentor, sample_image):
        """Test elastic deformation."""
        deformed = augmentor.elastic_deform(sample_image, alpha=10, sigma=3)

        assert deformed.shape == sample_image.shape
        # Should be different from original
        assert not np.allclose(deformed, sample_image)

    def test_augment_batch(self, augmentor, sample_image):
        """Test batch augmentation."""
        augmented = augmentor.augment(sample_image, n_augmented=5)

        # Should return original + 5 augmented
        assert len(augmented) == 6
        # First should be original
        np.testing.assert_array_equal(augmented[0], sample_image)

    def test_create_patches(self, augmentor):
        """Test patch creation."""
        # Create larger image
        image = np.random.rand(256, 256).astype(np.float32)

        patches = augmentor.create_patches(image, patch_size=32, num_patches=100, augment=False)

        assert patches.shape == (100, 1, 32, 32)

    def test_create_patches_with_channels(self, augmentor):
        """Test patch creation with channel dimension."""
        image = np.random.rand(3, 256, 256).astype(np.float32)

        patches = augmentor.create_patches(image, patch_size=32, num_patches=50)

        assert patches.shape == (50, 3, 32, 32)

    def test_create_patches_image_too_small(self, augmentor):
        """Test patch creation fails on small image."""
        small_image = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="Image too small"):
            augmentor.create_patches(small_image, patch_size=32, num_patches=10)


class TestAdvancedPreprocessing:
    """Tests for new advanced preprocessing features."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        config = PreprocessingConfig()
        return ImagePreprocessor(config)

    @pytest.fixture
    def sample_image(self):
        """Create sample grayscale image with distinct regions."""
        np.random.seed(42)
        image = np.zeros((256, 256), dtype=np.float32)
        # Create three distinct regions
        image[:85, :] = 0.2 + np.random.rand(85, 256) * 0.1
        image[85:170, :] = 0.5 + np.random.rand(85, 256) * 0.1
        image[170:, :] = 0.8 + np.random.rand(86, 256) * 0.1
        return image

    def test_denoise_bilateral(self, preprocessor, sample_image):
        """Test bilateral denoising."""
        # Add noise
        noisy = sample_image + np.random.normal(0, 0.1, sample_image.shape)

        # Denoise
        denoised = preprocessor.denoise_bilateral(noisy, sigma_spatial=3.0)

        assert denoised.shape == sample_image.shape
        # Denoised should be smoother
        assert np.std(denoised) < np.std(noisy)

    def test_denoise_method_selection(self, preprocessor, sample_image):
        """Test denoising method selection."""
        # Gaussian
        gaussian = preprocessor.denoise(sample_image, sigma=1.0, method="gaussian")
        assert gaussian.shape == sample_image.shape

        # Bilateral
        bilateral = preprocessor.denoise(sample_image, sigma=1.0, method="bilateral")
        assert bilateral.shape == sample_image.shape

    def test_segment_multiphase_improved(self, preprocessor, sample_image):
        """Test improved multi-phase segmentation."""
        try:
            segmented = preprocessor.segment_multiphase_improved(
                sample_image,
                n_phases=3,
                denoise_first=True,
                morphology_closing=True,
            )

            assert segmented.shape == sample_image.shape
            assert segmented.dtype == np.uint8
            unique_phases = np.unique(segmented)
            assert len(unique_phases) <= 3
        except ImportError:
            pytest.skip("scikit-image not available")


class TestTIFFSequenceLoader:
    """Tests for TIFF sequence loading."""

    @pytest.fixture
    def stack_processor(self):
        """Create stack processor instance."""
        config = PreprocessingConfig()
        return StackProcessor(config)

    def test_natural_sort_key(self):
        """Test natural sorting of filenames."""
        from preprocessing.image_processor import natural_sort_key

        paths = [
            Path("img_1.tif"),
            Path("img_10.tif"),
            Path("img_2.tif"),
            Path("img_20.tif"),
        ]

        sorted_paths = sorted(paths, key=natural_sort_key)
        expected = ["img_1.tif", "img_2.tif", "img_10.tif", "img_20.tif"]

        assert [p.name for p in sorted_paths] == expected

    def test_load_tiff_sequence(self, stack_processor):
        """Test loading TIFF sequence from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test TIFF sequence with non-sequential numbers
            for i in [1, 2, 10, 20]:
                img = Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8))
                img.save(tmpdir / f"slice_{i}.tif")

            # Load with natural sorting
            volume = stack_processor.load_tiff_sequence(tmpdir)

            # Should have 4 slices
            assert volume.shape[0] == 4
            assert volume.shape[1:] == (64, 64)

    def test_load_tiff_sequence_max_slices(self, stack_processor):
        """Test loading limited number of slices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 10 slices
            for i in range(10):
                img = Image.fromarray(np.random.randint(0, 256, (32, 32), dtype=np.uint8))
                img.save(tmpdir / f"slice_{i:03d}.tif")

            # Load only 5
            volume = stack_processor.load_tiff_sequence(tmpdir, max_slices=5)

            assert volume.shape[0] == 5

    def test_extract_training_slices(self, stack_processor):
        """Test extracting training slices from 3D volume."""
        volume = np.random.randint(0, 256, (20, 64, 64), dtype=np.uint8)

        # Extract z-slices
        slices_z = stack_processor.extract_training_slices(volume, axis='z', num_slices=10)
        assert len(slices_z) == 10
        assert slices_z[0].shape == (64, 64)

        # Extract y-slices
        slices_y = stack_processor.extract_training_slices(volume, axis='y', num_slices=5)
        assert len(slices_y) == 5
        assert slices_y[0].shape == (20, 64)

    def test_extract_orthogonal_slices(self, stack_processor):
        """Test extracting slices from all axes."""
        volume = np.random.randint(0, 256, (30, 40, 50), dtype=np.uint8)

        slices = stack_processor.extract_orthogonal_slices(volume, num_per_axis=5)

        # Should have 5 slices per axis = 15 total
        assert len(slices) == 15

    def test_volume_to_binary(self, stack_processor):
        """Test volume binarization."""
        volume = np.random.randint(0, 256, (10, 32, 32), dtype=np.uint8)

        binary = stack_processor.volume_to_binary(volume, method="otsu")

        assert binary.shape == volume.shape
        assert binary.dtype == np.uint8
        assert set(np.unique(binary)) == {0, 1}
