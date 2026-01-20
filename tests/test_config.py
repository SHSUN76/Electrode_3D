"""
Tests for configuration module.
"""

import tempfile
from pathlib import Path

import pytest

from electrode_generator.config import (
    Config,
    SliceGANConfig,
    PreprocessingConfig,
    PostprocessingConfig,
    COMSOLConfig,
)


class TestSliceGANConfig:
    """Tests for SliceGAN configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SliceGANConfig()

        assert config.nz == 512
        assert config.ngf == 64
        assert config.ndf == 64
        assert config.nc == 3
        assert config.img_size == 64
        assert config.batch_size == 8
        assert config.lr_g == 0.0001
        assert config.lr_d == 0.0001
        assert config.lambda_gp == 10.0
        assert config.critic_iters == 5
        assert config.num_epochs == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SliceGANConfig(
            nz=128,
            ngf=128,
            num_epochs=200,
        )

        assert config.nz == 128
        assert config.ngf == 128
        assert config.num_epochs == 200


class TestPreprocessingConfig:
    """Tests for preprocessing configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PreprocessingConfig()

        assert config.target_size == (256, 256)
        assert config.num_classes == 3
        assert config.denoise is True
        assert config.normalize is True

    def test_augmentation_settings(self):
        """Test augmentation settings."""
        config = PreprocessingConfig(
            augment=True,
            flip_horizontal=True,
            flip_vertical=False,
            rotation_range=30.0,
        )

        assert config.augment is True
        assert config.flip_horizontal is True
        assert config.flip_vertical is False
        assert config.rotation_range == 30.0


class TestPostprocessingConfig:
    """Tests for postprocessing configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PostprocessingConfig()

        assert config.voxel_size == 0.1
        assert config.level == 0.5
        assert config.step_size == 1
        assert "stl" in config.export_formats

    def test_smoothing_settings(self):
        """Test smoothing settings."""
        config = PostprocessingConfig(
            smooth_iterations=5,
            smooth_factor=0.3,
        )

        assert config.smooth_iterations == 5
        assert config.smooth_factor == 0.3


class TestCOMSOLConfig:
    """Tests for COMSOL configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = COMSOLConfig()

        assert config.cores == 4
        assert config.physics_type == "porous_electrode"
        assert config.mesh_size == "normal"

    def test_custom_values(self):
        """Test custom COMSOL values."""
        config = COMSOLConfig(
            cores=8,
            physics_type="lithium_ion",
        )

        assert config.cores == 8
        assert config.physics_type == "lithium_ion"


class TestConfig:
    """Tests for main configuration class."""

    def test_default_subconfigs(self):
        """Test that default subconfigs are created."""
        config = Config()

        assert config.slicegan is not None
        assert config.preprocessing is not None
        assert config.postprocessing is not None
        assert config.comsol is not None

    def test_yaml_save_load(self):
        """Test YAML save and load functionality."""
        config = Config(
            slicegan=SliceGANConfig(num_epochs=50),
            preprocessing=PreprocessingConfig(num_classes=4),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"

            # Save
            config.to_yaml(yaml_path)
            assert yaml_path.exists()

            # Load
            loaded_config = Config.from_yaml(yaml_path)

            assert loaded_config.slicegan.num_epochs == 50
            assert loaded_config.preprocessing.num_classes == 4

    def test_yaml_with_invalid_path(self):
        """Test loading YAML from non-existent path."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path/config.yaml")

    def test_ensure_dirs(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config = Config(
                data_dir=tmpdir / "data",
                output_dir=tmpdir / "output",
                checkpoint_dir=tmpdir / "checkpoints",
            )

            config.ensure_dirs()

            assert (tmpdir / "data" / "raw").exists()
            assert (tmpdir / "data" / "processed").exists()
            assert (tmpdir / "data" / "generated").exists()
            assert (tmpdir / "output").exists()
            assert (tmpdir / "checkpoints").exists()
